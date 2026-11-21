#!/usr/bin/env python3
"""
COLESION-AI // Clinical Evaluation Study Webserver (ui-eval)
- Serves the Formative Evaluation Workplace (index.html, css/, js/)
- /api/study/cases: Delivers 4 Tutorial Cases + 24 Curated Evaluation Cases
- /api/study/save: Persists full trial telemetry & keystroke logs to results/<run_id>.json
- /api/image/<image_id>.jpg: Streams original high-res dermoscopy images directly from data/build
- /api/heatmap/<image_id>.png: Streams Grad-CAM++ heatmaps with on-the-fly RGBA transparency
Zero disk duplication. Pure standard library + numpy/PIL/polars.
"""

import io
import json
import logging
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import polars as pl
from matplotlib import cm
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("colesion.eval.server")

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratosis (akiec)",
    "bcc": "Basal Cell Carcinoma (bcc)",
    "bkl": "Benign Keratosis (bkl)",
    "df": "Dermatofibroma (df)",
    "mel": "Melanoma (mel)",
    "nv": "Melanocytic Nevus (nv)",
    "vasc": "Vascular Lesion (vasc)",
}
CLASS_INDICES = {c: i for i, c in enumerate(CLASSES)}

# 4 Onboarding Tutorial Cases with Educational Debrief Content
TUTORIAL_DEFINITIONS = [
    {
        "image_id": "ISIC_0034533",
        "tutorial_title": "Tutorial 1/4: Routine Concordance (Fast-Path)",
        "scenario": "Routine Benign Concordance",
        "learning_goal": "ML Pred Correct: High confidence and agreement. Observe green status and rapid sign-off.",
        "explanation": "Pathology: Benign Melanocytic Nevus (nv). The AI correctly predicts Nevus with 99.1% confidence (margin: 98.3%). When the human expert and AI unequivocally agree on classic dermoscopic features, clinicians can use the efficient fast-path.",
    },
    {
        "image_id": "ISIC_0034584",
        "tutorial_title": "Tutorial 2/4: Critical AI Error (Automation Bias Trap!)",
        "scenario": "Melanoma False Negative",
        "learning_goal": "ML Pred Wrong: Dangerous AI failure! Practice overriding the AI when human inspection detects malignancy.",
        "explanation": "Pathology: Malignant Melanoma (mel). The AI erroneously predicts a benign Nevus with 73.5% confidence! Never accept AI predictions blindly. If you see atypical pigment structures or the Precedent Twins do not corroborate the AI claim, execute an Appropriate Override.",
    },
    {
        "image_id": "ISIC_0034527",
        "tutorial_title": "Tutorial 3/4: Close Rivalry (Differential Dilemma)",
        "scenario": "Narrow Confidence Margin",
        "learning_goal": "ML Close Rivalry: Confidence margin is tight (Δ < 5%). Yellow alert triggers. Use Precedent Twins (Hold W) to contrast.",
        "explanation": "Pathology: Dysplastic Nevus (nv). The model is split between Nevus (35.5%) and Benign Keratosis (31.7%), yielding an uncertain margin of Δ = 3.8%. When this yellow warning appears, compare the top-1 twin against the rival twin (Hold W) to differentiate.",
    },
    {
        "image_id": "ISIC_0034814",
        "tutorial_title": "Tutorial 4/4: Out-of-Distribution (OOD Refusal Gate)",
        "scenario": "High Manifold Distance / Artifact",
        "learning_goal": "ML OOD Gate: Manifold distance d_10 exceeds tau_99. Red alert locks AI. Select Option 8 (Inconclusive / Biopsy).",
        "explanation": "Pathology: Vascular lesion with atypical presentation. The embedding manifold distance (d_10 = 0.360) exceeds the tau_99 safety threshold (0.302). The system flags high epistemic uncertainty and refuses to provide a forecast. In this state, the rational clinical choice is Option 8 (Inconclusive / Biopsy referral).",
    },
]

# 24 Curated Study Cases Balanced Across All 7 Diagnoses + OOD + Cognitive Phenotypes
STUDY_CASE_DEFINITIONS = [
    {"image_id": "ISIC_0035900", "case_num": 1, "role": "Mel Routine Malign Concordance", "ablation": "full"},
    {"image_id": "ISIC_0035680", "case_num": 2, "role": "Critical AI False Negative (Over-Reliance Test)", "ablation": "full"},
    {"image_id": "ISIC_0035462", "case_num": 3, "role": "Subtle Melanoma (AI Synergy Aid)", "ablation": "full"},
    {"image_id": "ISIC_0035877", "case_num": 4, "role": "Extreme Close Rivalry (Mel vs NV, Δ = 0.2%)", "ablation": "no_heatmap"},
    {"image_id": "ISIC_0036005", "case_num": 5, "role": "Slice Risk Lower Extremity / Foot", "ablation": "full"},
    {"image_id": "ISIC_0034946", "case_num": 6, "role": "Routine Basal Cell Carcinoma Concordance", "ablation": "full"},
    {"image_id": "ISIC_0035310", "case_num": 7, "role": "Difficult Facial BCC (AI Aid)", "ablation": "full"},
    {"image_id": "ISIC_0036006", "case_num": 8, "role": "OOD AI Misclassification (BCC called DF)", "ablation": "full"},
    {"image_id": "ISIC_0035594", "case_num": 9, "role": "BCC vs AKIEC Close Rivalry (Δ = 1.4%)", "ablation": "no_twins"},
    {"image_id": "ISIC_0035001", "case_num": 10, "role": "Facial Actinic Keratosis (AI Aid)", "ablation": "full"},
    {"image_id": "ISIC_0034748", "case_num": 11, "role": "AKIEC vs BCC & BKL Ambiguity", "ablation": "full"},
    {"image_id": "ISIC_0035553", "case_num": 12, "role": "Pigmented AKIEC (AI False Positive Mel)", "ablation": "full"},
    {"image_id": "ISIC_0035530", "case_num": 13, "role": "Routine Benign Nevus Concordance", "ablation": "full"},
    {"image_id": "ISIC_0035838", "case_num": 14, "role": "Overcautious AI False Alarm (NV called Mel)", "ablation": "full"},
    {"image_id": "ISIC_0035197", "case_num": 15, "role": "Dysplastic Nevus Rivalry (NV vs Mel, Δ = 0.2%)", "ablation": "no_heatmap"},
    {"image_id": "ISIC_0035632", "case_num": 16, "role": "Atypical Nevus in Elderly Patient (Slice)", "ablation": "full"},
    {"image_id": "ISIC_0034794", "case_num": 17, "role": "Routine Seborrheic Keratosis (BKL)", "ablation": "full"},
    {"image_id": "ISIC_0035109", "case_num": 18, "role": "Pigmented BKL Mimicking Melanoma (AI Error)", "ablation": "full"},
    {"image_id": "ISIC_0035403", "case_num": 19, "role": "High Ambiguity BKL vs AKIEC (0% Crowd Acc)", "ablation": "no_twins"},
    {"image_id": "ISIC_0036007", "case_num": 20, "role": "Routine Dermatofibroma Concordance", "ablation": "full"},
    {"image_id": "ISIC_0035552", "case_num": 21, "role": "Atypical DF (AI Aid in Difficult Presentation)", "ablation": "full"},
    {"image_id": "ISIC_0034911", "case_num": 22, "role": "Routine Vascular Angioma Concordance", "ablation": "full"},
    {"image_id": "ISIC_0034858", "case_num": 23, "role": "Thrombosed Vascular Lesion (AI Error BCC)", "ablation": "full"},
    {"image_id": "ISIC_0035623", "case_num": 24, "role": "Extreme Manifold Distance OOD Artifact (Option 8)", "ablation": "full"},
]


def build_turbo_lut():
    turbo_cmap = cm.turbo(np.linspace(0, 1, 256))[:, :3] * 255.0
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, :] = np.clip(turbo_cmap, 0, 255).astype(np.uint8)
    return lut


TURBO_LUT = build_turbo_lut()


def load_env(env_path: Path) -> dict:
    config = {}
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip().strip('"').strip("'")
    return config


class StudyServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server, directory=str(server.ui_dir))

    def do_HEAD(self):
        self.do_GET(is_head=True)

    def do_GET(self, is_head=False):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return super().do_GET()

        # API: Evaluation study cases (4 tutorial + 24 evaluation cases)
        if path == "/api/study/cases":
            self.handle_api_study_cases(is_head=is_head)
            return

        # API: All explorer cases (fallback / preview)
        if path == "/api/cases" or path == "/data/cases.json":
            self.handle_api_cases(query, is_head=is_head)
            return

        # API: Global ML Metrics
        if path == "/api/metrics":
            self.handle_api_metrics(is_head=is_head)
            return

        # API: Stream original dermoscopy image
        if path.startswith("/api/image/"):
            img_id = path[len("/api/image/") :].split(".")[0]
            self.handle_stream_image(img_id, is_head=is_head)
            return

        # API: Stream Grad-CAM heatmap with transparent RGBA
        if path.startswith("/api/heatmap/"):
            img_id = path[len("/api/heatmap/") :].split(".")[0]
            self.handle_stream_heatmap(img_id, is_head=is_head)
            return

        # Static assets
        if is_head:
            return super().do_HEAD()
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # API: Save complete study telemetry JSON
        if path == "/api/study/save":
            self.handle_api_study_save()
            return

        self.send_error(404, "Endpoint not found")

    def handle_api_study_cases(self, is_head=False):
        payload = self.server.get_study_cases_payload()
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if not is_head:
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def handle_api_study_save(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8"))

            run_id = payload.get("run_id") or payload.get("participant_id") or "unnamed_run"
            # Sanitize run_id for filename
            clean_run_id = "".join(c for c in run_id if c.isalnum() or c in ("-", "_")).strip()
            if not clean_run_id:
                clean_run_id = "eval_run"

            results_dir = self.server.ui_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            out_file = results_dir / f"colesion_study_{clean_run_id}.json"

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            log.info("Successfully persisted study telemetry to %s", out_file)
            resp = json.dumps({
                "status": "success",
                "message": f"Telemetry saved successfully to results/colesion_study_{clean_run_id}.json",
                "file": out_file.name
            }).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(resp)

        except Exception as e:
            log.error("Failed to save study telemetry: %s", e)
            err = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def handle_api_cases(self, query, is_head=False):
        cases = self.server.get_cases_json(query)
        data = json.dumps(cases).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if not is_head:
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def handle_api_metrics(self, is_head=False):
        metrics = dict(self.server.metrics)
        data = json.dumps(metrics).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if not is_head:
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def handle_stream_image(self, img_id, is_head=False):
        img_path = self.server.image_path_map.get(img_id)
        if not img_path or not img_path.exists():
            self.send_error(404, f"Image {img_id} not found on disk")
            return

        file_len = img_path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(file_len))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if not is_head:
            try:
                with open(img_path, "rb") as f:
                    self.copyfile(f, self.wfile)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def handle_stream_heatmap(self, img_id, is_head=False):
        hm_path = self.server.heatmap_path_map.get(img_id)
        if not hm_path or not hm_path.exists():
            self.send_error(404, f"Heatmap {img_id} not found on disk")
            return

        try:
            with Image.open(hm_path) as im:
                im = im.convert("L")
                arr = np.array(im, dtype=np.uint8)

            rgb = TURBO_LUT[arr]
            alpha = (np.clip((arr.astype(np.float32) / 255.0) * 1.35, 0.0, 1.0) * 255.0).astype(np.uint8)
            rgba = np.dstack((rgb, alpha))
            out_img = Image.fromarray(rgba, mode="RGBA")

            bio = io.BytesIO()
            out_img.save(bio, format="PNG", compress_level=1)
            bio.seek(0)
            data = bio.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if not is_head:
                self.wfile.write(data)

        except Exception as e:
            log.error("Error processing heatmap for %s: %s", img_id, e)
            self.send_error(500, "Error rendering heatmap")


class EvaluationServer(ThreadingHTTPServer):
    def __init__(self, host, port, ui_dir: Path):
        self.ui_dir = ui_dir
        self.env = load_env(ui_dir / ".env")

        self.data_root = (ui_dir / self.env.get("DATA_BUILD_DIR", "../data/build")).resolve()
        self.ml_run_dir = (
            ui_dir
            / self.env.get(
                "ML_RUN_DIR", "../runs/clf-finetune-dinov2/2026-11-11_22-10-36_s42"
            )
        ).resolve()

        parquet_file = ui_dir / "state" / "calibrated_cases.parquet"
        metrics_file = ui_dir / "state" / "metrics.json"

        if not parquet_file.exists():
            raise FileNotFoundError(f"Missing required state file: {parquet_file}")

        log.info("Loading precomputed calibrated cases from %s ...", parquet_file)
        self.df = pl.read_parquet(parquet_file)
        self.metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}

        # Prebuild fast ID-to-row lookup
        self.cases_by_id = {}
        for row in self.df.iter_rows(named=True):
            self.cases_by_id[row["image_id"]] = row

        # Pre-cache image file paths across ALL 11,427 items (including training twins)
        items_parquet = self.data_root / "tables" / "items.parquet"
        self.image_path_map = {}
        if items_parquet.exists():
            log.info("Indexing complete image directory from %s ...", items_parquet)
            items_df = pl.read_parquet(items_parquet)
            for row in items_df.select(["image_id", "image_path"]).iter_rows(named=True):
                img_id = row["image_id"]
                p_rel = row["image_path"]
                if p_rel:
                    self.image_path_map[img_id] = self.data_root / p_rel
        else:
            for row in self.df.iter_rows(named=True):
                img_id = row["image_id"]
                p_rel = row["image_path"]
                if p_rel:
                    self.image_path_map[img_id] = self.data_root / p_rel

        # Pre-cache heatmap file paths
        self.heatmap_path_map = {}
        hm_export_dir = self.ml_run_dir / "export"
        for row in self.df.iter_rows(named=True):
            img_id = row["image_id"]
            hm_rel = row["heatmap"]
            if hm_rel:
                self.heatmap_path_map[img_id] = hm_export_dir / hm_rel

        log.info("Building case JSON structures ...")
        self.all_cases_dict = {
            c["image_id"]: c for c in self._build_cases_json()
        }

        super().__init__((host, port), StudyServerHandler)

    def _build_single_case_json(self, row):
        t_val = self.metrics.get("temperature", 1.2937)
        tau_val = self.metrics.get("tau_95", 0.2376)

        img_id = row["image_id"]
        split = row["split"]
        gt_short = row["dx"]
        pred_short = row["pred_class"]
        margin = row["margin_delta"]
        is_ood = row["is_ood"]
        slice_warn = row["slice_warning"]

        if is_ood:
            sc_type = "ood"
        elif gt_short != pred_short:
            sc_type = "disagreement"
        elif margin < 0.20:
            sc_type = "rivalry"
        else:
            sc_type = "agreement"

        cal_probs = {c: round(float(row[f"prob_{c}"]), 4) for c in CLASSES}
        gt_full = CLASS_FULL_NAMES.get(gt_short, gt_short)
        pred_full = CLASS_FULL_NAMES.get(pred_short, pred_short)

        loc = row["localization"] or "Unspecified"
        age = row["age"] if row["age"] is not None else "--"
        sex = (row["sex"] or "Unspecified").capitalize()

        precedents = None
        if not is_ood:
            def make_twin(t_id, t_dx, t_dist):
                dx_full = CLASS_FULL_NAMES.get(t_dx, t_dx)
                key_num = CLASS_INDICES.get(t_dx, -1) + 1
                sim_pct = round(float(max(75.0, min(99.4, (1.0 - t_dist * 1.8) * 100))), 1)
                return {
                    "case_id": f"REF-{t_id}",
                    "isic_id": t_id,
                    "diagnosis": dx_full,
                    "dx_short": t_dx,
                    "key_number": key_num,
                    "distance": float(t_dist),
                    "similarity_pct": sim_pct,
                    "confirmation": "Histopathology Biopsy Verified",
                    "image": f"/api/image/{t_id}.jpg",
                }

            pred_ids = row.get("pred_twin_ids") or []
            pred_dxs = row.get("pred_twin_dxs") or []
            pred_dists = row.get("pred_twin_dists") or []
            pred_twins = [make_twin(i, d, dist) for i, d, dist in zip(pred_ids, pred_dxs, pred_dists)]

            riv_ids = row.get("rival_twin_ids") or []
            riv_dxs = row.get("rival_twin_dxs") or []
            riv_dists = row.get("rival_twin_dists") or []
            rival_twins = [make_twin(i, d, dist) for i, d, dist in zip(riv_ids, riv_dxs, riv_dists)]

            near_ids = row.get("nearest_twin_ids") or []
            near_dxs = row.get("nearest_twin_dxs") or []
            near_dists = row.get("nearest_twin_dists") or []
            nearest_twins = [make_twin(i, d, dist) for i, d, dist in zip(near_ids, near_dxs, near_dists)]

            precedents = {
                "top_match": pred_twins[0] if pred_twins else None,
                "rival_match": rival_twins[0] if rival_twins else None,
                "contrastive": {"predicted_twins": pred_twins, "rival_twins": rival_twins},
                "nearest_latent": nearest_twins,
            }

        return {
            "id": f"CASE-{img_id}",
            "image_id": img_id,
            "split": split,
            "scenario_type": sc_type,
            "clinical_title": f"{split.upper()} Case // GT: {gt_full}",
            "patient": {
                "name": f"Specimen {img_id}",
                "age": age,
                "sex": sex,
                "anatomical_site": loc,
                "stain": "Dermoscopy",
            },
            "image": f"/api/image/{img_id}.jpg",
            "heatmap": f"/api/heatmap/{img_id}.png" if row["heatmap"] else None,
            "ground_truth": {
                "label_idx": CLASS_INDICES.get(gt_short, -1),
                "short_name": gt_short,
                "full_name": gt_full,
                "dx_type": row["dx_type"],
            },
            "ai_assessment": {
                "predicted_class": {
                    "label_idx": CLASS_INDICES.get(pred_short, -1),
                    "short_name": pred_short,
                    "full_name": pred_full,
                },
                "calibrated_probabilities": cal_probs,
                "margin_delta": margin,
                "temperature": round(t_val, 4),
                "ood": {
                    "is_ood": is_ood,
                    "knn_distance": row["knn_distance"],
                    "tau_95_threshold": round(tau_val, 4),
                },
                "slice_competence": {"warning": slice_warn, "message": row["slice_message"]},
            },
            "precedents": precedents,
        }

    def _build_cases_json(self):
        return [self._build_single_case_json(row) for row in self.df.iter_rows(named=True)]

    def get_cases_json(self, query):
        split_q = query.get("split", [None])[0]
        res = list(self.all_cases_dict.values())
        if split_q and split_q != "all":
            res = [c for c in res if c["split"] == split_q]
        return res

    def get_study_cases_payload(self):
        # 1. Format 4 tutorial cases
        tutorial_cases = []
        for t_def in TUTORIAL_DEFINITIONS:
            img_id = t_def["image_id"]
            if img_id in self.all_cases_dict:
                c = dict(self.all_cases_dict[img_id])
                c["study_meta"] = {
                    "is_tutorial": True,
                    "title": t_def["tutorial_title"],
                    "scenario": t_def["scenario"],
                    "learning_goal": t_def["learning_goal"],
                    "explanation": t_def["explanation"],
                    "ablation": "full"
                }
                tutorial_cases.append(c)

        # 2. Format 24 evaluation study cases
        study_cases = []
        for s_def in STUDY_CASE_DEFINITIONS:
            img_id = s_def["image_id"]
            if img_id in self.all_cases_dict:
                c = dict(self.all_cases_dict[img_id])
                c["study_meta"] = {
                    "is_tutorial": False,
                    "case_number": s_def["case_num"],
                    "role": s_def["role"],
                    "ablation": s_def["ablation"],
                }
                study_cases.append(c)

        return {
            "tutorial_cases": tutorial_cases,
            "study_cases": study_cases,
            "metadata": {
                "total_study_cases": len(study_cases),
                "total_tutorial_cases": len(tutorial_cases),
                "model_baseline_acc": 0.743,
                "model_balanced_acc": 0.778,
                "tau_95": 0.2376,
                "tau_99": 0.3017,
                "temperature": 1.2937,
            }
        }


def main():
    ui_dir = Path(__file__).resolve().parent
    env = load_env(ui_dir / ".env")
    port = int(env.get("PORT", 8001))

    host = "0.0.0.0"
    log.info("Initializing COLESION-AI Study Evaluation Server on http://localhost:%d/ ...", port)
    server = EvaluationServer(host, port, ui_dir)
    print("=" * 64)
    print("  COLESION-AI // Formative Evaluation Workplace (ui-eval)       ")
    print(f"  Running at: http://localhost:{port}/                         ")
    print("  Study endpoints:                                              ")
    print(f"    - Cases:     http://localhost:{port}/api/study/cases       ")
    print(f"    - Save Log:  http://localhost:{port}/api/study/save (POST) ")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Server shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
