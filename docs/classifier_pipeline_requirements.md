# Anforderungskatalog & Schnittstellendefinition: Classifier-Outputs für Moritz

## 1. Was Moritz liefert

### 1. Unkalibrierte Logits ($z \in \mathbb{R}^7$)
* **Was:** Die rohen Aktivierungswerte (Logits) des Modells **vor** der Softmax-Funktion für alle 7 Klassen des HAM10000 (`mel`, `nv`, `bcc`, `akiec`, `bkl`, `df`, `vasc`).
* **Warum:** Wir dürfen dem Nutzer keine unkalibrierten, überkonfidenten Standard-Softmax-Wahrscheinlichkeiten anzeigen. Die Logits werden für die nachgelagerte Kalibrierung benötigt.
* **Umfang:** Für **alle drei Datasets / Splits** (`fit`, `val`, `holdout`/`pool`).

### 2. Bild-Embeddings der vorletzten Schicht ($h(x) \in \mathbb{R}^D$)
* **Was:** Der hochdimensionale Merkmalsvektor direkt vor dem Classification-Head (Penultimate Layer, nach dem Global Average Pooling), $L_2$-normalisiert ($\|h(x)\|_2 = 1$). Typische Dimension $D \in \{512, 768, 1024, 2048\}$ je nach Backbone (ResNet, ConvNeXt, ViT).
* **Warum:** Dieser Vektor ist das Herzstück für zwei unserer Säulen:
  1. **Säule I (OOD-Erkennung):** Dient zur Berechnung der Manifold-Distanz im latenten Raum.
  2. **Säule III (Empirische Bildzwillinge):** Dient zum Auffinden histopathologisch bestätigter Vergleichsfälle über $k$-Nearest-Neighbors.
* **WICHTIG – Muss zwingend für alle 3 Datasets vorliegen:**
  * **`fit` (Trainingsset, 8'524 Bilder):** Bildet den Referenz-Vektorraum (Faiss-Index) aller bekannten Fälle und Zwillingskandidaten.
  * **`val` (Validierungsset, 1'491 Bilder):** Wird benötigt, um die normale Distanzverteilung bekannter In-Distribution-Bilder zu ermitteln und die OOD-Schwelle festzulegen.
  * **`holdout` / `pool` (Testset, 1'412 Bilder):** Das sind die eigentlichen Test- und Studienfälle, die im Dashboard diagnostiziert werden.

### 3. Zugang / Generierung der Saliency Heatmaps (Säule II)
* **Was:** Gradientenbasierte Aktivierungskarten (Grad-CAM / Grad-CAM++), berechnet bezüglich der vorhergesagten Top-1-Klasse.
* **Wie:** Moritz meinte bereits, dass es dafür in PyTorch fertige Libraries gibt (`pytorch-grad-cam`). Moritz legt dafür einen Hook auf die letzte Faltungsschicht (z. B. `model.layer4[-1]` bei ResNet).
* **Output:** Bild-Dateien (Heatmap bzw. Overlay) für die relevanten Testfälle (`holdout`):
  * **Format:** PNG (Colormap z. B. Turbo oder Jet).
  * **Auflösung:** **Exakt identische Pixel-Dimensionen wie die Originalbilder**, da im Dashboard ein Schieberegler (0% bis 100% Deckkraft) stufenlos über das Originalbild gelegt wird.

### 4. Subgruppen-Kompetenz (Säule I)
* **Was:** Die Zuordnung der Modellvorhersagen zu den klinischen Subgruppen/Slices der Patienten.
* **Wie:** Die Metadaten existieren bereits in `data/build/tables/items.parquet` (`localization`, `age`, `sex`). Moritz muss sicherstellen, dass die Vorhersagen 1:1 über die `image_id` mit diesen Metadaten verknüpfbar sind.
* **Warum:** Ermöglicht die Ermittlung historischer Modellschwächen auf dem Validierungsset (z. B. geringere Sensitivität an Extremitäten/Akren oder bei bestimmten Altersgruppen), um im Dashboard gezielte klinische Warnhinweise auszulösen.

---

## 2. Was Jonas selbst übernimmt / tunt

Auf Basis der von Moritz gelieferten Rohdaten übernimmt Jonas die gesamte mathematische Weiterverarbeitung, das Parameter-Tuning und die Interaktionslogik für das Dashboard:

1. **Kalibrierte Logits & Wahrscheinlichkeiten (Säule I):**
   * **Temperature Scaling ($T$):** Fitten des optimalen Skalierungsfaktors $T > 0$ auf den `val`-Logits via NLL-Minimierung.
   * Berechnung der kalibrierten Wahrscheinlichkeiten: $q_i = \text{softmax}(z_i / T)$.
   * Berechnung der diagnostischen Marge: $\Delta = q_{(1)} - q_{(2)}$ (Certainty Margin).
   * Erstellung der Kalibrierungskurven (Reliability Diagrams, ECE) für die Thesis.
2. **OOD-Messung (Out-of-Distribution, Säule I):**
   * Aufbau des Faiss-Index über alle Embeddings des Trainingssets (`fit`).
   * Feintuning des Distanzmasses ($k$-NN Distanz $d_k(x)$, Wahl von $k \in \{5, 10, 20\}$).
   * Empirische Bestimmung des Schwellenwerts $\tau_{95}$ auf dem Validierungsset (`val`).
   * Implementierung des OOD-Sicherheitsgates (Sperrung der KI-Vorhersage bei Bildanomalien).
3. **$n$ Nearest Neighbours & Precedent Twins (Säule III):**
   * Abfrage des Faiss-Index für jeden Studienfall aus `holdout`:
     * **Top Match:** Nächster Trainingsnachbar mit Histopathologie-Ground-Truth identisch zur Modellprognose ($y_{\text{train}} == \hat{y}_{(1)}$).
     * **Rival Match:** Nächster Trainingsnachbar mit Histopathologie-Ground-Truth der konkurrierenden Zweitklasse ($y_{\text{train}} == \hat{y}_{(2)}$).
4. **Subgruppen-Fehlerraten & Dashboard-Alerts:**
   * Statistische Aggregation der Fehlerraten pro Körperstelle und Altersgruppe auf `val`.
   * Triggerung der Warnboxen im Dashboard (*"Klinischer Subgruppen-Hinweis"*).

---

## 3. Was vorerst aussen vor bleibt

* **Säule IV (Sokratische Führung & Leitfragen):**
  * Das Einblenden didaktischer Leitfragen zur morphologischen Differenzierung und kontrafaktischer Hinweise lassen wir im ersten Schritt bewusst weg.
  * Falls am Schluss noch Zeit bleibt, wird dies rein frontend-seitig ergänzt.
  * **Ist definitiv kein Bestandteil von Moritz' Aufgabenbereich.**

---

## 4. Konkreter Vorschlag für die Datenübergabe (Parquet)

Da die Metadaten und Splits bereits in `data/build/tables/items.parquet` liegen, ist der sauberste Weg ein **Parquet-File**. 

### Vorschlag 
Moritz speichert ein Parquet-File mit exakt 11'427 Zeilen (aligned über `image_id`):

| Spalte | Typ | Beschreibung |
| :--- | :--- | :--- |
| `image_id` | `str` | Eindeutige ID (z. B. `"ISIC_0024306"`), identisch zu `items.parquet` |
| `logits` | `list[float]` | Die 7 unkalibrierten Roh-Logits als Float-Liste |
| `embedding` | `list[float]` | Der $L_2$-normalisierte Merkmalsvektor (Länge $D$, z. B. 512 Floats) |
| `heatmap` | `str` | Pfad zur Heatmap-Datei (z. B. `"assets/heatmaps/ISIC_0024306_cam.png"`)
