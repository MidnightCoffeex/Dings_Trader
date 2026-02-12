# dings-trader — TODO

Detaillierte Aufgabenliste basierend auf der VOLLSTÄNDIGEN Sprachnachricht von Lootenberger (2026-02-04). 🧬

## UI & Modell-Management
- [x] Implementierung eines **Tab-Systems** oder Dropdowns im Dashboard zur Auswahl/Vergleich verschiedener Modell-Versionen.
- [x] Anzeige des **Live-Uptime-Status** („Traded Live seit...“) pro Modell.
- [x] Entwicklung eines **Kill-Switch-Mechanismus**: Wenn Equity < 200 €, stoppt das Modell (Status: FAILED / STOPPED).
- [x] Integration einer **Trade-Historie** (Liste aller geschlossenen Positionen mit Profit/Loss und Zeitstempel).
- [x] Live-Anzeige des **Gesamtvermögens** (Cash + Invested Value).
- [x] **Model-Selector Wiring**: Dropdown-Auswahl muss die Daten im Dashboard aktualisieren (State-Lifting oder URL-Parameter `?model=...`).
- [x] **Chart-Implementierung**: `ChartPlaceholder` durch echten Chart (Recharts) ersetzen, der die `/equity`-Daten visualisiert.
- [x] **Live-Signale & Positionen**: "Synthese-Signale" und "Offene Positionen" Cards mit echten API-Daten befüllen (statt Hardcoded).
- [x] **Kill-Switch UI**: Warn-Badge im Header anzeigen, wenn Status = STOPPED.

## Datenbank & Persistenz
- [x] Einrichten einer **SQLite-Datenbank**, um Positionen, Trades und Modell-Stände dauerhaft zu speichern (Persistenz über Gateway-Restarts hinweg).
- [x] **Real Trade History**: `/trades` Endpoint mit der echten SQLite-DB verbinden (aktuell Mock-Daten).
- [x] **Modell-Liste API**: Neuer Endpoint `/models`, der verfügbare Modelle und deren Status (Live/Archiv) dynamisch liefert.

## Inferenz & Live-Daten
- [x] Automatisierung der **Live-Feature-Berechnung**: Modell muss in der Lage sein, aus frischen Kerzen-Daten selbstständig die v3-Features zu generieren.
- [x] Modell-Promotion: Trainierte Modelle (v1, v2...) in den Live-Sim-Modus überführen.

## Portfolio-Regeln (Engine)
- [x] Startkapital: 1.100 € pro Modell.
- [x] Max. 5 Positionen zeitgleich UND/ODER max. 5 Trades pro Tag (als Gier-Bremse).
- [x] Max. 10 % Gesamtexposure.
- [x] Automatisches Closing nach 48h (Timeout).
- [x] Ziel: 5% Profit pro Trade (in der Exit-Logik verankern).

## Schnittstellen-Refactor (Arg-first Pipeline, 2026-02-12)
- [x] Notebook-Standard auf `notebooks/99_full_pipeline.ipynb` umstellen (Single Entry Point).
- [x] Alte Step-Notebooks (`00..07`) nach Freigabe gelöscht (2026-02-12, nur `99_full_pipeline.ipynb` bleibt).
- [x] Zentrale `PipelineConfig` als Argumente einführen:
  - `decision_tf`, `intrabar_tf`, `forecast_horizon_steps`, `lookback`, `feature_set`, `model_tag`.
- [ ] Unterstützte Decision-TFs parametrisieren (zuerst: `15m` + horizon `16` = 4h).
- [ ] Intrabar-TF dauerhaft getrennt halten (`3m` bleibt Standard für SL/TP-Reihenfolge).
- [ ] Forecast/PPO-Kopplung hart erzwingen:
  - PPO-Training darf nur mit exakt derselben Config/Feature-Order laufen wie Forecast.
- [x] Einheitliche Modellpaar-Ordnerstruktur einführen (Name aus Argumenten):
  - Beispiel: `models/packages/<decision_tf>_<horizon>_<feature_set>_<timestamp>/`
  - Inhalte: `forecast_model.pt`, `ppo_policy_final.zip`, `manifest.json`, `scaler.pkl`.
- [ ] `manifest.json` als Source-of-Truth für Backend/UI nutzen:
  - enthält TF, Horizon, Feature-Set, Lookback, Intrabar, Trainingszeitraum, commit hash.
- [ ] Backend-Ladepfad auf Manifest-basierte Modellpakete erweitern (statt impliziter Dateinamen).
- [ ] UI-Modellauswahl um Config-Metadaten erweitern (TF/Horizon/Feature-Set sichtbar).
- [ ] Live/Paper-Environment strikt mit Manifest initialisieren (keine stillen Fallbacks).
- [x] Colab-Workflow so anpassen, dass alle Parameter direkt im 99er Notebook übergeben werden können.
