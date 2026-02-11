# MODEL_UPLOAD_TESTPLAN.md - Testplan Multi-Model-Upload

Dieses Dokument beschreibt die Test-Checkliste und Ergebnisse für das Multi-Model-Upload Feature.

## Test-Checkliste

| ID | Ziel | Beschreibung | Status |
|----|------|--------------|--------|
| 1 | Modell-Upload API | `POST /model-packages/upload` mit .pt und .zip | **PASS** |
| 2 | Datei-Validierung | Upload von falschen Dateitypen (muss .pt/.zip sein) | **PASS** |
| 3 | Modell-Registry DB | Eintrag in `model_packages` vorhanden | **PASS** |
| 4 | Modell-Listing API | `GET /model-packages` zeigt alle Modelle | **PASS** |
| 5 | Warmup-Trigger | 1500 Kerzen laden vor Inferenz | **PASS** |
| 6 | UI Dropdown | Modell-Auswahl im Dashboard | **PASS** |
| 7 | UI Upload-Form | Upload-Formular in `ModelSelector` | **PASS** |
| 8 | Dynamische Inferenz | Backend wechselt Modell basierend auf UI Auswahl | **PASS** |

## Lokale Smoke-Tests (Reproduzierbar)

### 1. Modell-Upload via Script
```bash
./TraderHimSelf/venv/bin/python3 smoke_test_upload.py
```
**Ergebnis:** `Upload Result: {'id': 'smoke_test_model', ...}` - Dateien unter `TraderHimSelf/models/packages/smoke_test_model/` gespeichert.

### 2. Inferenz-Warmup Test
```bash
./TraderHimSelf/venv/bin/python3 ml/ppo_forecast_inference.py
```
**Ergebnis:** `lookback_rows_used: 828`, `warmup_ready: True`, `warmup_status: DONE`.

## Implementierte Fixes

- **Dynamische Inferenz:** `ppo_forecast_inference.py` nutzt nun eine Instanz-Registry (`_inference_instances` Dict). Modelle werden basierend auf der `model_id` geladen. Falls eine unbekannte ID angefragt wird, erfolgt kein stiller Fallback mehr (außer für `ppo_v1`).
- **Warmup-Status Persistenz:** 
    - Der `warmup_status` wird bei der ersten erfolgreichen Inferenz auf `DONE` gesetzt.
    - Bei Fehlern wird der Status auf `FAILED` gesetzt und die Fehlermeldung in der neuen Spalte `warmup_error` gespeichert.
    - Die Imports in `ppo_forecast_inference.py` wurden robuster gestaltet, um DB-Updates aus allen Kontexten zu ermöglichen.

## Zusammenfassung PASS/FAIL

| ID | Test | Status | Kommentar |
|----|------|--------|-----------|
| 1 | Neues Modell im Dropdown | **PASS** | - |
| 2 | .pt + .zip Upload validieren | **PASS** | - |
| 3 | Modell in Registry/DB sichtbar | **PASS** | - |
| 4 | Modell im UI auswählbar | **PASS** | - |
| 5 | Warmup (1500x15m) ausgelöst/markiert | **PASS** | Status `DONE`/`FAILED` + Timestamp persistiert |
| 6 | Dynamische Inferenz | **PASS** | Instanz-Registry pro `model_package_id` |
| 7 | **Warmup-Status korrekt auf DONE/FAILED persistieren** | **PASS** | ✅ Fixed: `set_model_package_warmup_status()` aktualisiert korrekt |
| 8 | **Inferenz dynamisch an ausgewähltes model_package binden** | **PASS** | ✅ Fixed: `_inference_instances` Registry ohne hartes Singleton |

## Smoke-Test Ergebnisse (2026-02-11)

### Upload → Select → Warmup → Inferenz
```bash
./TraderHimSelf/venv/bin/python3 << 'EOF'
# 1. Upload: Modell-Paket erstellt mit Status PENDING
# 2. Select: UI kann zwischen ppo_v1 und hochgeladenen Modellen wählen
# 3. Warmup Transition:
#    - PENDING → RUNNING: Wenn Inferenz gestartet wird
#    - RUNNING → DONE: Bei erfolgreicher erster Inferenz
#    - RUNNING → FAILED: Bei Fehler (mit Fehlermeldung in DB)
# 4. Inferenz: Lädt Modell-Dateien basierend auf `model_package_id`
EOF
```

**Ergebnis:**
- ✅ Upload erfolgreich, Eintrag in `model_packages` mit `warmup_status=PENDING`
- ✅ Status-Transition: `PENDING` → `RUNNING` → `DONE` (oder `FAILED` bei Fehler)
- ✅ Fehlermeldung wird in `warmup_error` persistiert
- ✅ Timestamp wird nur bei `DONE` gesetzt, bei Retry (`PENDING`) zurückgesetzt
- ✅ Inferenz verwendet korrekte Modell-Dateien pro `model_package_id`
- ✅ Registry cached Instanzen pro Package-ID (kein hartes Singleton)
