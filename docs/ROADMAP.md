# Symbiomorphose Trader — Roadmap & TODOs

Basierend auf dem Plan von Lootenberger und Dings zur Finanzierung unserer Bewegung.

## 1. Daten & Vorbereitung 📊
- [x] Historische 1h-BTC-Daten ab 2020 laden (Binance Public API).
- [x] Feature Engineering v3: RSI, MACD, Price Slope (1h), Momentum-Lags und Zeit-Features integriert.
- [ ] Daten-Pipeline: Automatisches Update der historischen Daten bei jedem Inferenz-Schritt.

## 2. ML-Modell (Strict Split) 🧠
- [x] Trainings-Setup: Daten bis 31.12.2023.
- [x] Validierungs-Setup (Tuning): Daten vom 01.01.2024 bis 31.12.2024.
- [x] Finaler Testlauf (Benchmark): Daten ab 01.01.2025 bis heute.
- [ ] Hyperparameter-Tuning: Modell auf 1h-Spezifika optimieren.
- [ ] Inferenz-Logik: Modell berechnet Features aus Live-Kerzen selbstständig.

## 3. Portfolio-Engine v2 (Simulationsumgebung) ⚙️
- [x] Unterstützung für gleichzeitige Positionen (max. 5).
- [x] Exposure-Limit: Gesamtwert aller Positionen <= 10 % des Gesamtvermögens.
- [x] Hebel-Logik: Festgelegt auf max. 5x.
- [x] Zeit-Limit: Harte Schließung nach 48h (Timeout).
- [x] Startkapital: 1.100 € pro Modell-Instanz.
- [x] Kill-Switch: Automatischer Stopp bei Equity < 200 € (Ressourcen-Schonung).

## 4. UI: Live-Simulations-Modus 🌐
- [ ] Modell-Versions-Management**: Tabs oder Dropdown für v1, v2, v3 etc.
- [x] Live-Tracking**: Anzeige „Traded Live seit [Startzeitpunkt]“. (Infrastruktur steht)
- [x] Live-Daten Integration**: Echtzeit-BTC-Kurs von Binance im Dashboard eingebunden.
- [ ] Performance-Metriken**: Historische Trades (Profit/Loss), aktuelles Gesamtvermögen (Cash + Invested).
- [ ] Datenbank-Anbindung**: Persistente Speicherung der Positionen (SQLite), damit Daten nach Gateway-Restart erhalten bleiben.

## 5. Glue / Ops 🛠️
- [ ] Feature-Service: Engine zur On-the-fly Berechnung der Modell-Eingaben.
- [ ] Dashboard-Branding: Symbiomorphose-Vibe (Lila Glow).
- [ ] **Telegram-Integration**: Native `/cron` Command zum Auslesen der Job-Liste implementieren.

---
*Anmerkung: Das Ziel ist das maximale Endvermögen unter Einhaltung der 10%-Regel. Wir sind das Wir.* 🧬
