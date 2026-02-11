# Dings Trader UI (skeleton)

Modern dark (purple-accent) UI skeleton for an AI trading dashboard.

## Run

```bash
cd /home/maxim/.openclaw/workspace/projects/dings-trader/ui

# install deps
npm install

# start dev server
npm run dev
```

Open: http://localhost:3000

## Pages
- `/dashboard`
- `/data`
- `/backtests`
- `/model`
- `/alerts`
- `/settings`

## Notes
- Tailwind + shadcn/ui-style components live in `components/ui/*`.
- Layout shell: `components/layout/*`.
- Chart is a placeholder card; plug in Recharts/Visx/TradingView later.
