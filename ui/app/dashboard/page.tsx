export const dynamic = "force-dynamic";
export const revalidate = 0;

import { Suspense } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { PageHeader } from "@/components/layout/page-header";
import { ModelSelector } from "@/components/layout/model-selector";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ActivePositions,
  ClosedTrades,
  RiskMeter,
  NextSignalPreview,
  AccountBalance,
  LegacyTradeHistory,
  TradingSignalsTile,
  TradingChartLightweight,
  LivePriceBubble,
} from "@/components/dashboard";
import { PaperTradingStatus } from "@/components/layout/paper-trading-status";

type Trade = {
  time: string;
  side: string;
  profit: string;
  status: string;
};

type Signal = {
  symbol: string;
  action: string;
  confidence: string;
  variant: "default" | "secondary" | "destructive" | "outline" | "success" | "warning" | "purple";
};

type Position = {
  symbol: string;
  side: string;
  size: string;
  entry: string;
  unrealized_pnl: string;
};

type EquityPoint = {
  timestamp?: string;
  equity?: number;
  balance?: number;
  open_positions?: number;
  unrealized_pnl?: number;
  [key: string]: number | string | undefined;
};

// Paper Trading Dashboard Data Types
interface PaperAccount {
  model_id: string;
  initial_balance: number;
  balance_usdt: number;
  total_equity: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_return_pct: number;
  total_return_usdt: number;
  max_positions: number;
  default_leverage: number;
}

interface PaperPosition {
  id: string | number;
  symbol: string;
  side: string;
  entry_price: number;
  size_usdt: number;
  leverage: number;
  take_profit_pct?: number;
  stop_loss_pct?: number;
  unrealized_pnl_pct?: number;
  unrealized_pnl_usdt?: number;
  open_time: string;
}

interface PaperTrade {
  id: string | number;
  symbol: string;
  side: string;
  entry_price: number;
  exit_price: number;
  pnl_pct?: number;
  pnl_usdt?: number;
  close_reason?: string;
  close_time?: string;
  open_time?: string;
  size_usdt?: number;
}

interface PaperSignal {
  signal: string;
  confidence: number;
  sentiment: string;
  current_price: number;
  probabilities?: {
    short: number;
    flat: number;
    long: number;
  };
  timestamp?: string;
}

interface PaperPerformance {
  total_return_pct: number;
  win_rate: number;
  avg_pnl_pct: number;
  avg_win_pct: number;
  avg_loss_pct: number;
  open_exposure_pct: number;
}

interface PaperDashboardData {
  account: PaperAccount;
  ml_signal: PaperSignal;
  open_positions: PaperPosition[];
  open_positions_count: number;
  available_slots: number;
  recent_trades: PaperTrade[];
  performance: PaperPerformance;
}

function formatEur(value: number): string {
  const fixed = value.toFixed(2).replace(".", ",");
  const [intPart, fracPart] = fixed.split(",");
  const withSep = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  return `${withSep},${fracPart} €`;
}

async function getEquityData(runId: string, paperModelId?: string): Promise<EquityPoint[]> {
  try {
    // Use paper model ID if provided, otherwise use runId
    const equityRunId = paperModelId || runId;
    const res = await fetch(`http://127.0.0.1:8000/equity?run=${equityRunId}`, {
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error("equity fetch failed");
    }
    const data = await res.json();
    if (Array.isArray(data)) {
      return data as EquityPoint[];
    }
  } catch {
    // fallback
  }
  return [];
}

async function getSignals(runId: string): Promise<Signal[]> {
  try {
    const res = await fetch(`http://127.0.0.1:8000/signals?run=${runId}`, {
      cache: "no-store",
    });
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // fallback
  }
  return [];
}

async function getPositions(runId: string): Promise<Position[]> {
  try {
    const res = await fetch(`http://127.0.0.1:8000/positions?run=${runId}`, {
      cache: "no-store",
    });
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // fallback
  }
  return [];
}

async function getModelStatus(runId: string): Promise<string> {
  try {
    const res = await fetch(`http://127.0.0.1:8000/model-status?model_id=${runId}`, {
      cache: "no-store",
    });
    if (res.ok) {
      const data = await res.json();
      return data.status || "LIVE";
    }
  } catch {
    // fallback
  }
  return "LIVE";
}

function getLastEquity(data: EquityPoint[]): string {
  if (data.length > 0) {
    const last = data[data.length - 1];
    const direct = last?.equity;
    const value = typeof direct === "number" ? direct : Number(direct);
    if (Number.isFinite(value)) {
      return formatUsdt(value);
    }
    const numeric = Object.values(last).find((v) => typeof v === "number") as
      | number
      | undefined;
    if (typeof numeric === "number" && Number.isFinite(numeric)) {
      return formatUsdt(numeric);
    }
  }
  return "10,000.00 USDT";
}

function formatUsdt(value: number): string {
  return `${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USDT`;
}

async function getTrades(runId: string): Promise<Trade[]> {
  try {
    const res = await fetch(`http://127.0.0.1:8000/trades?run=${runId}`, {
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error("trade fetch failed");
    }
    const data = await res.json();
    if (Array.isArray(data)) {
      return data as Trade[];
    }
  } catch {
    // fallback to static rows
  }
  return [
    { time: "2025-02-04 10:00", side: "Long", profit: "+5.2%", status: "Closed" },
    { time: "2025-02-04 14:00", side: "Short", profit: "-1.1%", status: "Closed (SL)" },
  ];
}

// Fetch Paper Trading Dashboard Data
async function getPaperDashboardData(modelId: string): Promise<PaperDashboardData> {
  try {
    const res = await fetch(`http://127.0.0.1:8000/paper/dashboard/${modelId}`, {
      cache: "no-store",
      headers: {
        "Accept": "application/json",
      },
    });

    if (!res.ok) {
      // Try to create account if it doesn't exist
      if (res.status === 404) {
        const createRes = await fetch(`http://127.0.0.1:8000/paper/account/create`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_id: modelId,
            initial_balance: 10000.0,
            max_positions: 5,
            default_leverage: 7.0,
            profit_target_pct: 5.0,
            time_limit_hours: 48.0,
          }),
        });

        if (createRes.ok) {
          // Retry fetch
          const retryRes = await fetch(`http://127.0.0.1:8000/paper/dashboard/${modelId}`, {
            cache: "no-store",
          });
          if (retryRes.ok) {
            return await retryRes.json();
          }
        }
      }
      throw new Error(`Paper Trading API error: ${res.status}`);
    }

    return await res.json();
  } catch (error) {
    console.error("Paper Trading fetch error:", error);

    // Return default data
    return {
      account: {
        model_id: modelId,
        initial_balance: 10000,
        balance_usdt: 10000,
        total_equity: 10000,
        total_trades: 0,
        winning_trades: 0,
        losing_trades: 0,
        win_rate: 0,
        total_return_pct: 0,
        total_return_usdt: 0,
        max_positions: 5,
        default_leverage: 7,
      },
      ml_signal: {
        signal: "FLAT",
        confidence: 0,
        sentiment: "neutral",
        current_price: 0,
        probabilities: { short: 0, flat: 100, long: 0 },
      },
      open_positions: [],
      open_positions_count: 0,
      available_slots: 5,
      recent_trades: [],
      performance: {
        total_return_pct: 0,
        win_rate: 0,
        avg_pnl_pct: 0,
        avg_win_pct: 0,
        avg_loss_pct: 0,
        open_exposure_pct: 0,
      },
    };
  }
}

export default async function DashboardPage(props: {
  searchParams: Promise<{ model?: string | string[] }>;
}) {
  const searchParams = await props.searchParams;
  const modelParam = searchParams.model;
  const modelId = Array.isArray(modelParam) ? modelParam[0] : modelParam || "ppo_v1";

  // Map UI model IDs to API run IDs
  const runMapping: Record<string, string> = {
    v1: "val_2024",
    v2: "test_2025",
    v3: "test_2025",
  };

  const runId = runMapping[modelId] || "test_2025";

  // Paper trading uses model_id directly
  const paperModelId = `paper_${modelId}`;

  const [trades, equityData, signals, positions, status, paperData] = await Promise.all([
    getTrades(runId),
    getEquityData(runId, paperModelId),
    getSignals(runId),
    getPositions(runId),
    getModelStatus(runId),
    getPaperDashboardData(paperModelId),
  ]);

  const totalEquity = getLastEquity(equityData);

  // Calculate exposure per position for risk meter
  const exposurePerPosition = paperData.open_positions.map(p =>
    (p.size_usdt / paperData.account.total_equity) * 100
  );

  return (
    <AppShell>
      <div className="w-full space-y-6 2xl:space-y-8">
        <PageHeader
        title="Dings Trader Dashboard"
        subtitle="ML-gesteuertes Trading mit Echtzeit-Risk-Management"
        badge="Paper Trading v2.0"
        status={status}
        isLive={true}
        showLiveTimer={true}
        modelId={paperModelId}
        actions={
          <Suspense fallback={<div className="w-32 h-10 bg-muted/20 rounded animate-pulse" />}>
            <ModelSelector />
          </Suspense>
        }
      />

      {/* Obere Reihe: Trading Chart + Account Balance (Forecast + Timeframes sofort sichtbar) */}
      <div className="grid gap-4 lg:grid-cols-3 2xl:gap-6 w-full">
        <div className="lg:col-span-2 w-full flex flex-col gap-3">
          <div className="max-w-[420px]">
            <LivePriceBubble modelId={paperModelId} symbol="BTCUSDT" />
          </div>
          <TradingChartLightweight modelId={paperModelId} />
        </div>
        <AccountBalance account={paperData.account} modelId={paperModelId} />
      </div>

      {/* Mittlere Reihe: Positionen + Consolidated Card (Signal, Risk) */}
      <div className="grid gap-4 lg:grid-cols-2 2xl:gap-6 w-full">
        <ActivePositions
          positions={paperData.open_positions}
          maxPositions={paperData.account.max_positions}
          modelId={paperModelId}
        />
        
        {/* Three Cards Group: Trading Signals + Next Signal Preview + Risk Meter */}
        <div className="flex flex-col gap-4">
          {/* Card 1: Trading-Signale */}
          <Card className="border-primary/30 rounded-xl bg-card/40">
            <CardContent className="p-4 2xl:p-6">
              <TradingSignalsTile
                modelId={paperModelId}
                initialSignal={paperData.ml_signal}
                symbol="BTC/USDT"
              />
            </CardContent>
          </Card>

          {/* Card 2: Next Signal Preview */}
          <Card className="border-primary/30 rounded-xl bg-card/40">
            <CardContent className="p-4 2xl:p-6">
              <NextSignalPreview
                signal={paperData.ml_signal}
                symbol="BTC/USDT"
                modelId={paperModelId}
              />
            </CardContent>
          </Card>

          {/* Card 3: Risk Meter */}
          <Card className="border-primary/30 rounded-xl bg-card/40">
            <CardContent className="p-4 2xl:p-6">
              <RiskMeter
                totalExposure={paperData.performance.open_exposure_pct}
                positionCount={paperData.open_positions_count}
                exposurePerPosition={exposurePerPosition}
                maxPositions={paperData.account.max_positions}
                maxExposure={10}
                modelId={paperModelId}
              />
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Untere Reihe: Closed Trades (with Paper Trading data) */}
      <div className="w-full">
        <ClosedTrades trades={paperData.recent_trades} />
      </div>

      {/* Trade Historie Legacy - Client Component with Polling */}
      <LegacyTradeHistory 
        modelId={paperModelId}
        initialPositions={paperData.open_positions}
        initialTrades={paperData.recent_trades}
      />
      </div>
    </AppShell>
  );
}
