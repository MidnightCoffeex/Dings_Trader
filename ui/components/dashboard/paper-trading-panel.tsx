"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Wallet, 
  TrendingUp, 
  TrendingDown, 
  Activity,
  RefreshCw,
  AlertCircle,
  Zap,
  RotateCcw,
  Target,
  Percent
} from "lucide-react";

interface PaperAccountData {
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

interface MLSignalData {
  signal: string;
  confidence: number;
  sentiment: string;
  current_price: number;
  probabilities: {
    short: number;
    flat: number;
    long: number;
  };
  // Diagnostics
  warmup_ready?: boolean;
  lookback_rows_used?: number;
  action_raw?: any;
}

interface PaperPosition {
  id: number;
  symbol: string;
  side: string;
  entry_price: number;
  size_usdt: number;
  leverage: number;
  unrealized_pnl_pct: number;
  unrealized_pnl_usdt: number;
  open_time: string;
}

interface RecentTrade {
  id: number;
  symbol: string;
  side: string;
  entry_price: number;
  exit_price: number;
  pnl_pct: number;
  pnl_usdt: number;
  close_reason: string;
  close_time: string;
}

interface PaperDashboardData {
  account: PaperAccountData;
  ml_signal: MLSignalData;
  open_positions: PaperPosition[];
  open_positions_count: number;
  available_slots: number;
  recent_trades: RecentTrade[];
  performance: {
    total_return_pct: number;
    win_rate: number;
    avg_pnl_pct: number;
    avg_win_pct: number;
    avg_loss_pct: number;
    open_exposure_pct: number;
  };
}

interface PaperTradingPanelProps {
  modelId?: string;
}

export function PaperTradingPanel({ modelId = "paper_ppo_v1" }: PaperTradingPanelProps) {
  const [data, setData] = useState<PaperDashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [resetting, setResetting] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [isAutoTrading, setIsAutoTrading] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Use Next.js API route (proxy to Python backend)
      const res = await fetch(`/api/paper/dashboard/${modelId}`, {
        cache: "no-store",
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }

      const dashboardData = await res.json();
      setData(dashboardData);
      setLastUpdate(new Date());
    } catch (err) {
      console.error("Paper Trading fetch error:", err);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [modelId]);

  // Initial fetch
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Auto-refresh every 1 second (Live Tick)
  useEffect(() => {
    const interval = setInterval(() => {
      fetchData();
    }, 1000);

    return () => clearInterval(interval);
  }, [fetchData]);

  const handleReset = async () => {
    if (!confirm("Are you sure you want to reset this paper trading account? All history will be lost.")) return;
    
    try {
      setResetting(true);
      // Use Next.js API route proxy (we need to implement this or use direct call if configured)
      // Since we don't have a proxy for reset yet, let's assume we can add it or call direct.
      // But wait, we set up /api/backend-status proxy. We probably need /api/paper/reset proxy.
      // Or we can use the existing /api/paper/dashboard pattern but for POST.
      // Actually, paper_api.py has @router.post("/account/{model_id}/reset")
      
      // Let's assume we need to create a proxy or use the direct backend URL if we are in dev/local.
      // But for consistency with the dashboard fetch, we should probably add a proxy.
      // However, to save time, I will try to call the backend directly via the existing proxy mechanism if possible?
      // No, Next.js proxying needs a route handler.
      
      // I'll quickly add a reset route handler: projects/dings-trader/ui/app/api/paper/reset/route.ts
      
      const res = await fetch("/api/paper/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId, keep_history: false }),
      });

      if (!res.ok) throw new Error("Reset failed");
      
      await fetchData();
    } catch (err) {
      console.error("Reset error:", err);
      alert("Failed to reset account");
    } finally {
      setResetting(false);
    }
  };

  const formatUsd = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatPct = (value: number) => {
    const sign = value >= 0 ? "+" : "";
    return `${sign}${value.toFixed(2)}%`;
  };

  const getSignalBadge = (signal: string, confidence: number) => {
    if (signal === "LONG" || signal === "long") {
      return (
        <Badge variant="success" className="flex items-center gap-1 px-2 py-1">
          <TrendingUp className="w-3 h-3" />
          BULLISH
        </Badge>
      );
    } else if (signal === "SHORT" || signal === "short") {
      return (
        <Badge variant="destructive" className="flex items-center gap-1 px-2 py-1">
          <TrendingDown className="w-3 h-3" />
          BEARISH
        </Badge>
      );
    } else {
      return (
        <Badge variant="outline" className="flex items-center gap-1 px-2 py-1">
          <Activity className="w-3 h-3" />
          NEUTRAL
        </Badge>
      );
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 70) return "text-emerald-400";
    if (confidence >= 50) return "text-amber-400";
    return "text-muted-foreground";
  };

  if (loading && !data) {
    return (
      <Card className="bg-card/40 border-primary/20">
        <CardContent className="p-6">
          <div className="flex items-center justify-center h-40">
            <div className="animate-spin w-6 h-6 border-2 border-primary border-t-transparent rounded-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error && !data) {
    return (
      <Card className="bg-card/40 border-destructive/50">
        <CardContent className="p-6">
          <div className="flex flex-col items-center gap-3 text-destructive">
            <AlertCircle className="w-8 h-8" />
            <p className="text-sm">{error}</p>
            <Button variant="outline" size="sm" onClick={fetchData}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data) return null;

  const { account, ml_signal, open_positions, recent_trades, performance, available_slots } = data;

  return (
    <div className="space-y-4">
      {/* Paper Trading Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Badge variant="purple" className="text-xs px-3 py-1">
            <Zap className="w-3 h-3 mr-1" />
            PAPER TRADING
          </Badge>
          <span className="text-sm text-muted-foreground">
            Model: {account.model_id}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">
            {lastUpdate.toLocaleTimeString("de-DE")}
          </span>
          <Button 
            variant="outline" 
            size="sm" 
            className="h-8 border-destructive/50 hover:bg-destructive/10 text-destructive"
            onClick={handleReset}
            disabled={resetting}
          >
            <RotateCcw className={`w-3 h-3 mr-1 ${resetting ? 'animate-spin' : ''}`} />
            Reset
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={fetchData}>
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Account Overview & ML Signal */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Account Balance */}
        <Card className="bg-card/40 border-primary/20">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Wallet className="w-4 h-4" />
                Simulated Account
              </CardTitle>
              <Badge variant="outline" className="text-[10px]">
                {account.max_positions} Max Positions
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Total Equity */}
            <div className="space-y-1">
              <div className="text-xs text-muted-foreground">Total Equity</div>
              <div className="text-3xl font-bold tracking-tight">
                {formatUsd(account.total_equity)}
              </div>
              <div className={`text-sm ${account.total_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {formatPct(account.total_return_pct)} ({formatUsd(account.total_return_usdt)})
              </div>
            </div>

            {/* Balance Breakdown */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-background/40 border border-border/40">
                <div className="text-[11px] text-muted-foreground mb-1">Available</div>
                <div className="font-semibold">{formatUsd(account.balance_usdt)}</div>
              </div>
              <div className="p-3 rounded-lg bg-background/40 border border-border/40">
                <div className="text-[11px] text-muted-foreground mb-1">Open Exposure</div>
                <div className="font-semibold">{performance.open_exposure_pct.toFixed(1)}%</div>
              </div>
            </div>

            {/* Trade Stats */}
            <div className="pt-2 border-t border-border/40 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Total Trades</span>
                <span className="font-medium">{account.total_trades}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Win Rate</span>
                <span className={`font-medium ${account.win_rate >= 50 ? 'text-emerald-400' : ''}`}>
                  {account.win_rate.toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">W/L</span>
                <span className="font-medium text-emerald-400">{account.winning_trades}</span>
                <span className="text-muted-foreground">/</span>
                <span className="font-medium text-rose-400">{account.losing_trades}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* ML Signal */}
        <Card className={`bg-card/40 border-primary/20 ${
          ml_signal.sentiment === 'bullish' ? 'border-emerald-500/30' : 
          ml_signal.sentiment === 'bearish' ? 'border-rose-500/30' : ''
        }`}>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                ML Signal
              </CardTitle>
              <Badge variant="outline" className="text-[10px]">
                Live
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Signal & Confidence */}
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-muted-foreground mb-1">BTC/USDT</div>
                {getSignalBadge(ml_signal.signal, ml_signal.confidence)}
              </div>
              <div className="text-right">
                <div className={`text-3xl font-bold ${getConfidenceColor(ml_signal.confidence)}`}>
                  {ml_signal.confidence}%
                </div>
                <div className="text-[10px] text-muted-foreground">Confidence</div>
              </div>
            </div>

            {/* Current Price */}
            <div className="p-3 rounded-lg bg-background/40 border border-border/40">
              <div className="text-[11px] text-muted-foreground mb-1">Current Price</div>
              <div className="text-xl font-semibold">
                ${ml_signal.current_price?.toLocaleString() || "--"}
              </div>
            </div>

            {/* Probability Bars */}
            <div className="space-y-2">
              <div className="text-[11px] text-muted-foreground">Model Probabilities</div>
              
              {/* Long */}
              <div className="space-y-1">
                <div className="flex justify-between text-[10px]">
                  <span className="text-emerald-400">Long</span>
                  <span>{ml_signal.probabilities?.long.toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-emerald-500 transition-all"
                    style={{ width: `${ml_signal.probabilities?.long || 0}%` }}
                  />
                </div>
              </div>

              {/* Short */}
              <div className="space-y-1">
                <div className="flex justify-between text-[10px]">
                  <span className="text-rose-400">Short</span>
                  <span>{ml_signal.probabilities?.short.toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-rose-500 transition-all"
                    style={{ width: `${ml_signal.probabilities?.short || 0}%` }}
                  />
                </div>
              </div>

              {/* Flat */}
              <div className="space-y-1">
                <div className="flex justify-between text-[10px]">
                  <span className="text-muted-foreground">Flat</span>
                  <span>{ml_signal.probabilities?.flat.toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-muted-foreground transition-all"
                    style={{ width: `${ml_signal.probabilities?.flat || 0}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Diagnostics */}
            {(ml_signal.warmup_ready !== undefined || ml_signal.lookback_rows_used !== undefined) && (
              <div className="pt-3 border-t border-border/40 text-[10px] space-y-1">
                <div className="flex justify-between text-muted-foreground">
                  <span>Warmup:</span>
                  <span className={ml_signal.warmup_ready ? "text-emerald-400" : "text-amber-400"}>
                    {ml_signal.warmup_ready ? "Ready" : "Warming Up"}
                  </span>
                </div>
                {ml_signal.lookback_rows_used !== undefined && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Lookback:</span>
                    <span>{ml_signal.lookback_rows_used} rows</span>
                  </div>
                )}
                {ml_signal.action_raw !== undefined && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Raw Action:</span>
                    <span className="font-mono text-[9px] truncate max-w-[100px]" title={String(ml_signal.action_raw)}>
                      {String(ml_signal.action_raw)}
                    </span>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Open Positions */}
      <Card className="bg-card/40 border-primary/20">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Open Positions ({open_positions.length || 0}/{account.max_positions})
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant={available_slots > 0 ? "success" : "destructive"} className="text-[10px]">
                {available_slots} Slots Available
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {open_positions && open_positions.length > 0 ? (
            <div className="space-y-3">
              {open_positions.map((pos) => (
                <div 
                  key={pos.id}
                  className="rounded-lg border border-border/60 bg-background/40 p-3"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold">{pos.symbol}</span>
                      <Badge 
                        variant={pos.side === "Long" ? "success" : "destructive"}
                        className="text-[10px]"
                      >
                        {pos.side}
                      </Badge>
                      <span className="text-[10px] text-muted-foreground">{pos.leverage}x</span>
                    </div>
                    <div className={`text-right ${pos.unrealized_pnl_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      <div className="text-sm font-bold">
                        {pos.unrealized_pnl_pct >= 0 ? '+' : ''}{pos.unrealized_pnl_pct.toFixed(2)}%
                      </div>
                      <div className="text-[10px]">
                        {pos.unrealized_pnl_usdt >= 0 ? '+' : ''}{formatUsd(pos.unrealized_pnl_usdt)}
                      </div>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-[11px]">
                    <div>
                      <span className="text-muted-foreground">Entry</span>
                      <div className="font-medium">${pos.entry_price.toLocaleString()}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Size</span>
                      <div className="font-medium">{formatUsd(pos.size_usdt)}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Open</span>
                      <div className="font-medium">
                        {new Date(pos.open_time).toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" })}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-6 text-sm text-muted-foreground">
              No open positions
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recent Trades */}
      <Card className="bg-card/40 border-primary/20">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Recent Closed Trades
          </CardTitle>
        </CardHeader>
        <CardContent>
          {recent_trades && recent_trades.length > 0 ? (
            <div className="space-y-2">
              {recent_trades.slice(0, 5).map((trade) => (
                <div 
                  key={trade.id}
                  className="flex items-center justify-between p-2 rounded-md bg-background/30 border border-border/40"
                >
                  <div className="flex items-center gap-2">
                    <Badge 
                      variant={trade.side === "Long" ? "success" : "destructive"}
                      className="text-[9px]"
                    >
                      {trade.side}
                    </Badge>
                    <span className="text-xs font-medium">{trade.symbol}</span>
                    <span className="text-[10px] text-muted-foreground">
                      {trade.close_reason}
                    </span>
                  </div>
                  <div className={`text-sm font-medium ${trade.pnl_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct.toFixed(2)}%
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4 text-sm text-muted-foreground">
              No closed trades yet
            </div>
          )}
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <Card className="bg-card/40 border-primary/20">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            <Target className="w-4 h-4" />
            Performance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 rounded-lg bg-background/40 border border-border/40 text-center">
              <div className="text-[10px] text-muted-foreground mb-1">Total Return</div>
              <div className={`text-lg font-semibold ${performance.total_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {formatPct(performance.total_return_pct)}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-background/40 border border-border/40 text-center">
              <div className="text-[10px] text-muted-foreground mb-1">Win Rate</div>
              <div className={`text-lg font-semibold ${performance.win_rate >= 50 ? 'text-emerald-400' : ''}`}>
                {performance.win_rate.toFixed(1)}%
              </div>
            </div>
            <div className="p-3 rounded-lg bg-background/40 border border-border/40 text-center">
              <div className="text-[10px] text-muted-foreground mb-1">Avg Win</div>
              <div className="text-lg font-semibold text-emerald-400">
                +{performance.avg_win_pct?.toFixed(2) || 0}%
              </div>
            </div>
            <div className="p-3 rounded-lg bg-background/40 border border-border/40 text-center">
              <div className="text-[10px] text-muted-foreground mb-1">Avg Loss</div>
              <div className="text-lg font-semibold text-rose-400">
                {performance.avg_loss_pct?.toFixed(2) || 0}%
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
