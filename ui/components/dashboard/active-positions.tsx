"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Timer } from "lucide-react";
import { useEffect, useState, useCallback } from "react";

interface Position {
  id: string;
  symbol: string;
  side: "Long" | "Short";
  entryPrice: number;
  takeProfit: number;
  stopLoss: number;
  leverage: number;
  size: number;
  sizeUsd: number;
  openTime: Date;
  unrealizedPnl: number;
  unrealizedPnlUsd: number;
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

interface PaperDashboardData {
  account: {
    model_id: string;
    total_equity: number;
    max_positions: number;
  };
  open_positions: PaperPosition[];
  open_positions_count: number;
}

interface ActivePositionsProps {
  positions?: PaperPosition[];
  maxPositions?: number;
  modelId?: string;
}

function formatDuration(startTime: Date): string {
  const now = new Date();
  const diff = now.getTime() - new Date(startTime).getTime();
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  return `${hours}h ${minutes}m`;
}

// Ensure timestamp is treated as UTC
function toUtcTimestamp(rawTime: string): string {
  return rawTime.endsWith('Z') ? rawTime : `${rawTime}Z`;
}

function getTimeRemaining(startTime: Date, limitHours: number = 48): string {
  const now = new Date();
  const start = new Date(toUtcTimestamp(startTime.toISOString()));
  const deadline = new Date(start.getTime() + limitHours * 60 * 60 * 1000);
  const diff = deadline.getTime() - now.getTime();
  
  if (diff <= 0) return "Abgelaufen";
  
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  return `${hours}h ${minutes}m`;
}

// Calculate TP/SL prices based on entry and percentages
function calculateTP(entryPrice: number, side: string, tpPct: number, leverage: number): number {
  if (side === "Long") {
    return entryPrice * (1 + (tpPct / 100) / leverage);
  } else {
    return entryPrice * (1 - (tpPct / 100) / leverage);
  }
}

function calculateSL(entryPrice: number, side: string, slPct: number, leverage: number): number {
  if (side === "Long") {
    return entryPrice * (1 + (slPct / 100) / leverage);
  } else {
    return entryPrice * (1 - (slPct / 100) / leverage);
  }
}

export function ActivePositions({ 
  positions: initialPositions = [], 
  maxPositions: initialMaxPositions = 5,
  modelId = "paper_ppo_v1"
}: ActivePositionsProps) {
  const [positions, setPositions] = useState<PaperPosition[]>(initialPositions);
  const [maxPositions, setMaxPositions] = useState(initialMaxPositions);
  const [mounted, setMounted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Polling function to fetch data every 2 seconds
  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(`/api/paper/dashboard/${modelId}`, {
        cache: "no-store",
        headers: { "Accept": "application/json" },
      });
      
      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }
      
      const data: PaperDashboardData = await res.json();
      setPositions(data.open_positions || []);
      setMaxPositions(data.account?.max_positions || 5);
      setError(null);
    } catch (err) {
      console.error("Failed to fetch positions:", err);
      setError("Failed to update");
    }
  }, [modelId]);

  // Set up polling interval
  useEffect(() => {
    setMounted(true);
    
    // Fetch immediately on mount
    fetchData();
    
    // Poll every 2 seconds
    const interval = setInterval(fetchData, 2000);
    
    return () => clearInterval(interval);
  }, [fetchData]);

  // Transform API positions to display format
  const displayPositions: Position[] = positions.length > 0
    ? positions.map(p => ({
        id: String(p.id),
        symbol: p.symbol,
        side: p.side as "Long" | "Short",
        entryPrice: p.entry_price,
        size: (p.size_usdt / 10000) * 100, // Convert to % of 10k portfolio
        sizeUsd: p.size_usdt,
        leverage: p.leverage,
        takeProfit: calculateTP(p.entry_price, p.side, p.take_profit_pct || 5, p.leverage),
        stopLoss: calculateSL(p.entry_price, p.side, p.stop_loss_pct || -3, p.leverage),
        openTime: new Date(toUtcTimestamp(p.open_time)),
        unrealizedPnl: p.unrealized_pnl_pct || 0,
        unrealizedPnlUsd: p.unrealized_pnl_usdt || 0,
      }))
    : [];

  return (
    <Card className="bg-card/40 border-primary/20">
      <CardHeader className="pb-3 2xl:pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm 2xl:text-base font-medium text-muted-foreground">
            Aktive Positionen ({displayPositions.length}/{maxPositions})
          </CardTitle>
          <div className="flex items-center gap-2">
            {error && (
              <span className="text-[10px] text-rose-400">{error}</span>
            )}
            <Badge variant="outline" className="text-[10px] 2xl:text-xs">Max 10% Cap</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 2xl:space-y-4">
        <div className="max-h-[750px] 2xl:max-h-[900px] overflow-y-auto pr-1 space-y-3 2xl:space-y-4">
          {displayPositions.map((pos) => (
            <div
              key={pos.id}
              className="rounded-lg border border-border/60 bg-background/40 p-3 2xl:p-4 space-y-2 2xl:space-y-3"
            >
              {/* Header: Symbol + Side + Timer */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-semibold 2xl:text-base">{pos.symbol}</span>
                  <Badge
                    variant={pos.side === "Long" ? "success" : "destructive"}
                    className="text-[10px] 2xl:text-xs"
                  >
                    {pos.side}
                  </Badge>
                  <span className="text-[10px] 2xl:text-xs text-muted-foreground">
                    {pos.leverage}x
                  </span>
                </div>
                <div className="flex items-center gap-1 text-[10px] 2xl:text-xs text-muted-foreground">
                  <Timer className="w-3 h-3 2xl:w-3.5 2xl:h-3.5" />
                  {mounted ? getTimeRemaining(pos.openTime, 48) : "--"}
                </div>
              </div>

              {/* Entry / TP / SL */}
              <div className="grid grid-cols-3 gap-2 text-[11px] 2xl:text-xs">
                <div className="space-y-0.5">
                  <span className="text-muted-foreground">Entry</span>
                  <div className="font-medium">${pos.entryPrice.toLocaleString()}</div>
                </div>
                <div className="space-y-0.5">
                  <span className="text-muted-foreground">TP</span>
                  <div className="font-medium text-emerald-400">${pos.takeProfit.toLocaleString(undefined, {maximumFractionDigits: 0})}</div>
                </div>
                <div className="space-y-0.5">
                  <span className="text-muted-foreground">SL</span>
                  <div className="font-medium text-rose-400">${pos.stopLoss.toLocaleString(undefined, {maximumFractionDigits: 0})}</div>
                </div>
              </div>

              {/* P&L + Size */}
              <div className="flex items-center justify-between pt-1 border-t border-border/30">
                <div className="flex flex-col">
                  <span className="text-[11px] 2xl:text-xs text-muted-foreground">
                    Invest: ${pos.sizeUsd.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0})} ({pos.size.toFixed(1)}%)
                  </span>
                  <span className="text-[10px] 2xl:text-xs text-muted-foreground opacity-70">
                    {mounted ? formatDuration(pos.openTime) : "--"} Laufzeit
                  </span>
                </div>
                <div className={`text-right ${pos.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  <div className="text-sm 2xl:text-base font-bold">
                    {pos.unrealizedPnl >= 0 ? '+' : ''}{pos.unrealizedPnl.toFixed(2)}%
                  </div>
                  <div className="text-[10px] 2xl:text-xs">
                    ${pos.unrealizedPnlUsd >= 0 ? '+' : ''}{pos.unrealizedPnlUsd.toFixed(2)}
                  </div>
                </div>
              </div>

              {/* Countdown Bar */}
              <div className="relative h-1 2xl:h-1.5 bg-muted rounded-full overflow-hidden">
                <div
                  className="absolute inset-y-0 left-0 bg-primary/60 transition-all duration-1000"
                  style={{ width: `${Math.max(0, Math.min(100, (1 - (Date.now() - new Date(pos.openTime).getTime()) / (48 * 60 * 60 * 1000)) * 100))}%` }}
                />
              </div>
              <div className="text-[9px] 2xl:text-[11px] text-muted-foreground text-center">
                48h Limit · Auto-Exit bei Nichterreichen
              </div>
            </div>
          ))}

          {displayPositions.length === 0 && (
            <div className="text-center py-6 2xl:py-8 text-sm 2xl:text-base text-muted-foreground">
              Keine aktiven Positionen
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
