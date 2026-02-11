"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown } from "lucide-react";

interface ClosedTrade {
  id: string;
  symbol: string;
  side: "Long" | "Short";
  entryPrice: number;
  exitPrice: number;
  exitTime: Date;
  pnl: number;
  pnlUsd: number;
  exitReason: "TP" | "SL" | "Timeout" | "Manual";
  duration: string;
}

interface ClosedTradesProps {
  trades?: Array<{
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
  }>;
}

// Ensure timestamp is treated as UTC
function toUtcTimestamp(rawTime: string): string {
  return rawTime.endsWith('Z') ? rawTime : `${rawTime}Z`;
}

// Calculate duration between two timestamps
function calculateDuration(openTime?: string, closeTime?: string): string {
  if (!openTime || !closeTime) return "-";
  const start = new Date(toUtcTimestamp(openTime));
  const end = new Date(toUtcTimestamp(closeTime));
  const diff = end.getTime() - start.getTime();
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  return `${hours}h ${minutes}m`;
}

export function ClosedTrades({ trades = [] }: ClosedTradesProps) {
  // Transform API trades to display format - show ALL trades
  const displayTrades: ClosedTrade[] = trades.length > 0
    ? trades.map(t => {
        const pnl = t.pnl_pct || 0;
        return {
          id: String(t.id),
          symbol: t.symbol,
          side: t.side as "Long" | "Short",
          entryPrice: t.entry_price,
          exitPrice: t.exit_price,
          exitTime: t.close_time ? new Date(toUtcTimestamp(t.close_time)) : new Date(),
          pnl: pnl,
          pnlUsd: t.pnl_usdt || 0,
          exitReason: (t.close_reason === "TP" ? "TP" : 
                      t.close_reason === "SL" ? "SL" : 
                      t.close_reason === "EXPIRED" ? "Timeout" : "Manual") as "TP" | "SL" | "Timeout" | "Manual",
          duration: calculateDuration(t.open_time, t.close_time),
        };
      })
    : [];

  const getExitBadge = (reason: string) => {
    switch (reason) {
      case "TP":
        return <Badge variant="success" className="text-[10px] 2xl:text-xs">TP</Badge>;
      case "SL":
        return <Badge variant="destructive" className="text-[10px] 2xl:text-xs">SL</Badge>;
      case "Timeout":
        return <Badge variant="warning" className="text-[10px] 2xl:text-xs">48h</Badge>;
      default:
        return <Badge variant="outline" className="text-[10px] 2xl:text-xs">Manual</Badge>;
    }
  };

  const winCount = displayTrades.filter(t => t.pnl > 0).length;
  const winRate = displayTrades.length > 0 
    ? Math.round(winCount / displayTrades.length * 100) 
    : 0;

  return (
    <Card className="bg-card/40">
      <CardHeader className="pb-3 2xl:pb-4">
        <CardTitle className="text-sm 2xl:text-base font-medium text-muted-foreground">
          Closed Trades ({displayTrades.length})
        </CardTitle>
      </CardHeader>
      <CardContent className="2xl:min-h-[400px]">
        <div className="max-h-[400px] overflow-y-auto pr-1 space-y-2 2xl:space-y-4">
          {displayTrades.map((trade) => (
            <div
              key={trade.id}
              className="flex items-center justify-between p-2.5 2xl:p-3.5 rounded-lg border border-border/50 bg-background/30 hover:bg-background/50 transition-colors"
            >
              {/* Left: Symbol + Side */}
              <div className="flex items-center gap-2 2xl:gap-3 min-w-[100px]">
                <div className={`w-1 h-8 2xl:h-10 rounded-full ${trade.side === 'Long' ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                <div>
                  <div className="font-medium text-sm 2xl:text-base">{trade.symbol}</div>
                  <div className="text-[10px] 2xl:text-xs text-muted-foreground flex items-center gap-1">
                    {trade.side === 'Long' ? (
                      <TrendingUp className="w-3 h-3 2xl:w-4 2xl:h-4 text-emerald-500" />
                    ) : (
                      <TrendingDown className="w-3 h-3 2xl:w-4 2xl:h-4 text-rose-500" />
                    )}
                    {trade.side}
                  </div>
                </div>
              </div>

              {/* Middle: Exit Reason */}
              <div className="flex flex-col items-center gap-1">
                {getExitBadge(trade.exitReason)}
                <span className="text-[9px] 2xl:text-xs text-muted-foreground">
                  ${trade.exitPrice.toLocaleString(undefined, {maximumFractionDigits: 0})}
                </span>
              </div>

              {/* Right: P&L */}
              <div className={`text-right min-w-[70px] ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                <div className="font-bold text-sm 2xl:text-base">
                  {trade.pnl >= 0 ? '+' : ''}{trade.pnl.toFixed(1)}%
                </div>
                <div className="text-[10px] 2xl:text-xs">
                  {trade.pnlUsd >= 0 ? '+' : ''}${trade.pnlUsd.toFixed(0)}
                </div>
              </div>
            </div>
          ))}

          {displayTrades.length === 0 && (
            <div className="text-center py-4 2xl:py-6 text-sm 2xl:text-base text-muted-foreground">
              Noch keine abgeschlossenen Trades
            </div>
          )}
        </div>

        {/* Win Rate Summary */}
        {displayTrades.length > 0 && (
          <div className="mt-3 2xl:mt-4 pt-3 2xl:pt-4 border-t border-border/40 flex items-center justify-between text-xs 2xl:text-sm">
            <span className="text-muted-foreground">Win Rate (letzte {displayTrades.length})</span>
            <span className="font-medium">{winRate}%</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
