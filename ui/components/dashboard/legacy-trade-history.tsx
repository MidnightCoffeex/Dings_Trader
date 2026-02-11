"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { HelpCircle } from "lucide-react";

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

interface PaperDashboardData {
  open_positions: PaperPosition[];
  recent_trades: PaperTrade[];
}

interface LegacyTradeHistoryProps {
  modelId: string;
  initialPositions?: PaperPosition[];
  initialTrades?: PaperTrade[];
}

// Ensure timestamp is treated as UTC
function toUtcTimestamp(rawTime: string): string {
  return rawTime.endsWith('Z') ? rawTime : `${rawTime}Z`;
}

export function LegacyTradeHistory({ 
  modelId, 
  initialPositions = [], 
  initialTrades = [] 
}: LegacyTradeHistoryProps) {
  const [openPositions, setOpenPositions] = useState<PaperPosition[]>(initialPositions);
  const [recentTrades, setRecentTrades] = useState<PaperTrade[]>(initialTrades);
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
      setOpenPositions(data.open_positions || []);
      setRecentTrades(data.recent_trades || []);
      setError(null);
    } catch (err) {
      console.error("Failed to fetch trade history:", err);
      setError("Failed to update");
    }
  }, [modelId]);

  // Set up polling interval
  useEffect(() => {
    // Fetch immediately on mount
    fetchData();
    
    // Poll every 2 seconds
    const interval = setInterval(fetchData, 2000);
    
    return () => clearInterval(interval);
  }, [fetchData]);

  // Close reason badge variant
  const getCloseReasonVariant = (reason?: string): "default" | "secondary" | "destructive" | "outline" | "success" | "warning" | "purple" => {
    if (!reason) return 'outline';
    if (reason === 'TP') return 'success';
    if (reason === 'SL') return 'destructive';
    return 'secondary';
  };

  return (
    <Card className="bg-card/40">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Vollständige Trade‑Historie (Legacy)
          </CardTitle>
          {error && (
            <span className="text-[10px] text-rose-400">{error}</span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto rounded-md border border-border/70">
          <div className="max-h-[400px] overflow-y-auto">
            <div className="min-w-[900px]">
              {/* Table Header */}
              <div className="grid grid-cols-8 bg-background/40 px-3 py-2 text-xs text-muted-foreground sticky top-0 z-10">
                <div>Zeit</div>
                <div>Symbol</div>
                <div className="flex items-center gap-1">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="flex items-center gap-1 cursor-help hover:text-foreground transition-colors">
                          Richtung
                          <HelpCircle className="w-3 h-3" />
                        </span>
                      </TooltipTrigger>
                      <TooltipContent side="top" className="max-w-[220px]">
                        <div className="space-y-1.5">
                          <p className="font-semibold text-xs">📈 Richtung (Long/Short)</p>
                          <div className="space-y-1 text-[11px]">
                            <div className="flex items-center gap-2">
                              <Badge variant="success" className="text-[9px]">Long</Badge>
                              <span>Kauf - Erwartung steigender Preise</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant="destructive" className="text-[9px]">Short</Badge>
                              <span>Verkauf - Erwartung fallender Preise</span>
                            </div>
                          </div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <div className="text-right flex items-center justify-end gap-1">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="flex items-center gap-1 cursor-help hover:text-foreground transition-colors">
                          Status
                          <HelpCircle className="w-3 h-3" />
                        </span>
                      </TooltipTrigger>
                      <TooltipContent side="top" className="max-w-[200px]">
                        <div className="space-y-1.5">
                          <p className="font-semibold text-xs">📊 Trade Status</p>
                          <div className="space-y-1 text-[11px]">
                            <div className="flex items-center gap-2">
                              <Badge variant="warning" className="text-[9px]">OPEN</Badge>
                              <span>Position aktiv</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-[9px]">CLOSED</Badge>
                              <span>Position geschlossen</span>
                            </div>
                          </div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <div className="text-right flex items-center justify-end gap-1">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="flex items-center gap-1 cursor-help hover:text-foreground transition-colors">
                          Close Reason
                          <HelpCircle className="w-3 h-3" />
                        </span>
                      </TooltipTrigger>
                      <TooltipContent side="top" className="max-w-[240px]">
                        <div className="space-y-1.5">
                          <p className="font-semibold text-xs">🔚 Close Reason (Schlussgrund)</p>
                          <div className="space-y-1 text-[11px]">
                            <div className="flex items-center gap-2">
                              <Badge variant="success" className="text-[9px]">TP</Badge>
                              <span>Take Profit - Gewinnziel erreicht</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant="destructive" className="text-[9px]">SL</Badge>
                              <span>Stop Loss - Verlustbegrenzung</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant="secondary" className="text-[9px]">SIGNAL_FLIP</Badge>
                              <span>Signalrichtung geändert</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant="warning" className="text-[9px]">EXPIRED</Badge>
                              <span>48h Zeitlimit abgelaufen</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-[9px]">MANUAL</Badge>
                              <span>Manuell geschlossen</span>
                            </div>
                          </div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <div className="text-right">Investment</div>
                <div className="text-right">Dauer</div>
                <div className="text-right">PnL</div>
              </div>

              {/* Open Positions */}
              {openPositions.map((pos, idx) => {
                const openTime = new Date(toUtcTimestamp(pos.open_time));
                const timeStr = openTime.toLocaleString('de-DE', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });

                // Calculate duration since open
                const now = new Date();
                const diffMs = now.getTime() - openTime.getTime();
                const hours = Math.floor(diffMs / (1000 * 60 * 60));
                const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
                const durationStr = hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;

                const pnlPositive = (pos.unrealized_pnl_pct || 0) >= 0;

                return (
                  <div
                    key={`open-${pos.id}-${idx}`}
                    className="grid grid-cols-8 items-center px-3 py-2 text-sm border-t border-border/70 bg-background/20"
                  >
                    <div className="font-medium">{timeStr}</div>
                    <div>{pos.symbol}</div>
                    <div>
                      <Badge variant={pos.side === 'Long' ? 'success' : 'destructive'}>{pos.side}</Badge>
                    </div>
                    <div className="text-right">
                      <Badge variant="warning">OPEN</Badge>
                    </div>
                    <div className="text-right text-muted-foreground">-</div>
                    <div className="text-right">{pos.size_usdt ? `${Math.abs(pos.size_usdt).toFixed(0)} USDT` : '-'}</div>
                    <div className="text-right text-muted-foreground">{durationStr}</div>
                    <div
                      className={`text-right font-medium ${
                        pnlPositive ? "text-emerald-300" : "text-rose-300"
                      }`}
                    >
                      {pos.unrealized_pnl_pct !== undefined ? `${pos.unrealized_pnl_pct >= 0 ? '+' : ''}${pos.unrealized_pnl_pct.toFixed(2)}%` : '-'}
                      {pos.unrealized_pnl_usdt !== undefined && (
                        <span className="text-xs ml-1 text-muted-foreground">
                          ({pos.unrealized_pnl_usdt >= 0 ? '+' : ''}${pos.unrealized_pnl_usdt.toFixed(0)})
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}

              {/* Closed Trades */}
              {recentTrades.map((t, idx) => {
                const positive = (t.pnl_pct || 0) >= 0;
                const closeTime = t.close_time ? new Date(toUtcTimestamp(t.close_time)) : null;
                const timeStr = closeTime ? closeTime.toLocaleString('de-DE', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '-';

                // Calculate duration
                let durationStr = '-';
                if (t.open_time && t.close_time) {
                  const start = new Date(toUtcTimestamp(t.open_time));
                  const end = new Date(toUtcTimestamp(t.close_time));
                  const diffMs = end.getTime() - start.getTime();
                  const hours = Math.floor(diffMs / (1000 * 60 * 60));
                  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
                  durationStr = hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
                }

                return (
                  <div
                    key={`${t.id}-${idx}`}
                    className="grid grid-cols-8 items-center px-3 py-2 text-sm border-t border-border/70 bg-background/20"
                  >
                    <div className="font-medium">{timeStr}</div>
                    <div>{t.symbol}</div>
                    <div>
                      <Badge variant={t.side === 'Long' ? 'success' : 'destructive'}>{t.side}</Badge>
                    </div>
                    <div className="text-right">
                      <Badge variant="outline">CLOSED</Badge>
                    </div>
                    <div className="text-right">
                      {t.close_reason ? (
                        <Badge variant={getCloseReasonVariant(t.close_reason)}>{t.close_reason}</Badge>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </div>
                    <div className="text-right">{t.size_usdt ? `${Math.abs(t.size_usdt).toFixed(0)} USDT` : '-'}</div>
                    <div className="text-right text-muted-foreground">{durationStr}</div>
                    <div
                      className={`text-right font-medium ${
                        positive ? "text-emerald-300" : "text-rose-300"
                      }`}
                    >
                      {t.pnl_pct !== undefined ? `${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}%` : '-'}
                      {t.pnl_usdt !== undefined && (
                        <span className="text-xs ml-1 text-muted-foreground">
                          ({t.pnl_usdt >= 0 ? '+' : ''}${t.pnl_usdt.toFixed(0)})
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}

              {openPositions.length === 0 && recentTrades.length === 0 && (
                <div className="px-3 py-4 text-sm text-center text-muted-foreground">
                  Keine Trades vorhanden
                </div>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
