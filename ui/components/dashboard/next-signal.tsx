"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Activity, Clock, Target, RefreshCw, Info } from "lucide-react";

interface SignalData {
  direction: "bullish" | "neutral" | "bearish";
  confidence: number;
  timeframe: string;
  nextEntry?: {
    price: number;
    type: "Long" | "Short";
  };
  indicators: {
    rsi: number;
    macd: "bullish" | "bearish" | "neutral";
    momentum: number;
  };
  timestamp: Date;
}

interface NextSignalPreviewProps {
  signal?: {
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
  };
  symbol?: string;
  modelId?: string;
  noCard?: boolean;
}

export function NextSignalPreview({ 
  signal: initialSignal,
  symbol = "BTC/USDT",
  modelId = "paper_ppo_v1",
  noCard = false,
}: NextSignalPreviewProps) {
  const [signal, setSignal] = useState(initialSignal);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Fetch fresh signal data via API proxy
  const fetchSignal = useCallback(async () => {
    if (!modelId) return;
    
    try {
      setIsRefreshing(true);
      const res = await fetch(`/api/paper/dashboard/${modelId}`, {
        cache: "no-store",
        headers: { "Accept": "application/json" },
      });
      
      if (res.ok) {
        const data = await res.json();
        if (data.ml_signal) {
          setSignal(data.ml_signal);
          setLastUpdate(new Date());
        }
      }
    } catch (err) {
      console.error("Failed to refresh signal:", err);
    } finally {
      setIsRefreshing(false);
    }
  }, [modelId]);

  // Auto-refresh every 2 seconds for real-time feel
  useEffect(() => {
    // Initial fetch
    fetchSignal();
    
    // Set up interval
    const interval = setInterval(() => {
      fetchSignal();
    }, 2000); // 2 seconds

    return () => clearInterval(interval);
  }, [fetchSignal]);
  // Transform API signal to display format
  const getDirectionFromSignal = (sig: string): "bullish" | "neutral" | "bearish" => {
    switch (sig) {
      case "LONG": return "bullish";
      case "SHORT": return "bearish";
      default: return "neutral";
    }
  };

  // Use real signal data or fallback
  const direction = signal ? getDirectionFromSignal(signal.signal) : "neutral";
  const confidence = signal?.confidence || 0;
  const currentPrice = signal?.current_price || 0;
  
  // Build display data
  const data: SignalData = {
    direction,
    confidence,
    timeframe: "60s",
    nextEntry: currentPrice > 0 ? {
      price: currentPrice,
      type: direction === "bullish" ? "Long" : direction === "bearish" ? "Short" : "Long",
    } : undefined,
    indicators: {
      rsi: 50, // Would need RSI from API
      macd: direction === "bullish" ? "bullish" : direction === "bearish" ? "bearish" : "neutral",
      momentum: signal?.probabilities ? 
        (signal.probabilities.long - signal.probabilities.short) : 0,
    },
    timestamp: lastUpdate,
  };

  const getDirectionBadge = (dir: string) => {
    switch (dir) {
      case "bullish":
        return (
          <Badge variant="success" className="flex items-center gap-1 px-3 py-1">
            <TrendingUp className="w-3 h-3" />
            BULLISH
          </Badge>
        );
      case "bearish":
        return (
          <Badge variant="destructive" className="flex items-center gap-1 px-3 py-1">
            <TrendingDown className="w-3 h-3" />
            BEARISH
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="flex items-center gap-1 px-3 py-1">
            <Activity className="w-3 h-3" />
            FLAT
          </Badge>
        );
    }
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 70) return "text-emerald-400";
    if (conf >= 50) return "text-amber-400";
    return "text-muted-foreground";
  };

  const content = (
    <>
      {!noCard && (
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              Next Signal Preview
              <RefreshCw className={`w-3 h-3 ${isRefreshing ? 'animate-spin' : ''}`} />
            </CardTitle>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1 text-[10px] text-muted-foreground" title="Signal wird alle 2 Sekunden aktualisiert">
                <Clock className="w-3 h-3" />
                alle 2s
              </div>
              <div className="text-[9px] text-muted-foreground/60">
                {lastUpdate.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </div>
            </div>
          </div>
        </CardHeader>
      )}
      <CardContent className={`space-y-3 2xl:space-y-4 ${noCard ? '' : ''}`}>
        {/* Haupt-Signal */}
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs text-muted-foreground mb-1">{symbol}</div>
            {getDirectionBadge(data.direction)}
          </div>
          <div className="text-right">
            <div className={`text-2xl font-bold ${getConfidenceColor(data.confidence)}`}>
              {data.confidence}%
            </div>
            <div className="text-[10px] text-muted-foreground flex items-center gap-1"
                 title="Confidence = Wie sicher ist sich das ML-Modell (0-100%)&#10;• 80%+ = starkes Signal&#10;• 60-79% = moderat&#10;• <60% = schwach">
              Confidence
              <Info 
                className="w-3 h-3 cursor-help text-muted-foreground/60 hover:text-muted-foreground" 
              />
            </div>
          </div>
        </div>

        {/* Current Price / Entry Zone */}
        {data.nextEntry && currentPrice > 0 && (
          <div className="p-3 rounded-lg bg-background/40 border border-border/40">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
              <Target className="w-3 h-3" />
              Current Price
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-semibold">
                ${data.nextEntry.price.toLocaleString(undefined, {maximumFractionDigits: 0})}
              </span>
              <Badge 
                variant={data.direction === "bullish" ? "success" : data.direction === "bearish" ? "destructive" : "outline"}
                className="text-[10px]"
              >
                {data.direction === "bullish" ? "LONG" : data.direction === "bearish" ? "SHORT" : "WAIT"}
              </Badge>
            </div>
          </div>
        )}

        {/* Indikatoren - zeige Probability Distribution */}
        {signal?.probabilities && (
          <div className="space-y-2">
            <div className="text-[11px] text-muted-foreground">Signal Probabilities</div>
            <div className="space-y-1.5">
              {/* Long */}
              <div className="flex items-center gap-2">
                <span className="text-[10px] w-10 text-emerald-400">Long</span>
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${signal.probabilities.long}%` }} />
                </div>
                <span className="text-[10px] w-8 text-right">{signal.probabilities.long.toFixed(1)}%</span>
              </div>
              {/* Flat */}
              <div className="flex items-center gap-2">
                <span className="text-[10px] w-10 text-muted-foreground">Flat</span>
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-muted-foreground/50 rounded-full" style={{ width: `${signal.probabilities.flat}%` }} />
                </div>
                <span className="text-[10px] w-8 text-right">{signal.probabilities.flat.toFixed(1)}%</span>
              </div>
              {/* Short */}
              <div className="flex items-center gap-2">
                <span className="text-[10px] w-10 text-rose-400">Short</span>
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-rose-500 rounded-full" style={{ width: `${signal.probabilities.short}%` }} />
                </div>
                <span className="text-[10px] w-8 text-right">{signal.probabilities.short.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}

        {/* Letzte Aktualisierung */}
        <div className="pt-2 border-t border-border/30 text-[10px] text-muted-foreground text-center">
          Signal: {signal?.signal || "FLAT"} · 
          Auto-refresh: 2s ·
          Updated: {lastUpdate.toLocaleTimeString('de-DE', { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit'
          })}
        </div>
      </CardContent>
    </>
  );

  if (noCard) {
    return content;
  }

  return (
    <Card className={`bg-card/40 border-primary/20 ${
      data.direction === 'bullish' ? 'border-emerald-500/20' : 
      data.direction === 'bearish' ? 'border-rose-500/20' : ''
    }`}>
      {content}
    </Card>
  );
}
