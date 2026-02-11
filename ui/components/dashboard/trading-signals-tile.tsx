"use client";

import { useState, useEffect, useCallback } from "react";
import { Badge } from "@/components/ui/badge";
import { RefreshCw } from "lucide-react";

interface SignalData {
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
  action_raw?: number[];
}

interface PaperDashboardData {
  ml_signal: SignalData;
}

interface TradingSignalsTileProps {
  modelId: string;
  initialSignal?: SignalData;
  symbol?: string;
}

function SignalGauge({ value }: { value: number }) {
  // value is -1.0 to 1.0
  // Map to percentage 0 to 100
  const percentage = Math.min(Math.max(((value + 1) / 2) * 100, 0), 100);
  
  // Color determination
  let colorClass = "bg-slate-400";
  if (value >= 0.1) colorClass = "bg-emerald-500";
  else if (value <= -0.1) colorClass = "bg-rose-500";
  
  return (
    <div className="w-full flex flex-col gap-1.5 pt-2">
      <div className="flex justify-between text-[9px] text-muted-foreground px-0.5 uppercase tracking-wider font-semibold">
        <span>Short</span>
        <span>Flat</span>
        <span>Long</span>
      </div>
      <div className="relative h-2.5 w-full bg-secondary/50 rounded-full overflow-hidden border border-border/50">
        {/* Center marker */}
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-foreground/20 -translate-x-1/2 z-10" />
        
        {/* Bar from Center */}
        <div 
          className={`absolute top-0 bottom-0 transition-all duration-500 ${colorClass}`}
          style={{ 
            left: value < 0 ? `${percentage}%` : '50%',
            right: value > 0 ? `${100 - percentage}%` : '50%',
            opacity: 0.9
          }}
        />
      </div>
      <div className="flex justify-between items-center mt-0.5">
        <span className="text-[10px] text-muted-foreground">Decision Strength</span>
        <span className={`text-xs font-mono font-medium ${value >= 0.1 ? "text-emerald-400" : value <= -0.1 ? "text-rose-400" : "text-muted-foreground"}`}>
          {value > 0 ? "+" : ""}{value.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

export function TradingSignalsTile({ 
  modelId, 
  initialSignal,
  symbol = "BTC/USDT"
}: TradingSignalsTileProps) {
  const [signal, setSignal] = useState<SignalData | undefined>(initialSignal);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [error, setError] = useState<string | null>(null);

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
        const data: PaperDashboardData = await res.json();
        if (data.ml_signal) {
          setSignal(data.ml_signal);
          setLastUpdate(new Date());
          setError(null);
        }
      } else {
        throw new Error(`API error: ${res.status}`);
      }
    } catch (err) {
      console.error("Failed to refresh trading signals:", err);
      setError("Update failed");
    } finally {
      setIsRefreshing(false);
    }
  }, [modelId]);

  // Auto-refresh (reduced polling pressure)
  useEffect(() => {
    // Initial fetch
    fetchSignal();

    const REFRESH_MS = 15000;
    const interval = setInterval(() => {
      fetchSignal();
    }, REFRESH_MS);

    return () => clearInterval(interval);
  }, [fetchSignal]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          Trading-Signale
        </h3>
        <div className="flex items-center gap-2">
          {error && (
            <span className="text-[10px] text-rose-400">{error}</span>
          )}
          <Badge variant="purple">{signal?.signal || "FLAT"}</Badge>
          <RefreshCw className={`w-3 h-3 text-muted-foreground ${isRefreshing ? 'animate-spin' : ''}`} />
        </div>
      </div>
      
      <div className="flex flex-col gap-2 rounded-lg border border-border/70 bg-background/40 px-3 py-3 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="text-sm font-medium">{symbol}</div>
          <Badge variant="purple">{signal?.signal || "FLAT"}</Badge>
        </div>
        
        <SignalGauge value={signal?.action_raw?.[0] ?? (
          signal?.signal === "LONG" ? (signal?.confidence || 0) / 100 :
          signal?.signal === "SHORT" ? -(signal?.confidence || 0) / 100 : 0
        )} />
      </div>
      
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span className="italic">Fokus auf Bitcoin Trading - Altcoins deaktiviert</span>
        <span className="text-[9px]">
          Updated: {lastUpdate.toLocaleTimeString('de-DE', { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit'
          })}
        </span>
      </div>
    </div>
  );
}
