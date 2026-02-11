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
}

interface PaperDashboardData {
  ml_signal: SignalData;
}

interface TradingSignalsTileProps {
  modelId: string;
  initialSignal?: SignalData;
  symbol?: string;
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
      
      <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/40 px-3 py-2.5 shadow-sm">
        <div className="text-sm font-medium">{symbol}</div>
        <div className="flex items-center gap-2">
          <Badge variant="purple">{signal?.signal || "FLAT"}</Badge>
          <Badge variant="outline">{signal?.confidence || 0}% Confidence</Badge>
        </div>
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
