"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";

interface TickerData {
  symbol: string;
  lastPrice: number;
  priceChange: number;
  priceChangePercent: number;
}

interface LivePriceBubbleProps {
  modelId: string;
  symbol?: string; // e.g. BTCUSDT
}

function formatPrice(value: number) {
  if (!Number.isFinite(value)) return "-";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: value > 1000 ? 0 : 2,
    maximumFractionDigits: value > 1000 ? 2 : 4,
  });
}

export function LivePriceBubble({ modelId, symbol = "BTCUSDT" }: LivePriceBubbleProps) {
  const [ticker, setTicker] = useState<TickerData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchTicker = useCallback(async () => {
    try {
      const tickerRes = await fetch(`/api/binance/ticker?symbol=${symbol}`);
      if (tickerRes.ok) {
        const data = await tickerRes.json();
        if (Number.isFinite(data?.lastPrice)) {
          setTicker({
            symbol: data.symbol || symbol,
            lastPrice: Number(data.lastPrice),
            priceChange: Number(data.priceChange ?? 0),
            priceChangePercent: Number(data.priceChangePercent ?? 0),
          });
          setLoading(false);
          return;
        }
      }

      // Fallback: price from paper dashboard
      const dashRes = await fetch(`/api/paper/dashboard/${modelId}`);
      if (dashRes.ok) {
        const dashData = await dashRes.json();
        const p = Number(dashData?.ml_signal?.current_price);
        if (Number.isFinite(p) && p > 0) {
          setTicker((prev) => ({
            symbol,
            lastPrice: p,
            priceChange: prev?.priceChange ?? 0,
            priceChangePercent: prev?.priceChangePercent ?? 0,
          }));
          setLoading(false);
        }
      }
    } catch (err) {
      console.error("LivePriceBubble fetch error", err);
    } finally {
      setLoading(false);
    }
  }, [modelId, symbol]);

  useEffect(() => {
    fetchTicker();
    const id = setInterval(fetchTicker, 15000);
    return () => clearInterval(id);
  }, [fetchTicker]);

  const trend = useMemo(() => {
    const pct = Number(ticker?.priceChangePercent ?? 0);
    const up = pct >= 0;
    return {
      up,
      color: up ? "text-emerald-400" : "text-rose-400",
      bg: up ? "bg-emerald-500/10 border-emerald-500/30" : "bg-rose-500/10 border-rose-500/30",
      label: `${up ? "+" : ""}${pct.toFixed(2)}%`,
    };
  }, [ticker]);

  return (
    <Card className="group relative overflow-hidden border-violet-500/30 bg-card/50 transition-all duration-300 hover:border-violet-400/70 hover:shadow-[0_0_28px_rgba(168,85,247,0.35)]">
      <div className="pointer-events-none absolute -left-20 -top-20 h-56 w-56 rounded-full bg-violet-500/25 blur-3xl opacity-40 transition-opacity duration-300 group-hover:opacity-70" />
      <div className="pointer-events-none absolute -right-16 top-0 h-40 w-40 rounded-full bg-fuchsia-500/20 blur-3xl opacity-30 transition-opacity duration-300 group-hover:opacity-60" />

      <CardContent className="relative px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-[10px] uppercase tracking-wide text-muted-foreground/90">
              <span className="relative inline-flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
              </span>
              Live Preis Bubble
            </div>

            <div className="mt-1 flex items-center gap-2">
              <div className="rounded-full border border-violet-400/40 bg-violet-500/15 px-4 py-1.5 shadow-[0_0_18px_rgba(139,92,246,0.32)]">
                <span className="font-mono text-xl font-bold leading-none text-foreground sm:text-2xl">
                  {loading && !ticker ? "…" : `$${formatPrice(Number(ticker?.lastPrice ?? 0))}`}
                </span>
              </div>

              <div className={`rounded-full border px-3 py-1 text-sm font-semibold ${trend.bg} ${trend.color}`}>
                {trend.label}
              </div>
            </div>

            <p className="mt-1 text-[10px] text-muted-foreground/80">24h Trend (Binance-Standard)</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
