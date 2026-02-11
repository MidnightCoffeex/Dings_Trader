"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closeTime: number;
}

interface TickerData {
  symbol: string;
  lastPrice: number;
  priceChange: number;
  priceChangePercent: number;
  high24h: number;
  low24h: number;
  volume24h: number;
  quoteVolume24h: number;
  openPrice: number;
}

interface LiveTickerProps {
  symbol?: string;
}

export function LiveTicker({ symbol = "BTC/USDT" }: LiveTickerProps) {
  const [candles, setCandles] = useState<CandleData[]>([]);
  const [ticker, setTicker] = useState<TickerData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const symbolParam = symbol.replace("/", "");

      // Fetch candles and ticker in parallel
      const [candlesRes, tickerRes] = await Promise.all([
        fetch(`/api/binance/candles?symbol=${symbolParam}&interval=1h&limit=24`),
        fetch(`/api/binance/ticker?symbol=${symbolParam}`),
      ]);

      if (!candlesRes.ok) {
        const txt = await candlesRes.text();
        throw new Error(`Candles fetch failed: ${candlesRes.status} ${txt}`);
      }
      if (!tickerRes.ok) {
        const txt = await tickerRes.text();
        throw new Error(`Ticker fetch failed: ${tickerRes.status} ${txt}`);
      }

      const candlesData = await candlesRes.json();
      const tickerData = await tickerRes.json();

      setCandles(candlesData);
      setTicker(tickerData);
      setLastUpdate(new Date());
    } catch (err) {
      console.error("LiveTicker fetch error:", err);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  // Fetch only ticker/price for quick updates
  const fetchPriceOnly = useCallback(async () => {
    try {
      const symbolParam = symbol.replace("/", "");
      const tickerRes = await fetch(`/api/binance/ticker?symbol=${symbolParam}`);
      
      if (tickerRes.ok) {
        const tickerData = await tickerRes.json();
        setTicker(tickerData);
        setLastUpdate(new Date());
      }
    } catch (err) {
      console.error("Price fetch error:", err);
    }
  }, [symbol]);

  // Initial fetch
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Auto-refresh candles every 2 seconds for real-time feel
  useEffect(() => {
    const interval = setInterval(() => {
      fetchData();
    }, 2000); // 2 seconds

    return () => clearInterval(interval);
  }, [fetchData]);

  // Fast price refresh every 2 seconds
  useEffect(() => {
    const priceInterval = setInterval(() => {
      fetchPriceOnly();
    }, 2000); // 2 seconds

    return () => clearInterval(priceInterval);
  }, [fetchPriceOnly]);

  // Format price with appropriate decimals
  const formatPrice = (price: number) => {
    return price.toLocaleString("en-US", {
      minimumFractionDigits: price > 10000 ? 0 : price > 1000 ? 1 : 2,
      maximumFractionDigits: price > 10000 ? 0 : price > 1000 ? 1 : 2,
    });
  };

  // Format time for display
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString("de-DE", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  // Calculate chart dimensions
  const chartHeight = 400;
  const chartWidth = 100; // percentage
  const padding = { top: 10, right: 5, bottom: 20, left: 5 };

  // Calculate min/max for scaling
  const prices = candles.flatMap((c) => [c.high, c.low]);
  const minPrice = prices.length > 0 ? Math.min(...prices) * 0.999 : 0; // -0.1% padding
  const maxPrice = prices.length > 0 ? Math.max(...prices) * 1.001 : 0; // +0.1% padding
  const priceRange = maxPrice - minPrice || 1;

  // Scale price to Y coordinate
  const scaleY = (price: number) => {
    const availableHeight = chartHeight - padding.top - padding.bottom;
    const normalized = (maxPrice - price) / priceRange;
    return padding.top + normalized * availableHeight;
  };

  const isPositive = ticker ? ticker.priceChange >= 0 : true;

  return (
    <Card className="bg-card/40 border-primary/20">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full animate-pulse ${
                loading ? "bg-yellow-500" : "bg-emerald-500"
              }`}
            />
            Live {symbol}
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[10px]">
              1h Kerzen
            </Badge>
            <Badge variant="outline" className="text-[10px] text-emerald-400 border-emerald-400/30">
              Live Price: 2s
            </Badge>
            {error && (
              <Badge variant="destructive" className="text-[10px]">
                Error
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 2xl:space-y-6">
        {/* Preis-Anzeige mit Live-Ticker */}
        <div className="flex items-baseline gap-3">
          {ticker ? (
            <>
              <div className="flex items-center gap-2">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                </span>
                <span className="text-3xl font-bold tracking-tight">
                  ${formatPrice(ticker.lastPrice)}
                </span>
              </div>
              <div
                className={`flex items-center gap-1 text-sm font-medium ${
                  isPositive ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                <span>
                  {isPositive ? "+" : ""}
                  {formatPrice(ticker.priceChange)}
                </span>
                <span>
                  ({isPositive ? "+" : ""}
                  {ticker.priceChangePercent.toFixed(2)}%)
                </span>
              </div>
            </>
          ) : (
            <span className="text-3xl font-bold tracking-tight text-muted-foreground">
              Loading...
            </span>
          )}
        </div>

        {/* 24h Stats */}
        {ticker && (
          <div className="flex gap-4 text-xs text-muted-foreground">
            <div>
              <span className="text-muted-foreground/60">24h High: </span>
              <span className="font-medium">${formatPrice(ticker.high24h)}</span>
            </div>
            <div>
              <span className="text-muted-foreground/60">24h Low: </span>
              <span className="font-medium">${formatPrice(ticker.low24h)}</span>
            </div>
            <div>
              <span className="text-muted-foreground/60">24h Vol: </span>
              <span className="font-medium">
                {(ticker.volume24h / 1000).toFixed(1)}K
              </span>
            </div>
          </div>
        )}

        {/* Candle-Chart - Fixed container with proper aspect ratio handling */}
        <div 
          className="relative bg-background/30 rounded-lg border border-border/40 overflow-hidden h-[200px] 2xl:h-[400px]"
        >
          {candles.length > 0 ? (
            <svg
              width="100%"
              height="100%"
              viewBox={`0 0 100 ${chartHeight}`}
              preserveAspectRatio="none"
              style={{ display: 'block', width: '100%', height: '100%' }}
            >
              {/* Grid-Linien */}
              {[0, 1, 2, 3].map((i) => (
                <line
                  key={i}
                  x1="0"
                  y1={padding.top + (i * (chartHeight - padding.top - padding.bottom)) / 3}
                  x2="100"
                  y2={padding.top + (i * (chartHeight - padding.top - padding.bottom)) / 3}
                  stroke="currentColor"
                  strokeOpacity="0.1"
                  strokeWidth="0.2"
                />
              ))}

              {/* Candle-Sticks - Improved rendering for cross-browser compatibility */}
              {candles.map((candle, idx) => {
                const isGreen = candle.close >= candle.open;
                const totalCandles = candles.length;
                const candleWidth = 3; // Slightly wider for better visibility
                const availableWidth = 100 - padding.left - padding.right;
                const spacing = availableWidth / (totalCandles + 1);
                const x = padding.left + spacing * (idx + 0.5);

                const yHigh = scaleY(candle.high);
                const yLow = scaleY(candle.low);
                const yOpen = scaleY(candle.open);
                const yClose = scaleY(candle.close);

                const bodyTop = Math.min(yOpen, yClose);
                const bodyHeight = Math.max(Math.abs(yClose - yOpen), 0.8); // Minimum visible height

                const color = isGreen ? "#10b981" : "#f43f5e";

                return (
                  <g key={idx}>
                    {/* Upper Wick */}
                    <line
                      x1={x}
                      y1={yHigh}
                      x2={x}
                      y2={bodyTop}
                      stroke={color}
                      strokeWidth="0.5"
                      vectorEffect="non-scaling-stroke"
                    />
                    {/* Body - Using rect with explicit fill */}
                    <rect
                      x={x - candleWidth / 2}
                      y={bodyTop}
                      width={candleWidth}
                      height={bodyHeight}
                      fill={color}
                      stroke={color}
                      strokeWidth="0.2"
                      rx="0.3"
                      shapeRendering="geometricPrecision"
                    />
                    {/* Lower Wick */}
                    <line
                      x1={x}
                      y1={bodyTop + bodyHeight}
                      x2={x}
                      y2={yLow}
                      stroke={color}
                      strokeWidth="0.5"
                      vectorEffect="non-scaling-stroke"
                    />
                  </g>
                );
              })}

              {/* Current price line */}
              {ticker && candles.length > 0 && (
                <line
                  x1="0"
                  y1={scaleY(ticker.lastPrice)}
                  x2="100"
                  y2={scaleY(ticker.lastPrice)}
                  stroke="currentColor"
                  strokeOpacity="0.3"
                  strokeDasharray="2,2"
                  strokeWidth="0.3"
                />
              )}
            </svg>
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground bg-background/50 p-4 text-center">
              {error ? (
                <>
                  <span className="text-destructive font-bold mb-1">Fehler beim Laden</span>
                  <span className="text-[10px] opacity-70">{error}</span>
                </>
              ) : (
                "Lade Daten..."
              )}
            </div>
          )}

          {/* Last update indicator */}
          <div className="absolute bottom-2 right-2 text-[9px] text-muted-foreground/60 bg-background/80 px-2 py-1 rounded">
            {lastUpdate.toLocaleTimeString("de-DE")}
          </div>
        </div>

        {/* Zeit-Achse */}
        <div className="flex justify-between text-[10px] text-muted-foreground px-1">
          {candles.length > 0 ? (
            <>
              <span>{formatTime(candles[0]?.time || 0)}</span>
              <span>
                {formatTime(candles[Math.floor(candles.length / 2)]?.time || 0)}
              </span>
              <span>Jetzt</span>
            </>
          ) : (
            <>
              <span>1h vor</span>
              <span>45m</span>
              <span>30m</span>
              <span>15m</span>
              <span>Jetzt</span>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
