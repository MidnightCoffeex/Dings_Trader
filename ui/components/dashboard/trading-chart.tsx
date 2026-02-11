"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Bar,
  Area,
  ReferenceLine,
  Brush,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";

interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TradingChartProps {
  modelId: string;
  symbol?: string;
}

const TIMEFRAMES = ["15m", "1h", "4h", "1d"] as const;
type Timeframe = typeof TIMEFRAMES[number];

const TF_TO_15M_FACTOR: Record<Timeframe, number> = {
  "15m": 1,
  "1h": 4,
  "4h": 16,
  "1d": 96,
};

const TF_TO_LIMIT: Record<Timeframe, number> = {
  "15m": 4, // 4 * 15m = 1h
  "1h": 4, // 4 * 1h = 4h
  "4h": 12, // 12 * 4h = 48h (max available)
  "1d": 2, // 2 * 1d = 48h (max available)
};

const TF_TO_CANDLE_FETCH_LIMIT: Record<Timeframe, number> = {
  "15m": 120,
  "1h": 120,
  "4h": 120,
  "1d": 120,
};

const TF_TO_DEFAULT_VISIBLE_CANDLES: Record<Timeframe, number> = {
  "15m": 32,
  "1h": 32,
  "4h": 28,
  "1d": 20,
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const touchDistance = (
  a: { clientX: number; clientY: number },
  b: { clientX: number; clientY: number }
) => {
  const dx = a.clientX - b.clientX;
  const dy = a.clientY - b.clientY;
  return Math.hypot(dx, dy);
};

function CandleStickShape(props: any) {
  const { x, y, width, height, payload } = props ?? {};

  if (
    payload?.type !== "candle" ||
    !Number.isFinite(x) ||
    !Number.isFinite(y) ||
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0
  ) {
    return null;
  }

  const low = Number(payload.low);
  const high = Number(payload.high);
  const open = Number(payload.open);
  const close = Number(payload.close);

  if (![low, high, open, close].every(Number.isFinite) || high <= low) {
    return null;
  }

  const color = payload.candleColor ?? "#22c55e";
  const top = Math.min(y, y + height);
  const bottom = Math.max(y, y + height);
  const centerX = x + width / 2;

  const mapY = (price: number) => top + ((high - price) / (high - low)) * (bottom - top);
  const openY = mapY(open);
  const closeY = mapY(close);

  const bodyWidth = Math.max(3, width * 0.62);
  const bodyX = centerX - bodyWidth / 2;
  const bodyY = Math.min(openY, closeY);
  const bodyHeight = Math.max(2, Math.abs(openY - closeY));

  return (
    <g>
      <line x1={centerX} y1={top} x2={centerX} y2={bottom} stroke={color} strokeWidth={1.5} />
      <rect x={bodyX} y={bodyY} width={bodyWidth} height={bodyHeight} fill={color} stroke={color} rx={0.5} />
    </g>
  );
}

export function TradingChart({ modelId, symbol = "BTC/USDT" }: TradingChartProps) {
  const [timeframe, setTimeframe] = useState<Timeframe>("1h");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [forecast, setForecast] = useState<number[][]>([]);
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const [xRange, setXRange] = useState<{ startIndex: number; endIndex: number } | null>(null);
  const [manualXView, setManualXView] = useState<{ window: number; rightOffset: number } | null>(null);
  const [yZoom, setYZoom] = useState(1);
  const [isYDragging, setIsYDragging] = useState(false);
  const yDragRef = useRef<{ startY: number; startZoom: number } | null>(null);
  const chartViewportRef = useRef<HTMLDivElement | null>(null);
  const pinchRef = useRef<{
    startDistance: number;
    startRange: { startIndex: number; endIndex: number };
    startCenterRatio: number;
  } | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const cleanSymbol = symbol.replace("/", "");

        // 1) Candles
        let candleData: Candle[] = [];
        const limit = TF_TO_CANDLE_FETCH_LIMIT[timeframe] ?? 120;

        try {
          const res = await fetch(`/api/market-data/candles?symbol=${cleanSymbol}&interval=${timeframe}&limit=${limit}`);
          if (res.ok) candleData = await res.json();
          else throw new Error("Backend candles failed");
        } catch {
          const res = await fetch(`/api/binance/candles?symbol=${cleanSymbol}&interval=${timeframe}&limit=${limit}`);
          if (res.ok) candleData = await res.json();
        }

        if (Array.isArray(candleData) && candleData.length > 0) {
          setCandles(candleData);
          const lastClose = Number(candleData[candleData.length - 1]?.close);
          if (Number.isFinite(lastClose) && lastClose > 0) {
            setLivePrice(lastClose);
          }
        }

        // 2) Forecast + current price from dashboard
        const dashRes = await fetch(`/api/paper/dashboard/${modelId}`);
        if (dashRes.ok) {
          const dashData = await dashRes.json();
          const currentPrice = Number(dashData?.ml_signal?.current_price);
          const fv = dashData?.ml_signal?.forecast_values;

          if (Number.isFinite(currentPrice) && currentPrice > 0) {
            setLivePrice(currentPrice);
          }

          if (Array.isArray(fv) && Number.isFinite(currentPrice) && currentPrice > 0) {
            const priceForecast: number[][] = fv
              .filter((f: any) => Array.isArray(f) && f.length >= 3)
              .map((f: any) => {
                const q10 = Number(f[0]);
                const q50 = Number(f[1]);
                const q90 = Number(f[2]);
                if (![q10, q50, q90].every(Number.isFinite)) return null;

                // Keep forecast in a realistic near-term band so candles remain readable
                const clampRet = (x: number) => Math.max(-0.03, Math.min(0.03, x));
                return [
                  currentPrice * Math.exp(clampRet(q10)),
                  currentPrice * Math.exp(clampRet(q50)),
                  currentPrice * Math.exp(clampRet(q90)),
                ];
              })
              .filter((f: any): f is number[] => Array.isArray(f));

            setForecast(priceForecast);
          } else if (Array.isArray(fv)) {
            setForecast(fv);
          }
        }
      } catch (err) {
        console.error("Failed to fetch chart data", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => clearInterval(interval);
  }, [timeframe, modelId, symbol]);

  // On timeframe switch: reset view to default range/zoom
  useEffect(() => {
    setXRange(null);
    setManualXView(null);
    setYZoom(1);
  }, [timeframe]);

  const chartData = useMemo(() => {
    if (!candles.length) return [];

    const lastCandle = candles[candles.length - 1];
    const lastTime = lastCandle.time;

    // Map candles
    const data = candles.map((c, i) => {
      const isLast = i === candles.length - 1;
      const isUp = c.close >= c.open;
      return {
        ...c,
        type: "candle",

        // range-bars for candlestick rendering
        ocRange: [c.open, c.close],
        hlRange: [c.low, c.high],
        candleColor: isUp ? "#22c55e" : "#f43f5e",

        // Start forecast curves exactly at the tip of the last candle
        forecast_q10: isLast ? c.close : null,
        forecast_q50: isLast ? c.close : null,
        forecast_q90: isLast ? c.close : null,
        forecast_lower: isLast ? c.close : null,
        forecast_band: isLast ? 0 : null,
      };
    });

    const factor = TF_TO_15M_FACTOR[timeframe] ?? 1;
    const limit = TF_TO_LIMIT[timeframe] ?? 4;
    const stepMs = factor * 15 * 60 * 1000;

    const sampled = forecast
      .filter((_, i) => (i + 1) % factor === 0)
      .slice(0, limit);

    if (sampled.length > 0) {
      sampled.forEach((f, k) => {
        const q10 = Number(f?.[0]);
        const q50 = Number(f?.[1]);
        const q90 = Number(f?.[2]);
        if (![q10, q50, q90].every(Number.isFinite)) return;

        const time = lastTime + (k + 1) * stepMs;
        data.push({
          time,
          type: "forecast",
          // Keep bar-only series non-numeric for forecast points so bar domain logic ignores them.
          ocRange: [Number.NaN, Number.NaN],
          hlRange: [Number.NaN, Number.NaN],
          candleColor: null,
          forecast_q10: q10,
          forecast_q50: q50,
          forecast_q90: q90,
          forecast_lower: q10,
          forecast_band: Math.max(0, q90 - q10),
        } as any);
      });
    }

    return data;
  }, [candles, forecast, timeframe]);

  const defaultXRange = useMemo(() => {
    if (!chartData.length) return null;

    const forecastCount = chartData.filter((d: any) => d.type === "forecast").length;
    const visibleCandles = TF_TO_DEFAULT_VISIBLE_CANDLES[timeframe] ?? 32;
    const totalVisible = visibleCandles + forecastCount;

    const endIndex = chartData.length - 1;
    const startIndex = Math.max(0, endIndex - totalVisible + 1);

    return { startIndex, endIndex };
  }, [chartData, timeframe]);

  // Keep x-range valid when fresh data arrives
  useEffect(() => {
    if (!chartData.length) return;

    setXRange((prev) => {
      const maxIndex = chartData.length - 1;

      // If user manually set a viewport, keep it sticky across updates
      if (manualXView) {
        const window = clamp(manualXView.window, 2, maxIndex + 1);
        const rightOffset = clamp(manualXView.rightOffset, 0, maxIndex);
        const endIndex = clamp(maxIndex - rightOffset, 0, maxIndex);
        const startIndex = clamp(endIndex - window + 1, 0, endIndex);
        return { startIndex, endIndex };
      }

      if (!prev) return defaultXRange;

      const startIndex = clamp(prev.startIndex, 0, maxIndex);
      const endIndex = clamp(prev.endIndex, startIndex, maxIndex);
      return { startIndex, endIndex };
    });
  }, [chartData, defaultXRange, manualXView]);

  // Base domain (candles + full forecast range)
  const baseDomain = useMemo(() => {
    const candleValues = candles
      .flatMap((c) => [c.low, c.high])
      .filter((v) => Number.isFinite(v) && v > 0);

    const forecastValues = forecast
      .flatMap((f) => [Number(f?.[0]), Number(f?.[2])])
      .filter((v) => Number.isFinite(v) && v > 0);

    const values = [...candleValues, ...forecastValues];
    if (!values.length) return [0, 100] as [number, number];

    const min = Math.min(...values);
    const max = Math.max(...values);
    const padding = Math.max((max - min) * 0.08, min * 0.0012);
    return [min - padding, max + padding] as [number, number];
  }, [candles, forecast]);

  // User-zoomed Y domain
  const yDomain = useMemo(() => {
    const [min, max] = baseDomain;
    if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return baseDomain;

    const center = (min + max) / 2;
    const half = (max - min) / 2;
    const zoomHalf = half / clamp(yZoom, 0.4, 12);
    return [center - zoomHalf, center + zoomHalf] as [number, number];
  }, [baseDomain, yZoom]);

  const resetView = useCallback(() => {
    if (defaultXRange) {
      setXRange(defaultXRange);
    }
    setManualXView(null);
    setYZoom(1);
  }, [defaultXRange]);

  const onBrushChange = useCallback((range: { startIndex?: number; endIndex?: number }) => {
    if (typeof range?.startIndex === "number" && typeof range?.endIndex === "number") {
      const next = { startIndex: range.startIndex, endIndex: range.endIndex };
      setXRange(next);

      const maxIndex = Math.max(0, chartData.length - 1);
      const window = Math.max(2, next.endIndex - next.startIndex + 1);
      const rightOffset = Math.max(0, maxIndex - next.endIndex);
      setManualXView({ window, rightOffset });
    }
  }, [chartData.length]);

  const zoomXAroundCenter = useCallback((centerRatio: number, scale: number) => {
    const activeRange = xRange ?? defaultXRange;
    if (!activeRange || chartData.length < 2) return;

    const maxIndex = chartData.length - 1;
    const forecastCount = chartData.filter((d: any) => d.type === "forecast").length;
    const minWindow = Math.min(maxIndex + 1, Math.max(10, forecastCount + 8));
    const maxWindow = maxIndex + 1;

    const currentWindow = Math.max(2, activeRange.endIndex - activeRange.startIndex + 1);
    const safeScale = Number.isFinite(scale) && scale > 0 ? scale : 1;
    const newWindow = clamp(Math.round(currentWindow / safeScale), minWindow, maxWindow);

    const safeCenterRatio = clamp(centerRatio, 0, 1);
    const centerIndex = Math.round(activeRange.startIndex + safeCenterRatio * (currentWindow - 1));

    const maxStart = Math.max(0, maxIndex - newWindow + 1);
    const startIndex = clamp(Math.round(centerIndex - newWindow / 2), 0, maxStart);
    const endIndex = startIndex + newWindow - 1;

    setXRange({ startIndex, endIndex });
    setManualXView({
      window: Math.max(2, endIndex - startIndex + 1),
      rightOffset: Math.max(0, maxIndex - endIndex),
    });
  }, [chartData, xRange, defaultXRange]);

  const onChartWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (!xRange && !defaultXRange) return;

    e.preventDefault();

    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0.5;
    const scale = Math.exp(-e.deltaY * 0.0022);
    zoomXAroundCenter(ratio, scale);
  };

  const onChartTouchStart = (e: React.TouchEvent<HTMLDivElement>) => {
    if (e.touches.length !== 2) return;

    const activeRange = xRange ?? defaultXRange;
    if (!activeRange) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const [a, b] = [e.touches[0], e.touches[1]];
    const midX = (a.clientX + b.clientX) / 2;
    const centerRatio = rect.width > 0 ? clamp((midX - rect.left) / rect.width, 0, 1) : 0.5;

    pinchRef.current = {
      startDistance: touchDistance(a, b),
      startRange: activeRange,
      startCenterRatio: centerRatio,
    };
  };

  const onChartTouchMove = (e: React.TouchEvent<HTMLDivElement>) => {
    if (e.touches.length !== 2 || !pinchRef.current || chartData.length < 2) return;

    e.preventDefault();

    const pinch = pinchRef.current;
    const [a, b] = [e.touches[0], e.touches[1]];

    const rect = e.currentTarget.getBoundingClientRect();
    const midX = (a.clientX + b.clientX) / 2;
    const centerRatio = rect.width > 0 ? clamp((midX - rect.left) / rect.width, 0, 1) : pinch.startCenterRatio;

    const maxIndex = chartData.length - 1;
    const forecastCount = chartData.filter((d: any) => d.type === "forecast").length;
    const minWindow = Math.min(maxIndex + 1, Math.max(10, forecastCount + 8));
    const maxWindow = maxIndex + 1;

    const currentDistance = touchDistance(a, b);
    const safeStartDistance = Math.max(1, pinch.startDistance);
    const scale = currentDistance / safeStartDistance;

    const baseWindow = Math.max(2, pinch.startRange.endIndex - pinch.startRange.startIndex + 1);
    const newWindow = clamp(Math.round(baseWindow / Math.max(0.1, scale)), minWindow, maxWindow);

    const centerShift = (centerRatio - pinch.startCenterRatio) * baseWindow;
    const centerIndex = Math.round(
      pinch.startRange.startIndex + pinch.startCenterRatio * (baseWindow - 1) + centerShift
    );

    const maxStart = Math.max(0, maxIndex - newWindow + 1);
    const startIndex = clamp(Math.round(centerIndex - newWindow / 2), 0, maxStart);
    const endIndex = startIndex + newWindow - 1;

    setXRange({ startIndex, endIndex });
    setManualXView({
      window: Math.max(2, endIndex - startIndex + 1),
      rightOffset: Math.max(0, maxIndex - endIndex),
    });
  };

  const onChartTouchEnd = () => {
    pinchRef.current = null;
  };

  // Right-axis drag -> Y zoom
  const startYDrag = (clientY: number) => {
    yDragRef.current = { startY: clientY, startZoom: yZoom };
    setIsYDragging(true);
  };

  const onYRailMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    startYDrag(e.clientY);
  };

  const onYRailTouchStart = (e: React.TouchEvent<HTMLDivElement>) => {
    if (!e.touches.length) return;
    startYDrag(e.touches[0].clientY);
  };

  useEffect(() => {
    if (!isYDragging) return;

    const updateZoom = (clientY: number) => {
      const drag = yDragRef.current;
      if (!drag) return;

      const delta = drag.startY - clientY;
      const zoomFactor = Math.exp(delta * 0.01);
      const nextZoom = clamp(drag.startZoom * zoomFactor, 0.4, 12);
      setYZoom(nextZoom);
    };

    const onMouseMove = (e: MouseEvent) => updateZoom(e.clientY);
    const onTouchMove = (e: TouchEvent) => {
      if (!e.touches.length) return;
      updateZoom(e.touches[0].clientY);
    };

    const stopDrag = () => {
      yDragRef.current = null;
      setIsYDragging(false);
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", stopDrag);
    window.addEventListener("touchmove", onTouchMove, { passive: true });
    window.addEventListener("touchend", stopDrag);
    window.addEventListener("touchcancel", stopDrag);

    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", stopDrag);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", stopDrag);
      window.removeEventListener("touchcancel", stopDrag);
    };
  }, [isYDragging]);

  const currentPrice = useMemo(() => {
    if (Number.isFinite(livePrice) && (livePrice ?? 0) > 0) return livePrice as number;

    const lastClose = Number(candles[candles.length - 1]?.close);
    if (Number.isFinite(lastClose) && lastClose > 0) return lastClose;

    return null;
  }, [livePrice, candles]);

  const formatPrice = (value: number) => {
    if (!Number.isFinite(value)) return "-";
    return value.toLocaleString("en-US", {
      minimumFractionDigits: value > 1000 ? 0 : 2,
      maximumFractionDigits: value > 1000 ? 2 : 4,
    });
  };

  // Format tick
  const formatXAxis = (tickItem: number | string) => {
    const ts = typeof tickItem === "string" ? Number(tickItem) : tickItem;
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <Card className="bg-card/40 border-primary/20 h-full flex flex-col">
      <CardHeader className="py-3 px-4 flex flex-row items-center justify-between border-b border-border/50">
        <div className="flex items-center gap-4">
          <CardTitle className="text-sm font-medium text-muted-foreground">{symbol} Forecast & Analysis</CardTitle>
        </div>
        <div className="flex items-center gap-2">
          {loading && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
          <Badge variant="outline" className="text-[10px] font-mono">
            {timeframe}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="flex-1 p-0 min-h-[320px] relative">
        {/* Timeframe buttons */}
        <div className="absolute top-2 left-4 z-10 flex bg-background/60 backdrop-blur-sm rounded-md p-0.5 border border-border/50 shadow-sm">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`px-3 py-1 text-[10px] font-bold rounded-sm transition-colors ${
                timeframe === tf
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {tf}
            </button>
          ))}
        </div>

        {/* Reset + zoom hints */}
        <div className="absolute top-2 right-4 z-10 flex items-center gap-2">
          <div className="hidden sm:block px-2 py-1 rounded-md border border-border/50 bg-background/70 text-[10px] text-muted-foreground">
            Wheel/Pinch = X Zoom
          </div>
          <div className="px-2 py-1 rounded-md border border-border/50 bg-background/70 text-[10px] font-mono text-muted-foreground">
            Y x{yZoom.toFixed(2)}
          </div>
          <button
            onClick={resetView}
            className="px-2 py-1 text-[10px] font-semibold rounded-md border border-border/50 bg-background/70 hover:bg-background text-foreground"
          >
            Recalibrate
          </button>
        </div>

        <div
          ref={chartViewportRef}
          className="h-[320px] w-full relative"
          onWheel={onChartWheel}
          onTouchStart={onChartTouchStart}
          onTouchMove={onChartTouchMove}
          onTouchEnd={onChartTouchEnd}
          onTouchCancel={onChartTouchEnd}
        >
          {/* Right rail: drag up/down to zoom Y */}
          <div
            className="absolute right-0 top-0 z-20 h-[320px] w-5 cursor-ns-resize"
            onMouseDown={onYRailMouseDown}
            onTouchStart={onYRailTouchStart}
            title="Y-axis zoom (drag up/down)"
          >
            <div className="absolute right-[6px] top-10 bottom-10 w-[2px] rounded bg-primary/40" />
          </div>

          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={chartData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />

              <XAxis
                dataKey="time"
                type="category"
                tickFormatter={formatXAxis}
                stroke="#525252"
                tick={{ fill: "#a3a3a3", fontSize: 10 }}
              />

              <YAxis
                domain={yDomain}
                allowDataOverflow
                orientation="right"
                stroke="#525252"
                tick={{ fill: "#a3a3a3", fontSize: 10 }}
                tickFormatter={(val) => val.toFixed(0)}
              />

              <Tooltip
                contentStyle={{ backgroundColor: "#18181b", borderColor: "#27272a", fontSize: "12px" }}
                labelFormatter={(label) => {
                  const ts = typeof label === "string" ? Number(label) : (label as number);
                  return Number.isFinite(ts) ? new Date(ts).toLocaleString() : String(label);
                }}
                formatter={(val, name) => [typeof val === "number" ? val.toFixed(2) : "-", name]}
              />

              {/* Vertical line at "Now" */}
              {candles.length > 0 && (
                <ReferenceLine
                  x={candles[candles.length - 1].time}
                  stroke="#525252"
                  strokeDasharray="3 3"
                  label={{ position: "top", value: "NOW", fill: "#737373", fontSize: 10 }}
                />
              )}

              {/* Current price marker */}
              {currentPrice !== null && (
                <ReferenceLine
                  y={currentPrice}
                  stroke="#a78bfa"
                  strokeDasharray="4 4"
                  strokeOpacity={0.65}
                  label={{
                    position: "right",
                    value: `$${formatPrice(currentPrice)}`,
                    fill: "#c4b5fd",
                    fontSize: 10,
                  }}
                />
              )}

              {/* Candles */}
              <Bar
                dataKey="hlRange"
                barSize={10}
                minPointSize={3}
                shape={<CandleStickShape />}
                isAnimationActive={false}
                legendType="none"
              />

              {/* Forecast band (q10..q90) + median */}
              <Area
                type="monotone"
                dataKey="forecast_lower"
                stackId="forecastBand"
                stroke="none"
                fill="transparent"
                isAnimationActive={false}
                connectNulls={false}
                name="Forecast Lower"
              />
              <Area
                type="monotone"
                dataKey="forecast_band"
                stackId="forecastBand"
                stroke="none"
                fill="#6366f1"
                fillOpacity={0.2}
                isAnimationActive={false}
                connectNulls={false}
                name="Forecast Range"
              />
              <Line
                type="monotone"
                dataKey="forecast_q50"
                stroke="#818cf8"
                strokeWidth={3}
                strokeDasharray="5 5"
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
                name="Forecast (Median)"
              />
              <Line
                type="monotone"
                dataKey="forecast_q10"
                stroke="#4338ca"
                strokeWidth={1}
                strokeDasharray="3 3"
                dot={false}
                strokeOpacity={0.4}
                connectNulls={false}
                isAnimationActive={false}
                name="Lower Bound"
              />
              <Line
                type="monotone"
                dataKey="forecast_q90"
                stroke="#4338ca"
                strokeWidth={1}
                strokeDasharray="3 3"
                dot={false}
                strokeOpacity={0.4}
                connectNulls={false}
                isAnimationActive={false}
                name="Upper Bound"
              />

              {/* X-axis pan/zoom handles */}
              {xRange && (
                <Brush
                  dataKey="time"
                  height={18}
                  stroke="#6366f1"
                  travellerWidth={8}
                  startIndex={xRange.startIndex}
                  endIndex={xRange.endIndex}
                  onChange={onBrushChange}
                  tickFormatter={formatXAxis}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
