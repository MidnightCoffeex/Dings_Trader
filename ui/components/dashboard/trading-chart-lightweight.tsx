"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createChart,
  ColorType,
  type IChartApi,
  type ISeriesApi,
  type IPriceLine,
  type CandlestickData,
  type LineData,
  type LogicalRange,
  type UTCTimestamp,
  type Time,
  LineStyle,
} from "lightweight-charts";
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

interface TradingChartLightweightProps {
  modelId: string;
  symbol?: string;
}

const TIMEFRAMES = ["15m", "1h", "4h", "1d"] as const;
type Timeframe = (typeof TIMEFRAMES)[number];

const INITIAL_TIMEFRAME: Timeframe = "1h";

const TF_TO_15M_FACTOR: Record<Timeframe, number> = {
  "15m": 1,
  "1h": 4,
  "4h": 16,
  "1d": 96,
};

const TF_TO_LIMIT: Record<Timeframe, number> = {
  "15m": 4, // 4 * 15m = 1h
  "1h": 4, // 4 * 1h = 4h
  "4h": 12, // 12 * 4h = 48h
  "1d": 2, // 2 * 1d = 48h
};

const TF_TO_CANDLE_FETCH_LIMIT: Record<Timeframe, number> = {
  "15m": 120,
  "1h": 120,
  "4h": 120,
  "1d": 120,
};

const asUtc = (ms: number) => Math.floor(ms / 1000) as UTCTimestamp;

const clampRet = (x: number) => Math.max(-0.03, Math.min(0.03, x));
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export function TradingChartLightweight({
  modelId,
  symbol = "BTC/USDT",
}: TradingChartLightweightProps) {
  const [timeframe, setTimeframe] = useState<Timeframe>(INITIAL_TIMEFRAME);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [forecast, setForecast] = useState<number[][]>([]);
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [yZoomFactor, setYZoomFactor] = useState(1);
  const [isYDragging, setIsYDragging] = useState(false);
  const [forecastView, setForecastView] = useState<"lines" | "fan">("lines");

  const containerRef = useRef<HTMLDivElement | null>(null);
  const fanCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const q10SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const q50SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const q90SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const currentPriceLineRef = useRef<IPriceLine | null>(null);

  const applyingVisibleRangeRef = useRef(false);
  const userInteractedRef = useRef(false);
  const savedVisibleRangeRef = useRef<LogicalRange | null>(null);
  const fittedInitiallyRef = useRef(false);

  const yZoomFactorRef = useRef(1);
  const yDragRef = useRef<{ startY: number; startZoom: number } | null>(null);

  const autoscaleInfoProvider = (original: any) => {
    const res = original?.();
    if (!res?.priceRange) return res;

    const factor = yZoomFactorRef.current;
    if (!Number.isFinite(factor) || Math.abs(factor - 1) < 0.0001) {
      return res;
    }

    const min = Number(res.priceRange.minValue);
    const max = Number(res.priceRange.maxValue);
    if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return res;

    const center = (min + max) / 2;
    const half = (max - min) / 2;
    const zoomHalf = half / factor;

    return {
      ...res,
      priceRange: {
        minValue: center - zoomHalf,
        maxValue: center + zoomHalf,
      },
    };
  };

  const applyYZoom = (nextZoom: number) => {
    const z = clamp(nextZoom, 0.5, 8);
    yZoomFactorRef.current = z;
    setYZoomFactor(z);

    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    if (!chart || !candleSeries) return;

    candleSeries.applyOptions({});

    const currentRange = chart.timeScale().getVisibleLogicalRange();
    if (currentRange) {
      applyingVisibleRangeRef.current = true;
      try {
        chart.timeScale().setVisibleLogicalRange(currentRange);
      } finally {
        applyingVisibleRangeRef.current = false;
      }
    }
  };

  const startYDrag = (clientY: number) => {
    yDragRef.current = {
      startY: clientY,
      startZoom: yZoomFactorRef.current,
    };
    setIsYDragging(true);
  };

  const onYRailTouchStart = (e: React.TouchEvent<HTMLDivElement>) => {
    const touch = e.touches?.[0];
    if (!touch) return;
    e.preventDefault();
    startYDrag(touch.clientY);
  };

  const onYRailMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    startYDrag(e.clientY);
  };

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const cleanSymbol = symbol.replace("/", "");
        const limit = TF_TO_CANDLE_FETCH_LIMIT[timeframe] ?? 120;

        // 1) Candles
        let candleData: Candle[] = [];
        try {
          const res = await fetch(
            `/api/market-data/candles?symbol=${cleanSymbol}&interval=${timeframe}&limit=${limit}`
          );
          if (res.ok) candleData = await res.json();
          else throw new Error("Backend candles failed");
        } catch {
          const res = await fetch(
            `/api/binance/candles?symbol=${cleanSymbol}&interval=${timeframe}&limit=${limit}`
          );
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

                return [
                  currentPrice * Math.exp(clampRet(q10)),
                  currentPrice * Math.exp(clampRet(q50)),
                  currentPrice * Math.exp(clampRet(q90)),
                ];
              })
              .filter((f: any): f is number[] => Array.isArray(f));

            setForecast(priceForecast);
          }
        }
      } catch (err) {
        console.error("Failed to fetch lightweight chart data", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => clearInterval(interval);
  }, [timeframe, modelId, symbol]);

  useEffect(() => {
    if (!isYDragging) return;

    const update = (clientY: number) => {
      const drag = yDragRef.current;
      if (!drag) return;

      const delta = drag.startY - clientY;
      const factor = Math.exp(delta * 0.01);
      applyYZoom(drag.startZoom * factor);
    };

    const onMouseMove = (e: MouseEvent) => update(e.clientY);
    const onMouseUp = () => {
      yDragRef.current = null;
      setIsYDragging(false);
    };

    const onTouchMove = (e: TouchEvent) => {
      if (!e.touches?.length) return;
      e.preventDefault();
      update(e.touches[0].clientY);
    };
    const onTouchEnd = () => {
      yDragRef.current = null;
      setIsYDragging(false);
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    window.addEventListener("touchmove", onTouchMove, { passive: false });
    window.addEventListener("touchend", onTouchEnd);
    window.addEventListener("touchcancel", onTouchEnd);

    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onTouchEnd);
      window.removeEventListener("touchcancel", onTouchEnd);
    };
  }, [isYDragging]);

  // Reset interaction state on scope switches
  useEffect(() => {
    userInteractedRef.current = false;
    savedVisibleRangeRef.current = null;
    fittedInitiallyRef.current = false;
    yDragRef.current = null;
    setIsYDragging(false);
    applyYZoom(1);
  }, [timeframe, modelId, symbol]);

  const seriesData = useMemo(() => {
    if (!candles.length) {
      return {
        candleData: [] as CandlestickData[],
        q10Data: [] as LineData[],
        q50Data: [] as LineData[],
        q90Data: [] as LineData[],
      };
    }

    const candleData: CandlestickData[] = candles
      .filter((c) => [c.time, c.open, c.high, c.low, c.close].every(Number.isFinite))
      .map((c) => ({
        time: asUtc(c.time),
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }));

    const lastCandle = candles[candles.length - 1];
    const lastTimeSec = asUtc(lastCandle.time);
    const factor = TF_TO_15M_FACTOR[timeframe] ?? 1;
    const limit = TF_TO_LIMIT[timeframe] ?? 4;
    const stepSec = factor * 15 * 60;

    const sampled = forecast
      .filter((_, i) => (i + 1) % factor === 0)
      .slice(0, limit);

    const q10Data: LineData[] = [{ time: lastTimeSec, value: lastCandle.close }];
    const q50Data: LineData[] = [{ time: lastTimeSec, value: lastCandle.close }];
    const q90Data: LineData[] = [{ time: lastTimeSec, value: lastCandle.close }];

    sampled.forEach((f, k) => {
      const q10 = Number(f?.[0]);
      const q50 = Number(f?.[1]);
      const q90 = Number(f?.[2]);
      if (![q10, q50, q90].every(Number.isFinite)) return;

      const t = (lastTimeSec + ((k + 1) * stepSec as UTCTimestamp)) as UTCTimestamp;
      q10Data.push({ time: t, value: q10 });
      q50Data.push({ time: t, value: q50 });
      q90Data.push({ time: t, value: q90 });
    });

    return { candleData, q10Data, q50Data, q90Data };
  }, [candles, forecast, timeframe]);

  const drawFanOverlay = useCallback(() => {
    const canvas = fanCanvasRef.current;
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;

    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;

    const targetW = Math.floor(cssWidth * dpr);
    const targetH = Math.floor(cssHeight * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    if (forecastView !== "fan") return;
    if (!chart || !candleSeries) return;

    const timeScale = chart.timeScale();
    const { q10Data, q90Data } = seriesData;
    const n = Math.min(q10Data.length, q90Data.length);
    if (n < 2) return;

    const points: Array<{ x: number; low: number; high: number }> = [];

    for (let i = 0; i < n; i++) {
      const lower = q10Data[i] as LineData;
      const upper = q90Data[i] as LineData;
      if (lower == null || upper == null) continue;
      if (!("time" in lower) || !("time" in upper) || !("value" in lower) || !("value" in upper)) continue;

      const x = timeScale.timeToCoordinate(lower.time as Time);
      const lowY = candleSeries.priceToCoordinate(Number(lower.value));
      const highY = candleSeries.priceToCoordinate(Number(upper.value));

      if (x == null || lowY == null || highY == null) continue;
      if (!Number.isFinite(x) || !Number.isFinite(lowY) || !Number.isFinite(highY)) continue;

      points.push({ x, low: lowY, high: highY });
    }

    if (points.length < 2) return;

    const grad = ctx.createLinearGradient(0, 0, 0, cssHeight);
    grad.addColorStop(0, "rgba(129, 140, 248, 0.28)");
    grad.addColorStop(1, "rgba(99, 102, 241, 0.10)");

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].high);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].high);
    }
    for (let i = points.length - 1; i >= 0; i--) {
      ctx.lineTo(points[i].x, points[i].low);
    }
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }, [forecastView, seriesData]);

  useEffect(() => {
    let raf = 0;

    const frame = () => {
      drawFanOverlay();
      raf = window.requestAnimationFrame(frame);
    };

    if (typeof window !== "undefined") {
      raf = window.requestAnimationFrame(frame);
    }

    return () => {
      if (raf) window.cancelAnimationFrame(raf);
    };
  }, [drawFanOverlay]);

  useEffect(() => {
    if (!containerRef.current || chartRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#a3a3a3",
      },
      width: containerRef.current.clientWidth,
      height: 320,
      rightPriceScale: {
        borderColor: "rgba(163,163,163,0.2)",
        textColor: "#a3a3a3",
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      timeScale: {
        borderColor: "rgba(163,163,163,0.2)",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 2,
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.04)" },
      },
      crosshair: {
        vertLine: { color: "rgba(167,139,250,0.35)", width: 1 },
        horzLine: { color: "rgba(167,139,250,0.35)", width: 1 },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: { time: true, price: true },
        axisDoubleClickReset: { time: true, price: true },
        mouseWheel: true,
        pinch: true,
      },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#f43f5e",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#f43f5e",
      priceLineVisible: false,
      lastValueVisible: false,
      autoscaleInfoProvider,
    });

    const q50Series = chart.addLineSeries({
      color: "#818cf8",
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      autoscaleInfoProvider,
    });

    const q10Series = chart.addLineSeries({
      color: "#4338ca",
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      autoscaleInfoProvider,
    });

    const q90Series = chart.addLineSeries({
      color: "#4338ca",
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      autoscaleInfoProvider,
    });

    chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (!range || applyingVisibleRangeRef.current) return;
      userInteractedRef.current = true;
      savedVisibleRangeRef.current = range;
    });

    const ro = new ResizeObserver(() => {
      if (!containerRef.current) return;
      chart.applyOptions({ width: containerRef.current.clientWidth });
    });
    ro.observe(containerRef.current);

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    q10SeriesRef.current = q10Series;
    q50SeriesRef.current = q50Series;
    q90SeriesRef.current = q90Series;

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      q10SeriesRef.current = null;
      q50SeriesRef.current = null;
      q90SeriesRef.current = null;
      currentPriceLineRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    const q10Series = q10SeriesRef.current;
    const q50Series = q50SeriesRef.current;
    const q90Series = q90SeriesRef.current;

    if (!chart || !candleSeries || !q10Series || !q50Series || !q90Series) return;

    const { candleData, q10Data, q50Data, q90Data } = seriesData;

    const timeScale = chart.timeScale();
    const preserveRange = userInteractedRef.current
      ? savedVisibleRangeRef.current ?? timeScale.getVisibleLogicalRange()
      : null;

    candleSeries.setData(candleData);
    q10Series.setData(q10Data);
    q50Series.setData(q50Data);
    q90Series.setData(q90Data);

    if (forecastView === "fan") {
      q10Series.applyOptions({
        color: "rgba(67, 56, 202, 0.45)",
        lineWidth: 1,
        lineStyle: LineStyle.Solid,
      });
      q90Series.applyOptions({
        color: "rgba(67, 56, 202, 0.45)",
        lineWidth: 1,
        lineStyle: LineStyle.Solid,
      });
      q50Series.applyOptions({
        color: "#a5b4fc",
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
      });
    } else {
      q10Series.applyOptions({
        color: "#4338ca",
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
      });
      q90Series.applyOptions({
        color: "#4338ca",
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
      });
      q50Series.applyOptions({
        color: "#818cf8",
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
      });
    }

    if (currentPriceLineRef.current) {
      candleSeries.removePriceLine(currentPriceLineRef.current);
      currentPriceLineRef.current = null;
    }

    if (livePrice && Number.isFinite(livePrice) && livePrice > 0) {
      currentPriceLineRef.current = candleSeries.createPriceLine({
        price: livePrice,
        color: "#a78bfa",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "NOW",
      });
    }

    if (preserveRange) {
      applyingVisibleRangeRef.current = true;
      try {
        timeScale.setVisibleLogicalRange(preserveRange);
      } catch {
        // no-op
      } finally {
        applyingVisibleRangeRef.current = false;
      }
    } else if (!fittedInitiallyRef.current && candleData.length > 0) {
      applyingVisibleRangeRef.current = true;
      timeScale.fitContent();
      applyingVisibleRangeRef.current = false;
      fittedInitiallyRef.current = true;
    }
  }, [seriesData, livePrice, forecastView]);

  const resetView = () => {
    const chart = chartRef.current;
    if (!chart) return;
    userInteractedRef.current = false;
    savedVisibleRangeRef.current = null;
    yDragRef.current = null;
    setIsYDragging(false);
    applyYZoom(1);

    applyingVisibleRangeRef.current = true;
    chart.timeScale().fitContent();
    applyingVisibleRangeRef.current = false;
  };

  return (
    <Card className="bg-card/40 border-primary/20 h-full flex flex-col">
      <CardHeader className="py-3 px-4 flex flex-row items-center justify-between border-b border-border/50">
        <div className="flex items-center gap-4">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            BTC/USDT Forecast & Analysis (Lightweight Prototype)
          </CardTitle>
        </div>
        <div className="flex items-center gap-2">
          {loading && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
          <Badge variant="outline" className="text-[10px] font-mono">
            {timeframe}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="flex-1 p-0 min-h-[320px] relative">
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

        <div className="absolute top-2 right-4 z-10 flex items-center gap-2">
          <div className="hidden sm:block px-2 py-1 rounded-md border border-border/50 bg-background/70 text-[10px] text-muted-foreground">
            Native zoom/pan active
          </div>

          <div className="flex bg-background/70 rounded-md p-0.5 border border-border/50 shadow-sm">
            <button
              onClick={() => setForecastView("lines")}
              className={`px-2 py-1 text-[10px] font-semibold rounded-sm transition-colors ${
                forecastView === "lines"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              lines
            </button>
            <button
              onClick={() => setForecastView("fan")}
              className={`px-2 py-1 text-[10px] font-semibold rounded-sm transition-colors ${
                forecastView === "fan"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              fan
            </button>
          </div>

          <div className="px-2 py-1 rounded-md border border-border/50 bg-background/70 text-[10px] text-muted-foreground">
            Y: x{yZoomFactor.toFixed(2)}
          </div>
          <button
            onClick={resetView}
            className="px-2 py-1 text-[10px] font-semibold rounded-md border border-border/50 bg-background/70 hover:bg-background text-foreground"
          >
            Recalibrate
          </button>
        </div>

        <div ref={containerRef} className="h-[320px] w-full" />
        <canvas
          ref={fanCanvasRef}
          className={`absolute inset-0 h-[320px] w-full pointer-events-none transition-opacity ${
            forecastView === "fan" ? "opacity-100" : "opacity-0"
          }`}
        />

        {/* Right-side Y-zoom rail for mobile/touch */}
        <div
          className={`absolute right-0 top-0 z-20 h-[320px] w-7 cursor-ns-resize ${isYDragging ? "bg-primary/10" : ""}`}
          onMouseDown={onYRailMouseDown}
          onTouchStart={onYRailTouchStart}
          title="Y-axis zoom (drag up/down)"
        >
          <div className="absolute right-[10px] top-10 bottom-10 w-[2px] rounded bg-primary/40" />
        </div>
      </CardContent>
    </Card>
  );
}
