"use client";

import {
  Area,
  AreaChart,
  Brush,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type EquityPoint = {
  timestamp?: string;
  equity?: number;
  balance?: number;
  open_positions?: number;
  unrealized_pnl?: number;
  [key: string]: string | number | undefined;
};

type EquityChartProps = {
  data: EquityPoint[];
  title?: string;
};

function formatTimestamp(ts: string | undefined): string {
  if (!ts) return "";
  try {
    const date = new Date(ts);
    // Format as "HH:MM" for time or "MM-DD HH:MM" for date+time
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    
    if (isToday) {
      return date.toLocaleTimeString("en-US", { 
        hour: "2-digit", 
        minute: "2-digit",
        hour12: false 
      });
    } else {
      return date.toLocaleDateString("en-US", { 
        month: "short", 
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false
      });
    }
  } catch {
    return ts || "";
  }
}

function formatUSDT(value: number): string {
  return `${value.toFixed(2)} USDT`;
}

export function EquityChart({ data, title = "Equity Curve" }: EquityChartProps) {
  // Safe accessor for equity value
  const getEquityValue = (item: EquityPoint): number => {
    if (typeof item.equity === "number") return item.equity;
    // fallback to searching for first numeric key if 'equity' missing
    const val = Object.values(item).find((v) => typeof v === "number");
    return typeof val === "number" ? val : 0;
  };

  // Pre-process data to ensure we have a valid key for Recharts
  const chartData = data.map((d) => ({
    ...d,
    _value: getEquityValue(d),
    _time: formatTimestamp(d.timestamp),
    _rawTime: d.timestamp || "",
  }));

  // Calculate min/max for Y-axis padding
  const values = chartData.map(d => d._value);
  const minValue = Math.min(...values, 0);
  const maxValue = Math.max(...values, 0);
  const padding = (maxValue - minValue) * 0.1;

  return (
    <Card className="bg-card/40 border-primary/20 h-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="h-[calc(100%-3rem)]">
        <div className="h-[320px] 2xl:h-[500px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
              <defs>
                <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
                stroke="rgba(255,255,255,0.05)"
              />
              <XAxis
                dataKey="_time"
                axisLine={false}
                tickLine={false}
                tick={{ fill: "#6b7280", fontSize: 10 }}
                angle={-45}
                textAnchor="end"
                height={50}
                interval="preserveStartEnd"
                minTickGap={30}
              />
              <YAxis
                domain={[minValue - padding, maxValue + padding]}
                orientation="right"
                tick={{ fill: "#6b7280", fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(val) => `${val.toFixed(0)}`}
                width={60}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#09090b",
                  border: "1px solid #27272a",
                  borderRadius: "6px",
                  fontSize: "12px",
                }}
                itemStyle={{ color: "#e5e7eb" }}
                formatter={(value) => [
                  formatUSDT(Number(value)),
                  "Equity",
                ]}
                labelFormatter={(label) => String(label)}
              />
              <Area
                type="monotone"
                dataKey="_value"
                stroke="#8b5cf6"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorEquity)"
              />
              <Brush
                dataKey="_time"
                height={24}
                stroke="#8b5cf6"
                fill="#27272a"
                travellerWidth={8}
                tickFormatter={() => ""}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
