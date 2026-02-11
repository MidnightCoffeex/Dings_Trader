"use client";

import dynamic from "next/dynamic";

const EquityChart = dynamic(
  () => import("./equity-chart").then((mod) => mod.EquityChart),
  { 
    ssr: false, 
    loading: () => (
      <div className="h-[320px] 2xl:h-[500px] bg-card/40 animate-pulse rounded-md" />
    ) 
  }
);

type EquityPoint = {
  timestamp?: string;
  equity?: number;
  balance?: number;
  open_positions?: number;
  unrealized_pnl?: number;
  [key: string]: string | number | undefined;
};

interface ChartLoaderProps {
  data: EquityPoint[];
}

export function ChartLoader({ data }: ChartLoaderProps) {
  return <EquityChart data={data} />;
}
