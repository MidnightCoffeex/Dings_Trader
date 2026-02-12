"use client";

import { useEffect, useMemo, useState } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";

type CapacityResponse = {
  ts: string;
  host: {
    cpuCores: number;
    load1m: number;
    load5m: number;
    load15m: number;
    load1mPct: number;
    memory: {
      totalBytes: number;
      usedBytes: number;
      freeBytes: number;
      usedPct: number;
    };
  };
  services: Array<{
    name: string;
    status: string;
    pid: number | null;
    cpuPct: number;
    memoryBytes: number;
  }>;
  servicesSummary: {
    count: number;
    totalCpuPct: number;
    totalMemoryBytes: number;
  };
  error?: string;
};

const fmtBytes = (bytes: number) => {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 MB";
  const mb = bytes / 1024 / 1024;
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(1)} MB`;
};

export default function CapacityPage() {
  const [data, setData] = useState<CapacityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    const fetchCapacity = async () => {
      try {
        const res = await fetch("/api/system-capacity");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = (await res.json()) as CapacityResponse;
        if (!active) return;
        setData(json);
        setError(json.error ?? null);
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : "fetch failed");
      } finally {
        if (active) setLoading(false);
      }
    };

    fetchCapacity();
    const id = setInterval(fetchCapacity, 10000);

    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const cpuPct = useMemo(() => Number(data?.host?.load1mPct ?? 0), [data]);
  const memPct = useMemo(() => Number(data?.host?.memory?.usedPct ?? 0), [data]);

  return (
    <AppShell>
      <div className="space-y-6">
        <PageHeader title="Kapazität" subtitle="Live-Auslastung von Host + Trading-Services" />

      <div className="grid gap-4 md:grid-cols-3">
        <Card className="bg-card/40 border-primary/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Host CPU (1m)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{cpuPct.toFixed(1)}%</div>
            <div className="mt-1 text-xs text-muted-foreground">
              Load: {Number(data?.host?.load1m ?? 0).toFixed(2)} / {data?.host?.cpuCores ?? 0} Cores
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/40 border-primary/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Host RAM</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{memPct.toFixed(1)}%</div>
            <div className="mt-1 text-xs text-muted-foreground">
              {fmtBytes(Number(data?.host?.memory?.usedBytes ?? 0))} / {fmtBytes(Number(data?.host?.memory?.totalBytes ?? 0))}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/40 border-primary/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Services gesamt</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{Number(data?.servicesSummary?.totalCpuPct ?? 0).toFixed(1)}%</div>
            <div className="mt-1 text-xs text-muted-foreground">
              {fmtBytes(Number(data?.servicesSummary?.totalMemoryBytes ?? 0))} RAM
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="bg-card/40 border-primary/20">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-sm text-muted-foreground">PM2 Services</CardTitle>
            <div className="flex items-center gap-2">
              {loading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
              <Badge variant="outline" className="text-[10px]">Auto-Refresh 10s</Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {error ? (
            <div className="rounded-md border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-300">
              Fehler: {error}
            </div>
          ) : null}

          <div className="space-y-2">
            {(data?.services ?? []).map((s) => (
              <div
                key={s.name}
                className="rounded-md border border-border/70 bg-background/30 p-3"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold">{s.name}</div>
                    <div className="text-xs text-muted-foreground">pid: {s.pid ?? "-"}</div>
                  </div>
                  <Badge variant={s.status === "online" ? "success" : "destructive"}>{s.status}</Badge>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-muted-foreground sm:grid-cols-4">
                  <div>CPU: <span className="text-foreground">{s.cpuPct.toFixed(1)}%</span></div>
                  <div>RAM: <span className="text-foreground">{fmtBytes(s.memoryBytes)}</span></div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 text-xs text-muted-foreground">
            Last update: {data?.ts ? new Date(data.ts).toLocaleTimeString() : "-"}
          </div>
        </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
