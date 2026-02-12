"use client";

import { AppShell } from "@/components/layout/app-shell";
import { PageHeader } from "@/components/layout/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Suspense } from "react";

function DataContent() {
  return (
    <div className="space-y-6">
      <PageHeader title="Data" subtitle="Ingestion, feature stores, and dataset health." />

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="bg-card/40 lg:col-span-2">
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground">Datasets</CardTitle>
            <Button size="sm" variant="secondary">Add source</Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {[
                { name: "OHLCV — 1m", rows: "88M", status: "Healthy" },
                { name: "Orderbook — L2", rows: "12M", status: "Lagging" },
                { name: "News sentiment", rows: "2.1M", status: "Healthy" },
              ].map((d) => (
                <div
                  key={d.name}
                  className="flex items-center justify-between rounded-md border border-border/70 bg-background/30 px-3 py-2"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium">{d.name}</div>
                    <div className="text-xs text-muted-foreground">Rows: {d.rows}</div>
                  </div>
                  <Badge variant={d.status === "Healthy" ? "success" : "warning"}>{d.status}</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/40">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-muted-foreground">Pipeline</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="rounded-md border border-border/70 bg-background/30 p-3">
              <div className="text-xs text-muted-foreground">Last sync</div>
              <div className="text-sm font-medium">2 minutes ago</div>
            </div>
            <div className="rounded-md border border-border/70 bg-background/30 p-3">
              <div className="text-xs text-muted-foreground">Feature refresh</div>
              <div className="text-sm font-medium">Every 5m</div>
            </div>
            <Button className="w-full" variant="outline">Run ingestion (stub)</Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default function DataPage() {
  return (
    <AppShell>
      <Suspense fallback={<div>Loading data...</div>}>
        <DataContent />
      </Suspense>
    </AppShell>
  );
}
