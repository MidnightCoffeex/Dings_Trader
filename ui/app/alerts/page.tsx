"use client";

import { AppShell } from "@/components/layout/app-shell";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Suspense } from "react";

function AlertsContent() {
  return (
    <div className="space-y-6">
      <PageHeader title="Alerts" subtitle="Risk, drift, and execution notifications." />

      <Card className="bg-card/40">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-muted-foreground">Recent Alerts</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[
              {
                t: "Model drift detected",
                s: "PSI exceeded threshold on features: volatility_1m, spread",
                sev: "warning",
              },
              {
                t: "Order rejected",
                s: "Insufficient margin for BTC-USD — simulated account",
                sev: "default",
              },
              {
                t: "Risk limit OK",
                s: "Exposure within bounds across all symbols",
                sev: "success",
              },
            ].map((a) => (
              <div key={a.t} className="rounded-md border border-border/70 bg-background/30 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium">{a.t}</div>
                    <div className="mt-1 text-xs text-muted-foreground">{a.s}</div>
                  </div>
                  <Badge variant={a.sev as any}>{a.sev}</Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function AlertsPage() {
  return (
    <AppShell>
      <Suspense fallback={<div>Loading alerts...</div>}>
        <AlertsContent />
      </Suspense>
    </AppShell>
  );
}
