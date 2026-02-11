import { PageHeader } from "@/components/layout/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function BacktestsPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Backtests"
        subtitle="Run, compare, and analyze strategy performance."
      />

      <Card className="bg-card/40">
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Recent Runs
          </CardTitle>
          <Button size="sm">New backtest</Button>
        </CardHeader>
        <CardContent>
          <div className="overflow-hidden rounded-md border border-border/70">
            <div className="grid grid-cols-5 bg-background/40 px-3 py-2 text-xs text-muted-foreground">
              <div>Run</div>
              <div>Universe</div>
              <div>Period</div>
              <div>Status</div>
              <div className="text-right">Sharpe</div>
            </div>
            {[
              { id: "bt-1042", uni: "BTC/ETH", per: "90d", st: "Completed", sh: "1.42" },
              { id: "bt-1041", uni: "Top 20", per: "180d", st: "Completed", sh: "0.98" },
              { id: "bt-1040", uni: "SOL", per: "30d", st: "Queued", sh: "—" },
            ].map((r) => (
              <div
                key={r.id}
                className="grid grid-cols-5 items-center border-t border-border/70 bg-background/20 px-3 py-2 text-sm"
              >
                <div className="font-medium">{r.id}</div>
                <div>{r.uni}</div>
                <div>{r.per}</div>
                <div>
                  <Badge variant={r.st === "Completed" ? "success" : "default"}>
                    {r.st}
                  </Badge>
                </div>
                <div className="text-right">{r.sh}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
