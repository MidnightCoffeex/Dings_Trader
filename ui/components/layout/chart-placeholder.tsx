import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function ChartPlaceholder({
  title = "Equity Curve (placeholder)",
}: {
  title?: string;
}) {
  return (
    <Card className="bg-card/40">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative h-[320px] overflow-hidden rounded-md border border-border/70 bg-background/40">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/15 via-transparent to-cyan-500/10" />
          <div className="absolute inset-0 grid place-items-center">
            <div className="text-center">
              <div className="text-sm font-medium">Chart goes here</div>
              <div className="mt-1 text-xs text-muted-foreground">
                Plug in Recharts, Visx, or TradingView
              </div>
            </div>
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-background/80 to-transparent" />
        </div>
      </CardContent>
    </Card>
  );
}
