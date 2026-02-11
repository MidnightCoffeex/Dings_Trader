import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function MetricCard({
  title,
  value,
  hint,
  trend,
}: {
  title: string;
  value: string;
  hint?: string;
  trend?: "up" | "down" | "flat";
}) {
  const variant = trend === "up" ? "success" : trend === "down" ? "warning" : "default";

  return (
    <Card className="bg-card/40">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline justify-between gap-3">
          <div className="text-2xl font-semibold tracking-tight">{value}</div>
          {hint ? <Badge variant={variant as any}>{hint}</Badge> : null}
        </div>
      </CardContent>
    </Card>
  );
}
