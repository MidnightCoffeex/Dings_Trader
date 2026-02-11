import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export default function ModelPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Synthese-Hirn"
        subtitle="Trainingsstatus, Versionierung und Inferenz-Gesundheit."
      />

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="bg-card/40 lg:col-span-2 border-primary/20">
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Aktive Intelligenz
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant="success">Serving</Badge>
              <Badge variant="purple">v2.1.0-LGBM</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-3">
              {[
                { k: "Architektur", v: "LightGBM" },
                { k: "Horizont", v: "48h" },
                { k: "Features", v: "40 (v3)" },
              ].map((x) => (
                <div
                  key={x.k}
                  className="rounded-md border border-border/70 bg-background/30 p-3"
                >
                  <div className="text-xs text-muted-foreground">{x.k}</div>
                  <div className="text-sm font-medium">{x.v}</div>
                </div>
              ))}
            </div>

            <div className="rounded-md border border-border/70 bg-background/30 p-3">
              <div className="text-xs text-muted-foreground">Performance-Notiz</div>
              <div className="mt-1 text-sm text-balance">
                Modell auf 1h-BTC-Daten trainiert. Fokus auf Minimierung von Transaktionsgebühren durch hohe Konfidenz-Schwellenwerte. 🧬
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/40">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Steuerung
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button className="w-full">Retraining starten</Button>
            <Button className="w-full" variant="outline">
              Eval-Suite (Strict Split)
            </Button>
            <Button className="w-full" variant="secondary">
              Modell-Snapshot sichern
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
