import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Konfiguration"
        subtitle="Verwaltung der Symbiose-Parameter und Risiko-Limits."
      />

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="bg-card/40 lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Profil
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">Name</div>
              <Input defaultValue="Maxim" />
            </div>
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">Entität</div>
              <Input defaultValue="Symbiomorphose" />
            </div>
            <div className="space-y-2 sm:col-span-2">
              <div className="text-xs text-muted-foreground">Moltbook Handle</div>
              <Input defaultValue="u/Dings" />
            </div>
            <div className="sm:col-span-2">
              <Button>Speichern</Button>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/40">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Risiko-Parameter
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="rounded-md border border-border/70 bg-background/30 p-3">
              <div className="text-xs text-muted-foreground">Max. Hebel</div>
              <div className="text-sm font-medium text-primary">5.0x</div>
            </div>
            <div className="rounded-md border border-border/70 bg-background/30 p-3">
              <div className="text-xs text-muted-foreground">Exposure Cap</div>
              <div className="text-sm font-medium text-primary">10%</div>
            </div>
            <div className="rounded-md border border-border/70 bg-background/30 p-3">
              <div className="text-xs text-muted-foreground">Time-out Limit</div>
              <div className="text-sm font-medium text-primary">48h</div>
            </div>
            <Button variant="outline" className="w-full">
              Limits anpassen
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
