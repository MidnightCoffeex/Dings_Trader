"use client";

import { useEffect, useMemo, useState, Suspense } from "react";
import { ChevronDown, Plus, Upload, X } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

type ModelPackage = {
  id: string;
  name: string;
  status: string;
  created_at: string;
  warmup_required: number | boolean;
  warmup_status?: string;
  warmup_completed_at?: string | null;
};

type ModelOption = {
  id: string;
  name: string;
  statusLabel: "Live" | "Live-Sim" | "Archiv";
  since?: string;
  warmupStatus?: string;
};

const statusVariant: Record<ModelOption["statusLabel"], "success" | "purple" | "warning"> = {
  Live: "success",
  "Live-Sim": "purple",
  Archiv: "warning",
};

function formatDateIso(iso: string | undefined): string | undefined {
  if (!iso) return undefined;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return undefined;
  return d.toLocaleDateString("de-DE");
}

function mapPackageToOption(p: ModelPackage): ModelOption {
  const status = (p.status || "").toUpperCase();
  const statusLabel: ModelOption["statusLabel"] = status === "ARCHIVED" ? "Archiv" : "Live-Sim";
  return {
    id: p.id,
    name: p.name,
    statusLabel,
    since: formatDateIso(p.created_at),
    warmupStatus: p.warmup_status,
  };
}

function FileDropzone({
  label,
  accept,
  file,
  onChange,
  hint,
}: {
  label: string;
  accept: string;
  file: File | null;
  onChange: (file: File | null) => void;
  hint?: string;
}) {
  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    onChange(f);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0] || null;
    onChange(f);
  };

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  return (
    <div className="space-y-2">
      <div className="text-sm font-medium">{label}</div>
      <div
        className="rounded-md border border-dashed border-border/70 bg-card/30 p-4"
        onDrop={onDrop}
        onDragOver={onDragOver}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="text-sm text-muted-foreground">
              {file ? (
                <span className="text-foreground">
                  {file.name} <span className="text-xs text-muted-foreground">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                </span>
              ) : (
                "Drag & Drop oder Datei auswählen"
              )}
            </div>
            {hint ? <div className="text-[11px] text-muted-foreground mt-1">{hint}</div> : null}
          </div>
          <label className="shrink-0">
            <input
              type="file"
              accept={accept}
              onChange={onPick}
              className="hidden"
            />
            <Button type="button" variant="outline" size="sm" className="gap-2">
              <Upload className="h-4 w-4" />
              Upload
            </Button>
          </label>
        </div>
        {file ? (
          <div className="mt-3">
            <Button type="button" variant="ghost" size="sm" onClick={() => onChange(null)} className="gap-2 text-muted-foreground">
              <X className="h-4 w-4" />
              Datei entfernen
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function AddModelModal({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (created: { id: string; name: string }) => void;
}) {
  const [name, setName] = useState("");
  const [forecastFile, setForecastFile] = useState<File | null>(null);
  const [ppoFile, setPpoFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setError(null);
    }
  }, [open]);

  const validate = (): string | null => {
    if (!name.trim()) return "Bitte Modellname angeben.";
    if (!forecastFile) return "Bitte Forecast-Modell (.pt) auswählen.";
    if (!forecastFile.name.endsWith(".pt")) return "Forecast-Modell muss eine .pt Datei sein.";
    if (!ppoFile) return "Bitte PPO-Modell (.zip) auswählen.";
    if (!ppoFile.name.endsWith(".zip")) return "PPO-Modell muss eine .zip Datei sein.";
    return null;
  };

  const submit = async () => {
    const v = validate();
    if (v) {
      setError(v);
      return;
    }

    try {
      setSubmitting(true);
      setError(null);

      const form = new FormData();
      form.append("name", name.trim());
      form.append("forecast_model", forecastFile as File);
      form.append("ppo_model", ppoFile as File);

      const res = await fetch("/api/model-packages", {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const t = await res.text().catch(() => "");
        throw new Error(t || "Upload failed");
      }

      const created = (await res.json()) as { id: string; name: string };
      onCreated({ id: created.id, name: created.name });

      // Reset modal state
      setName("");
      setForecastFile(null);
      setPpoFile(null);
      onClose();
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Upload failed";
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div className="w-full max-w-lg rounded-xl border border-border/70 bg-background shadow-xl">
          <div className="flex items-center justify-between border-b border-border/60 px-5 py-4">
            <div className="min-w-0">
              <div className="text-sm font-semibold">Neues Modell hinzufügen</div>
              <div className="text-[11px] text-muted-foreground">
                Forecast (.pt) + PPO (.zip) werden als gemeinsames Modellpaket gespeichert.
              </div>
            </div>
            <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close">
              <X className="h-4 w-4" />
            </Button>
          </div>

          <div className="px-5 py-4 space-y-4">
            <div className="space-y-2">
              <div className="text-sm font-medium">Modellname</div>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="z.B. PPO v2 (Experiment)"
              />
            </div>

            <FileDropzone
              label="Forecast-Modell (.pt)"
              accept=".pt"
              file={forecastFile}
              onChange={setForecastFile}
              hint="Wird als forecast_model.pt gespeichert."
            />

            <FileDropzone
              label="PPO-Modell (.zip)"
              accept=".zip"
              file={ppoFile}
              onChange={setPpoFile}
              hint="Wird als ppo_policy.zip gespeichert."
            />

            {error ? (
              <div className="rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
                {error}
              </div>
            ) : null}
          </div>

          <div className="flex items-center justify-end gap-2 border-t border-border/60 px-5 py-4">
            <Button variant="outline" onClick={onClose} disabled={submitting}>
              Abbrechen
            </Button>
            <Button onClick={submit} disabled={submitting} className="gap-2">
              <Plus className="h-4 w-4" />
              {submitting ? "Upload…" : "Modellpaket speichern"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ModelSelectorContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const currentId = searchParams.get("model") || "ppo_v1";

  const [packages, setPackages] = useState<ModelPackage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);

  const options = useMemo(() => packages.map(mapPackageToOption), [packages]);

  const current = useMemo(() => {
    const found = options.find((m) => m.id === currentId);
    return found || options[0] || { id: currentId, name: currentId, statusLabel: "Live-Sim" as const };
  }, [options, currentId]);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch("/api/model-packages", { cache: "no-store" });
      if (!res.ok) throw new Error("failed");
      const data = (await res.json()) as ModelPackage[];
      setPackages(Array.isArray(data) ? data : []);
    } catch (e) {
      setError("Konnte Modelle nicht laden");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleSelect = (modelId: string) => {
    router.push(`?model=${modelId}`);
  };

  const handleCreated = async (created: { id: string }) => {
    await load();
    handleSelect(created.id);
  };

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" className="gap-2" disabled={loading}>
            Modell: {current.id.toUpperCase()}
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-72">
          <DropdownMenuLabel>Modelle</DropdownMenuLabel>
          <DropdownMenuSeparator />

          {error ? (
            <div className="px-2 py-2 text-xs text-rose-300">{error}</div>
          ) : null}

          {options.map((model) => (
            <DropdownMenuItem
              key={model.id}
              onSelect={() => handleSelect(model.id)}
              className="flex items-center justify-between gap-3"
            >
              <div>
                <div className="text-sm font-medium">{model.name}</div>
                <div className="text-xs text-muted-foreground">{model.id.toUpperCase()}</div>
                <div className="text-[10px] text-muted-foreground">
                  {model.since ? `${model.statusLabel} seit ${model.since}` : ""}
                  {model.warmupStatus ? ` • Warmup: ${model.warmupStatus}` : ""}
                </div>
              </div>
              <Badge variant={statusVariant[model.statusLabel]}>{model.statusLabel}</Badge>
            </DropdownMenuItem>
          ))}

          <DropdownMenuSeparator />
          <DropdownMenuItem
            onSelect={(e) => {
              e.preventDefault();
              setAddOpen(true);
            }}
            className="gap-2"
          >
            <Plus className="h-4 w-4" />
            Neues Modell hinzufügen
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <AddModelModal open={addOpen} onClose={() => setAddOpen(false)} onCreated={handleCreated} />
    </>
  );
}

export function ModelSelector() {
  return (
    <Suspense fallback={<Button variant="outline" disabled>Loading Models...</Button>}>
      <ModelSelectorContent />
    </Suspense>
  );
}
