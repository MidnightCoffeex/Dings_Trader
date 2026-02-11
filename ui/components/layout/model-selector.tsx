"use client";

import { useState, useEffect } from "react";
import { ChevronDown } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

type ModelOption = {
  id: string;
  name: string;
  status: "Live" | "Live-Sim" | "Archiv";
  since?: string;
};

const MODELS: ModelOption[] = [
  { id: "ppo_v1", name: "PPO v1 (ML)", status: "Live-Sim", since: "11.02.2026" },
];

const statusVariant: Record<ModelOption["status"], "success" | "purple" | "warning"> = {
  Live: "success",
  "Live-Sim": "purple",
  Archiv: "warning",
};

export function ModelSelector() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const currentId = searchParams.get("model") || "ppo_v1";
  
  const [current, setCurrent] = useState(
    MODELS.find((m) => m.id === currentId) || MODELS[0]
  );

  useEffect(() => {
    const model = MODELS.find((m) => m.id === currentId);
    if (model) setCurrent(model);
  }, [currentId]);

  const handleSelect = (model: ModelOption) => {
    setCurrent(model);
    router.push(`?model=${model.id}`);
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" className="gap-2">
          Modell: {current.id.toUpperCase()}
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64">
        <DropdownMenuLabel>Modelle</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {MODELS.map((model) => (
          <DropdownMenuItem
            key={model.id}
            onSelect={() => handleSelect(model)}
            className="flex items-center justify-between gap-3"
          >
            <div>
              <div className="text-sm font-medium">{model.name}</div>
              <div className="text-xs text-muted-foreground">
                {model.id.toUpperCase()}
              </div>
              {model.since ? (
                <div className="text-[10px] text-muted-foreground">
                  {model.status} seit {model.since}
                </div>
              ) : (
                <div className="text-[10px] text-muted-foreground">Archiviert</div>
              )}
            </div>
            <Badge variant={statusVariant[model.status]}>{model.status}</Badge>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
