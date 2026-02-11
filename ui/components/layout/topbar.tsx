"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Bell, Menu, Plus, Search, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

export function Topbar({
  onOpenMobileNav,
}: {
  onOpenMobileNav?: () => void;
}) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [resetting, setResetting] = useState(false);

  const modelParam = searchParams.get("model");
  const modelId = Array.isArray(modelParam) ? modelParam[0] : modelParam || "v2";
  const paperModelId = `paper_${modelId}`;

  const handleReset = async () => {
    if (!confirm(`Reset ${paperModelId.toUpperCase()}? All history will be lost.`)) return;
    
    try {
      setResetting(true);
      const res = await fetch("/api/paper/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: paperModelId, keep_history: false }),
      });

      if (!res.ok) throw new Error("Reset failed");
      
      // Force refresh
      router.refresh();
      window.location.reload(); // Hard reload to ensure fresh state
    } catch (err) {
      console.error("Reset error:", err);
      alert("Failed to reset account");
    } finally {
      setResetting(false);
    }
  };

  return (
    <header className="sticky top-0 z-20 border-b border-border/70 bg-background/50 backdrop-blur supports-[backdrop-filter]:bg-background/35">
      <div className="flex h-16 w-full items-center gap-3 px-4 sm:px-6">
        <Button
          variant="outline"
          size="icon"
          className="lg:hidden"
          aria-label="Open navigation"
          onClick={onOpenMobileNav}
        >
          <Menu className="h-4 w-4" />
        </Button>

        <div className="relative hidden flex-1 md:block">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search symbols, strategies, runs…"
            className="pl-9"
          />
        </div>

        <div className="ml-auto flex items-center gap-2">
          <Button 
            variant="destructive" 
            className="hidden sm:inline-flex"
            onClick={handleReset}
            disabled={resetting}
          >
            <RotateCcw className={`h-4 w-4 mr-2 ${resetting ? 'animate-spin' : ''}`} />
            Reset Run
          </Button>
          <Button variant="outline" size="icon" aria-label="Notifications">
            <Bell className="h-4 w-4" />
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex items-center gap-2 rounded-full border border-border/70 bg-card/40 p-1 pl-1.5 pr-2 hover:bg-card/60">
                <Avatar className="h-8 w-8">
                  <AvatarFallback>MX</AvatarFallback>
                </Avatar>
                <div className="hidden text-left sm:block">
                  <div className="text-xs font-medium leading-4">Maxim</div>
                  <div className="text-[11px] text-muted-foreground leading-4">
                    quant@local
                  </div>
                </div>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-52">
              <DropdownMenuLabel>Account</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <Link href="/settings">Settings</Link>
              </DropdownMenuItem>
              <DropdownMenuItem>API keys</DropdownMenuItem>
              <DropdownMenuItem>Billing</DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="text-muted-foreground">
                Sign out (stub)
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
}
