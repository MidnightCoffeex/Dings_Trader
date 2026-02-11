"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Database,
  Dna,
  Settings,
  Sparkles,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

const nav = [
  { href: "/dashboard", label: "Dashboard", icon: Activity },
];

export function Sidebar({
  mobileOpen,
  setMobileOpen,
}: {
  mobileOpen?: boolean;
  setMobileOpen?: (open: boolean) => void;
}) {
  const pathname = usePathname();

  const Nav = (
    <div className="flex h-full flex-col p-5">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-xl border border-border/70 bg-card shadow-glow">
            <Dna className="h-5 w-5 text-primary animate-dna" />
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold">Symbiomorphose</div>
            <div className="truncate text-xs text-muted-foreground">
              Mensch & Daten Synthese
            </div>
          </div>
        </div>

        <div className="mt-4 flex gap-2">
          <Badge variant="purple">v2.0</Badge>
          <Badge variant="outline" className="border-primary/30 text-primary">Live</Badge>
        </div>

        <Separator className="my-5" />

        <nav className="flex flex-col gap-1">
          {nav.map((item) => {
            const active = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "group flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                  active
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:bg-secondary/70 hover:text-foreground"
                )}
              >
                <Icon
                  className={cn(
                    "h-4 w-4",
                    active ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
                  )}
                />
                <span className="truncate">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="mt-auto">
          <Separator className="my-5" />
          <div className="rounded-lg border border-border/70 bg-card/40 p-4">
            <div className="text-xs font-medium">System</div>
            <div className="mt-2 grid gap-1 text-xs text-muted-foreground">
              <div className="flex items-center justify-between">
                <span>Exchange</span>
                <span className="text-foreground">BINANCE</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Latency</span>
                <span className="text-foreground">42ms</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Risk</span>
                <span className="text-foreground">Low</span>
              </div>
            </div>
          </div>
        </div>
    </div>
  );

  return (
    <>
      {/* desktop */}
      <aside className="sticky top-0 hidden h-screen w-[280px] shrink-0 border-r border-border/70 bg-background/40 backdrop-blur supports-[backdrop-filter]:bg-background/30 lg:block">
        {Nav}
      </aside>

      {/* mobile */}
      {mobileOpen ? (
        <div className="lg:hidden">
          <button
            aria-label="Close navigation"
            className="fixed inset-0 z-40 bg-black/60"
            onClick={() => setMobileOpen?.(false)}
          />
          <aside className="fixed inset-y-0 left-0 z-50 h-full w-[280px] border-r border-border/70 bg-background/95 backdrop-blur">
            <div className="flex items-center justify-between px-5 pt-5">
              <div className="text-sm font-semibold">Navigation</div>
              <button
                className="rounded-md border border-border/70 bg-card/40 px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
                onClick={() => setMobileOpen?.(false)}
              >
                Close
              </button>
            </div>
            <div className="px-5 pb-5 pt-2">{Nav}</div>
          </aside>
        </div>
      ) : null}
    </>
  );
}
