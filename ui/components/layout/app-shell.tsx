"use client";

import * as React from "react";

import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";

export function AppShell({
  children,
}: {
  children: React.ReactNode;
}) {
  const [mobileNavOpen, setMobileNavOpen] = React.useState(false);

  return (
    <div className="min-h-screen bg-dashboard">
      <div className="flex min-h-screen w-full">
        <Sidebar mobileOpen={mobileNavOpen} setMobileOpen={setMobileNavOpen} />
        <div className="flex min-w-0 flex-1 flex-col">
          <Topbar onOpenMobileNav={() => setMobileNavOpen(true)} />
          <main className="min-w-0 flex-1 p-4 sm:p-6 2xl:p-20">
            <div className="w-full">{children}</div>
          </main>
        </div>
      </div>
    </div>
  );
}
