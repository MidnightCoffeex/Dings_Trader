"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

interface PaperAccount {
  model_id: string;
  model_package_id?: string;
  warmup_required?: boolean;
  warmup_status?: string;
  initial_balance: number;
  balance_usdt: number;
  total_equity: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_return_pct: number;
  total_return_usdt: number;
  max_positions: number;
  default_leverage: number;
}

interface PaperDashboardData {
  account: PaperAccount;
  open_positions_count: number;
}

interface PaperTradingStatusProps {
  modelId: string;
  initialData: PaperDashboardData;
}

export function PaperTradingStatus({ modelId, initialData }: PaperTradingStatusProps) {
  const [data, setData] = useState<PaperDashboardData>(initialData);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/paper/dashboard/${modelId}`, {
          cache: "no-store",
          headers: { "Accept": "application/json" },
        });
        if (res.ok) {
          const newData = await res.json();
          setData(newData);
        }
      } catch (error) {
        console.error("Paper Trading poll error:", error);
      } finally {
        setLoading(false);
      }
    };

    // Poll every 15 seconds
    const interval = setInterval(fetchData, 15000);

    return () => clearInterval(interval);
  }, [modelId]);

  const account = data.account;
  const isProfitable = account.total_return_pct >= 0;

  return (
    <Card className="bg-card/40 border-primary/20">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Paper Trading Status
          </CardTitle>
          {loading && (
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Startkapital</span>
            <span className="font-medium">{account.initial_balance.toLocaleString()} USDT</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Aktuelle Equity</span>
            <span className={`font-medium ${isProfitable ? 'text-emerald-400' : 'text-rose-400'}`}>
              {account.total_equity.toLocaleString()} USDT
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Positionen</span>
            <span className="font-medium">{data.open_positions_count} / {account.max_positions}</span>
          </div>
          {account.warmup_required ? (
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Warmup</span>
              <span className={`font-medium ${account.warmup_status === 'DONE' ? 'text-emerald-400' : 'text-amber-300'}`}>
                {account.warmup_status || 'PENDING'}
              </span>
            </div>
          ) : null}
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Win Rate</span>
            <span className="font-medium">{account.win_rate.toFixed(1)}%</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Trades gesamt</span>
            <span className="font-medium">{account.total_trades}</span>
          </div>
          <div className="pt-2 border-t border-border/40 text-[10px] text-muted-foreground">
            Paper Trading simuliert Trades mit 10k USDT Startkapital. 
            Max 5 Positionen, 7-10x Hebel, 48h Limit.
          </div>
        </div>
      </CardContent>
    </Card>
  );
}