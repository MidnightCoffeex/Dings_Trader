"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Wallet, Target, TrendingUp, PiggyBank, Info } from "lucide-react";
import { useEffect, useState, useCallback } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface PaperAccount {
  model_id: string;
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

interface OpenPosition {
  id: string;
  symbol: string;
  side: string;
  size_usdt: number;
  entry_price: number;
  current_price: number;
  pnl_usdt: number;
  pnl_pct: number;
  leverage: number;
  opened_at: string;
}

interface PaperDashboardData {
  account: PaperAccount;
  open_positions: OpenPosition[];
  performance: {
    total_return_pct: number;
  };
}

interface AccountBalanceProps {
  account?: PaperAccount;
  modelId?: string;
}

export function AccountBalance({ account: initialAccount, modelId = "paper_ppo_v1" }: AccountBalanceProps) {
  const [account, setAccount] = useState<PaperAccount | null>(initialAccount || null);
  const [openPositions, setOpenPositions] = useState<OpenPosition[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Polling function (reduced pressure on /api/paper/dashboard)
  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(`/api/paper/dashboard/${modelId}`, {
        cache: "no-store",
        headers: { "Accept": "application/json" },
      });
      
      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }
      
      const data: PaperDashboardData = await res.json();
      setAccount(data.account);
      setOpenPositions(data.open_positions || []);
      setError(null);
    } catch (err) {
      console.error("Failed to fetch account data:", err);
      setError("Failed to update");
    }
  }, [modelId]);

  // Set up polling interval
  useEffect(() => {
    // Fetch immediately on mount
    fetchData();
    
    // Poll every 15 seconds
    const interval = setInterval(fetchData, 15000);
    
    return () => clearInterval(interval);
  }, [fetchData]);

  // Use real data from paper trading API, fallback to demo if not available
  const defaultAccount = {
    model_id: "default",
    initial_balance: 10000,
    total_equity: 10000,
    balance_usdt: 10000,
    total_trades: 0,
    winning_trades: 0,
    losing_trades: 0,
    win_rate: 0,
    total_return_pct: 0,
    total_return_usdt: 0,
    max_positions: 5,
    default_leverage: 7,
  };
  
  const data = account || defaultAccount;

  // Calculate display values
  const startBalance = data.initial_balance;
  const currentBalance = data.total_equity;
  const totalGrowth = data.total_return_pct;
  
  // Calculate invested amount from open positions (sum of size_usdt)
  const totalInvested = openPositions.reduce((sum, pos) => sum + (pos.size_usdt || 0), 0);
  
  // Trading cap is 10% of total equity
  const tradingCap = currentBalance * 0.10;
  
  // Available for trading is trading cap minus what's already invested
  const availableForTrading = Math.max(0, tradingCap - totalInvested);
  
  // Daily target is 3% of start balance
  const dailyTarget = 3;
  const dailyTargetUsd = startBalance * 0.03;
  
  // Estimate today's P&L (in real app would come from daily tracking)
  const todayPnl = totalGrowth;
  const todayPnlUsd = data.total_return_usdt;
  
  const targetProgress = Math.min((todayPnl / dailyTarget) * 100, 100);
  const isTargetMet = todayPnl >= dailyTarget;
  const isProfitDay = todayPnl >= 0;

  function formatUsd(value: number): string {
    const fixed = value.toFixed(2).replace(".", ",");
    const [intPart, fracPart] = fixed.split(",");
    const withSep = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, ".");
    return `${withSep},${fracPart} USDT`;
  }

  return (
    <Card className="bg-card/40 border-primary/20">
      <CardHeader className="pb-3 2xl:pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm 2xl:text-base font-medium text-muted-foreground flex items-center gap-2">
            <Wallet className="w-4 h-4 2xl:w-5 2xl:h-5" />
            Account Balance
          </CardTitle>
          <div className="flex items-center gap-2">
            {error && (
              <span className="text-[10px] text-rose-400">{error}</span>
            )}
            <Badge
              variant={isProfitDay ? "success" : "destructive"}
              className="text-[10px] 2xl:text-xs"
            >
              {isProfitDay ? '+' : ''}{totalGrowth.toFixed(1)}% Gesamt
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 2xl:space-y-6 2xl:min-h-[500px]">
        {/* Haupt-Balance */}
        <div className="space-y-1 2xl:space-y-1.5">
          <div className="text-xs 2xl:text-sm text-muted-foreground">Total Equity (Paper)</div>
          <div className="text-3xl 2xl:text-4xl font-bold tracking-tight">
            {formatUsd(currentBalance)}
          </div>
          <div className={`text-sm 2xl:text-base ${totalGrowth >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {totalGrowth >= 0 ? '+' : ''}{totalGrowth.toFixed(1)}% seit Start
          </div>
        </div>

        {/* Cash vs Invested */}
        <TooltipProvider>
          <div className="grid grid-cols-2 gap-3 2xl:gap-4">
            <div className="p-3 2xl:p-4 rounded-lg bg-background/40 border border-border/40">
              <div className="flex items-center gap-1.5 text-[11px] 2xl:text-xs text-muted-foreground mb-1 2xl:mb-1.5">
                <PiggyBank className="w-3 h-3 2xl:w-4 2xl:h-4" />
                <Tooltip>
                  <TooltipTrigger className="flex items-center gap-1 cursor-help">
                    Available for Trading
                    <Info className="w-3 h-3 text-muted-foreground/60" />
                  </TooltipTrigger>
                  <TooltipContent side="top">
                    <p className="text-xs">Based on 10% Trading Cap of Total Equity</p>
                    <p className="text-xs text-muted-foreground">Cap: {formatUsd(tradingCap)}</p>
                  </TooltipContent>
                </Tooltip>
              </div>
              <div className="font-semibold 2xl:text-lg">{formatUsd(availableForTrading)}</div>
              <div className="text-[10px] text-muted-foreground mt-0.5">
                Cap: {formatUsd(tradingCap)}
              </div>
            </div>
            <div className="p-3 2xl:p-4 rounded-lg bg-background/40 border border-border/40">
              <div className="flex items-center gap-1.5 text-[11px] 2xl:text-xs text-muted-foreground mb-1 2xl:mb-1.5">
                <TrendingUp className="w-3 h-3 2xl:w-4 2xl:h-4" />
                Invested
              </div>
              <div className="font-semibold 2xl:text-lg">{formatUsd(totalInvested)}</div>
              <div className="text-[10px] text-muted-foreground mt-0.5">
                {openPositions.length} position{openPositions.length !== 1 ? 's' : ''}
              </div>
            </div>
          </div>
        </TooltipProvider>

        {/* Daily Target Progress */}
        <div className="space-y-2 2xl:space-y-3">
          <div className="flex items-center justify-between text-xs 2xl:text-sm">
            <div className="flex items-center gap-1.5 text-muted-foreground">
              <Target className="w-3 h-3 2xl:w-4 2xl:h-4" />
              Daily Target ({dailyTarget}%)
            </div>
            <div className="flex items-center gap-2">
              <span className={isTargetMet ? 'text-emerald-400 font-medium' : ''}>
                {totalGrowth.toFixed(1)}%
              </span>
              <span className="text-muted-foreground">/ {dailyTarget}%</span>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="relative h-3 2xl:h-4 bg-muted rounded-full overflow-hidden">
            {/* Target Marker */}
            <div className="absolute inset-y-0 w-[2px] bg-primary/50" style={{ left: '100%' }} />
            {/* Progress */}
            <div
              className={`absolute inset-y-0 left-0 transition-all duration-500 rounded-full ${
                isTargetMet ? 'bg-emerald-500' : 'bg-primary/70'
              }`}
              style={{ width: `${Math.min(targetProgress, 100)}%` }}
            />
          </div>

          <div className="flex justify-between text-[10px] 2xl:text-xs text-muted-foreground">
            <span>0%</span>
            <span>50%</span>
            <span className={isTargetMet ? 'text-emerald-400 font-medium' : ''}>
              {isTargetMet ? '✓ Target erreicht!' : 'Target: ' + dailyTarget + '%'}
            </span>
          </div>
        </div>

        {/* P&L Breakdown */}
        <div className="pt-2 2xl:pt-3 border-t border-border/40 space-y-1.5 2xl:space-y-2">
          <div className="flex items-center justify-between text-xs 2xl:text-sm">
            <span className="text-muted-foreground">Gesamt P&L</span>
            <span className={`font-medium ${isProfitDay ? 'text-emerald-400' : 'text-rose-400'}`}>
              {isProfitDay ? '+' : ''}{formatUsd(data.total_return_usdt)}
            </span>
          </div>
          <div className="flex items-center justify-between text-xs 2xl:text-sm">
            <span className="text-muted-foreground">Win Rate</span>
            <span className="font-medium">{data.win_rate?.toFixed(1) || 0}%</span>
          </div>
          <div className="flex items-center justify-between text-xs 2xl:text-sm">
            <span className="text-muted-foreground">Trades</span>
            <span className="text-muted-foreground">{data.total_trades || 0} ({data.winning_trades || 0}W / {data.losing_trades || 0}L)</span>
          </div>
          <div className="flex items-center justify-between text-xs 2xl:text-sm">
            <span className="text-muted-foreground">Startkapital</span>
            <span className="text-muted-foreground">{formatUsd(startBalance)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
