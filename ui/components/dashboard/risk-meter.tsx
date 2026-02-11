"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Shield, HelpCircle, Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useEffect, useState, useCallback } from "react";

interface PaperPosition {
  id: string | number;
  symbol: string;
  side: string;
  size_usdt: number;
}

interface PaperDashboardData {
  account: {
    model_id: string;
    total_equity: number;
    max_positions: number;
  };
  open_positions: PaperPosition[];
  open_positions_count: number;
  performance: {
    total_return_pct: number;
    win_rate: number;
    avg_pnl_pct: number;
    avg_win_pct: number;
    avg_loss_pct: number;
    open_exposure_pct: number;
  };
}

interface RiskMeterProps {
  totalExposure?: number;
  positionCount?: number;
  exposurePerPosition?: number[];
  marginUsed?: number;
  marginAvailable?: number;
  maxPositions?: number;
  maxExposure?: number;
  modelId?: string;
  noCard?: boolean;
}

export function RiskMeter({
  totalExposure: initialTotalExposure,
  positionCount: initialPositionCount,
  exposurePerPosition: initialExposurePerPosition,
  marginUsed,
  marginAvailable,
  maxPositions: initialMaxPositions = 5,
  maxExposure = 50,
  modelId = "paper_ppo_v1",
  noCard = false,
}: RiskMeterProps) {
  const [totalExposure, setTotalExposure] = useState(initialTotalExposure ?? 0);
  const [positionCount, setPositionCount] = useState(initialPositionCount ?? 0);
  const [exposurePerPosition, setExposurePerPosition] = useState<number[]>(initialExposurePerPosition ?? []);
  const [maxPositions, setMaxPositions] = useState(initialMaxPositions);
  const [error, setError] = useState<string | null>(null);

  // Polling function to fetch data every 2 seconds
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
      
      // Calculate exposure per position
      const totalEquity = data.account?.total_equity || 10000;
      const perPositionExposure = (data.open_positions || []).map(p => 
        (p.size_usdt / totalEquity) * 100
      );
      
      setTotalExposure(data.performance?.open_exposure_pct || 0);
      setPositionCount(data.open_positions_count || 0);
      setExposurePerPosition(perPositionExposure);
      setMaxPositions(data.account?.max_positions || 5);
      setError(null);
    } catch (err) {
      console.error("Failed to fetch risk data:", err);
      setError("Failed to update");
    }
  }, [modelId]);

  // Set up polling interval
  useEffect(() => {
    // Fetch immediately on mount
    fetchData();
    
    // Poll every 2 seconds
    const interval = setInterval(fetchData, 2000);
    
    return () => clearInterval(interval);
  }, [fetchData]);

  // Use provided values or defaults
  const exposure = totalExposure;
  const positions = positionCount;
  const perPosition = exposurePerPosition;
  
  // 50% = Maximum
  const exposurePercent = Math.min((exposure / maxExposure) * 100, 100);
  
  // Farbe basierend auf Exposure-Level
  const getColorClass = (pct: number) => {
    if (pct > 80) return "bg-rose-500";
    if (pct > 60) return "bg-amber-500";
    return "bg-emerald-500";
  };

  const isHighRisk = exposurePercent > 80;
  const isMediumRisk = exposurePercent > 60 && exposurePercent <= 80;

  const content = (
    <>
      {!noCard && (
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              {isHighRisk ? (
                <AlertTriangle className="w-4 h-4 text-rose-500" />
              ) : (
                <Shield className="w-4 h-4 text-emerald-500" />
              )}
              Risk-Meter
            </CardTitle>
            <div className="flex items-center gap-2">
              {error && (
                <span className="text-[10px] text-rose-400">{error}</span>
              )}
              <Badge 
                variant={isHighRisk ? "destructive" : isMediumRisk ? "warning" : "success"}
                className="text-[10px]"
              >
                {isHighRisk ? "HIGH" : isMediumRisk ? "MEDIUM" : "LOW"}
              </Badge>
            </div>
          </div>
        </CardHeader>
      )}
      <CardContent className={`space-y-3 2xl:space-y-4 ${noCard ? '' : '2xl:min-h-[400px]'}`}>
        {/* Haupt-Exposure Balken */}
        <div className="space-y-2 2xl:space-y-3">
          <div className="flex items-center justify-between text-xs 2xl:text-sm">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-muted-foreground flex items-center gap-1.5 cursor-help hover:text-foreground transition-colors">
                    Total Exposure
                    <Info className="w-3.5 h-3.5 text-primary/70" />
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-[280px] p-3">
                  <div className="space-y-2">
                    <p className="font-semibold text-xs">📊 Total Exposure Erklärung</p>
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      Zeigt an, wie viel % deines Kapitals aktuell in Trades gebunden ist.
                    </p>
                    <div className="space-y-1 pt-1 border-t border-border/50">
                      <div className="flex items-center gap-2 text-[11px]">
                        <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                        <span><strong>0-40%:</strong> Sicherer Bereich</span>
                      </div>
                      <div className="flex items-center gap-2 text-[11px]">
                        <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                        <span><strong>40-50%:</strong> Warnzone - Vorsicht</span>
                      </div>
                      <div className="flex items-center gap-2 text-[11px]">
                        <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                        <span><strong>&gt;50%:</strong> Maximum erreicht!</span>
                      </div>
                    </div>
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <span className={`font-medium ${isHighRisk ? 'text-rose-400' : isMediumRisk ? 'text-amber-400' : 'text-emerald-400'}`}>
              {exposure.toFixed(1)}% / {maxExposure}% Max
            </span>
          </div>
          
          {/* Progress Bar mit Markern */}
          <div className="relative h-4 2xl:h-5 bg-muted rounded-full overflow-hidden border border-border/30">
            {/* Background markers - 25% Schritte */}
            <div className="absolute inset-0 flex">
              <div className="flex-1 border-r border-background/50" />
              <div className="flex-1 border-r border-background/50" />
              <div className="flex-1 border-r border-background/50" />
              <div className="flex-1" />
            </div>
            {/* Progress */}
            <div 
              className={`absolute inset-y-0 left-0 transition-all duration-500 ${getColorClass(exposurePercent)}`}
              style={{ width: `${exposurePercent}%` }}
            />
            {/* 40% Warning Marker */}
            <div 
              className="absolute inset-y-0 w-0.5 bg-amber-500 z-10 shadow-[0_0_6px_rgba(245,158,11,0.8)]" 
              style={{ left: '80%' }}
            >
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="absolute -top-1 -translate-x-1/2 cursor-help">
                      <div className="bg-amber-500 text-amber-950 text-[9px] font-bold px-1.5 py-0.5 rounded-full shadow-md">
                        40%
                      </div>
                      <div className="w-0.5 h-1.5 bg-amber-500 mx-auto"></div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="top">
                    <span className="text-[11px]">⚠️ Warnschwelle bei 40% Exposure</span>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
          
          {/* Skala */}
          <div className="flex justify-between text-[10px] text-muted-foreground px-0.5">
            <span>0%</span>
            <span className="text-amber-500/70">12.5%</span>
            <span className="text-amber-500/70">25%</span>
            <span className="text-amber-500/70">37.5%</span>
            <span className="text-rose-400 font-medium">50% Max</span>
          </div>
        </div>

        {/* Position Count */}
        <div className="space-y-1.5 2xl:space-y-2">
          <div className="flex items-center justify-between text-xs 2xl:text-sm">
            <span className="text-muted-foreground">Aktive Positionen</span>
            <span className={`font-medium ${positions >= 4 ? 'text-amber-400' : ''}`}>
              {positions} / {maxPositions}
            </span>
          </div>
          <div className="relative h-2 2xl:h-3 bg-muted rounded-full overflow-hidden">
            <div 
              className={`absolute inset-y-0 left-0 transition-all duration-500 ${
                positions >= 4 ? 'bg-amber-500' : 'bg-primary/60'
              }`}
              style={{ width: `${(positions / maxPositions) * 100}%` }}
            />
          </div>
        </div>

        {/* Per-Position Caps mit Tooltip Erklärung */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-xs text-muted-foreground flex items-center gap-1.5 cursor-help hover:text-foreground transition-colors">
                    Position Slots
                    <Info className="w-3.5 h-3.5 text-primary/70" />
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-[260px] p-3">
                  <div className="space-y-2">
                    <p className="font-semibold text-xs">🎯 Position Slots Erklärung</p>
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      Jeder Slot repräsentiert eine offene Position. Maximal 5 gleichzeitige Positionen erlaubt.
                    </p>
                    <div className="space-y-1 pt-1 border-t border-border/50">
                      <div className="flex items-center gap-2 text-[11px]">
                        <span className="w-6 h-4 rounded bg-primary/20 border border-primary/40 flex items-center justify-center text-[8px]">10%</span>
                        <span>Belegt (max 10% pro Position)</span>
                      </div>
                      <div className="flex items-center gap-2 text-[11px]">
                        <span className="w-6 h-4 rounded bg-amber-500/20 border border-amber-500/50 flex items-center justify-center text-[8px]">12%</span>
                        <span className="text-amber-400">Über dem Limit!</span>
                      </div>
                      <div className="flex items-center gap-2 text-[11px]">
                        <span className="w-6 h-4 rounded border border-dashed border-border/40 flex items-center justify-center text-[8px]">-</span>
                        <span>Freier Slot</span>
                      </div>
                    </div>
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <span className="text-[10px] text-muted-foreground">
              Max 10% / Position
            </span>
          </div>
          
          <div className="grid grid-cols-5 gap-1.5 2xl:gap-2">
            {[1, 2, 3, 4, 5].map((slot) => {
              const hasPosition = slot <= perPosition.length;
              const positionExposure = perPosition[slot - 1] || 0;
              const isOverLimit = positionExposure > 10;
              
              return (
                <TooltipProvider key={slot}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex flex-col items-center gap-1 2xl:gap-1.5 cursor-help">
                        <div 
                          className={`w-full h-9 2xl:h-11 rounded-md flex items-center justify-center text-[10px] 2xl:text-xs font-medium transition-all border ${
                            hasPosition 
                              ? isOverLimit
                                ? 'bg-amber-500/20 text-amber-400 border-amber-500/50 shadow-[0_0_8px_rgba(245,158,11,0.2)]' 
                                : 'bg-primary/20 text-primary border-primary/40 shadow-[0_0_8px_rgba(59,130,246,0.15)]'
                              : 'bg-muted/20 text-muted-foreground/40 border-dashed border-border/40 hover:border-border/60'
                          }`}
                        >
                          {hasPosition ? (
                            <div className="flex flex-col items-center leading-none gap-0.5">
                              <span>{positionExposure.toFixed(0)}%</span>
                              {isOverLimit && <span className="text-[7px] text-amber-400">⚠️</span>}
                            </div>
                          ) : (
                            <span className="text-[10px] opacity-50">{slot}</span>
                          )}
                        </div>
                        <span className="text-[9px] 2xl:text-[11px] text-muted-foreground">Slot {slot}</span>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      {hasPosition ? (
                        <div className="text-[11px]">
                          <span className="font-medium">Position {slot}:</span>
                          <span className={isOverLimit ? 'text-amber-400 ml-1' : ' ml-1'}>
                            {positionExposure.toFixed(1)}% Exposure
                            {isOverLimit && ' ⚠️ Über 10%!'}
                          </span>
                        </div>
                      ) : (
                        <span className="text-[11px]">Slot {slot} verfügbar</span>
                      )}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              );
            })}
          </div>
        </div>

        {/* Risk Rules mit Tooltips */}
        <div className="pt-3 2xl:pt-4 border-t border-border/40 space-y-1.5 2xl:space-y-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center justify-between text-[10px] text-muted-foreground cursor-help hover:text-foreground transition-colors">
                  <span className="flex items-center gap-1">
                    Max 10% pro Position
                    <HelpCircle className="w-3 h-3 opacity-50" />
                  </span>
                  <span className={perPosition.every(e => e <= 10) ? 'text-emerald-500 font-medium' : 'text-amber-500 font-medium'}>
                    {perPosition.every(e => e <= 10) ? '✓ OK' : '⚠️ Über Limit'}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="left">
                <span className="text-[11px]">Maximales Risiko pro einzelner Position: 10%</span>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center justify-between text-[10px] text-muted-foreground cursor-help hover:text-foreground transition-colors">
                  <span className="flex items-center gap-1">
                    Max 50% Total Exposure
                    <HelpCircle className="w-3 h-3 opacity-50" />
                  </span>
                  <span className={exposure <= maxExposure ? 'text-emerald-500 font-medium' : exposure <= maxExposure * 1.1 ? 'text-amber-500 font-medium' : 'text-rose-500 font-medium'}>
                    {exposure <= maxExposure ? '✓ OK' : exposure <= maxExposure * 1.1 ? '⚠️ Grenze' : '✗ Überschritten'}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="left">
                <span className="text-[11px]">Maximales Gesamt-Exposure über alle Positionen: 50%</span>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center justify-between text-[10px] text-muted-foreground cursor-help hover:text-foreground transition-colors">
                  <span className="flex items-center gap-1">
                    Max {maxPositions} Positionen
                    <HelpCircle className="w-3 h-3 opacity-50" />
                  </span>
                  <span className={positions <= maxPositions ? 'text-emerald-500 font-medium' : 'text-rose-500 font-medium'}>
                    {positions <= maxPositions ? '✓ OK' : '✗ Limit erreicht'}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="left">
                <span className="text-[11px]">Maximal {maxPositions} gleichzeitige Positionen erlaubt</span>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        {/* Warnmeldungen basierend auf Risk-Level */}
        {isHighRisk && (
          <div className="text-[10px] 2xl:text-xs text-rose-400 text-center bg-rose-500/10 border border-rose-500/20 rounded-md py-2 2xl:py-2.5 px-3 animate-pulse">
            <span className="font-semibold">⚠️ Risk Warning</span>
            <span className="block text-[9px] 2xl:text-[11px] text-rose-400/80 mt-0.5">
              Exposure über 40% – Reduziere um Maximum von 50% nicht zu überschreiten
            </span>
          </div>
        )}
        {!isHighRisk && isMediumRisk && (
          <div className="text-[10px] 2xl:text-xs text-amber-400 text-center bg-amber-500/10 border border-amber-500/20 rounded-md py-2 2xl:py-2.5 px-3">
            <span className="font-semibold">⚡ Achtung</span>
            <span className="block text-[9px] 2xl:text-[11px] text-amber-400/80 mt-0.5">
              Du näherst dich der 40% Warnschwelle
            </span>
          </div>
        )}
      </CardContent>
    </>
  );

  if (noCard) {
    return content;
  }

  return (
    <Card className={`bg-card/40 ${isHighRisk ? 'border-rose-500/40' : 'border-primary/20'}`}>
      {content}
    </Card>
  );
}
