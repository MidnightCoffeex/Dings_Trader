"use client";

import type { ReactNode } from "react";
import { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";

// Model-specific start time cache
const cachedModelStartTimes: Record<string, Date> = {};

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

export function PageHeader({
  title,
  subtitle,
  badge,
  status,
  actions,
  isLive = true,
  showLiveTimer = true,
  modelId = "paper_ppo_v1",
}: {
  title: string;
  subtitle?: string;
  badge?: string;
  status?: string;
  actions?: ReactNode;
  isLive?: boolean;
  showLiveTimer?: boolean;
  modelId?: string;
}) {
  const [liveSince, setLiveSince] = useState<string>("00:00:00");
  const [modelStartTime, setModelStartTime] = useState<Date | null>(cachedModelStartTimes[modelId] || null);

  // Fetch model-specific start time from backend
  useEffect(() => {
    if (!showLiveTimer) return;

    const fetchModelStatus = async () => {
      try {
        const res = await fetch(`/api/paper/account/${modelId}`, {
          cache: "no-store",
        });
        if (res.ok) {
          const data = await res.json();
          const rawTime = data.reset_at || data.created_at;
          if (rawTime) {
            // Ensure ISO format is parsed as UTC if no timezone is present
            const dateStr = rawTime.includes('Z') || rawTime.includes('+') 
              ? rawTime 
              : `${rawTime}Z`;
            const startTime = new Date(dateStr);
            cachedModelStartTimes[modelId] = startTime;
            setModelStartTime(startTime);
          }
        }
      } catch (err) {
        console.error("Failed to fetch model status:", err);
      }
    };

    fetchModelStatus();
    
    // Refresh start time every 10 seconds to catch resets
    const refreshInterval = setInterval(fetchModelStatus, 10000);
    return () => clearInterval(refreshInterval);
  }, [showLiveTimer, modelId]);

  useEffect(() => {
    if (!showLiveTimer || !modelStartTime) return;

    const updateTimer = () => {
      const now = new Date();
      const diff = now.getTime() - modelStartTime.getTime();
      setLiveSince(formatDuration(diff));
    };

    updateTimer();
    const interval = setInterval(updateTimer, 1000);
    return () => clearInterval(interval);
  }, [showLiveTimer, modelStartTime]);

  return (
    <div className="mb-6 flex items-start justify-between gap-4">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <h1 className="truncate text-xl font-semibold tracking-tight">{title}</h1>
          {badge ? <Badge variant="purple">{badge}</Badge> : null}
          {isLive ? (
            <Badge variant="success" className="animate-pulse flex items-center gap-1">
              <span className="w-2 h-2 bg-green-400 rounded-full"></span>
              🟢 Live
            </Badge>
          ) : (
            <Badge variant="destructive" className="animate-pulse">
              🔴 Offline
            </Badge>
          )}
          {showLiveTimer && isLive && (
            <Badge variant="outline" className="font-mono text-xs" title={`Model ${modelId} - resets on account reset`}>
              Live seit: {liveSince}
            </Badge>
          )}
          {status && status !== "LIVE" && !isLive ? (
            <Badge variant="destructive" className="animate-pulse">
              {status}
            </Badge>
          ) : null}
        </div>
        {subtitle ? (
          <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
        ) : null}
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </div>
  );
}
