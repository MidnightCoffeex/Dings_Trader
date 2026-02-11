import { NextResponse } from "next/server";
import os from "os";
import { exec } from "child_process";

export const runtime = "nodejs";

type Pm2Monit = {
  cpu?: number;
  memory?: number;
};

type Pm2Proc = {
  name?: string;
  pid?: number;
  pm2_env?: {
    status?: string;
  };
  monit?: Pm2Monit;
};

function runPm2Jlist(): Promise<Pm2Proc[]> {
  return new Promise((resolve) => {
    exec("pm2 jlist", { timeout: 2500, maxBuffer: 2 * 1024 * 1024 }, (err, stdout) => {
      if (err || !stdout) {
        resolve([]);
        return;
      }

      try {
        const parsed = JSON.parse(stdout);
        resolve(Array.isArray(parsed) ? parsed : []);
      } catch {
        resolve([]);
      }
    });
  });
}

export async function GET() {
  try {
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const usedMem = Math.max(0, totalMem - freeMem);
    const cpuCores = os.cpus().length;
    const load = os.loadavg(); // [1m, 5m, 15m]

    const pm2 = await runPm2Jlist();

    const services = pm2
      .filter((p) => p?.name)
      .map((p) => ({
        name: p.name as string,
        status: p.pm2_env?.status ?? "unknown",
        pid: p.pid ?? null,
        cpuPct: Number(p.monit?.cpu ?? 0),
        memoryBytes: Number(p.monit?.memory ?? 0),
      }))
      .sort((a, b) => a.name.localeCompare(b.name));

    const totalServiceMem = services.reduce((acc, s) => acc + (Number.isFinite(s.memoryBytes) ? s.memoryBytes : 0), 0);
    const totalServiceCpu = services.reduce((acc, s) => acc + (Number.isFinite(s.cpuPct) ? s.cpuPct : 0), 0);

    return NextResponse.json(
      {
        ts: new Date().toISOString(),
        host: {
          cpuCores,
          load1m: load[0],
          load5m: load[1],
          load15m: load[2],
          load1mPct: cpuCores > 0 ? (load[0] / cpuCores) * 100 : 0,
          memory: {
            totalBytes: totalMem,
            usedBytes: usedMem,
            freeBytes: freeMem,
            usedPct: totalMem > 0 ? (usedMem / totalMem) * 100 : 0,
          },
        },
        services,
        servicesSummary: {
          count: services.length,
          totalCpuPct: totalServiceCpu,
          totalMemoryBytes: totalServiceMem,
        },
      },
      {
        headers: {
          "Cache-Control": "no-cache, no-store, must-revalidate",
          Pragma: "no-cache",
        },
      }
    );
  } catch (error) {
    return NextResponse.json(
      {
        error: "Failed to read system capacity",
        details: error instanceof Error ? error.message : "unknown",
      },
      { status: 500 }
    );
  }
}
