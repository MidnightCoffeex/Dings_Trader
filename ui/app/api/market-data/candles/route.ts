import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const symbol = searchParams.get("symbol") || "BTCUSDT";
  const interval = searchParams.get("interval") || "15m";
  const limit = searchParams.get("limit") || "100";

  // Map UI timeframe to backend interval if needed
  // 15m -> 15m, 1h -> 1h, 1d -> 1d

  try {
    const res = await fetch(
      `http://127.0.0.1:8000/paper/market-data/candles?symbol=${symbol}&interval=${interval}&limit=${limit}`,
      { cache: "no-store" }
    );

    if (!res.ok) {
      console.warn("Backend /paper/market-data/candles failed");
      return NextResponse.json({ error: "Backend unavailable" }, { status: 502 });
    }

    const data = await res.json();
    // Backend returns {symbol, interval, candles:[...]}. UI wants array.
    return NextResponse.json(data.candles ?? []);
  } catch (error) {
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
