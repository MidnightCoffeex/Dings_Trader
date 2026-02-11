import { NextResponse } from "next/server";

// Binance 24h Ticker Statistics API Proxy
// Endpoint: /api/binance/ticker?symbol=BTCUSDT

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const symbol = searchParams.get("symbol") || "BTCUSDT";

  try {
    const response = await fetch(
      `https://api.binance.com/api/v3/ticker/24hr?symbol=${symbol}`,
      {
        headers: {
          "Accept": "application/json",
        },
        next: { revalidate: 0 }, // No cache
      }
    );

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();
    
    // Extract relevant fields
    const ticker = {
      symbol: data.symbol,
      lastPrice: parseFloat(data.lastPrice),
      priceChange: parseFloat(data.priceChange),
      priceChangePercent: parseFloat(data.priceChangePercent),
      high24h: parseFloat(data.highPrice),
      low24h: parseFloat(data.lowPrice),
      volume24h: parseFloat(data.volume),
      quoteVolume24h: parseFloat(data.quoteVolume),
      openPrice: parseFloat(data.openPrice),
    };

    return NextResponse.json(ticker, {
      headers: {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
      },
    });
  } catch (error) {
    console.error("Binance ticker fetch error:", error);
    return NextResponse.json(
      { error: "Failed to fetch ticker data" },
      { status: 500 }
    );
  }
}
