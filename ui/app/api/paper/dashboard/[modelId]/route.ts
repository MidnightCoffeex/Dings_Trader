import { NextResponse } from "next/server";

// Paper Trading Dashboard API
// Returns all data needed for the Paper Trading Dashboard

const API_BASE = "http://127.0.0.1:8000";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ modelId: string }> }
) {
  const { modelId } = await params;

  try {
    const response = await fetch(`${API_BASE}/paper/dashboard/${modelId}`, {
      headers: {
        "Accept": "application/json",
      },
      next: { revalidate: 0 },
    });

    if (!response.ok) {
      // If account doesn't exist, create it
      if (response.status === 404) {
        // Try to create account
        const createRes = await fetch(`${API_BASE}/paper/account/create`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_id: modelId,
            initial_balance: 10000.0,
            max_positions: 5,
            default_leverage: 7.0,
            profit_target_pct: 5.0,
            time_limit_hours: 48.0,
          }),
        });

        if (createRes.ok) {
          // Fetch again
          const retryRes = await fetch(`${API_BASE}/paper/dashboard/${modelId}`, {
            headers: { "Accept": "application/json" },
            next: { revalidate: 0 },
          });

          if (retryRes.ok) {
            const data = await retryRes.json();
            return NextResponse.json(data, {
              headers: {
                "Cache-Control": "no-cache, no-store, must-revalidate",
              },
            });
          }
        }
      }

      throw new Error(`Paper Trading API error: ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json(data, {
      headers: {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
      },
    });
  } catch (error) {
    console.error("Paper Trading Dashboard fetch error:", error);
    
    // Return fallback data
    return NextResponse.json({
      account: {
        model_id: modelId,
        initial_balance: 10000,
        balance_usdt: 10000,
        total_equity: 10000,
        total_trades: 0,
        winning_trades: 0,
        losing_trades: 0,
        win_rate: 0,
        total_return_pct: 0,
        total_return_usdt: 0,
        max_positions: 5,
        default_leverage: 7,
      },
      ml_signal: {
        signal: "FLAT",
        confidence: 0,
        sentiment: "neutral",
        current_price: 0,
        probabilities: { short: 0, flat: 100, long: 0 },
      },
      open_positions: [],
      open_positions_count: 0,
      available_slots: 5,
      recent_trades: [],
      performance: {
        total_return_pct: 0,
        win_rate: 0,
        avg_pnl_pct: 0,
        avg_win_pct: 0,
        avg_loss_pct: 0,
        open_exposure_pct: 0,
      },
      error: "API not available",
    }, { status: 200 });
  }
}
