import { NextResponse } from "next/server";

// Paper Trading API Proxy
// Proxies requests to the Python FastAPI backend

const API_BASE = "http://127.0.0.1:8000";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const path = searchParams.get("path") || "dashboard/paper_v1";

  try {
    const response = await fetch(`${API_BASE}/paper/${path}`, {
      headers: {
        "Accept": "application/json",
      },
      next: { revalidate: 0 },
    });

    if (!response.ok) {
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
    console.error("Paper Trading API fetch error:", error);
    return NextResponse.json(
      { error: "Failed to fetch paper trading data" },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { path = "signal", ...data } = body;

    const response = await fetch(`${API_BASE}/paper/${path}`, {
      method: "POST",
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`Paper Trading API error: ${response.status}`);
    }

    const result = await response.json();

    return NextResponse.json(result, {
      headers: {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
      },
    });
  } catch (error) {
    console.error("Paper Trading API post error:", error);
    return NextResponse.json(
      { error: "Failed to process paper trading request" },
      { status: 500 }
    );
  }
}
