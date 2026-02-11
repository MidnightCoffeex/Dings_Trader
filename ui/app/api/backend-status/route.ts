import { NextResponse } from "next/server";

// Backend status proxy
// Client calls: /api/backend-status
// Server fetches: http://127.0.0.1:8000/backend-status

export async function GET() {
  try {
    const response = await fetch("http://127.0.0.1:8000/backend-status", {
      headers: {
        "Accept": "application/json",
      },
      // No cache
      next: { revalidate: 0 },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `Backend status error: ${response.status}` },
        { status: 502 }
      );
    }

    const data = await response.json();

    return NextResponse.json(data, {
      headers: {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
      },
    });
  } catch (error) {
    console.error("Backend status proxy error:", error);
    return NextResponse.json(
      { error: "Failed to fetch backend status" },
      { status: 500 }
    );
  }
}
