import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { model_id, keep_history } = body;

    const response = await fetch(`http://127.0.0.1:8000/paper/account/${model_id}/reset?keep_history=${keep_history}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Reset proxy error:", error);
    return NextResponse.json(
      { error: "Failed to reset account" },
      { status: 500 }
    );
  }
}
