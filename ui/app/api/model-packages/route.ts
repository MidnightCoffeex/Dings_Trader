import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const API_BASE = "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${API_BASE}/model-packages`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      return NextResponse.json(
        { error: "Failed to fetch model packages", detail: text },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, {
      headers: {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        Pragma: "no-cache",
      },
    });
  } catch (error) {
    console.error("Model packages GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch model packages" },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const form = await request.formData();

    const response = await fetch(`${API_BASE}/model-packages/upload`, {
      method: "POST",
      headers: { Accept: "application/json" },
      body: form,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      return NextResponse.json(
        { error: "Failed to upload model package", detail: text },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, {
      headers: {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        Pragma: "no-cache",
      },
    });
  } catch (error) {
    console.error("Model packages POST error:", error);
    return NextResponse.json(
      { error: "Failed to upload model package" },
      { status: 500 }
    );
  }
}
