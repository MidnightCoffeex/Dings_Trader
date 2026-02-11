import { NextResponse } from "next/server";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ modelId: string }> }
) {
  const modelId = (await params).modelId;
  try {
    const res = await fetch(`http://127.0.0.1:8000/paper/account/${modelId}`, {
      cache: "no-store",
    });
    if (!res.ok) {
      return NextResponse.json({ error: "Failed to fetch account" }, { status: res.status });
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
