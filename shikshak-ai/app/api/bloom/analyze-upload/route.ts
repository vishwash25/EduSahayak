import { NextRequest, NextResponse } from "next/server";
import { normalizeBloomData } from "../normalize";

const PYTHON_BACKEND =
  process.env.PYTHON_BACKEND_URL || "http://127.0.0.1:8000";

export const maxDuration = 120;

export async function POST(req: NextRequest) {
  try {
    const incoming = await req.formData();
    const file = incoming.get("file") as File | null;
    const subject = String(incoming.get("subject") || "");
    const level = String(incoming.get("level") || "");

    if (!file) {
      return NextResponse.json(
        { status: "error", message: "No file provided." },
        { status: 400 }
      );
    }

    const form = new FormData();
    form.append("file", file, file.name);
    form.append("subject", subject);
    form.append("level", level);

    const response = await fetch(
      `${PYTHON_BACKEND}/api/bloom/analyze-upload`,
      {
        method: "POST",
        body: form,
        signal: AbortSignal.timeout(110_000),
      }
    );

    const data = await response.json();

    if (!response.ok || data.status !== "success") {
      return NextResponse.json(
        {
          status: "error",
          message: data.detail || data.message || "Python Bloom analysis failed.",
        },
        { status: response.status || 500 }
      );
    }

    const normalized = normalizeBloomData(data, "");

    return NextResponse.json({
      status: "success",
      ...normalized,
      filename: file.name,
      subject,
      level,
    });
  } catch (error: any) {
    console.error("[BLOOM UPLOAD]", error);

    return NextResponse.json(
      {
        status: "error",
        message: error.message || "Failed to analyze uploaded document.",
      },
      { status: 500 }
    );
  }
}
