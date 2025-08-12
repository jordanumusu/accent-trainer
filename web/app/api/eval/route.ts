import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const form = await req.formData();
  const audio = form.get("audio") as File;
  const text = form.get("text") as string;
  const lang = form.get("lang") as string;

  const fd = new FormData();
  fd.append("audio", audio, "u.webm");
  fd.append("text", text);
  fd.append("lang", lang);

  const r = await fetch(
    `${process.env.PRON_SERVER_URL ?? "http://localhost:8000"}/eval`,
    {
      method: "POST",
      body: fd,
    }
  );

  if (!r.ok) {
    const errorText = await r.text();
    console.error(`Scorer failed (${r.status}):`, errorText);
    return NextResponse.json(
      { error: "scorer failed", details: errorText },
      { status: 500 }
    );
  }

  return NextResponse.json(await r.json());
}
