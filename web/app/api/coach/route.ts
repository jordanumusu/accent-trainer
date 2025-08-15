// app/api/coach/route.ts
import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { z } from "zod";
import { zodTextFormat } from "openai/helpers/zod";


const Payload = z.object({
  target: z.string(),
  hypothesis: z.string(),
  target_ipa: z.string(),
  hypothesis_ipa: z.string(),
  score: z.number(),
  meta: z.record(z.any()).optional(),
  targetText: z.string(),
  lang: z.string().optional(), // e.g., "en-us", "fr-fr"
});

const PhoneScore = z.object({
  p: z.string(), // IPA symbol or digraph
  score: z.number().min(0).max(1),
  feedback: z.string(),
});

const Coach = z.object({
  message: z.string(),
  score: z.number().min(0).max(1),
  targetIPA: z.string(),
  hypothesisIPA: z.string(),
  focusPhones: z.array(z.string()).default([]),
  phones: z.array(PhoneScore).default([]),
  tip: z.string(),
});

const client = new OpenAI();

export async function POST(req: NextRequest) {
  try {
    const json = await req.json();
    const { target, hypothesis, target_ipa, hypothesis_ipa, score,targetText, lang } = Payload.parse(json);

    const system = [
      "You are an encouraging pronunciation coach for second-language learners.",
      "You ONLY use the IPA provided; do not invent new IPA.",
      "Be constructive and supportive without giving empty praise.",
      "Acknowledge genuine progress and offer helpful guidance for improvement.",
      "Adapt your coaching style to the target language being learned.",
      "If hypothesisIPA is empty, explain that recognition failed and give a practical tip."
    ].join(" ");

    const userText = [
      `LANG: ${lang ?? "en-us"}`,
      `TARGET: ${targetText}`,
      `ASR_HYP: ${hypothesis || "(empty)"}`,
      `TARGET_IPA: ${target_ipa}`,
      `HYPOTHESIS_IPA: ${hypothesis_ipa || "(empty)"}`,
      `RAW_SCORE: ${score.toFixed(4)}`,
      "",
      "TASK:",
      "- Acknowledge what they got right, if anything.",
      "- Compare HYPOTHESIS_IPA to TARGET_IPA and identify key differences.",
      "- Focus on 1-2 most important areas for improvement.",
      "- Give specific, actionable tips appropriate for the target language.",
      "- Consider language-specific pronunciation challenges (e.g., rolled Rs in Spanish, nasal vowels in French).",
      "- Keep 'message' encouraging but realistic (2-3 sentences).",
      "- Reuse RAW_SCORE unless you need to slightly adjust based on severity.",
      "- Output MUST match the Coach schema."
    ].join("\n");

    const r = await client.responses.parse({
      model: "gpt-4o-mini",
      input: [
        { role: "system", content: system },
        { role: "user", content: userText },
      ],
      temperature: 0.2,
      text: { format: zodTextFormat(Coach, "coach") },
    });

    const raw = (r as any).output_parsed ?? "{}";
    const parsedCoach = Coach.parse(raw);

    const responseBody = {
      ...parsedCoach,
      targetIPA: parsedCoach.targetIPA || target_ipa,
      hypothesisIPA: parsedCoach.hypothesisIPA || (hypothesis_ipa ?? ""),
      score: parsedCoach.score ?? score,
    };

    return NextResponse.json(responseBody);
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message ?? "Invalid request" },
      { status: 400 }
    );
  }
}
