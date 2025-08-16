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
  meta: z.record(z.any()).optional(),
  targetText: z.string(),
  lang: z.string().optional(), // e.g., "en-us", "fr-fr"
});

const Coach = z.object({
  message: z.string(),
  score: z.number().min(0).max(1),
  targetIPA: z.string(),
  hypothesisIPA: z.string(),
  focusPhones: z.array(z.string()).default([]),
  tip: z.string(),
});

const client = new OpenAI();

export async function POST(req: NextRequest) {
  try {
    const json = await req.json();
    const { target, hypothesis, target_ipa, hypothesis_ipa, targetText, lang } = Payload.parse(json);

    const system = [
      "You are an encouraging pronunciation coach for language learners.",
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
      "",
      "TASK:",
      "- Acknowledge what they got right, if anything.",
      "- Compare HYPOTHESIS_IPA to TARGET_IPA and identify key differences.",
      "- Focus on 1-2 most important areas for improvement in 'focusPhones' array.",
      "- Give specific, actionable tips appropriate for the target language in 'tip'.",
      "- Consider language-specific pronunciation challenges (e.g., rolled Rs in Spanish, nasal vowels in French).",
      "- For 'score': Give ONE holistic score 0.0-1.0. BE GENEROUS with similar words.",
      "- EXAMPLES: 'hallo' vs 'hello' = 0.8+, 'jello' vs 'hello' = 0.8+, 'goodnight' vs 'hello' = 0.0-0.1.",
      "- Minor vowel changes (æ→ə) or consonant substitutions (j→h) should score 0.8+.",
      "- Only score below 0.7 if the word sounds completely different or unrecognizable.",
      "- If TARGET_IPA and HYPOTHESIS_IPA are identical, score MUST be 1.0 (perfect match).",
      "- Scoring guide: 1.0 (identical IPA), 0.9 (near perfect), 0.8-0.9 (minor differences), 0.5-0.7 (major errors but recognizable), 0.1-0.4 (barely related), 0.0-0.1 (wrong word entirely).",
      "- Keep 'message' encouraging but realistic (2-3 sentences).",
      "- If score is above 0.90, give only congratulatory feedback - no constructive criticism needed.",
      "- Use everyday language: mention sounds with familiar word examples (e.g. 'the vowel sound in CAT' not just 'æ').",
      "- NEVER mention scores, percentages, or ratings in your message - only provide coaching guidance.",
      "- NEVER mention what the system 'detected' - focus only on pronunciation guidance.",
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
    };

    return NextResponse.json(responseBody);
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message ?? "Invalid request" },
      { status: 400 }
    );
  }
}
