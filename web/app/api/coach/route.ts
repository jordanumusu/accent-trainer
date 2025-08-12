import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

const PhoneScore = z.object({
  target: z.string(),
  hyp: z.string(),
  score: z.number(),
});

const EvalRes = z.object({
  avg: z.number(),
  targetIPA: z.string(),
  hypothesisIPA: z.string(),
  perPhone: z.array(PhoneScore),
  tip: z.string(),
  prosody: z
    .object({
      user: z.any().optional(),
      ref: z.any().optional(),
      compare: z.any().optional(),
    })
    .optional(),
  accentDistance: z.number().nullable().optional(),
});

const Payload = z.object({
  eval: EvalRes,
  targetText: z.string(),
  lang: z.string().default("en-us"),
});

const Coach = z.object({
  overallVerdict: z.enum(["pass", "retry"]),
  detectedAccent: z.string(),
  oneSentenceTip: z.string(),
  nativeAnalogy: z.string(),
  intonationCue: z.string(),
  focusPhones: z.array(z.string()),
  nextAttemptCriteria: z.object({
    minAvg: z.number(),
    noPhoneBelow: z.number(),
  }),
  coachMessage: z.string(),
});

export async function POST(req: NextRequest) {
  let parsed: z.infer<typeof Payload>;
  try {
    parsed = Payload.parse(await req.json());
  } catch (e) {
    return NextResponse.json({ error: "Bad payload", details: String(e) }, { status: 400 });
  }

  const realPhones = parsed.eval.perPhone.filter((p) => p.target !== "-");
  const minPhone = realPhones.length ? Math.min(...realPhones.map((p) => p.score)) : parsed.eval.avg;
  if (parsed.eval.avg >= 0.99 || minPhone >= 0.99) {
    return NextResponse.json({
      overallVerdict: "pass",
      detectedAccent: "Native-like for this word",
      oneSentenceTip: "Excellent! At this level, focus on relaxed, natural flow.",
      nativeAnalogy:
        "Say it like you would in a casual sentence rather than an isolated word.",
      intonationCue: "Keep the same stress and let the end relax naturally.",
      focusPhones: [],
      nextAttemptCriteria: { minAvg: 0.99, noPhoneBelow: 1.0 },
      coachMessage: `Sounds native‑like on “${parsed.targetText}” — great job! Keep the easy rhythm; no segment‑level tweaks needed.`,
    });
  }

  const avg = parsed.eval.avg;
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

  const system =
    `You are an accent and pronunciation coach.\n` +
    `You ALWAYS return a JSON object matching the provided schema.\n` +
    `You coach BEGINNER to INTERMEDIATE learners, being kind, specific, and actionable.\n` +
    `NEVER guess precise nationality—keep accents vague but mention a vague accent if giving medium/strong feedback ("Southern American English", "British English", "Spanish‑influenced English", etc.).\n` +
    `If avg >= 0.95, give praise and skip all corrections except tiny polish tips.\n` +
    `If 0.8 <= avg < 0.95, give light corrections (max 2 focusPhones).\n` +
    `If avg < 0.8, give stronger targeted corrections.\n` +
    `coachMessage style: "Looks like your accent is probably <CATEGORY> — nice try! Try pronouncing '<targetText>' with more emphasis on <part>, almost like '<analogy>' (<simple anatomy tip>)."`;  

  const worstPhones = parsed.eval.perPhone
    .filter((p) => p.target !== "-" && p.hyp !== "-" && p.score < 1)
    .map((p) => `${p.target}->${p.hyp}`)
    .slice(0, 8);

  const userText =
        `
        DATA:
        targetText: ${parsed.targetText}
        lang: ${parsed.lang}
        avg: ${avg.toFixed(3)}
        targetIPA: ${parsed.eval.targetIPA}
        hypothesisIPA: ${parsed.eval.hypothesisIPA}
        worstPhoneMappings: ${worstPhones.join(", ") || "none"}
        tipFromDSP: ${parsed.eval.tip}
        accentDistance: ${parsed.eval.accentDistance ?? "null"}

        INSTRUCTIONS:
        1) Respect score thresholds in the system message.
        2) detectedAccent must be vague (e.g., "British English", "Spanish‑influenced English").
        3) focusPhones = up to 4 (or fewer if high score).
        4) Return ONLY valid JSON matching the schema.
        5) Only return any feedback on pronunciation or accents if fairly confident in the input`;

    const r = await client.responses.parse({
      model: "gpt-4o-mini",
      input: [
        { role: "system", content: system },
        { role: "user", content: userText },
      ],
      temperature: 0.2,
      text: {
        format: zodTextFormat(Coach, "coach")
      }
    });

    const raw = (r as any).output_parsed ?? "{}";
    const parsedCoach = Coach.parse(raw);
    return NextResponse.json(parsedCoach);
 
}
