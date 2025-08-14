import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

// Updated schema to match new API output
const PhoneScore = z.object({
  p: z.string(),           // changed from target/hyp to single phone with score/feedback
  score: z.number(),
  feedback: z.string(),
});

const ProsodyResult = z.object({
  score: z.number(),
  stress_ok: z.boolean(),
  timing_ratio: z.number(),
});

const EvalRes = z.object({
  score: z.number(),                    // changed from avg to score
  targetIPA: z.string(),
  hypothesisIPA: z.string(),
  phones: z.array(PhoneScore),          // changed from perPhone to phones
  prosody: ProsodyResult,               // updated structure
  tip: z.string(),
  playback: z.object({                  // added playback urls
    user: z.string(),
    reference: z.string(),
  }).optional(),
});

const Payload = z.object({
  eval: EvalRes,
  targetText: z.string(),
  lang: z.string().default("en"),       // changed default from en-us to en
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

  // Updated to work with new phone structure
  const realPhones = parsed.eval.phones.filter((p) => p.p !== "-");
  const minPhone = realPhones.length ? Math.min(...realPhones.map((p) => p.score)) : parsed.eval.score;
  
  // Changed from avg to score
  if (parsed.eval.score >= 0.99 || minPhone >= 0.99) {
    return NextResponse.json({
      overallVerdict: "pass",
      detectedAccent: "Native-like for this word",
      oneSentenceTip: "Excellent! At this level, focus on relaxed, natural flow.",
      nativeAnalogy:
        "Say it like you would in a casual sentence rather than an isolated word.",
      intonationCue: "Keep the same stress and let the end relax naturally.",
      focusPhones: [],
      nextAttemptCriteria: { minAvg: 0.99, noPhoneBelow: 1.0 },
      coachMessage: `Sounds native‑like on "${parsed.targetText}" — great job! Keep the easy rhythm; no segment‑level tweaks needed.`,
    });
  }

  const score = parsed.eval.score;  // changed from avg
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

  const system =
    `You are an accent and pronunciation coach.\n` +
    `You ALWAYS return a JSON object matching the provided schema.\n` +
    `You coach BEGINNER to INTERMEDIATE learners, being kind, specific, and actionable.\n` +
    `NEVER guess precise nationality—keep accents vague but mention a vague accent if giving medium/strong feedback ("Southern American English", "British English", "Spanish‑influenced English", etc.).\n` +
    `If score >= 0.95, give praise and skip all corrections except tiny polish tips.\n` +
    `If 0.8 <= score < 0.95, give light corrections (max 2 focusPhones).\n` +
    `If score < 0.8, give stronger targeted corrections.\n`

  // Updated to work with new phone structure (no hyp field, just feedback)
  const problemPhones = parsed.eval.phones
    .filter((p) => p.p !== "-" && p.score < 0.8)
    .map((p) => `${p.p} (${p.feedback})`)
    .slice(0, 8);

  const userText =
        `
        DATA:
        targetText: ${parsed.targetText}
        lang: ${parsed.lang}
        score: ${score.toFixed(3)}
        targetIPA: ${parsed.eval.targetIPA}
        hypothesisIPA: ${parsed.eval.hypothesisIPA}
        problemPhones: ${problemPhones.join(", ") || "none"}
        tipFromAPI: ${parsed.eval.tip}
        prosodyScore: ${parsed.eval.prosody.score}
        stressOK: ${parsed.eval.prosody.stress_ok}
        timingRatio: ${parsed.eval.prosody.timing_ratio}

        INSTRUCTIONS:
        1) Respect score thresholds in the system message.
        3) focusPhones = up to 4 phones that need work (or fewer if high score).
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