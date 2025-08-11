"use client";
import { useEffect, useRef, useState } from "react";

type PhoneScore = { target: string; hyp: string; score: number };
type EvalRes = {
  avg: number;
  targetIPA: string;
  hypothesisIPA: string;
  perPhone: PhoneScore[];
  tip: string;
};

type Coach = {
  overallVerdict: "pass" | "retry";
  oneSentenceTip: string;
  bodyCue: string;
  microDrill: string;
  focusPhones: string[];
  nextAttemptCriteria: { minAvg: number; noPhoneBelow: number };
}

const CARDS = [
  { text: "bonjour", lang: "fr" },
  { text: "rue", lang: "fr" },
  { text: "thought", lang: "en" },
];

export default function App() {
  const [i, setI] = useState(0);
  const [rec, setRec] = useState<MediaRecorder | null>(null);
  const chunks = useRef<BlobPart[]>([]);
  const [evalRes, setEvalRes] = useState<EvalRes | null>(null);
  const [coach, setCoach] = useState<Coach | null>(null);
  const busy = useRef(false);

  useEffect(() => {
    (async () => {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      const r = new MediaRecorder(s, { mimeType: "audio/webm" });
      r.ondataavailable = (e) => chunks.current.push(e.data);
      r.onstop = async () => {
        if (busy.current) return;
        busy.current = true;
        const blob = new Blob(chunks.current, { type: "audio/webm" });
        chunks.current = [];
        const fd = new FormData();
        fd.append("audio", blob, "u.webm");
        fd.append("text", CARDS[i].text);
        fd.append("lang", CARDS[i].lang);
        const r1 = await fetch("/api/eval", { method: "POST", body: fd });
        const e1: EvalRes = await r1.json();
        setEvalRes(e1);

        const r2 = await fetch("/api/coach", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            word: CARDS[i].text,
            lang: CARDS[i].lang,
            targetIPA: e1.targetIPA,
            hypothesisIPA: e1.hypothesisIPA,
            perPhone: e1.perPhone
          })
        });
        const c1: Coach = await r2.json();
        setCoach(c1);
        busy.current = false;
      };
      setRec(r);
    })();
  }, [i]);

  const speak = () => {
    const u = new SpeechSynthesisUtterance(CARDS[i].text);
    u.lang = CARDS[i].lang === "fr" ? "fr-FR" : "en-US";
    speechSynthesis.speak(u);
  };

  const reset = () => { setEvalRes(null); setCoach(null); };

  return (
    <main style={{maxWidth: 720, margin: "0 auto", padding: 24, fontFamily: "system-ui"}}>
      <h1 style={{fontSize: 24, fontWeight: 600}}>Pronounce Cards</h1>
      <div style={{border: "1px solid #e5e7eb", borderRadius: 12, padding: 16, marginTop: 12}}>
        <div style={{fontSize: 36}}>{CARDS[i].text}</div>
        <div style={{opacity: 0.6}}>{CARDS[i].lang.toUpperCase()}</div>
        <div style={{display:"flex", gap: 8, marginTop: 12}}>
          <button onClick={speak}>🔊 Play</button>
          <button onClick={()=>{reset(); rec?.start();}}>⏺️ Record</button>
          <button onClick={()=>rec?.stop()}>⏹️ Stop</button>
          <button onClick={()=>{reset(); setI((k)=>(k+1)%CARDS.length);}}>Next ▶</button>
        </div>

        {evalRes && (
          <div style={{marginTop: 16}}>
            <div>Avg score: <b>{Math.round(evalRes.avg*100)}</b>/100</div>
            <div>Target: <code>{evalRes.targetIPA}</code></div>
            <div>You said: <code>{evalRes.hypothesisIPA}</code></div>
            <div>Tip: <span style={{color:"#b91c1c"}}>{evalRes.tip}</span></div>
            <details style={{marginTop:8}}>
              <summary>Per‑phone</summary>
              <ul>
                {evalRes.perPhone.map((p, idx)=>(
                  <li key={idx}><code>{p.target}</code> ← <code>{p.hyp}</code> : {Math.round(p.score*100)}</li>
                ))}
              </ul>
            </details>
          </div>
        )}

        {coach && (
          <div style={{marginTop: 16, padding:12, border:"1px solid #e5e7eb", borderRadius:8}}>
            <div><b>{coach.overallVerdict === "pass" ? "✅ Pass" : "🔁 Try again"}</b></div>
            <div><b>Coach:</b> {coach.oneSentenceTip}</div>
            <div><b>Body cue:</b> {coach.bodyCue}</div>
            <div><b>Drill:</b> {coach.microDrill}</div>
            {coach.focusPhones.length>0 && <div><b>Focus:</b> {coach.focusPhones.join(", ")}</div>}
          </div>
        )}
      </div>
    </main>
  );
}
