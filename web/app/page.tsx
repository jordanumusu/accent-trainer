"use client";

import React, { useState, useRef } from "react";

const LANG_OPTIONS = [
  { value: "en", label: "English (US)", flag: "🇺🇸" },
  { value: "fr", label: "Français", flag: "🇫🇷" },
  { value: "es", label: "Español", flag: "🇪🇸" },
  { value: "de", label: "Deutsch", flag: "🇩🇪" },
  { value: "it", label: "Italiano", flag: "🇮🇹" },
];

// Hacky and needs to be replaced
const PHONE_HELP: Record<string, { desc: string; examples: Record<string, string> }> = {
  θ: { desc: "Tongue lightly between teeth, blow air.", examples: { en: "think", fr: "think", es: "think" } },
  ð: { desc: "Tongue between teeth, keep voicing.", examples: { en: "this", fr: "this", es: "this" } },
  ʁ: { desc: "Uvular 'r'—tongue back near uvula.", examples: { en: "red", fr: "rue", es: "rojo" } },
  y: { desc: "Rounded 'ee' sound—say 'ee' then round lips.", examples: { en: "you", fr: "lune", es: "tú" } },
  æ: { desc: "Open mouth wide, tongue front and low.", examples: { en: "cat", fr: "cat", es: "gato" } },
  ɛ: { desc: "Open 'e' sound.", examples: { en: "bed", fr: "père", es: "perro" } },
  ɪ: { desc: "Short 'i' sound.", examples: { en: "bit", fr: "si", es: "ir" } },
  ɑ: { desc: "Open 'a' sound.", examples: { en: "father", fr: "pâte", es: "casa" } },
  ɔ: { desc: "Open 'o' sound.", examples: { en: "thought", fr: "port", es: "ojo" } },
  ʊ: { desc: "Short 'u' sound.", examples: { en: "book", fr: "vous", es: "luz" } },
  ə: { desc: "Neutral vowel sound.", examples: { en: "about", fr: "le", es: "casa" } },
  ʌ: { desc: "Open-mid back vowel.", examples: { en: "but", fr: "peu", es: "amor" } },
  ɒ: { desc: "Rounded open back vowel.", examples: { en: "lot", fr: "comme", es: "oro" } },
  ɜ: { desc: "Open-mid central vowel.", examples: { en: "bird", fr: "peur", es: "ser" } },
  r: { desc: "Alveolar trill.", examples: { en: "red", fr: "rouge", es: "rojo" } },
  ɹ: { desc: "Approximant 'r'.", examples: { en: "run", fr: "red", es: "red" } },
  l: { desc: "Lateral approximant.", examples: { en: "love", fr: "lune", es: "luz" } },
  w: { desc: "Labio-velar approximant.", examples: { en: "water", fr: "oui", es: "huevo" } },
  j: { desc: "Palatal approximant.", examples: { en: "yes", fr: "yeux", es: "yo" } },
  h: { desc: "Voiceless glottal fricative.", examples: { en: "hello", fr: "hello", es: "jota" } },
  f: { desc: "Voiceless labiodental fricative.", examples: { en: "fish", fr: "feu", es: "feo" } },
  v: { desc: "Voiced labiodental fricative.", examples: { en: "voice", fr: "vous", es: "vino" } },
  s: { desc: "Voiceless alveolar fricative.", examples: { en: "sun", fr: "soir", es: "sol" } },
  z: { desc: "Voiced alveolar fricative.", examples: { en: "zoo", fr: "rose", es: "mismo" } },
  ʃ: { desc: "Voiceless postalveolar fricative.", examples: { en: "ship", fr: "chien", es: "show" } },
  ʒ: { desc: "Voiced postalveolar fricative.", examples: { en: "measure", fr: "je", es: "beige" } },
  tʃ: { desc: "Voiceless postalveolar affricate.", examples: { en: "church", fr: "match", es: "mucho" } },
  dʒ: { desc: "Voiced postalveolar affricate.", examples: { en: "judge", fr: "juge", es: "general" } },
  p: { desc: "Voiceless bilabial plosive.", examples: { en: "pen", fr: "papa", es: "papa" } },
  b: { desc: "Voiced bilabial plosive.", examples: { en: "ball", fr: "beau", es: "bien" } },
  t: { desc: "Voiceless alveolar plosive.", examples: { en: "top", fr: "tout", es: "todo" } },
  d: { desc: "Voiced alveolar plosive.", examples: { en: "dog", fr: "deux", es: "dos" } },
  k: { desc: "Voiceless velar plosive.", examples: { en: "cat", fr: "car", es: "casa" } },
  g: { desc: "Voiced velar plosive.", examples: { en: "go", fr: "gare", es: "gato" } },
  m: { desc: "Bilabial nasal.", examples: { en: "man", fr: "mère", es: "mama" } },
  n: { desc: "Alveolar nasal.", examples: { en: "no", fr: "nous", es: "no" } },
  ŋ: { desc: "Velar nasal.", examples: { en: "sing", fr: "parking", es: "mango" } },
};

export default function Page() {
  const [lang, setLang] = useState("en");
  const [targetText, setTargetText] = useState("hello");
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [score, setScore] = useState<number | null>(null);
  const [focusPhones, setFocusPhones] = useState<string[]>([]);
  const [coachMessage, setCoachMessage] = useState<string>("");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const playReference = () => {
    const utter = new SpeechSynthesisUtterance(targetText);
    utter.lang = lang;
    speechSynthesis.speak(utter);
  };

  const playPhone = (phone: string) => {
    console.log("Playing phone:", phone, "Available in PHONE_HELP:", !!PHONE_HELP[phone]);
    const help = PHONE_HELP[phone];
    if (help) {
      const example = help.examples[lang] || help.examples["en"];
      const utter = new SpeechSynthesisUtterance(example);
      const speechLang = lang === "en" ? "en-US" : lang === "fr" ? "fr-FR" : lang === "es" ? "es-ES" : "en-US";
      utter.lang = speechLang;
      speechSynthesis.speak(utter);
    } else {
      console.log("No example found for phone:", phone);
      const utter = new SpeechSynthesisUtterance(phone);
      const speechLang = lang === "en" ? "en-US" : lang === "fr" ? "fr-FR" : lang === "es" ? "es-ES" : "en-US";
      utter.lang = speechLang;
      speechSynthesis.speak(utter);
    }
  };

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    chunksRef.current = [];
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    mediaRecorder.onstop = handleStop;
    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  const handleStop = async () => {
    setLoading(true);
    const audioBlob = new Blob(chunksRef.current, { type: "audio/wav" });
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    formData.append("text", targetText);
    formData.append("lang", lang);

    const evalRes = await fetch("/api/eval", {
      method: "POST",
      body: formData,
    });
    const evalData = await evalRes.json();

    const coachRes = await fetch("/api/coach", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...evalData, targetText, lang }),
    });
    const coachData = await coachRes.json();

    setScore(coachData.score);
    setFocusPhones(coachData.focusPhones || []);
    setCoachMessage(coachData.message || "");
    setLoading(false);
  };

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-2xl font-bold">Speech Trainer</h1>

      <div>
        <label className="block mb-1">Language:</label>
        <select
          value={lang}
          onChange={(e) => setLang(e.target.value)}
          className="border p-1 rounded"
        >
          {LANG_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.flag} {opt.label}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="block mb-1">Target Text:</label>
        <input
          className="border p-1 rounded"
          value={targetText}
          onChange={(e) => setTargetText(e.target.value)}
        />
        <button
          className="ml-2 px-2 py-1 bg-blue-500 text-white rounded"
          onClick={playReference}
        >
          ▶ Play Reference
        </button>
      </div>

      <div>
        {recording ? (
          <button
            onClick={stopRecording}
            className="px-4 py-2 bg-red-500 text-white rounded"
          >
            ⏹ Stop
          </button>
        ) : loading ? (
          <button
            disabled
            className="px-4 py-2 bg-gray-400 text-white rounded cursor-not-allowed"
          >
            Processing...
          </button>
        ) : (
          <button
            onClick={startRecording}
            className="px-4 py-2 bg-green-500 text-white rounded"
          >
            🎙 Record
          </button>
        )}
      </div>

      {score !== null && (
        <div className="mt-4">
          <h2 className="text-lg font-semibold">
            Score: {(score * 100).toFixed(1)}%
          </h2>
          {coachMessage && <p className="mt-2">{coachMessage}</p>}

          {focusPhones.length > 0 && (
            <div className="mt-3">
              <h3 className="font-semibold">Focus Sounds:</h3>
              <ul className="list-disc pl-5">
                {focusPhones.map((p) => (
                  <li key={p} className="mb-1">
                    <span className="font-mono">{p}</span> —{" "}
                    <button
                      className="ml-2 px-2 py-1 bg-gray-200 dark:bg-gray-600 text-black dark:text-white rounded"
                      onClick={() => playPhone(p)}
                    >
                      🔊 Example
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
