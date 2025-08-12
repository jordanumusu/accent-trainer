"use client";

import React, { useState, useRef } from "react";

const LANG_OPTIONS = [
  { value: "en-us", label: "English (US)", flag: "🇺🇸" },
  { value: "en-gb", label: "English (UK)", flag: "🇬🇧" },
  { value: "fr-fr", label: "Français", flag: "🇫🇷" },
  { value: "es", label: "Español", flag: "🇪🇸" },
  { value: "de", label: "Deutsch", flag: "🇩🇪" },
  { value: "it", label: "Italiano", flag: "🇮🇹" },
];

const PHONE_HELP: Record<string, { desc: string; example: string }> = {
  θ: { desc: "Tongue lightly between teeth, blow air.", example: "think" },
  ð: { desc: "Tongue between teeth, keep voicing.", example: "this" },
  ʁ: { desc: "Uvular 'r'—tongue back near uvula.", example: "rue" },
  y: { desc: "Rounded 'ee' sound—say 'ee' then round lips.", example: "lune" },
  æ: { desc: "Open mouth wide, tongue front and low.", example: "cat" },
};

export default function AccentTrainerPage() {
  const [lang, setLang] = useState("en-us");
  const [targetText, setTargetText] = useState("hello");
  const [recording, setRecording] = useState(false);
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
    const help = PHONE_HELP[phone];
    if (help) {
      const utter = new SpeechSynthesisUtterance(help.example);
      utter.lang = lang;
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
    const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" });
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    formData.append("text", targetText);
    formData.append("lang", lang);

    const evalRes = await fetch("/api/eval", {
      method: "POST",
      body: formData,
    });
    const evalData = await evalRes.json();

    setScore(evalData.avg);
    const coachRes = await fetch("/api/coach", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ eval: evalData, targetText, lang }),
    });
    const coachData = await coachRes.json();

    setFocusPhones(coachData.focusPhones || []);
    setCoachMessage(coachData.coachMessage || "");
  };

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-2xl font-bold">Accent Trainer</h1>

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
                    {PHONE_HELP[p]?.desc || "No description available"}
                    <button
                      className="ml-2 px-2 py-1 bg-gray-200 rounded"
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
