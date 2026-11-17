// app/api/bloom/analyze/route.ts
import { NextRequest, NextResponse } from "next/server";

const PYTHON_BACKEND = process.env.PYTHON_BACKEND_URL || "http://127.0.0.1:8080";
const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
import { normalizeBloomData } from "../normalize";

const GROQ_MODELS = [
  "openai/gpt-oss-120b",
  "openai/gpt-oss-20b",
  "qwen/qwen3.8-27b",
];

async function callDirectLLM(text: string, subject?: string, level?: string) {
  const subjectHint = subject ? `Subject / Domain: ${subject}.` : "";
  const levelHint = level ? `Target Student Level: ${level}.` : "";

  const prompt = `You are an expert pedagogical assessment specialist, senior curriculum auditor, and authoritative Bloom's Taxonomy taxonomist specializing in the Revised Bloom's Taxonomy (Anderson & Krathwohl, 2001).

Your task is to analyze the provided question paper or questions with MAXIMUM PEDAGOGICAL PRECISION.
${subjectHint} ${levelHint}

COGNITIVE PROCESS LEVELS:
- L1 - Remember (Level 1): Retrieving relevant knowledge from long-term memory without alteration.
- L2 - Understand (Level 2): Constructing meaning; explaining concepts in own words, summarizing, comparing basic ideas.
- L3 - Apply (Level 3): Executing or implementing a procedure, algorithm, or mathematical formula in a given problem scenario.
- L4 - Analyze (Level 4): Breaking material into constituent parts, examining relationships, comparing architectural trade-offs.
- L5 - Evaluate (Level 5): Making judgments based on explicit criteria, defending decisions, critiquing approaches.
- L6 - Create (Level 6): Synthesizing elements into a novel whole, designing original algorithms, architectures, or models.

PRECISION & HUMANIZATION RULES:
1. Parse every distinct question, sub-question, or numbered problem.
2. CONTEXT over keywords: Look at the actual cognitive demand, not merely the verb.
3. HUMANIZE THE CLASSIFICATION: Real teachers view cognitive demand as a spectrum. Provide:
   - "educator_spectrum": e.g. "L2 (Understand) â†” L4 (Analyze)"
   - "spectrum_lower": adjacent lower level number or null
   - "spectrum_higher": adjacent higher level number or null
   - "lower_perspective_rationale": why an educator might classify this lower (e.g. routine textbook problem)
   - "higher_perspective_rationale": why an educator might classify this higher (e.g. unguided multi-step synthesis)
   - "contextual_factors": 2-3 factors that shift human perception of this question
   - "is_ambiguous": true if on a cognitive borderline
4. Provide rationale, level_up_suggestion, and full paper summary metrics.

QUESTIONS TO ANALYZE:
"""
${text.slice(0, 8000)}
"""

Respond ONLY with valid JSON (no markdown fences):
{
  "questions": [
    {
      "id": "q1",
      "question_number": "Q1",
      "text": "Full question text",
      "marks": 5,
      "bloom_level": 1,
      "bloom_name": "Remember",
      "action_verb": "Define",
      "knowledge_dimension": "Factual",
      "difficulty": "Low",
      "confidence": 95,
      "is_ambiguous": false,
      "alternative_level": 2,
      "ambiguity_note": null,
      "educator_spectrum": "L1 (Remember) â†” L2 (Understand)",
      "spectrum_lower": 1,
      "spectrum_higher": 2,
      "lower_perspective_rationale": "Requires pure recall from long-term memory without manipulation.",
      "higher_perspective_rationale": "If personal synthesis or illustration is required, could touch Level 2.",
      "contextual_factors": ["Instructional Scaffolding", "Problem Familiarity"],
      "rationale": "Requires pure recall from long-term memory.",
      "level_up_suggestion": "Elevated question formulation"
    }
  ],
  "summary": {
    "total_questions": 1,
    "total_marks": 5,
    "lots_percentage": 100,
    "hots_percentage": 0,
    "lots_count": 1,
    "hots_count": 0,
    "distribution": {
      "1": { "count": 1, "percentage": 100, "name": "Remember" },
      "2": { "count": 0, "percentage": 0, "name": "Understand" },
      "3": { "count": 0, "percentage": 0, "name": "Apply" },
      "4": { "count": 0, "percentage": 0, "name": "Analyze" },
      "5": { "count": 0, "percentage": 0, "name": "Evaluate" },
      "6": { "count": 0, "percentage": 0, "name": "Create" }
    },
    "dominant_level": "Remember (L1)",
    "balance_rating": "Recall-Heavy",
    "pedagogical_critique": "Assessment relies heavily on rote recall.",
    "recommendations": ["Add application and analytical questions."],
    "subject_detected": "General",
    "ambiguous_count": 0,
    "humanized_perception_note": "Cognitive rigor varies based on whether problems were practiced in homework."
  }
}`;

  if (GROQ_API_KEY) {
    for (const model of GROQ_MODELS) {
      try {
        const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${GROQ_API_KEY}`,
          },
          body: JSON.stringify({
            model,
            messages: [
              {
                role: "system",
                content: "You are an expert Bloom Taxonomy evaluator. Output ONLY valid JSON, no markdown fences.",
              },
              { role: "user", content: prompt },
            ],
            response_format: { type: "json_object" },
            temperature: 0.1,
            max_tokens: 4096,
          }),
          signal: AbortSignal.timeout(60_000),
        });

        if (response.ok) {
          const data = await response.json();
          let rawText = data.choices?.[0]?.message?.content || "";
          rawText = rawText.replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, "").trim();
          if (rawText.startsWith("```")) {
            rawText = rawText.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "");
          }
          const parsed = JSON.parse(rawText);
          return normalizeBloomData(parsed, text);
        }
      } catch (e) {
        console.warn(`[API/BLOOM] Groq model ${model} failed:`, e);
      }
    }
  }

  // Gemini Fallback
  if (GEMINI_API_KEY) {
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`;
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: { responseMimeType: "application/json", temperature: 0.1 },
        }),
        signal: AbortSignal.timeout(60_000),
      });

      if (resp.ok) {
        const data = await resp.json();
        let rawText = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
        if (rawText.startsWith("```")) {
          rawText = rawText.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "");
        }
        const parsed = JSON.parse(rawText);
        return normalizeBloomData(parsed, text);
      }
    } catch (e) {
      console.warn("[API/BLOOM] Gemini fallback failed:", e);
    }
  }

  throw new Error("All LLM providers failed for Bloom's Taxonomy analysis.");
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    if (!body || !body.text || !body.text.trim()) {
      return NextResponse.json(
        { status: "error", message: "Please provide question paper text for analysis." },
        { status: 400 }
      );
    }

    const { text, subject, level } = body;

    // 1. Try Python backend first
    try {
      const pyResp = await fetch(`${PYTHON_BACKEND}/api/bloom/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, subject, level }),
        signal: AbortSignal.timeout(45_000),
      });

      if (pyResp.ok) {
        const pyData = await pyResp.json();
        if (pyData.status === "success" && pyData.questions) {
          const normalized = normalizeBloomData(pyData, text);
          return NextResponse.json({ status: "success", ...normalized });
        }
      }
    } catch (pyErr) {
      console.warn("[API/BLOOM] Python backend unreachable or failed, falling back to Next.js LLM:", pyErr);
    }

    // 2. Direct LLM fallback (Groq / Gemini)
    const result = await callDirectLLM(text, subject, level);
    return NextResponse.json({ status: "success", ...result });
  } catch (error: any) {
    console.error("[API/BLOOM/ANALYZE] Error:", error);
    return NextResponse.json(
      { status: "error", message: error.message || "Bloom's analysis failed." },
      { status: 500 }
    );
  }
}
