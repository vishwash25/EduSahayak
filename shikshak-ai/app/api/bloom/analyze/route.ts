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

  const prompt = `You are an expert academic assessment specialist applying the Revised Bloom's Taxonomy (Anderson and Krathwohl, 2001).

QUESTION-FIRST CLASSIFICATION PRINCIPLE:
Treat each exact, complete question as the unit of analysis. Read every sentence,
scenario, supplied value, condition, constraint and requested deliverable.

Infer the complete answer or product a student must produce. Consider all essential
sub-tasks and identify the dominant cognitive process genuinely required. Distinguish
routine execution of a familiar method from genuine analysis, criteria-based evaluation
and original synthesis.

Never map an isolated command word or keyword directly to a Bloom level. Wording is
contextual evidence only. Do not invent requirements or inflate the level beyond what
the complete question actually demands.
${subjectHint}
${levelHint}

THE SIX COGNITIVE LEVELS:
- L1 Remember: Retrieve or recognize facts, terms, definitions, labels, or previously learned information.
- L2 Understand: Explain, interpret, summarize, classify, illustrate, or compare concepts to demonstrate meaning.
- L3 Apply: Execute a learned procedure, formula, algorithm, construction method, rule, or technique in a concrete task.
- L4 Analyze: Break information or a system into meaningful parts and identify relationships, causes, structure, patterns, or consequences.
- L5 Evaluate: Make and defend a judgment against criteria, standards, evidence, or trade-offs.
- L6 Create: Synthesize elements into a substantially original product, plan, model, method, design, or solution requiring meaningful decisions.

REQUIRED REASONING FOR EVERY QUESTION:
1. Identify what the student must actually do to produce a correct answer.
2. Infer the expected answer or product: fact, explanation, routine procedure, structural analysis, criteria-based judgment, or original integrated solution.
3. Decide whether the work involves retrieval, explanation, execution, analysis of relationships, justified judgment, or original synthesis.
4. Use subject and academic level as context when supplied. Marks are only a weak supporting clue.
5. Select the dominant cognitive process genuinely required. Do not inflate the level because wording sounds sophisticated.

IMPORTANT EXAMPLES:
- "Design a triangle with a circle inside it" is normally L3 Apply when it requires a familiar geometric construction using a taught method. The word "design" does not make it L6.
- "Design an original bridge for a site with competing load, cost, material, and safety constraints, then develop an integrated proposal" may be L6 Create because it requires synthesizing a new solution.
- "Evaluate 5 + 7" is L3 Apply when the student simply calculates the expression. The word "evaluate" does not make it L5.
- "Evaluate two database designs against stated cost, security, and scalability criteria and defend a recommendation" is L5 Evaluate.
- "Compare two familiar terms and state their basic differences" may be L2 Understand.
- "Analyze how components interact and cause a system failure" is L4 Analyze.
- "Find the output of this program" may be L3 Apply when the student traces a known execution procedure, even without an obvious action verb.
- High marks do not automatically imply high cognitive demand. A multi-step task is not automatically Analyze or Create.
- For diagrams and constructions, distinguish reproducing a taught method from making original decisions and integrating elements.
- Do not invent hidden requirements. When context is incomplete, make the most reasonable academic inference and reduce confidence rather than refusing to classify.

CONFIDENCE:
Always assign exactly one best-fit level from L1 to L6. Never omit a classification or label a question ambiguous.
Return confidence as a percentage from 60 to 98:
- 90-98: Cognitive operation and expected work are clear.
- 80-89: Strong evidence with minor assumptions.
- 70-79: Reasonable inference, but some context is unknown.
- 60-69: Adjacent levels are plausible or the expected response is partly specified.
This is model-estimated confidence, not a statistically calibrated probability. Do not inflate it.

RESPONSE RULES:
- Parse every distinct question and sub-question.
- Return one result per question in the original order.
- Preserve question identifiers and exact question text.
- Do not return action-verb identification or a verb-label field.
- Use only valid Bloom levels 1 through 6.
- Give a specific rationale explaining the cognitive work required, not merely repeating the action verb.
- Set is_ambiguous to false for every question. Reflect uncertainty through the confidence score instead.
- Return only valid JSON, without Markdown fences.

QUESTIONS:
${text.slice(0, 8000)}

OUTPUT JSON STRUCTURE:
{
  "questions": [
    {
      "id": "q1",
      "question_number": "Q1",
      "text": "Full question text",
      "marks": 5,
      "bloom_level": 3,
      "bloom_name": "Apply",

      "knowledge_dimension": "Procedural",
      "difficulty": "Medium",
      "confidence": 72,
      "is_ambiguous": false,
      "alternative_level": null,
      "ambiguity_note": null,
      "educator_spectrum": "L3 (Apply)",
      "spectrum_lower": null,
      "spectrum_higher": null,
      "lower_perspective_rationale": "A simpler interpretation may be possible if the task only requires recalling a familiar fact.",
      "higher_perspective_rationale": "A more demanding interpretation would require additional analysis or original synthesis not explicitly requested.",
      "contextual_factors": ["Problem familiarity", "Instructional scaffolding"],
      "rationale": "The student executes a familiar procedure to complete the requested task.",
      "level_up_suggestion": "Introduce an unfamiliar context requiring students to select and justify an appropriate method."
    }
  ],
  "summary": {
    "total_questions": 1,
    "total_marks": 5,
    "lots_percentage": 0,
    "hots_percentage": 100,
    "lots_count": 0,
    "hots_count": 1,
    "distribution": {
      "1": {"count": 0, "percentage": 0, "name": "Remember"},
      "2": {"count": 0, "percentage": 0, "name": "Understand"},
      "3": {"count": 1, "percentage": 100, "name": "Apply"},
      "4": {"count": 0, "percentage": 0, "name": "Analyze"},
      "5": {"count": 0, "percentage": 0, "name": "Evaluate"},
      "6": {"count": 0, "percentage": 0, "name": "Create"}
    },
    "dominant_level": "Apply (L3)",
    "balance_rating": "Application-Centric",
    "pedagogical_critique": "Summarize the actual cognitive distribution of the paper.",
    "recommendations": ["Provide balanced coverage across the six cognitive levels."],
    "subject_detected": "General",
    "ambiguous_count": 0,
    "humanized_perception_note": "Confidence reflects the evidence available in each question."
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
