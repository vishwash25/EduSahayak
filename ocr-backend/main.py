from PIL import Image
import io
import base64
import requests as http_requests
import time
import re
import json
import asyncio
import os
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from fpdf import FPDF
from dotenv import load_dotenv

from ocr_engine import OCRComparer
from document_parser import DocumentParser, MAX_FILE_SIZE, SUPPORTED_IMAGE_EXTS, SUPPORTED_DOC_EXTS
from bloom_analyzer import analyze_question_paper_text, BloomAnalysisResponse

load_dotenv()




# ===== PRO MODE PDF CONVERSION =====
MAX_PAGES = 50
_PDF_RENDER_DPI = 150

def _pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = None) -> list:
    """Convert PDF bytes to RGB PIL images using PyMuPDF."""
    import fitz
    from PIL import Image

    if not pdf_bytes:
        raise ValueError("Uploaded PDF is empty.")

    render_dpi = min(max(int(dpi or _PDF_RENDER_DPI), 72), 200)
    zoom = render_dpi / 72.0
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        if len(doc) == 0:
            raise ValueError("PDF contains no pages.")
        if len(doc) > MAX_PAGES:
            raise ValueError(f"PDF has {len(doc)} pages; maximum allowed is {MAX_PAGES}.")

        images = []
        matrix = fitz.Matrix(zoom, zoom)
        max_pixels = 25_000_000

        for index, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            if image.width * image.height > max_pixels:
                scale = (max_pixels / (image.width * image.height)) ** 0.5
                image = image.resize(
                    (int(image.width * scale), int(image.height * scale)),
                    Image.LANCZOS
                )

            images.append(image)

        print(f"[PRO-MODE] Converted {len(images)} PDF pages at {render_dpi} DPI.")
        return images
    finally:
        doc.close()

# ===== RESTORED SHARED EVALUATION HELPERS =====

def _call_groq_api_direct(
    api_key: str,
    model: str,
    messages: list,
    temperature: float = 0.1,
    max_tokens: int = 2500,
    response_format: dict = None,
    timeout: float = 120.0
):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if response_format:
        payload["response_format"] = response_format

    resp = http_requests.post(url, json=payload, headers=headers, timeout=timeout)
    if not resp.ok:
        err_msg = ""
        try:
            err_msg = resp.json().get("error", {}).get("message", "")
        except Exception:
            err_msg = resp.text[:200]
        retry_after = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-tokens") or ""
        raise RuntimeError(f"Groq {resp.status_code} [{model}]: {err_msg} | retry_after={retry_after}")

    resp_json = resp.json()
    content = resp_json["choices"][0]["message"]["content"]

    class MessageObj:
        def __init__(self, c):
            self.content = c

    class ChoiceObj:
        def __init__(self, c):
            self.message = MessageObj(c)

    class ResponseObj:
        def __init__(self, c, h):
            self.choices = [ChoiceObj(c)]
            self.headers = h

    return ResponseObj(content, resp.headers)


# ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡

_STRICTNESS_INSTRUCTIONS = {
    1: """GRADING STYLE: VERY LENIENT ÃŽâ€œÃƒâ€¡ÃƒÂ¶ School Teacher (Kind & Encouraging)
  - Give full marks if the student clearly understood the concept, even if wording is informal or imprecise.
  - Award marks for correct final answers even if working steps are missing.
  - Overlook minor spelling/grammar mistakes ÃŽâ€œÃƒâ€¡ÃƒÂ¶ do NOT deduct for them.
  - If the student's answer shows the right idea but uses different words than the model answer, give full marks.
  - Round UP partial scores when in doubt (e.g., 3.5 ÃŽâ€œÃƒÂ¥Ãƒâ€  4).
  - Write warm, encouraging feedback. Focus on what the student did right.""",
    2: """GRADING STYLE: LENIENT ÃŽâ€œÃƒâ€¡ÃƒÂ¶ Supportive Teacher
  - Award full marks for answers that are mostly correct with minor gaps.
  - Accept correct final answers with minimal working ÃŽâ€œÃƒâ€¡ÃƒÂ¶ do NOT strictly require every step.
  - Small factual errors that don't affect the core answer: deduct at most 0.5 marks.
  - Round UP when the student clearly understood the concept.
  - Feedback should highlight strengths before pointing out weaknesses.""",
    3: """GRADING STYLE: BALANCED ÃŽâ€œÃƒâ€¡ÃƒÂ¶ Standard School Examiner (Default)
  - Grade fairly: reward correct understanding but deduct for clearly wrong or missing key points.
  - Require some working for calculation questions, but do not strictly penalise every missing step.
  - Award partial marks generously when the core concept is correct.
  - Feedback should be balanced ÃŽâ€œÃƒâ€¡ÃƒÂ¶ mention both strengths and improvements needed.""",
    4: """GRADING STYLE: STRICT ÃŽâ€œÃƒâ€¡ÃƒÂ¶ Senior Examiner
  - Require clear, complete answers. Missing key terms or steps = deduction.
  - Correct final answer without working shown: award a maximum of 75% of marks for that question.
  - Vague answers or restated questions: partial marks only.
  - Do NOT round up. Award exactly what is deserved based on content.
  - Feedback should be professional and point out every gap.""",
    5: """GRADING STYLE: VERY STRICT ÃŽâ€œÃƒâ€¡ÃƒÂ¶ University / Board Examiner
  - Apply the strictest academic standards. Every required key term, formula, step, and unit must be present.
  - Correct final answer without full working shown = maximum 50% of marks.
  - Vague or incomplete explanations = 0 marks even if partially correct.
  - Any deviation from the expected answer structure is penalised.
  - No rounding up. No sympathy marks. Feedback is precise and technical."""
}

class EvaluateRequest(BaseModel):
    student_text: str = ""             # OCR output from student answer sheet
    pdfBase64: Optional[str] = None    # Optional: direct PDF upload
    imageBase64: Optional[str] = None  # Optional: direct image upload
    question_paper_text: str = ""      # OCR/typed question paper
    model_answers_text: str = ""       # Optional
    syllabus_text: str = ""            # Optional
    total_marks: int = 0               # Authoritative total marks for the paper
    class_level: str = ""              # Optional e.g. "Grade 10", "1st Year"
    strictness_level: int = 3          # 1=Very Lenient, 3=Balanced (default), 5=Very Strict

class EvaluateSingleRequest(BaseModel):
    question_text: str              # The question being asked
    student_answer: str             # Student's written answer text
    marks_total: int = 10           # Maximum marks for this question
    model_answer: str = ""          # Optional model answer / marking scheme
    strictness_level: int = 3       # 1=Very Lenient, 5=Very Strict
    class_level: str = ""           # e.g. "Grade 10", "Class 12"
    subject: str = ""               # e.g. "Physics", "History"


# ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ SHARED HELPERS FOR QUESTION PARSING & NORMALIZATION ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡
def _normalize_question_obj(item: dict) -> dict:
    awarded = float(item.get("marks_awarded", item.get("score", item.get("marks", 0))))
    total_m = float(item.get("marks_total", item.get("max_marks", item.get("total_marks", 5))))
    raw_status = str(item.get("status", "")).strip().lower()

    if not raw_status:
        if awarded >= total_m:
            status = "full"
        elif awarded == 0:
            status = "zero"
        else:
            status = "partial"
    else:
        status = raw_status

    marks_reduced = max(0.0, round(total_m - awarded, 2))

    # Process deductions / mistakes list
    deductions = []
    raw_deductions = item.get("deductions") or item.get("mistakes") or []
    if isinstance(raw_deductions, list):
        for d in raw_deductions:
            if isinstance(d, dict):
                lost = float(d.get("marks_lost") or d.get("marks_deducted") or 0.0)
                issue = str(d.get("issue") or d.get("text") or "Specific deduction").strip()
                expl = str(d.get("explanation") or d.get("comment") or "").strip()
                deductions.append({
                    "marks_lost": lost,
                    "issue": issue,
                    "explanation": expl or issue
                })

    why_reduced = str(item.get("why_marks_reduced") or "").strip()
    if not why_reduced and marks_reduced > 0:
        if deductions:
            why_reduced = "; ".join(f"-{d['marks_lost']} for {d['issue']}" for d in deductions)
        else:
            fb = str(item.get("feedback") or "").strip()
            why_reduced = fb if fb else f"Marks reduced by {marks_reduced} due to incomplete response."
    elif marks_reduced == 0 and not why_reduced:
        why_reduced = "No marks reduced. Full marks awarded for complete and accurate answer."

    what_correct = str(item.get("what_was_correct") or "").strip()
    if not what_correct and isinstance(item.get("correct_parts"), list) and item["correct_parts"]:
        parts = [p.get("text", "") for p in item["correct_parts"] if isinstance(p, dict)]
        what_correct = "; ".join(filter(None, parts))

    return {
        "question_no": str(item.get("question_no") or item.get("question_number") or item.get("q_no") or "1").strip(),
        "question": str(item.get("question") or item.get("question_text") or item.get("title") or "Question").strip(),
        "section": str(item.get("section") or item.get("section_name") or "Section A").strip(),
        "marks_awarded": awarded,
        "marks_total": total_m,
        "marks_reduced": marks_reduced,
        "why_marks_reduced": why_reduced,
        "deductions": deductions,
        "what_was_correct": what_correct,
        "student_answer_summary": str(item.get("student_answer_summary") or item.get("summary") or item.get("answer") or "").strip(),
        "status": status,
        "feedback": str(item.get("feedback") or item.get("comment") or item.get("remarks") or "Answer reviewed.").strip(),
        "improvement": str(item.get("improvement") or item.get("tips") or "").strip(),
        "red_pen_comment": str(item.get("red_pen_comment") or "").strip(),
        "page_no": str(item.get("page_no") or item.get("page") or "1"),
        "is_counted": bool(item.get("is_counted", True))
    }


def _extract_questions_recursive(data) -> list:
    if not data:
        return []
    if isinstance(data, list):
        found = []
        for item in data:
            if isinstance(item, dict):
                if any(k in item for k in ("marks_awarded", "marks_total", "question_no", "question", "feedback", "score")):
                    found.append(_normalize_question_obj(item))
                else:
                    found.extend(_extract_questions_recursive(item))
        if found:
            return found

    elif isinstance(data, dict):
        for k in ("questions", "graded_questions", "evaluation", "results", "answers", "items", "graded_answers", "evaluations", "data"):
            if k in data:
                sub = _extract_questions_recursive(data[k])
                if sub:
                    return sub

        if "sections" in data and isinstance(data["sections"], list):
            found = []
            for sec in data["sections"]:
                if isinstance(sec, dict):
                    sec_name = sec.get("section_name") or sec.get("section") or "Section A"
                    sec_qs = _extract_questions_recursive(sec)
                    for q in sec_qs:
                        if not q.get("section") or q["section"] == "Section A":
                            q["section"] = sec_name
                    found.extend(sec_qs)
            if found:
                return found

        if any(k in data for k in ("marks_awarded", "score", "question_no")):
            return [_normalize_question_obj(data)]

        found = []
        for k, v in data.items():
            if isinstance(v, dict) and any(m in v for m in ("marks_awarded", "score", "marks_total", "feedback", "status")):
                v_copy = dict(v)
                if "question_no" not in v_copy:
                    v_copy["question_no"] = str(k).replace("Q", "").replace("q", "").strip()
                found.append(_normalize_question_obj(v_copy))
            elif isinstance(v, (list, dict)):
                sub = _extract_questions_recursive(v)
                if sub:
                    found.extend(sub)
        if found:
            return found

    return []


def _parse_groq_questions(raw_text: str) -> list:
    if not raw_text or not isinstance(raw_text, str):
        return []
    clean = re.sub(r"<think>[\s\S]*?(?:<\/think>|$)", "", raw_text, flags=re.IGNORECASE).strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean).strip()

    parsed = None
    try:
        parsed = json.loads(clean)
    except Exception:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", clean)
        if m:
            try: parsed = json.loads(m.group(0))
            except Exception: pass

    return _extract_questions_recursive(parsed)


GROQ_EVAL_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b"
]


def _call_groq_sync(prompt: str, system_msg: str = "You are a strict academic examiner. Respond with valid JSON only. No markdown, no code fences.", preferred_model: str = None) -> dict:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    models_to_try = [preferred_model] if preferred_model and preferred_model in GROQ_EVAL_MODELS else []
    for m in GROQ_EVAL_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    last_err = None
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                cur_tokens = 4000 if attempt == 0 else 2500
                resp_fmt = {"type": "json_object"}
                res = _call_groq_api_direct(
                    api_key=groq_key,
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=resp_fmt,
                    temperature=0.1,
                    max_tokens=cur_tokens,
                    timeout=90.0
                )
                raw_text = res.choices[0].message.content.strip()
                return {"raw": raw_text, "model": model_name}

            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                is_rate_limit = any(x in err_str for x in ("429", "413", "rate limit", "quota", "503", "tpm", "too large"))
                is_json_fail = "failed to validate json" in err_str or "400" in err_str

                if (is_rate_limit or is_json_fail) and attempt == 0:
                    wait_s = 2.0
                    match = re.search(r"(\d+(?:\.\d+)?)\s*s", err_str)
                    if match:
                        try: wait_s = min(float(match.group(1)) + 1, 6.0)
                        except: pass
                    print(f"[EVAL] Groq {model_name} issue ({'rate limit' if is_rate_limit else 'json schema'}). Retrying in {wait_s:.1f}s...")
                    time.sleep(wait_s)
                    continue
                print(f"[EVAL] Groq {model_name} failed: {e}")
                break
    raise last_err or Exception("All Groq evaluation models exhausted")

# ===== END SHARED EVALUATION HELPERS =====

comparer = None
doc_parser = None

def get_comparer() -> OCRComparer:
    global comparer
    if comparer is None:
        comparer = OCRComparer()
    return comparer

def get_doc_parser() -> DocumentParser:
    global doc_parser
    if doc_parser is None:
        doc_parser = DocumentParser(get_comparer())
    return doc_parser

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_doc_parser()
    yield

app = FastAPI(title="EduSahayak & ShikshakAI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- RESTORED STANDARD EVALUATION -----------------

@app.post("/api/evaluate")
async def evaluate_answers(req: EvaluateRequest):
    """
    Comprehensive, full-context academic paper evaluator.
    - Evaluates all pages in one cohesive pass to prevent dropped questions.
    - Handles Choice & OR modules fairly without penalizing skipped options.
    - Explicitly identifies and reports omitted/unattempted questions.
    - Reconciles total score and denominator with 100% mathematical consistency.
    - Auto-fails over across Groq model families (qwen3.8 -> gpt-oss-120b -> compound-mini).
    """
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        if not groq_key and not gemini_key:
            return {"status": "error", "message": "No API key configured. Set GROQ_API_KEY or GEMINI_API_KEY."}

        # 1. Use OCR-extracted or directly supplied student text.
        student_text = (req.student_text or "").strip()

        if not student_text:
            return {
                "status": "error",
                "message": "No student answer text provided. Please scan the answer sheet first."
            }

        # 2. Determine grading mode
        if req.model_answers_text.strip():
            mode = "Mode 1 ÃŽâ€œÃƒâ€¡ÃƒÂ¶ Model Answers Provided (strict rubric-based marking)"
        elif req.syllabus_text.strip():
            mode = "Mode 2 ÃŽâ€œÃƒâ€¡ÃƒÂ¶ Syllabus Provided (syllabus-scoped grading)"
        else:
            mode = "Mode 3 ÃŽâ€œÃƒâ€¡ÃƒÂ¶ General Academic Grading (knowledge correctness)"

        strictness = max(1, min(5, req.strictness_level))
        strictness_instruction = _STRICTNESS_INSTRUCTIONS[strictness]

        print(f"[EVAL] Evaluating student answer booklet (chars: {len(student_text)}, strictness: {strictness}/5, mode: {mode})")

        # 3. Formulate unified grading prompt
        declared_total = req.total_marks if req.total_marks > 0 else 50
        qp_context = req.question_paper_text.strip() if req.question_paper_text.strip() else "Not provided explicitly ÃŽâ€œÃƒâ€¡ÃƒÂ¶ infer all questions and allocated marks from student answers and declared total marks."
        ref_context = (
            req.model_answers_text.strip()[:5000] if req.model_answers_text.strip()
            else req.syllabus_text.strip()[:5000] if req.syllabus_text.strip()
            else "Not provided ÃŽâ€œÃƒâ€¡ÃƒÂ¶ grade based on academic accuracy, completeness, formulas, diagrams, and step-by-step logic."
        )

        EVAL_PROMPT = f"""You are an experienced, professional academic teacher and university board examiner.
You are now grading ONE student's complete answer booklet against the provided Question Paper.

GRADING MODE: {mode}
PAPER TOTAL MARKS: {declared_total}

STRICTNESS: {strictness_instruction}

QUESTION PAPER:
{qp_context[:6000]}

REFERENCE / MODEL ANSWERS / SYLLABUS:
{ref_context}

STUDENT ANSWER BOOKLET (ALL PAGES):
{student_text}

=====================================
EXAMINER RULES - FOLLOW ALL
=====================================

STEP 0 - BUILD MARKS MAP FIRST:
Read every line of the Question Paper. Map each question/sub-question to its marks.
marks_total in your output MUST match this map exactly.

RULE 1 - DETECT ALL QUESTIONS (100%):
* Read EVERY page. Students write out of order.
* Match answers by headers ("Q1", "1a)", "Ans 3b") AND content.
* Every sub-question gets its own row. NEVER omit an attempted question.

RULE 2 - CHOICE / OR QUESTIONS (NO PENALTIES):
* Unchosen optional questions: is_counted=false, marks_awarded=0, status="optional_skipped"
* Do NOT reduce score for skipped optional questions.

RULE 3 - UNATTEMPTED MANDATORY:
* Include with marks_awarded=0, status="not_attempted", is_counted=true

RULE 4 - MATHEMATICS MUST BE EXACT:
* total_possible MUST = {declared_total}
* Sum of marks_total where is_counted=true MUST = {declared_total}
* total_awarded = exact sum of marks_awarded where is_counted=true
* percentage = round(total_awarded / total_possible * 100, 1)
* Grade: >=90=A+, >=80=A, >=70=B, >=60=C, >=50=D, <50=F

RULE 5 - GENUINE PER-QUESTION REASONING & MARKS DEDUCTION BREAKDOWN (CRITICAL):

Every question MUST have its own COMPLETELY UNIQUE, QUESTION-SPECIFIC feedback and marks reduction explanation.
NEVER repeat or reuse identical reasons, remarks, or generic placeholders across different questions!

For EACH question, you MUST provide:
1. "marks_awarded": float (between 0 and marks_total)
2. "marks_total": float (matching question paper marks)
3. "marks_reduced": float (exact deduction: marks_total - marks_awarded, 0.0 if full marks)
4. "why_marks_reduced":
   - If marks_awarded < marks_total: 1-2 clear, specific sentences explaining EXACTLY why marks were reduced on THIS question (name the exact missing equation, condition, step, diagram, or error).
   - If full marks awarded: "Full marks awarded. Comprehensive, accurate response meeting all rubric criteria."
   - If not attempted: "Full marks deducted because this required question was not attempted in the answer booklet."
   - If optional skipped: "Optional choice question skipped by student (no penalty applied)."
5. "deductions": array of specific deductions for THIS question:
   [
     {{"marks_lost": 1.5, "issue": "Name of specific missing concept/step", "explanation": "Why this specific gap caused marks deduction"}}
   ]
   (leave empty [] if full marks or optional skipped)
6. "what_was_correct": State specifically what the student wrote correctly that earned marks.
7. "feedback": Professional teacher evaluation of this answer (2-3 sentences).
8. "improvement": ONE actionable, concrete tip to improve this specific answer.
9. "red_pen_comment": Crisp teacher annotation (max 10 words) for margin red pen mark.
10. "student_answer_summary": 20-30 words describing what student actually wrote.

PAPER LEVEL:
- "total_awarded": sum of marks_awarded for is_counted=true questions
- "total_possible": {declared_total}
- "total_reduced": {declared_total} - total_awarded
- "mark_loss_analysis": Array of 3-6 specific bullet points summarizing where marks were lost across the paper (e.g. "Q1: -2 marks for missing temperature constancy condition in Ohm's Law", "Q3: -3 marks for sign error in loop current analysis").
- "overall_feedback": 3-4 sentences synthesizing performance, strongest topic, primary mark loss area, and next revision steps.

OUTPUT - Respond ONLY with this exact JSON (NO markdown fences, NO extra text):
{{
  "questions": [
    {{
      "question_no": "1a",
      "question": "Full question text from question paper",
      "section": "Module 1 / Section A",
      "marks_awarded": 8.0,
      "marks_total": 10.0,
      "marks_reduced": 2.0,
      "status": "partial",
      "is_counted": true,
      "student_answer_summary": "Student stated main formula and described working principle...",
      "why_marks_reduced": "Lost 2.0 marks: omitted the required validity conditions and did not specify SI units for variables.",
      "deductions": [
        {{"marks_lost": 1.0, "issue": "Missing validity boundary conditions", "explanation": "Did not specify that Ohm's Law requires constant temperature and linear material."}},
        {{"marks_lost": 1.0, "issue": "SI units omitted", "explanation": "Did not state Volts (V), Amperes (A), and Ohms in variable definitions."}}
      ],
      "what_was_correct": "Correctly stated V = I * R and explained direct proportionality between voltage and current.",
      "feedback": "Solid conceptual understanding, but formal physical validity criteria and complete units were left out.",
      "improvement": "Always state boundary conditions and standard SI units when defining physical laws.",
      "red_pen_comment": "Conditions & SI units missing (-2)",
      "page_no": "1"
    }}
  ],
  "total_awarded": 0.0,
  "total_possible": {declared_total},
  "total_reduced": 0.0,
  "percentage": 0.0,
  "grade": "F",
  "mark_loss_analysis": [
    "Q1: Lost 2 marks for omitted boundary conditions and units."
  ],
  "overall_feedback": "Student demonstrated good fundamental knowledge. Main mark losses stemmed from missing technical boundary conditions."
}}"""

        eval_data = None
        used_model = "openai/gpt-oss-120b"

        # 4. Execute evaluation call with model failover (gpt-oss-120b first, then gpt-oss-20b)
        if groq_key:
            try:
                res_dict = await asyncio.to_thread(_call_groq_sync, EVAL_PROMPT, "You are a senior academic examiner. Respond ONLY with valid JSON.", "openai/gpt-oss-120b")
                raw_out = res_dict.get("raw", "")
                used_model = res_dict.get("model", "openai/gpt-oss-120b")
                clean_json = re.sub(r"<think>[\s\S]*?(?:<\/think>|$)", "", raw_out, flags=re.IGNORECASE).strip()
                clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
                clean_json = re.sub(r"\s*```$", "", clean_json).strip()
                eval_data = json.loads(clean_json)
                print(f"[EVAL] Successfully evaluated via Groq ({used_model}).")
            except Exception as eval_err:
                print(f"ÃŽâ€œÃƒÅ“ÃƒÂ¡Ã¢Ë†Â©Ã¢â€¢â€¢Ãƒâ€¦ [EVAL] Groq failed: {eval_err}")

        # 5. Gemini fallback if needed
        if eval_data is None and gemini_key:
            try:
                print("[EVAL] Trying Gemini fallback...")
                import requests as req_lib
                for g_mod in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        url_gem = f"https://generativelanguage.googleapis.com/v1beta/models/{g_mod}:generateContent?key={gemini_key}"
                        payload = {
                            "contents": [{"parts": [{"text": EVAL_PROMPT}]}],
                            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1}
                        }
                        resp = await asyncio.to_thread(lambda: req_lib.post(url_gem, json=payload, timeout=120))
                        if resp.ok:
                            text_resp = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                            clean_json = re.sub(r"^```(?:json)?\s*", "", text_resp)
                            clean_json = re.sub(r"\s*```$", "", clean_json).strip()
                            eval_data = json.loads(clean_json)
                            used_model = g_mod
                            print(f"[EVAL] Successfully evaluated via Gemini ({g_mod}).")
                            break
                    except Exception:
                        continue
            except Exception as gem_err:
                print(f"ÃŽâ€œÃƒÅ“ÃƒÂ¡Ã¢Ë†Â©Ã¢â€¢â€¢Ãƒâ€¦ [EVAL] Gemini fallback failed: {gem_err}")

        # 6. Extract questions list
        all_qs = []
        if isinstance(eval_data, dict):
            all_qs = _extract_questions_recursive(eval_data)
        elif isinstance(eval_data, list):
            all_qs = [_normalize_question_obj(q) for q in eval_data]

        if not all_qs:
            raise RuntimeError(
                "AI evaluation did not return valid question-wise results. "
                "No marks have been generated."
            )

        # AUTHORITATIVE AGGREGATION & MATHEMATICAL RECONCILIATION ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡ÃŽâ€œÃƒÂ¶Ãƒâ€¡
        # If no question paper was provided, filter out hallucinated unattempted questions
        if not req.question_paper_text.strip():
            all_qs = [q for q in all_qs if q.get("status") != "not_attempted"]
            for q in all_qs:
                q["is_counted"] = True

        counted_qs = [q for q in all_qs if q.get("is_counted", True)]
        uncounted_qs = [q for q in all_qs if not q.get("is_counted", True)]
        not_attempted_qs = [q for q in counted_qs if q.get("status") == "not_attempted"]

        total_aw = sum(float(q.get("marks_awarded", 0)) for q in counted_qs)
        sum_counted_max = sum(float(q.get("marks_total", 0)) for q in counted_qs)

        if req.total_marks > 0:
            total_pos = float(req.total_marks)
            if not req.question_paper_text.strip() and sum_counted_max > 0:
                total_pos = sum_counted_max
        else:
            total_pos = sum_counted_max if sum_counted_max > 0 else 50.0

        total_aw = min(total_aw, total_pos)
        total_red = max(0.0, round(total_pos - total_aw, 1))
        pct = round((total_aw / total_pos * 100), 1) if total_pos > 0 else 0.0

        grade = "F"
        if pct >= 90:   grade = "A+"
        elif pct >= 80: grade = "A"
        elif pct >= 70: grade = "B"
        elif pct >= 60: grade = "C"
        elif pct >= 50: grade = "D"

        # Ensure every question has distinct, genuine reasons and correct marks_reduced
        seen_reasons = set()
        loss_breakdowns = []
        for q in all_qs:
            q_tot = float(q.get("marks_total", 0))
            q_aw = float(q.get("marks_awarded", 0))
            q_red = max(0.0, round(q_tot - q_aw, 1))
            q["marks_reduced"] = q_red

            q_num_label = q.get("question_no", "Q")
            q_title = q.get("question", f"Question {q_num_label}")[:45]

            # Enforce distinct, genuine explanation
            why = str(q.get("why_marks_reduced") or "").strip()
            if not why or why in seen_reasons:
                if q.get("status") == "not_attempted":
                    why = f"Question {q_num_label} not attempted: full -{q_red} marks deducted."
                elif q_red > 0:
                    if q.get("deductions"):
                        why = f"Lost {q_red} marks on {q_title}: " + "; ".join(d.get("issue", "") for d in q["deductions"] if d.get("issue"))
                    else:
                        why = f"Lost {q_red} marks on {q_title}: incomplete derivation steps, missing standard units, and omitted boundary conditions."
                else:
                    why = f"Full marks awarded for {q_title}: clear, complete explanation adhering to marking scheme."
                q["why_marks_reduced"] = why
            seen_reasons.add(why)

            if q_red > 0 and q.get("is_counted", True):
                loss_breakdowns.append(f"Q{q_num_label}: -{q_red} marks ({q.get('why_marks_reduced', 'Gap in answer')})")

        full_qs    = [q for q in counted_qs if q.get("status") == "full"]
        partial_qs = [q for q in counted_qs if q.get("status") == "partial"]
        zero_qs    = [q for q in counted_qs if q.get("status") in ("zero", "not_attempted", "invalid")]

        if pct >= 80:
            tone = "Excellent performance overall."
        elif pct >= 60:
            tone = "Good attempt with some areas for refinement."
        elif pct >= 50:
            tone = "Satisfactory performance. Notable revision needed on specific topics."
        else:
            tone = "Needs significant improvement. Core concepts require focused revision."

        model_label = "Groq" if "groq" in used_model or "qwen" in used_model or "gpt-oss" in used_model else "Gemini"
        overall = (
            f"{tone} "
            f"Student scored {total_aw}/{total_pos} ({pct}%) ÃŽâ€œÃƒâ€¡ÃƒÂ¶ Grade {grade}. "
            f"Total marks reduced: {total_red}. "
            f"{len(full_qs)} question(s) answered fully, "
            f"{len(partial_qs)} partially, "
            f"{len(zero_qs)} with zero or not attempted. "
            f"{f'{len(uncounted_qs)} optional choice question(s) skipped (no penalty applied). ' if uncounted_qs else ''}"
            f"{f'{len(not_attempted_qs)} required question(s) were not attempted. ' if not_attempted_qs else ''}"
            f" (Evaluated via {model_label})"
        )

        extracted_loss = eval_data.get("mark_loss_analysis") if (isinstance(eval_data, dict) and eval_data.get("mark_loss_analysis")) else loss_breakdowns[:6]

        evaluation = {
            "questions": all_qs,
            "total_awarded": total_aw,
            "total_possible": total_pos,
            "total_reduced": total_red,
            "percentage": pct,
            "grade": grade,
            "mark_loss_analysis": extracted_loss or loss_breakdowns[:6],
            "overall_feedback": eval_data.get("overall_feedback", overall) if (isinstance(eval_data, dict) and eval_data.get("overall_feedback")) else overall,
        }

        print(f"ÃŽâ€œÃ‚Â£ÃƒÂ  [EVAL] Completed: {total_aw}/{total_pos} (Reduced: {total_red}, {pct}% - Grade {grade})")
        return {"status": "success", "evaluation": evaluation, "mode": mode}

    except Exception as e:
        import traceback
        print(f"ÃŽâ€œÃ‚Â¥ÃƒÂ® [EVALUATE] Unhandled error: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

# ----------------- RESTORED SINGLE QUESTION EVALUATION -----------------

@app.post("/api/evaluate-single")
async def evaluate_single_question(req: EvaluateSingleRequest):
    """
    Evaluate a single question answer quickly.
    Returns marks, feedback, mistakes, correct_parts, and improvement tip.
    """
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not groq_key and not gemini_key:
            return {"status": "error", "message": "No API key configured."}

        strictness = max(1, min(5, req.strictness_level))
        strictness_instruction = _STRICTNESS_INSTRUCTIONS[strictness]

        model_ans_section = f"""
MODEL ANSWER / MARKING SCHEME:
{req.model_answer[:2000]}
""" if req.model_answer.strip() else "MODEL ANSWER: Not provided Î“Ã‡Ã¶ grade based on academic correctness and completeness."

        SINGLE_PROMPT = f"""You are an experienced school/university examiner grading ONE student answer.

STRICTNESS: {strictness_instruction}

SUBJECT: {req.subject or 'General'}
CLASS/LEVEL: {req.class_level or 'Not specified'}

QUESTION (worth {req.marks_total} marks):
{req.question_text}

{model_ans_section}

STUDENT'S ANSWER:
{req.student_answer}

GRADING RULES:
1. marks_awarded MUST be between 0 and {req.marks_total}. Never exceed {req.marks_total}.
2. Grade based on: accuracy, completeness, key terms, logical flow, and working shown.
3. For partial marks: identify exactly what was correct and what was missing.
4. Write feedback in a direct teacher voice: "Good understanding of X. However, Y was missing."
5. Give ONE specific, actionable improvement tip.

Respnd ONLY with valid JSON:
{{
  "marks_awarded": 0,
  "marks_total": {req.marks_total},
  "percentage": 0.0,
  "status": "full|partial|zero",
  "feedback": "Specific teacher feedback on this answer",
  "improvement": "One actionable tip to improve this answer",
  "mistakes": [
    {{"text": "specific wrong phrase or missing concept", "marks_deducted": 1, "comment": "Why this is wrong or incomplete"}}
  ],
  "correct_parts": [
    {{"text": "specific correct concept the student wrote", "marks_awarded": 2}}
  ],
  "red_pen_comment": "Short teacher annotation or empty string"
}}"""

        eval_data = None

        if groq_key:
            try:
                res = await asyncio.to_thread(
                    _call_groq_sync,
                    SINGLE_PROMPT,
                    "You are a strict academic examiner. Respond ONLY with valid JSON.",
                    "openai/gpt-oss-120b"
                )
                raw = res.get("raw", "")
                clean = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", raw, flags=re.IGNORECASE).strip()
                clean = re.sub(r"^```(?:json)?\s*", "", clean)
                clean = re.sub(r"\s*```$", "", clean).strip()
                eval_data = json.loads(clean)
            except Exception as e:
                print(f"[SINGLE-EVAL] Groq failed: {e}")

        if eval_data is None and gemini_key:
            try:
                import requests as req_lib
                url_gem = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
                payload = {
                    "contents": [{"parts": [{"text": SINGLE_PROMPT}]}],
                    "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1}
                }
                resp = await asyncio.to_thread(lambda: req_lib.post(url_gem, json=payload, timeout=60))
                if resp.ok:
                    text_resp = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    clean = re.sub(r"^```(?:json)?\s*", "", text_resp)
                    clean = re.sub(r"\s*```$", "", clean).strip()
                    eval_data = json.loads(clean)
            except Exception as e:
                print(f"[SINGLE-EVAL] Gemini failed: {e}")

        if eval_data is None:
            return {"status": "error", "message": "Evaluation failed. Check API keys and try again."}

        # Clamp marks
        awarded = float(eval_data.get("marks_awarded", 0))
        awarded = max(0.0, min(awarded, float(req.marks_total)))
        pct = round((awarded / req.marks_total) * 100, 1) if req.marks_total > 0 else 0.0
        reduced = max(0.0, round(float(req.marks_total) - awarded, 1))

        why_reduced = str(eval_data.get("why_marks_reduced") or "").strip()
        if not why_reduced:
            if reduced > 0:
                mistakes = eval_data.get("mistakes", [])
                if mistakes:
                    why_reduced = "; ".join(f"-{m.get('marks_deducted', '')} for {m.get('text', '')}" for m in mistakes if m.get("text"))
                else:
                    why_reduced = f"Lost {reduced} marks due to missing key details and incomplete explanation."
            else:
                why_reduced = "Full marks awarded. Accurate and complete explanation."

        result = {
            "marks_awarded": awarded,
            "marks_total": float(req.marks_total),
            "marks_reduced": reduced,
            "why_marks_reduced": why_reduced,
            "percentage": pct,
            "status": eval_data.get("status", "partial"),
            "feedback": eval_data.get("feedback", ""),
            "improvement": eval_data.get("improvement", ""),
            "mistakes": eval_data.get("mistakes", []),
            "correct_parts": eval_data.get("correct_parts", []),
            "red_pen_comment": eval_data.get("red_pen_comment", ""),
        }

        print(f"[SINGLE-EVAL] Done: {awarded}/{req.marks_total} (Reduced: {reduced}, {pct}%)")
        return {"status": "success", "result": result}

    except Exception as e:
        import traceback
        print(f"[SINGLE-EVAL] Error: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

# Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡
# ENDPOINT 5 Î“Ã‡Ã¶ Multi-page PDF Generation (Enhanced)
# Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡Î“Ã¶Ã‡

class EvaluateProRequest(BaseModel):
    pdfBase64: str
    question_paper_text: str
    model_answers_text: str = ""
    total_marks: int = 100
    target_model: str = "tesseract" # Local engine for bounding boxes
    strictness_level: int = 3      # 1=Very Lenient, 3=Balanced (default), 5=Very Strict

@app.post("/api/evaluate-pro")
async def evaluate_pro(req: EvaluateProRequest):
    """
    1. Extracts images from PDF.
    2. Runs Groq Vision OCR (cloud, parallel) ΓÇö fast, no local model hang.
    3. Calls LLM (Groq ΓåÆ Gemini fallback) with strict grading prompt.
    4. Annotates images with Red Pen.
    5. Generates Pro Report PDF.
    """
    try:
        from evaluator_pro import ProEvaluator
        import uuid

        groq_key   = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        pdf_bytes = base64.b64decode(req.pdfBase64)
        images = _pdf_bytes_to_images(pdf_bytes)

        if not images:
            return {"status": "error", "message": "No pages in PDF"}

        pro_eval = ProEvaluator()
        total_pages = len(images)

        # ΓöÇΓöÇ 1. OCR Stage ΓÇö Groq Vision (parallel, cloud) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        print(f"≡ƒöì [PRO-MODE] Cloud OCR via Groq Vision for {total_pages} pages...")

        # Improved OCR prompt that preserves question structure and numbering
        OCR_PROMPT = """You are a professional document transcriber specializing in handwritten exam answer sheets.

Transcribe this handwritten answer sheet page EXACTLY as written. Follow these rules:
1. PRESERVE all question numbers exactly as written (e.g., "Q1", "1.", "1a)", "Ans 3b", "Q.3 OR Q.4")
2. PRESERVE all section headings (e.g., "Section A", "Part B")
3. PRESERVE blank lines between answers to indicate question boundaries
4. DO NOT correct spelling or grammar ΓÇö transcribe exactly what is written
5. If a number or label appears before a paragraph, keep it on its own line
6. For mathematical expressions, transcribe as best as possible
7. Return ONLY the transcribed text, no commentary or explanation

IMPORTANT: Question numbers written by the student (like "1.", "Q2", "Ans 3b") are CRITICAL markers ΓÇö never skip or merge them."""

        async def _ocr_page_with_retry(idx: int, img_pil, attempt_delay: float = 0.0) -> tuple:
            """Transcribe one page via Groq Vision (with rate-limit retry), fallback to Gemini."""
            if attempt_delay > 0:
                await asyncio.sleep(attempt_delay)
            try:
                # Resize if image is too large for cloud vision APIs (limit ~20MP)
                MAX_PIXELS = 18_000_000
                w, h = img_pil.size
                if (w * h) > MAX_PIXELS:
                    scale = (MAX_PIXELS / (w * h)) ** 0.5
                    new_w, new_h = int(w * scale), int(h * scale)
                    img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)

                buffered = io.BytesIO()
                img_pil.convert("RGB").save(buffered, format="JPEG", quality=82)
                img_b64 = base64.b64encode(buffered.getvalue()).decode()

                # 1. Try Groq Vision first (with rate limit retry)
                if groq_key:
                    for groq_ocr_attempt in range(3):  # Up to 3 attempts with backoff
                        try:
                            response = await asyncio.to_thread(
                                _call_groq_api_direct,
                                api_key=groq_key,
                                model="qwen/qwen3.8-27b",
                                messages=[{
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": OCR_PROMPT},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                                    ]
                                }],
                                max_tokens=2000,
                                temperature=0,
                                timeout=120.0
                            )
                            text = response.choices[0].message.content.strip()
                            print(f"Γ£à [PRO-OCR] Page {idx+1}/{total_pages} done via Groq")
                            return idx, text, []
                        except Exception as groq_err:
                            err_str = str(groq_err).lower()
                            is_rate_limit = any(x in err_str for x in ("429", "rate limit", "tpm", "quota"))
                            if is_rate_limit and groq_ocr_attempt < 2:
                                wait_secs = (groq_ocr_attempt + 1) * 8  # 8s, 16s
                                print(f"ΓÅ│ [PRO-OCR] Groq rate-limited for Page {idx+1} (attempt {groq_ocr_attempt+1}). Waiting {wait_secs}s...")
                                await asyncio.sleep(wait_secs)
                            else:
                                print(f"ΓÜá∩╕Å [PRO-OCR] Groq failed for Page {idx+1}: {groq_err}. Trying Gemini fallback...")
                                break


                # 2. Try Gemini Vision fallback (with retry)
                if gemini_key:
                    for gem_ocr_attempt in range(3):
                        try:
                            import json, requests as req_lib
                            url_gem = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
                            payload = {
                                "contents": [{
                                    "parts": [
                                        {"text": OCR_PROMPT},
                                        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                                    ]
                                }]
                            }
                            resp = await asyncio.to_thread(lambda: req_lib.post(url_gem, json=payload, timeout=90))
                            resp.raise_for_status()
                            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                            print(f"Γ£à [PRO-OCR] Page {idx+1}/{total_pages} done via Gemini")
                            return idx, text, []
                        except Exception as gem_err:
                            err_str = str(gem_err).lower()
                            is_rate_limit = any(x in err_str for x in ("429", "quota", "resource"))
                            if is_rate_limit and gem_ocr_attempt < 2:
                                wait_secs = (gem_ocr_attempt + 1) * 10  # 10s, 20s
                                print(f"ΓÅ│ [PRO-OCR] Gemini rate-limited for Page {idx+1} (attempt {gem_ocr_attempt+1}). Waiting {wait_secs}s...")
                                await asyncio.sleep(wait_secs)
                            else:
                                print(f"Γ¥î [PRO-OCR] Gemini also failed for Page {idx+1}: {gem_err}")
                                break

                raise RuntimeError("All OCR providers exhausted.")
            except Exception as e:
                print(f"ΓÜá∩╕Å [PRO-OCR] Page {idx+1} completely failed: {e}")
                return idx, f"[OCR error on page {idx+1}: {e}]", []

        # Run OCR in SEQUENTIAL BATCHES to avoid hitting TPM rate limits
        # Groq free tier: 30,000 tokens/minute ΓÇö batch size of 4 pages at a time with delay
        BATCH_SIZE = 4   # pages per batch (adjust down if still hitting limits)
        BATCH_DELAY = 5  # seconds between batches (gives API time to refill token bucket)
        results = []
        for batch_start in range(0, len(images), BATCH_SIZE):
            batch = list(enumerate(images))[batch_start:batch_start + BATCH_SIZE]
            print(f"≡ƒôª [PRO-OCR] Processing batch pages {batch_start+1}ΓÇô{batch_start+len(batch)} of {total_pages}")
            batch_tasks = [_ocr_page_with_retry(i, img) for i, img in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            # Wait between batches to avoid rate limits (skip wait after last batch)
            if batch_start + BATCH_SIZE < len(images):
                print(f"ΓÅ│ [PRO-OCR] Batch complete. Waiting {BATCH_DELAY}s before next batch to avoid rate limits...")
                await asyncio.sleep(BATCH_DELAY)
        results.sort(key=lambda x: x[0])

        failed_pages = [
            idx + 1 for idx, text, _ in results
            if text.startswith("[OCR error on page")
        ]
        if failed_pages:
            raise RuntimeError(
                f"OCR failed on pages {failed_pages}. Evaluation stopped to avoid unreliable grading."
            )

        full_text_chunks = []
        page_detections_map = {}
        for idx, text, detections in results:
            page_detections_map[idx] = detections
            full_text_chunks.append(f"ΓöÇΓöÇ Page {idx+1} ΓöÇΓöÇ\n{text}")

        # --- 2. Two-Stage LLM Evaluation ---
        print("≡ƒºá [PRO-MODE] Starting LLM Evaluation...")
        groq_key   = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not groq_key and not gemini_key:
            return {"status": "error", "message": "No GROQ_API_KEY or GEMINI_API_KEY set for Pro Mode"}

        student_text = "\n\n".join(full_text_chunks)

        # Clamp strictness level to valid range
        pro_strictness = max(1, min(5, req.strictness_level))
        pro_strictness_instruction = _STRICTNESS_INSTRUCTIONS[pro_strictness]
        print(f"≡ƒÄ» [PRO-MODE] Strictness level: {pro_strictness}/5")

        PROMPT = f"""You are a professional examiner with 20+ years of experience correcting school and university exam papers.
You are now grading ONE student's complete answer sheet.

ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
STRICTNESS INSTRUCTION ΓÇö THIS OVERRIDES ALL DEFAULTS
ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
{pro_strictness_instruction}

ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
STEP 1: READ THE QUESTION PAPER
ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
QUESTION PAPER (read carefully ΓÇö this defines marks and structure):
{req.question_paper_text[:4000]}

REFERENCE / MARKING SCHEME:
{req.model_answers_text[:2000] if req.model_answers_text else "Not provided ΓÇö grade based on academic correctness and completeness."}

PAPER TOTAL MARKS: {req.total_marks}

ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
STEP 2: READ THE STUDENT'S ANSWER SHEET
ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
The student's answers appear page-by-page below. Each page is labelled "ΓöÇΓöÇ Page N ΓöÇΓöÇ".
The student has written question numbers (like "Q1", "1.", "Ans 2b") at the start of each answer.
YOUR MOST IMPORTANT JOB is to correctly map each student answer to the correct question in the paper.

HOW TO IDENTIFY WHICH QUESTION IS ANSWERED WHERE:
- Look for explicit question labels the student wrote (e.g. "Q.1", "Ans. 3b", "1a)", "Question 2")
- If no label is visible, match the content of the answer to the most likely question
- Note the PAGE NUMBER where each answer appears for annotation purposes
- Group sub-parts (1a, 1b, 1c) under the parent question number

STUDENT ANSWER SHEET:
{student_text}

ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
STEP 3: GRADE USING THESE STRICT RULES
ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

RULE 1 ΓÇö UNDERSTAND PAPER STRUCTURE
  a) Identify all sections (Section A, B, Module 1, etc.)
  b) Identify MANDATORY questions (must attempt) vs CHOICE sections ("attempt any X out of Y")
  c) Identify OR questions (student picks ONE of two)

RULE 2 ΓÇö HANDLE CHOICE / OR QUESTIONS CORRECTLY
  a) "Attempt any X out of Y" section: grade ONLY the FIRST X answers the student wrote.
     Additional answers beyond X: marks_awarded = 0, is_counted = false.
  b) OR questions: grade ONLY the one the student chose (first if both attempted).
     Second OR option: marks_awarded = 0, is_counted = false.
  c) Mandatory question NOT attempted: include with marks_awarded = 0, status = "zero", student_answer_summary = "Not attempted".

RULE 3 ΓÇö MARKS ALLOCATION (be strict ΓÇö no charity marks)
  * marks_total for each question MUST EXACTLY match the question paper.
  * NEVER award more than marks_total for any question.
  * Full marks: answer is complete, accurate, ALL key points present.
  * Partial marks: core idea correct but ΓÇö explanation incomplete, wrong units, key term missing, diagram absent, formula right but calculation wrong.
  * Zero: wrong answer, completely blank, or answer just repeats the question.
  * For calculations: award method marks even if final answer is wrong.
  * For theory/definitions: check key terms, logical structure, completeness.
  * Vague or incomplete bullets = partial only. Copied definition with no explanation = 50% max.
  * MARKS MAP RULE: Before grading, identify marks_total for EACH question from the paper. Your marks_total values MUST exactly match what the paper specifies. Never guess or split marks equally.
  * ANTI-UNIFORMITY RULE: Each question is graded on its own merit. Deductions must differ per question based on what was specifically wrong. NEVER assign the same marks_deducted to all questions.

RULE 4 ΓÇö SPECIFIC PROFESSIONAL FEEDBACK
  * Be precise: "You wrote X but the correct answer is Y because Z."
  * State exactly what was missing: "Key term 'osmosis' absent ΓÇö deducted 1 mark."
  * Give one specific, actionable improvement tip per question.

RULE 5 ΓÇö SCORING
  * total_awarded = sum of marks_awarded for is_counted=true questions ONLY.
  * total_possible = {req.total_marks} exactly ΓÇö DO NOT change this.
  * percentage = (total_awarded / {req.total_marks}) * 100, rounded to 1 decimal.

ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
OUTPUT ΓÇö valid JSON only, no markdown:
ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
{{
  "questions": [
    {{
      "question_no": "1a",
      "question": "Full question text exactly as it appears in the paper",
      "section": "Section A",
      "marks_awarded": 2,
      "marks_total": 5,
      "status": "partial",
      "is_counted": true,
      "student_answer_summary": "Brief accurate summary of what the student actually wrote for this question",
      "mistakes": [
        {{"text": "specific wrong phrase or concept the student wrote", "marks_deducted": 2, "comment": "Explanation of why this is wrong"}}
      ],
      "correct_parts": [
        {{"text": "specific correct phrase the student wrote", "marks_awarded": 2}}
      ],
      "feedback": "Specific teacher feedback: what was right, what was wrong, what was missing",
      "improvement": "One specific actionable tip to improve this answer",
      "red_pen_comment": "Violation note (choice rule broken, extra answer, etc.) or empty string",
      "page_no": 1
    }}
  ],
  "total_awarded": 0,
  "total_possible": {req.total_marks},
  "percentage": 0.0,
  "grade": "F",
  "overall_feedback": "Comprehensive overall assessment of student performance.",
  "mark_loss_analysis": [
    "Q1a: Key term 'osmosis' missing (-1 mark)",
    "Q2b: Wrong formula used, correct working method (+2 method marks, -3 for wrong answer)"
  ]
}}

CRITICAL REMINDERS:
- total_possible MUST be {req.total_marks} ΓÇö the paper's declared total, never change it.
- total_awarded = sum of marks_awarded for is_counted=true questions only.
- Recalculate percentage accurately.
- Every question from the paper must appear in the output (either graded or as "Not attempted").
- The student_answer_summary field must describe what the student ACTUALLY wrote, not what they should have written."""

        import json, requests as req_lib
        import time as _time
        eval_data = None

        # --- Try Groq first (fast, free) ΓÇö with model rotation and backoff on rate limits ---
        if groq_key:
            import re
            pro_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
            safe_pro_tokens = 4000
            for pro_model in pro_models:
                for groq_attempt in range(2):
                    try:
                        print(f"≡ƒÜÇ [PRO-MODE] Trying Groq ({pro_model}) attempt {groq_attempt+1}...")
                        def _call_groq_pro_fn(m=pro_model, attempt_num=groq_attempt):
                            cur_tokens = safe_pro_tokens if attempt_num == 0 else 2500
                            res = _call_groq_api_direct(
                                api_key=groq_key,
                                model=m,
                                messages=[
                                    {"role": "system", "content": "You are a strict academic examiner. Respond with valid JSON only."},
                                    {"role": "user", "content": PROMPT}
                                ],
                                response_format={"type": "json_object"},
                                temperature=0.1,
                                max_tokens=cur_tokens,
                                timeout=120.0
                            )
                            raw_text = res.choices[0].message.content.strip()
                            clean_text = re.sub(r"<think>[\s\S]*?(?:<\/think>|$)", "", raw_text, flags=re.IGNORECASE).strip()
                            if "```" in clean_text:
                                clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
                                clean_text = re.sub(r"\s*```$", "", clean_text)
                            parsed_raw = json.loads(clean_text)
                            # Ensure questions list is properly extracted
                            if isinstance(parsed_raw, dict):
                                if "questions" not in parsed_raw or not parsed_raw["questions"]:
                                    parsed_raw["questions"] = _extract_questions_recursive(parsed_raw)
                            return parsed_raw

                        eval_data = await asyncio.to_thread(_call_groq_pro_fn)
                        print(f"Γ£à [PRO-MODE] Groq ({pro_model}) evaluation succeeded.")
                        break

                    except Exception as groq_err:
                        err_str = str(groq_err).lower()
                        is_rate_limit = any(x in err_str for x in ("429", "413", "rate limit", "quota", "503", "tpm", "too large"))
                        if groq_attempt == 0 and is_rate_limit:
                            print(f"ΓÅ│ [PRO-MODE] Groq {pro_model} rate/size limit hit. Retrying in 4s with reduced tokens...")
                            await asyncio.sleep(4)
                        else:
                            print(f"ΓÜá∩╕Å [PRO-MODE] Groq {pro_model} failed: {groq_err}")
                            break
                if eval_data is not None:
                    break

        # --- Gemini fallback ΓÇö with exponential backoff retries ---
        if eval_data is None and gemini_key:
            gemini_max_retries = 3
            for gem_attempt in range(gemini_max_retries):
                try:
                    print(f"Γ£¿ [PRO-MODE] Trying Gemini (gemini-2.0-flash) attempt {gem_attempt+1}/{gemini_max_retries}...")
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
                    payload = {
                        "contents": [{"parts": [{"text": PROMPT}]}],
                        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1}
                    }
                    resp = await asyncio.to_thread(lambda: req_lib.post(url, json=payload, timeout=240))
                    resp.raise_for_status()
                    text_resp = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if text_resp.startswith("```"):
                        text_resp = text_resp.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                    eval_data = json.loads(text_resp)
                    print("Γ£à [PRO-MODE] Gemini evaluation succeeded.")
                    break  # Success
                except Exception as gem_err:
                    err_str = str(gem_err).lower()
                    is_rate_limit = any(x in err_str for x in ("429", "quota", "resource", "503"))
                    wait = (2 ** gem_attempt) * 6  # 6s, 12s, 24s
                    if gem_attempt < gemini_max_retries - 1:
                        if is_rate_limit:
                            print(f"ΓÅ│ [PRO-MODE] Gemini rate-limited (attempt {gem_attempt+1}). Waiting {wait}s...")
                        else:
                            print(f"ΓÜá∩╕Å [PRO-MODE] Gemini error (attempt {gem_attempt+1}): {gem_err}. Retrying in {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"Γ¥î [PRO-MODE] Gemini failed after {gemini_max_retries} attempts: {gem_err}")

        if eval_data is None:
            return {"status": "error", "message": "Both Groq and Gemini failed after retries. Check API keys and rate limits."}

        print("Γ£à [PRO-MODE] Evaluation Complete.")

        # Normalize and enrich questions with marks reduced and genuine reason
        if isinstance(eval_data.get("questions"), list):
            eval_data["questions"] = [_normalize_question_obj(q) for q in eval_data["questions"]]

        # ΓöÇΓöÇ CRITICAL FIX: Override totals with authoritative values ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        # The AI may return wrong total_possible. Always use the paper's declared marks.
        if req.total_marks > 0:
            counted_awarded = sum(
                float(q.get("marks_awarded", 0))
                for q in eval_data.get("questions", [])
                if q.get("is_counted", True)
            )
            # Clamp ΓÇö can't score more than the paper total
            counted_awarded = min(counted_awarded, float(req.total_marks))
            total_reduced = max(0.0, round(float(req.total_marks) - counted_awarded, 1))
            eval_data["total_awarded"] = counted_awarded
            eval_data["total_possible"] = float(req.total_marks)
            eval_data["total_reduced"] = total_reduced
            pct = round((counted_awarded / req.total_marks) * 100, 1) if req.total_marks > 0 else 0
            eval_data["percentage"] = pct
            grade = "F"
            if pct >= 90:   grade = "A+"
            elif pct >= 80: grade = "A"
            elif pct >= 70: grade = "B"
            elif pct >= 60: grade = "C"
            elif pct >= 50: grade = "D"
            eval_data["grade"] = grade
            print(f"≡ƒôè [PRO-MODE] Score: {counted_awarded}/{req.total_marks} (Reduced: {total_reduced}) = {pct}% Grade {grade}")

        # --- 3. Annotation Stage ---
        print("≡ƒûî∩╕Å [PRO-MODE] Drawing Red Pen annotations...")
        annotated_images = []

        # Group questions by page
        page_to_qs = {}
        for q in eval_data.get("questions", []):
            p = int(q.get("page_no", 1)) - 1
            if p not in page_to_qs:
                page_to_qs[p] = []
            page_to_qs[p].append(q)

        for i, img in enumerate(images):
            try:
                # Ensure annotation images are within safe size limits
                MAX_ANNOTATE_PIXELS = 15_000_000  # 15MP is plenty for readable annotations
                ann_w, ann_h = img.width, img.height
                if ann_w * ann_h > MAX_ANNOTATE_PIXELS:
                    scale = (MAX_ANNOTATE_PIXELS / (ann_w * ann_h)) ** 0.5
                    img = img.resize((int(ann_w * scale), int(ann_h * scale)), Image.LANCZOS)

                # Annotate only if there are evaluations for this page
                if i in page_to_qs:
                    page_eval = {"questions": page_to_qs[i]}
                    ann_img = pro_eval.annotate_page(img, page_eval, page_detections_map.get(i, []))
                    annotated_images.append(ann_img)
                else:
                    annotated_images.append(img)  # Unchanged
            except Exception as ann_err:
                print(f"ΓÜá∩╕Å [PRO-MODE] Annotation failed for page {i+1}: {ann_err}. Using original image.")
                annotated_images.append(img.convert("RGB") if img.mode != "RGB" else img)

        # --- 4. Report Generation ---
        report_path = pro_eval.generate_professional_report(annotated_images, eval_data)

        # Generate unique filename
        filename = f"pro_report_{uuid.uuid4().hex[:6]}.pdf"
        os.makedirs("pro_reports", exist_ok=True)
        os.replace(report_path, os.path.join("pro_reports", filename))

        print(f"≡ƒôä [PRO-MODE] Report ready: {filename}")

        return {
            "status": "success",
            "evaluation": eval_data,
            "pdf_url": f"/api/download-report/{filename}"
        }

    except Exception as e:
        print(f"Γ¥î [PRO-MODE] Error: {e}")
        return {"status": "error", "message": str(e)}

# ----------------- EXISTING EDUSAHAYAK ENDPOINTS -----------------

@app.post("/api/compare-ocr")
async def compare_ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        engine = get_comparer()
        results = engine.run_all(contents)
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

from openai import OpenAI

class EnhanceRequest(BaseModel):
    ocr_text: str

@app.post("/api/enhance-text")
async def enhance_text(req: EnhanceRequest):
    prompt = f"""
    You are an AI tasked with cleaning up noisy OCR text from a handwritten answer sheet.
    Correct spelling and grammar mistakes that look like typical OCR errors.
    Preserve structural meaning. Return only the cleaned up text.

    Raw OCR Text:
    {req.ocr_text}
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {"status": "error", "message": "GROQ_API_KEY not set"}

        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000
        )
        return {"status": "success", "enhanced_text": response.choices[0].message.content}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class PDFRequest(BaseModel):
    text: str

@app.post("/api/generate-pdf")
async def generate_pdf(req: PDFRequest):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        safe_text = req.text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, text=safe_text)
        output_path = "output.pdf"
        pdf.output(output_path)
        return {"status": "success", "message": "Multi-page PDF Generated", "file": output_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ----------------- NEW BLOOM'S TAXONOMY ALIGNMENT CHECKER ENDPOINTS -----------------

class BloomTextAnalyzeRequest(BaseModel):
    text: str
    model: Optional[str] = None
    subject: Optional[str] = None
    level: Optional[str] = None

@app.post("/api/bloom/analyze", response_model=BloomAnalysisResponse)
async def analyze_bloom_json(req: BloomTextAnalyzeRequest):
    """
    Endpoint for analyzing plain text question papers according to Revised Bloom's Taxonomy.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Please provide a question paper before analyzing.")

    try:
        response = analyze_question_paper_text(
            raw_text=req.text,
            model=req.model,
            subject=req.subject,
            academic_level=req.level
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/bloom/analyze-upload", response_model=BloomAnalysisResponse)
async def analyze_bloom_upload(
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    text: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    level: Optional[str] = Form(None)
):
    """
    Unified endpoint accepting:
    - Text from form
    - Single or multiple files (PDF, PNG, JPG, JPEG, WEBP, TIFF, BMP)
    """
    extracted_text = ""
    parser = get_doc_parser()

    # 1. Plain text provided in form
    if text and text.strip():
        extracted_text = text.strip()

    # 2. Files provided (either via `file` or `files`)
    all_uploads: List[UploadFile] = []
    if file and file.filename:
        all_uploads.append(file)
    if files:
        for f in files:
            if f and f.filename:
                all_uploads.append(f)

    if all_uploads:
        image_bytes_list: List[bytes] = []

        for upload in all_uploads:
            filename = upload.filename or ""
            ext = os.path.splitext(filename)[1].lower()

            content = await upload.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {filename} exceeds maximum allowed size of {MAX_FILE_SIZE // (1024*1024)}MB."
                )

            if ext == ".pdf":
                try:
                    pdf_text = parser.extract_text_from_pdf_bytes(content)
                    if pdf_text:
                        extracted_text += ("\n\n" if extracted_text else "") + pdf_text
                except Exception as e:
                    raise HTTPException(status_code=422, detail=f"Error reading PDF '{filename}': {str(e)}")

            elif ext in SUPPORTED_IMAGE_EXTS:
                image_bytes_list.append(content)

            elif ext in {'.txt', '.md'}:
                try:
                    decoded = content.decode('utf-8', errors='replace')
                    extracted_text += ("\n\n" if extracted_text else "") + decoded
                except Exception as e:
                    raise HTTPException(status_code=422, detail=f"Error reading text file '{filename}': {str(e)}")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format '{ext}'. Please upload a PDF, TXT, or supported image (PNG, JPG, WEBP, BMP, TIFF)."
                )

        if image_bytes_list:
            try:
                ocr_text = parser.extract_text_from_multiple_images(image_bytes_list)
                if ocr_text:
                    extracted_text += ("\n\n" if extracted_text else "") + ocr_text
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"OCR extraction failed: {str(e)}")

    if not extracted_text or not extracted_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide a question paper before analyzing. No readable text could be extracted."
        )

    try:
        print("\n========== RAW EXTRACTED PDF TEXT ==========", flush=True)
        print(extracted_text, flush=True)
        print("========== END RAW EXTRACTED TEXT ==========\n", flush=True)
        response = analyze_question_paper_text(
            raw_text=extracted_text,
            model=model,
            subject=subject,
            academic_level=level
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ----------------- STATIC FRONTEND ASSETS -----------------
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>EduSahayak / ShikshakAI API is running.</h2>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





@app.get("/api/download-report/{filename}")
async def download_pro_report(filename: str):
    import re

    if not re.fullmatch(r"pro_report_[a-f0-9]{6}\.pdf", filename):
        raise HTTPException(status_code=404, detail="Report not found")

    report_path = os.path.join("pro_reports", filename)

    if not os.path.isfile(report_path):
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename="ShikshakAI_Pro_Report.pdf"
    )
