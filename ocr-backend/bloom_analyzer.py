import os
import re
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Ensure .env is reliably loaded from ocr-backend directory or project root
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_env_backend = os.path.join(_curr_dir, ".env")
_env_root = os.path.join(os.path.dirname(_curr_dir), ".env")

if os.path.exists(_env_backend):
    load_dotenv(dotenv_path=_env_backend)
elif os.path.exists(_env_root):
    load_dotenv(dotenv_path=_env_root)
else:
    load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard Revised Bloom's Taxonomy Level Mapping
BLOOM_LEVELS: Dict[int, str] = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create"
}

BLOOM_NAME_TO_LEVEL: Dict[str, int] = {
    "remember": 1,
    "remembering": 1,
    "knowledge": 1,
    "recall": 1,
    "understand": 2,
    "understanding": 2,
    "comprehension": 2,
    "apply": 3,
    "applying": 3,
    "application": 3,
    "analyze": 4,
    "analyzing": 4,
    "analysis": 4,
    "evaluate": 5,
    "evaluating": 5,
    "evaluation": 5,
    "create": 6,
    "creating": 6,
    "creation": 6,
    "synthesis": 6
}

# Regex patterns for Bloom's level extraction
BLOOM_PATTERNS = [
    re.compile(r'\[\s*(?:(?:CO\s*[-:\s]?\s*\d+[\s,/]+)?(?:BL|BLOOM|BLOOM\'?S?\s*LEVEL|LEVEL|B\.L\.)\s*[-:\s]?\s*([1-6])(?:\s*[,/]\s*CO\s*[-:\s]?\s*\d+)?)\s*\]', re.IGNORECASE),
    re.compile(r'\(\s*(?:(?:CO\s*[-:\s]?\s*\d+[\s,/]+)?(?:BL|BLOOM|BLOOM\'?S?\s*LEVEL|LEVEL|B\.L\.)\s*[-:\s]?\s*([1-6])(?:\s*[,/]\s*CO\s*[-:\s]?\s*\d+)?)\s*\)', re.IGNORECASE),
    re.compile(r'(?:\b(?:BL|BLOOM|BLOOM\'?S?\s*LEVEL|LEVEL|B\.L\.)\s*[-:\s]?\s*([1-6]))\b', re.IGNORECASE),
    re.compile(r'\[\s*(Remember(?:ing)?|Understand(?:ing)?|Apply(?:ing)?|Analyze|Analyzing|Analysis|Evaluate|Evaluating|Evaluation|Create|Creating|Creation)\s*\]', re.IGNORECASE),
    re.compile(r'\(\s*(Remember(?:ing)?|Understand(?:ing)?|Apply(?:ing)?|Analyze|Analyzing|Analysis|Evaluate|Evaluating|Evaluation|Create|Creating|Creation)\s*\)', re.IGNORECASE),
]

# Regex patterns for Marks extraction
MARKS_PATTERNS = [
    re.compile(r'\[\s*(\d{1,3})\s*(?:Marks?|M|marks?|pts?)\s*\]', re.IGNORECASE),
    re.compile(r'\(\s*(\d{1,3})\s*(?:Marks?|M|marks?|pts?)\s*\)', re.IGNORECASE),
    re.compile(r'\b(\d{1,3})\s*(?:Marks?|marks?|pts?)\b', re.IGNORECASE),
    re.compile(r'\[\s*(\d{1,3})\s*\]'),
]

# Header line keywords with strict word boundaries
HEADER_LINE_KEYWORDS = [
    r'\bUNIVERSITY\b', r'\bCOLLEGE\b', r'\bINSTITUTE\b', r'\bACADEMY\b',
    r'\bDEPARTMENT\b', r'\bEXAMINATION\b', r'\bSEMESTER\b',
    r'\bDEGREE\b', r'\bB\.?E\.?\b', r'\bB\.?TECH\b', r'\bM\.?TECH\b', r'\bB\.?SC\b',
    r'\bBRANCH\b', r'\bPROGRAMME\b', r'\bSUBJECT\b', r'\bCOURSE\b',
    r'\bMODULE\s*[-–:\d]+', r'\bTIME\s*:\s*\d', r'\bMAX(?:IMUM)?\s*MARKS\b',
    r'\bTOTAL\s*MARKS\b', r'\bDURATION\s*:', r'\bDATE\s*:', r'\bUSN\s*:',
    r'\bROLL\s*NO\b', r'\bREG(?:ISTRATION)?\s*NO\b',
    r'\bEND\s*TERM\b', r'\bMID\s*TERM\b', r'\bFINAL\s*EXAM\b'
]
HEADER_LINE_REGEX = re.compile('|'.join(HEADER_LINE_KEYWORDS), re.IGNORECASE)

INSTRUCTION_HEADER_REGEX = re.compile(
    r'^\s*(?:INSTRUCTIONS?(?:\s+TO\s+CANDIDATES)?|NOTE\s*:|GENERAL\s+INSTRUCTIONS|IMPORTANT\s+NOTE)',
    re.IGNORECASE
)

INSTRUCTION_LINE_REGEX = re.compile(
    r'(?:Answer\s+(?:all|any|the following|each|questions)|Assume\s+suitable\s+data|Draw\s+neat\s+(?:diagrams|sketches)|All\s+questions\s+carry\s+equal\s+marks|Use\s+of\s+(?:steam|normal|scientific|tables|calculator|charts)|(?:Scientific\s+)?Calculator\s+is\s+(?:permitted|allowed|prohibited)|Non-programmable\s+calculator|Candidates\s+should\s+verify|Write\s+your\s+USN|Missing\s+data\s+may\s+be\s+suitably\s+assumed)',
    re.IGNORECASE
)

FOOTER_REGEX = re.compile(
    r'(?:^\s*Page\s*\d+\s*(?:of\s*\d+|\/\s*\d+)?\s*$|\*\*\*\s*END\s*OF\s*QUESTION\s*PAPER\s*\*\*\*|---.*END.*---|^\s*Department\s+of\b)',
    re.IGNORECASE
)

SECTION_REGEX = re.compile(
    r'^\s*(?:SECTION|PART|MODULE|UNIT|GROUP)\s*[-–:\s]*\s*([A-Z0-9]+)\s*$',
    re.IGNORECASE
)

# Question matchers
MAIN_Q_START_REGEX = re.compile(
    r'^\s*(?:(?:Question|Q)\.?\s*(\d+)|(\d+))\s*[\.:\)-]\s*(.*)$',
    re.IGNORECASE
)
STANDALONE_Q_NUM_REGEX = re.compile(
    r'^\s*(?:Question|Q)\.?\s*(\d+)\s*$',
    re.IGNORECASE
)
SUB_Q_PREFIX_REGEX = re.compile(
    r'^\s*(?:(?:Question|Q)\.?\s*\d+\s*[\.:\-]?\s*)?(?:\(([a-zA-Z]|[ivxIVX]{1,5})\)|([a-zA-Z]|[ivxIVX]{1,5})\)|\b([a-zA-Z]|[ivxIVX]{1,5})\.(?=\s+[A-Za-z]))\s*[\.:\-]?\s*(.*)$'
)

CODE_LINE_INDICATORS = re.compile(
    r'(?:;\s*$|{\s*$|^\s*}\s*$|^\s*@\w+|^\s*(?:public|private|protected|static|final|abstract|class|interface|enum|record|struct|void|int|float|double|boolean|bool|char|String|string|auto|def|fn|function|import|export|package|using|include|return|if|else|while|for|switch|case|try|catch|finally|SELECT|FROM|WHERE|INSERT|UPDATE|DELETE)\b|=>|->|::|==|!=|\+\+|--|\bnew\s+\w+)',
    re.IGNORECASE
)

class QuestionItem(BaseModel):
    questionNumber: str
    subQuestionNumber: Optional[str] = None
    fullIdentifier: str
    parentId: Optional[str] = None
    questionText: str
    rawText: Optional[str] = None
    marks: Optional[int] = None
    assignedBloomLevel: Optional[int] = None
    assignedBloomName: Optional[str] = None
    aiBloomLevel: Optional[int] = None
    aiBloomName: Optional[str] = None
    status: str = Field(default="NOT_SPECIFIED", description="MATCH | MISMATCH | NOT_SPECIFIED | AMBIGUOUS")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    reason: str = ""

class BloomSummary(BaseModel):
    totalQuestions: int
    matches: int
    mismatches: int
    notSpecified: int
    ambiguous: int

class BloomAnalysisResponse(BaseModel):
    status: str
    summary: BloomSummary
    questions: List[QuestionItem]
    modelUsed: Optional[str] = None
    note: Optional[str] = None


def normalize_question_id(qid: str) -> str:
    """
    Normalizes question identifiers for flexible cross-matching.
    Examples: 'Q1' -> '1', 'q1' -> '1', 'Q2(a)' -> '2(a)', '2.a' -> '2a'.
    """
    if not qid:
        return ""
    s = str(qid).strip().lower()
    s = re.sub(r'^(?:question|q)\.?\s*', '', s)
    s = re.sub(r'[\s\.\:\-\_]', '', s)
    return s


def parse_groq_json_response(content: str) -> List[Dict[str, Any]]:
    """
    Robustly parses raw Groq response content into a list of question analysis dictionaries.
    Handles markdown-fenced code blocks (```json ... ```) and direct JSON arrays/objects.
    """
    if not content or not content.strip():
        raise ValueError("Groq returned an empty response.")

    cleaned = content.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Groq AI response as JSON: {str(e)}. Raw content: {cleaned[:200]}")

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "analysis" in data and isinstance(data["analysis"], list):
            return data["analysis"]
        elif "questions" in data and isinstance(data["questions"], list):
            return data["questions"]
        elif "results" in data and isinstance(data["results"], list):
            return data["results"]
        else:
            raise ValueError(f"Groq response JSON dictionary does not contain 'analysis' or 'questions' list. Available keys: {list(data.keys())}")
    else:
        raise ValueError(f"Unexpected Groq response structure: expected JSON object or array, got {type(data).__name__}")


def extract_bloom_and_marks(line: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Extracts assigned Bloom's level and marks from a question line,
    returning (cleaned_text, assigned_bloom_level, marks).
    """
    assigned_bl = None
    marks = None
    cleaned = line

    # 1. Search Bloom's level
    for pattern in BLOOM_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            val = match.group(1).strip()
            if val.isdigit() and int(val) in BLOOM_LEVELS:
                assigned_bl = int(val)
            elif val.lower() in BLOOM_NAME_TO_LEVEL:
                assigned_bl = BLOOM_NAME_TO_LEVEL[val.lower()]

            cleaned = cleaned[:match.start()] + " " + cleaned[match.end():]
            break

    # 2. Search Marks
    for pattern in MARKS_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            try:
                m_val = int(match.group(1).strip())
                if 1 <= m_val <= 100:
                    marks = m_val
                    cleaned = cleaned[:match.start()] + " " + cleaned[match.end():]
                    break
            except Exception:
                pass

    # Only clean empty square brackets, never strip valid parentheses () from code
    cleaned = re.sub(r'\[\s*\]', '', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned, assigned_bl, marks


def parse_question_paper_structure(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parses questions and subquestions while ignoring headers, footers, instructions,
    and university exam metadata. Preserves hierarchy (e.g. Q1 -> Q1(a), Q1(b)).
    """
    lines = [ln.strip() for ln in raw_text.split('\n') if ln.strip()]
    parsed_items: List[Dict[str, Any]] = []

    current_main_num: Optional[str] = None
    current_sub_num: Optional[str] = None
    current_item: Optional[Dict[str, Any]] = None
    has_seen_first_question: bool = False
    in_instruction_block: bool = False

    def commit_current():
        nonlocal current_item
        if current_item:
            txt = current_item.get("questionText", "").strip()
            clean_txt, bl, m = extract_bloom_and_marks(txt)
            clean_txt = re.sub(r'^[\.:\-\)\s]+', '', clean_txt).strip()
            if bl and not current_item.get("assignedBloomLevel"):
                current_item["assignedBloomLevel"] = bl
                current_item["assignedBloomName"] = BLOOM_LEVELS.get(bl)
            if m and not current_item.get("marks"):
                current_item["marks"] = m
            current_item["questionText"] = clean_txt

            if clean_txt and len(clean_txt) >= 2:
                parsed_items.append(current_item)
            current_item = None

    for line in lines:
        # 1. Ignore Footers
        if FOOTER_REGEX.search(line):
            commit_current()
            continue

        # 2. Section Headers (e.g. PART A, SECTION B)
        if SECTION_REGEX.search(line):
            in_instruction_block = False
            commit_current()
            continue

        # 3. Instruction Header (e.g. INSTRUCTIONS TO CANDIDATES:, NOTE:)
        if INSTRUCTION_HEADER_REGEX.search(line):
            in_instruction_block = True
            commit_current()
            continue

        # 4. Instruction lines
        if INSTRUCTION_LINE_REGEX.search(line):
            continue

        # 5. If in instruction block, skip non-question lines or numbered instructions
        if in_instruction_block:
            if re.search(r'\bQ(?:uestion)?\.?\s*\d+', line, re.IGNORECASE):
                in_instruction_block = False
            else:
                continue

        # 6. If before the first question, check header regex
        if not has_seen_first_question and HEADER_LINE_REGEX.search(line):
            continue

        # 7. Check if line is a subquestion or main question
        sub_match = SUB_Q_PREFIX_REGEX.match(line)
        main_match = MAIN_Q_START_REGEX.match(line)
        standalone_q_match = STANDALONE_Q_NUM_REGEX.match(line)

        is_sub = False
        sub_id = None
        sub_rest = ""
        if sub_match:
            sub_id = sub_match.group(1) or sub_match.group(2) or sub_match.group(3)
            sub_rest = sub_match.group(4)
            if len(sub_id) <= 4 and not CODE_LINE_INDICATORS.search(line):
                is_sub = True

        # Pre-question filter: If no question marker has been seen yet, ignore any non-question lines
        if not has_seen_first_question:
            if main_match or standalone_q_match or (is_sub and current_main_num is not None):
                has_seen_first_question = True
            else:
                continue

        if is_sub and current_main_num is not None:
            commit_current()
            current_sub_num = sub_id
            full_id = f"Q{current_main_num}({current_sub_num})"
            clean_line, bl, marks = extract_bloom_and_marks(sub_rest)
            clean_line = re.sub(r'^[\.:\-\)\s]+', '', clean_line).strip()

            current_item = {
                "questionNumber": f"Q{current_main_num}",
                "subQuestionNumber": current_sub_num,
                "fullIdentifier": full_id,
                "parentId": f"Q{current_main_num}",
                "questionText": clean_line,
                "rawText": line,
                "marks": marks,
                "assignedBloomLevel": bl,
                "assignedBloomName": BLOOM_LEVELS.get(bl) if bl else None
            }
        elif main_match or standalone_q_match:
            commit_current()
            if main_match:
                q_num = main_match.group(1) or main_match.group(2)
                rest = main_match.group(3).strip()
            else:
                q_num = standalone_q_match.group(1)
                rest = ""

            current_main_num = q_num
            current_sub_num = None

            nested_sub = SUB_Q_PREFIX_REGEX.match(rest)
            if nested_sub and not CODE_LINE_INDICATORS.search(rest):
                nested_sub_id = nested_sub.group(1) or nested_sub.group(2) or nested_sub.group(3)
                nested_sub_rest = nested_sub.group(4)
                current_sub_num = nested_sub_id
                full_id = f"Q{current_main_num}({nested_sub_id})"
                clean_line, bl, marks = extract_bloom_and_marks(nested_sub_rest)
                clean_line = re.sub(r'^[\.:\-\)\s]+', '', clean_line).strip()
                current_item = {
                    "questionNumber": f"Q{current_main_num}",
                    "subQuestionNumber": nested_sub_id,
                    "fullIdentifier": full_id,
                    "parentId": f"Q{current_main_num}",
                    "questionText": clean_line,
                    "rawText": line,
                    "marks": marks,
                    "assignedBloomLevel": bl,
                    "assignedBloomName": BLOOM_LEVELS.get(bl) if bl else None
                }
            else:
                clean_line, bl, marks = extract_bloom_and_marks(rest)
                clean_line = re.sub(r'^[\.:\-\)\s]+', '', clean_line).strip()
                full_id = f"Q{current_main_num}"
                current_item = {
                    "questionNumber": f"Q{current_main_num}",
                    "subQuestionNumber": None,
                    "fullIdentifier": full_id,
                    "parentId": None,
                    "questionText": clean_line,
                    "rawText": line,
                    "marks": marks,
                    "assignedBloomLevel": bl,
                    "assignedBloomName": BLOOM_LEVELS.get(bl) if bl else None
                }
        else:
            # Continuation line of the active question
            if current_item:
                current_item["questionText"] += " " + line

    commit_current()

    # Fallback only if no questions were identified and text exists
    if not parsed_items and raw_text.strip():
        paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
        for i, p in enumerate(paragraphs, 1):
            if not (HEADER_LINE_REGEX.search(p) or INSTRUCTION_LINE_REGEX.search(p) or INSTRUCTION_HEADER_REGEX.search(p) or FOOTER_REGEX.search(p)):
                clean_p, bl, marks = extract_bloom_and_marks(p)
                clean_p = re.sub(r'^[\.:\-\)\s]+', '', clean_p).strip()
                if len(clean_p) >= 2:
                    parsed_items.append({
                        "questionNumber": f"Q{i}",
                        "subQuestionNumber": None,
                        "fullIdentifier": f"Q{i}",
                        "parentId": None,
                        "questionText": clean_p,
                        "rawText": p,
                        "marks": marks,
                        "assignedBloomLevel": bl,
                        "assignedBloomName": BLOOM_LEVELS.get(bl) if bl else None
                    })

    return parsed_items


SYSTEM_BLOOM_PROMPT = """You are an academic expert in educational assessment, curriculum design, and the Revised Bloom's Taxonomy (Anderson, Krathwohl, et al., 2001).
Your task is to analyze examination questions and determine their precise cognitive process level according to the Revised Bloom's Taxonomy.

THE 6 COGNITIVE PROCESS LEVELS OF REVISED BLOOM'S TAXONOMY:
1. Remember (Level 1):
   - Definition: Retrieving relevant knowledge from long-term memory.
   - Cognitive Processes: Recognizing, recalling, listing, stating definitions, identifying basic facts.
   - Key Characteristic: Requires purely reproductive recall without requiring the student to demonstrate conceptual comprehension, transformation, or procedural application.

2. Understand (Level 2):
   - Definition: Constructing meaning from instructional messages, including oral, written, and graphic communication.
   - Cognitive Processes: Interpreting, exemplifying, classifying, summarizing, inferring, comparing, explaining concepts, mechanisms, or principles.
   - Key Characteristic: Requires the student to demonstrate conceptual comprehension (e.g. explaining why/how something works conceptually, translating across representations, summarizing).

3. Apply (Level 3):
   - Definition: Carrying out or using a procedure in a given situation.
   - Cognitive Processes: Executing (routine procedure on familiar task), implementing (applying procedure/algorithm/formula to solve a concrete problem or scenario).
   - Key Characteristic: Student executes a concrete computational, algorithmic, or mathematical procedure or applies a standard rule/pattern to a given dataset or concrete problem.

4. Analyze (Level 4):
   - Definition: Breaking material into its constituent parts and determining how the parts relate to one another and to an overall structure or purpose.
   - Cognitive Processes: Differentiating, organizing, attributing, identifying root causes, analyzing time/space complexity bottlenecks, analyzing structural relationships, fault diagnosis.
   - Key Characteristic: Requires deconstructing a system, diagram, or algorithm, investigating internal mechanics/flaws, or contrasting multi-component relationships.

5. Evaluate (Level 5):
   - Definition: Making judgments based on criteria and standards.
   - Cognitive Processes: Checking, critiquing, assessing trade-offs, judging which architecture/algorithm/solution is optimal for given constraints and justifying that judgment with formal criteria.
   - Key Characteristic: Requires defending a technical stance, evaluating conflicting requirements, validating against strict criteria, or grading comparative alternatives.

6. Create (Level 6):
   - Definition: Putting elements together to form a novel, coherent, or functional whole; reorganizing elements into a new pattern or structure.
   - Cognitive Processes: Generating hypotheses, planning, designing novel architectures/schemas/systems, constructing full solutions from open-ended specifications.
   - Key Characteristic: Student synthesizes an original design, formulates a new algorithm, or builds an integrated architectural schema.

CRITICAL ACADEMIC EVALUATION RULES:
1. DO NOT classify questions solely by action verbs! Analyze the actual cognitive demand of the entire question in context.
2. If a question is ambiguous, missing vital context, or its cognitive demand cannot be reliably determined without unseen context, set `isAmbiguous: true`, provide the closest estimated level, and explain why in `reason`.
3. Output concise, objective, academically sound reasons (1-2 sentences) highlighting the specific cognitive demand required.
4. Classify ONLY the question entries supplied in the user message. Do NOT independently split, expand, merge, or invent questions or subquestions.
5. Return EXACTLY one analysis object for every supplied input object, in the same order.
6. Copy each input object's `id` EXACTLY into its corresponding output object. Never create or modify an ID.
7. Even if question text contains parts (a), (b), (i), (ii), classify the entire supplied entry as one question unless those parts are separate input objects.

Respond strictly with valid JSON conforming to this schema:
{
  "analysis": [
    {
      "id": "Q1",
      "aiBloomLevel": 2,
      "aiBloomName": "Understand",
      "confidence": 0.92,
      "reason": "The question requires students to explain the conceptual mechanism of dynamic binding rather than merely stating a memorized definition.",
      "isAmbiguous": false
    }
  ]
}
"""


def assess_questions_with_groq(
    questions: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """
    Calls Groq LLM to assess cognitive levels of the extracted questions.
    Raises RuntimeError / ValueError on missing key, API failure, or invalid response.
    Never silently fabricates Level 2 results.
    Returns (analyzed_questions, model_used, error_or_note).
    """
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    groq_model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # Strict check: Never silently fabricate classifications when GROQ_API_KEY is missing
    if not api_key or not api_key.strip():
        logger.error("GROQ_API_KEY is not configured in the active environment.")
        raise RuntimeError("GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your environment or .env file.")

    question_payload = [
        {
            "id": q.get("fullIdentifier") or q.get("questionNumber", f"Q{i+1}"),
            "questionText": q.get("questionText", ""),
            "marks": q.get("marks")
        }
        for i, q in enumerate(questions)
    ]

    user_prompt = f"""Analyze the following examination questions according to Revised Bloom's Taxonomy:

{json.dumps(question_payload, indent=2)}

Return a JSON object containing the "analysis" array with each question's evaluation."""

    try:
        logger.info(f"Groq classification request started: {len(question_payload)} questions sent to model '{groq_model}'.")
        client = OpenAI(
            api_key=api_key.strip(),
            base_url="https://api.groq.com/openai/v1",
            timeout=45.0
        )

        response = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_BLOOM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=4000
        )

        content = response.choices[0].message.content
        logger.info(f"Groq response received successfully from model '{groq_model}'.")

    except Exception as e:
        logger.error(f"Groq API call failed: {str(e)}")

        print("\n========== GROQ ERROR DIAGNOSTIC ==========", flush=True)
        print("ERROR TYPE:", type(e).__name__, flush=True)
        print("ERROR MESSAGE:", str(e), flush=True)

        response = getattr(e, "response", None)
        if response is not None:
            try:
                error_data = response.json()
                print("GROQ ERROR RESPONSE:", json.dumps(error_data, indent=2), flush=True)
                error_details = error_data.get("error", {})
                if isinstance(error_details, dict):
                    print("FAILED GENERATION:", error_details.get("failed_generation"), flush=True)
            except Exception:
                print("GROQ RESPONSE TEXT:", getattr(response, "text", ""), flush=True)

        print("========== END GROQ ERROR =================\n", flush=True)
        raise RuntimeError(f"Groq AI service unavailable or returned an error: {str(e)}")

    # Parse and validate response JSON structure
    analysis_list = parse_groq_json_response(content)
    logger.info(f"Successfully parsed {len(analysis_list)} question classifications from Groq response.")
    print("DEBUG GROQ COUNT:", len(analysis_list), flush=True)
    print("DEBUG GROQ IDS:", [
        item.get("id") or item.get("question_id") or
        item.get("questionNumber") or item.get("question_number")
        for item in analysis_list
    ], flush=True)
    print("DEBUG INPUT IDS:", [
        q.get("fullIdentifier") or q.get("questionNumber")
        for q in questions
    ], flush=True)

    logger.info("GROQ QUESTION IDS: %s", [item.get("id") for item in analysis_list])
    logger.info("GROQ RESPONSE FIELDS: %s", [list(item.keys()) for item in analysis_list])

    # Build flexible lookup using multiple possible ID field names.
    analysis_by_id: Dict[str, Dict[str, Any]] = {}

    for item in analysis_list:
        raw_id = str(
            item.get("id")
            or item.get("question_id")
            or item.get("questionId")
            or item.get("questionNumber")
            or item.get("question_number")
            or item.get("fullIdentifier")
            or ""
        ).strip()

        if raw_id:
            analysis_by_id[raw_id] = item
            norm_id = normalize_question_id(raw_id)
            if norm_id:
                analysis_by_id[norm_id] = item

    logger.info(
        "Groq classification count: %s; input question count: %s",
        len(analysis_list), len(questions)
    )

    enriched_questions = []

    for index, q in enumerate(questions):
        qid = q.get("fullIdentifier") or q.get("questionNumber") or ""
        norm_qid = normalize_question_id(qid)

        eval_item = analysis_by_id.get(qid) or analysis_by_id.get(norm_qid)

        # Match by order only when every input question has a response.
        if not eval_item and len(analysis_list) == len(questions):
            eval_item = analysis_list[index]
            logger.warning(
                "Question ID '%s' matched by response order.",
                qid
            )

        if not eval_item:
            raise RuntimeError(
                f"Groq AI response did not include a classification for question ID '{qid}'. "
                f"Received {len(analysis_list)} classifications for {len(questions)} questions."
            )

        ai_level_raw = eval_item.get("aiBloomLevel")
        if isinstance(ai_level_raw, str) and ai_level_raw.isdigit():
            ai_level = int(ai_level_raw)
        elif isinstance(ai_level_raw, str) and ai_level_raw.lower() in BLOOM_NAME_TO_LEVEL:
            ai_level = BLOOM_NAME_TO_LEVEL[ai_level_raw.lower()]
        elif isinstance(ai_level_raw, int) and ai_level_raw in BLOOM_LEVELS:
            ai_level = ai_level_raw
        else:
            raise RuntimeError(f"Groq AI returned an invalid or unrecognized Bloom level '{ai_level_raw}' for question ID '{qid}'.")

        ai_name = BLOOM_LEVELS.get(ai_level)
        if not ai_name:
            raise RuntimeError(f"Invalid Bloom level integer {ai_level} for question ID '{qid}'. Must be between 1 and 6.")

        confidence = float(eval_item.get("confidence", 0.90))
        reason = eval_item.get("reason", f"Assessed as Level {ai_level} ({ai_name}) based on cognitive demand.")
        is_ambiguous = bool(eval_item.get("isAmbiguous", False))

        assigned_bl = q.get("assignedBloomLevel")

        if is_ambiguous:
            status = "AMBIGUOUS"
        elif assigned_bl is None:
            status = "NOT_SPECIFIED"
        elif assigned_bl == ai_level:
            status = "MATCH"
        else:
            status = "MISMATCH"

        enriched_questions.append({
            **q,
            "aiBloomLevel": ai_level,
            "aiBloomName": ai_name,
            "status": status,
            "confidence": round(confidence, 2),
            "reason": reason
        })

    return enriched_questions, groq_model, None


def fallback_cognitive_assessment(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic rule-based cognitive analyzer.
    Used ONLY for explicit testing or standalone benchmarking.
    Never invoked as a silent substitute during normal live Bloom analysis.
    """
    enriched = []
    for q in questions:
        text = q.get("questionText", "").strip().lower()
        assigned_bl = q.get("assignedBloomLevel")

        ai_level = 2
        ai_name = "Understand"
        confidence = 0.85
        reason = "The question requires conceptual comprehension and explanation."
        is_ambiguous = False

        if len(text.split()) <= 2 or len(text) < 10:
            is_ambiguous = True
            reason = "Question is too brief or underspecified to definitively evaluate student cognitive demand without additional context."
        elif any(k in text for k in ["design a", "develop a system", "formulate a", "construct an architecture", "propose a novel", "build a comprehensive", "plan and implement a new", "design an object-oriented architecture"]):
            ai_level = 6
            ai_name = "Create"
            reason = "Requires synthesizing components to design an original system, model, or schema."
        elif any(k in text for k in ["justify whether", "evaluate whether", "critique", "which is more appropriate", "assess the trade-off", "validate the effectiveness", "justify your decision", "judge which"]):
            ai_level = 5
            ai_name = "Evaluate"
            reason = "Requires making evaluative judgments and justifying decisions against formal criteria."
        elif any(k in text for k in ["analyze the", "identify its time complexity", "identify the causes of", "worst-case time complexity", "break down", "differentiate between", "compare and contrast", "troubleshoot", "bottlenecks in"]):
            ai_level = 4
            ai_name = "Analyze"
            reason = "Requires deconstructing the concept into constituent parts and analyzing component relationships or complexity."
        elif any(k in text for k in ["apply ", "calculate ", "compute ", "solve the following", "using dijkstra", "find the shortest path", "trace the execution", "derive the equation for a given", "apply the factory design pattern"]):
            ai_level = 3
            ai_name = "Apply"
            reason = "Requires applying a standard procedural method, algorithm, or mathematical rule to a concrete scenario."
        elif re.match(r'^(?:define|list|state|name|what is meant by|write the formula for)\b', text) and len(text.split()) < 12 and "explain how" not in text and "justify" not in text:
            ai_level = 1
            ai_name = "Remember"
            reason = "Requires direct retrieval of facts or standard terminology from long-term memory without requiring procedural application."
        else:
            ai_level = 2
            ai_name = "Understand"
            reason = "Requires understanding and explaining core conceptual principles and underlying mechanisms."

        if is_ambiguous:
            status = "AMBIGUOUS"
        elif assigned_bl is None:
            status = "NOT_SPECIFIED"
        elif assigned_bl == ai_level:
            status = "MATCH"
        else:
            status = "MISMATCH"

        enriched.append({
            **q,
            "aiBloomLevel": ai_level,
            "aiBloomName": ai_name,
            "status": status,
            "confidence": confidence,
            "reason": reason
        })

    return enriched


def analyze_question_paper_text(
    raw_text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> BloomAnalysisResponse:
    """
    End-to-end pipeline:
    1. Parses structure (questions, subquestions, ignores headers/footers/metadata).
    2. Extracts assigned Bloom levels & marks.
    3. Runs AI cognitive analysis via Groq (raises error if Groq is unavailable).
    4. Compares assigned vs AI-assessed levels.
    5. Computes summary statistics.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Please provide a question paper before analyzing.")

    parsed_raw_questions = parse_question_paper_structure(raw_text)

    print("\n========== PARSED QUESTION DIAGNOSTIC ==========", flush=True)
    for index, question in enumerate(parsed_raw_questions, 1):
        print(
            f"ITEM {index} | ID: {question.get('fullIdentifier')} | "
            f"TEXT: {question.get('questionText', '')[:250]}",
            flush=True
        )
    print("========== END DIAGNOSTIC ======================\n", flush=True)

    if not parsed_raw_questions:
        raise ValueError("No analyzable questions found in the provided input. Please check the question paper format.")

    analyzed_questions, model_used, note = assess_questions_with_groq(
        parsed_raw_questions,
        api_key=api_key,
        model=model
    )

    total = len(analyzed_questions)
    matches = sum(1 for q in analyzed_questions if q["status"] == "MATCH")
    mismatches = sum(1 for q in analyzed_questions if q["status"] == "MISMATCH")
    not_specified = sum(1 for q in analyzed_questions if q["status"] == "NOT_SPECIFIED")
    ambiguous = sum(1 for q in analyzed_questions if q["status"] == "AMBIGUOUS")

    summary = BloomSummary(
        totalQuestions=total,
        matches=matches,
        mismatches=mismatches,
        notSpecified=not_specified,
        ambiguous=ambiguous
    )

    typed_questions = [QuestionItem(**q) for q in analyzed_questions]

    return BloomAnalysisResponse(
        status="success",
        summary=summary,
        questions=typed_questions,
        modelUsed=model_used,
        note=note
    )
