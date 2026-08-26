# EduSahayak - Bloom's Taxonomy Alignment Checker

EduSahayak (ShikshakAI) is an academic assessment suite designed to assist educators in analyzing question papers, verifying cognitive taxonomy alignment, and processing academic documents.

This repository features the **Bloom's Taxonomy Alignment Checker**, which analyzes university examination question papers to verify whether the Bloom's Taxonomy level already assigned to each question matches its true cognitive demand according to the **Revised Bloom's Taxonomy (Anderson & Krathwohl, 2001)**.

---

## 1. What the Bloom's Feature Does

- **Question Paper Structure Detection**: Distinguishes between examination metadata (university name, department, semester, instructions, footers, page numbers) and actual questions/subquestions.
- **Hierarchy Preservation**: Preserves question grouping and sub-question nesting (e.g., `Q1` &rarr; `Q1(a)`, `Q1(b)` &rarr; `i`, `ii`).
- **Assigned Bloom's Level Extraction**: Automatically detects annotated tags in formats such as `[BL-3]`, `(BL-2)`, `Bloom's Level 1`, `Level 4`, `CO1/BL2`, etc.
- **Cognitive Demand Assessment**: Leverages Groq LLM reasoning (LLaMA 3.3) to independently evaluate the true cognitive depth of each question, avoiding simplistic keyword/verb-matching pitfalls.
- **Alignment Verification**: Compares the assigned level with the AI assessment to report clear match statuses:
  - `MATCH`: Assigned level matches cognitive demand.
  - `MISMATCH`: Assigned level differs from the actual cognitive depth.
  - `NOT_SPECIFIED`: No Bloom level tag was annotated in the paper.
  - `AMBIGUOUS`: Question is too brief or context-dependent to evaluate reliably.
- **Summary & Detailed Rationale**: Generates concise summary metrics (total, matches, mismatches, unspecified, ambiguous) and observable academic justifications for each question.

---

## 2. Supported Inputs

1. **Paste Plain Text**: Complete question paper text with questions, subquestions, and optional marks/Bloom tags.
2. **Upload PDF**: Both digital text PDFs and scanned document PDFs (via automated high-resolution OCR rendering).
3. **Upload Images**: Single or multi-page photographed/scanned question papers (PNG, JPG, JPEG, WEBP, TIFF, BMP).

---

## 3. Revised Bloom's Taxonomy Reference

| Level | Cognitive Process | Definition & Scope |
|:-----:|:------------------|:-------------------|
| **1** | **Remember** | Retrieving relevant knowledge from memory (recalling definitions, terms, facts without transformation). |
| **2** | **Understand** | Constructing meaning from instructional messages (explaining concepts, summarizing, comparing mechanisms). |
| **3** | **Apply** | Executing a standard procedure or implementing algorithms/calculations in a concrete scenario. |
| **4** | **Analyze** | Breaking material into parts, determining relationships, analyzing time/space complexity bottlenecks. |
| **5** | **Evaluate** | Making judgments based on criteria and standards, defending technical decisions, assessing trade-offs. |
| **6** | **Create** | Synthesizing components into a novel whole, designing scalable architectures, formulating original algorithms. |

---

## 4. Backend Architecture & Groq Configuration

- **Backend Framework**: FastAPI (Python 3.12)
- **AI Provider**: Groq OpenAI-compatible API (`https://api.groq.com/openai/v1`)
- **Default Model**: `openai/gpt-oss-120b` (configurable via environment variable)

### Environment Variables

Create a `.env` file inside `ocr-backend/` (or set environment variables):

```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

> **Security Note**: `GROQ_API_KEY` is loaded strictly on the backend and is never exposed to the frontend/client.

---

## 5. API Endpoints

### 1. `POST /api/bloom/analyze`
Accepts a JSON payload with plain text.

**Request Body**:
```json
{
  "text": "Q1. What is OOP? (2 Marks) [BL-3]\nQ2. Design a banking system using OOP principles. (10 Marks) [BL-6]"
}
```

### 2. `POST /api/bloom/analyze-upload`
Accepts `multipart/form-data` for file uploads (PDF, images) or form text.

**Form Parameters**:
- `file`: (Optional) Single PDF or image file.
- `files`: (Optional) Multiple image files for multi-page papers.
- `text`: (Optional) Question paper text.

### Expected Response Structure

```json
{
  "status": "success",
  "summary": {
    "totalQuestions": 2,
    "matches": 1,
    "mismatches": 1,
    "notSpecified": 0,
    "ambiguous": 0
  },
  "questions": [
    {
      "questionNumber": "Q1",
      "subQuestionNumber": null,
      "fullIdentifier": "Q1",
      "parentId": null,
      "questionText": "What is OOP?",
      "rawText": "Q1. What is OOP? (2 Marks) [BL-3]",
      "marks": 2,
      "assignedBloomLevel": 3,
      "assignedBloomName": "Apply",
      "aiBloomLevel": 2,
      "aiBloomName": "Understand",
      "status": "MISMATCH",
      "confidence": 0.94,
      "reason": "The question tests conceptual understanding of object-oriented concepts rather than requiring the student to apply a specific procedure."
    },
    {
      "questionNumber": "Q2",
      "subQuestionNumber": null,
      "fullIdentifier": "Q2",
      "parentId": null,
      "questionText": "Design a banking system using OOP principles.",
      "rawText": "Q2. Design a banking system using OOP principles. (10 Marks) [BL-6]",
      "marks": 10,
      "assignedBloomLevel": 6,
      "assignedBloomName": "Create",
      "aiBloomLevel": 6,
      "aiBloomName": "Create",
      "status": "MATCH",
      "confidence": 0.96,
      "reason": "Requires synthesizing components to design an original, multi-entity software architecture."
    }
  ],
  "modelUsed": "openai/gpt-oss-120b"
}
```

---

## 6. How to Run and Test

### Starting the Server
Navigate to the `ocr-backend` directory and run the FastAPI server:

```powershell
cd ocr-backend
.\venv\Scripts\python.exe main.py
```
Or with uvicorn:
```powershell
.\venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
`http://localhost:8000`

### Running the Test Suite
The project includes automated unit and integration tests covering:
- Plain text classification
- Subquestions & hierarchy extraction
- Header, instruction, and footer filtering
- PDF extraction
- Missing Bloom's level handling
- Mismatch detection
- Ambiguous question handling
- FastAPI endpoint responses

Run the test suite:

```powershell
cd ocr-backend
.\venv\Scripts\python.exe test_bloom.py
.\venv\Scripts\python.exe test_api_endpoints.py
```
