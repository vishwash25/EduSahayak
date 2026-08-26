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
            model=req.model
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
    model: Optional[str] = Form(None)
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
        response = analyze_question_paper_text(
            raw_text=extracted_text,
            model=model
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
