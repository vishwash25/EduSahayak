# EduSahayak
### AI-Assisted Academic Assessment and Evaluation Platform

EduSahayak is an academic assessment platform designed to assist educators in analysing question papers, processing student answer scripts, evaluating responses and examining alignment with Revised Bloom's Taxonomy.

The project combines a Next.js web application with a Python FastAPI backend to provide document processing, OCR-assisted workflows, AI-supported evaluation and report generation.

---

## Key Features

- **Question Paper Analysis:** Analyse question papers and identify their structure, questions and subquestions.
- **Bloom's Taxonomy Analysis:** Examine cognitive levels using the Revised Bloom's Taxonomy framework.
- **OCR Processing:** Extract text from scanned academic documents and images using configured OCR engines.
- **Answer Evaluation:** Evaluate student responses against questions and expected answers.
- **Single Question Evaluation:** Evaluate individual answers independently.
- **Pro Mode:** Support advanced answer-script evaluation workflows.
- **OCR Comparison:** Compare OCR output and support text enhancement.
- **PDF Reports:** Generate and download evaluation reports.
- **Academic Document Processing:** Work with supported PDF and image-based inputs.

*Available features depend on the configured AI providers, OCR engines and application settings.*

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | Next.js 16, React 19, TypeScript |
| Styling | Tailwind CSS |
| Backend | Python, FastAPI |
| AI integration | Groq API, Gemini API; additional provider integrations in applicable frontend routes |
| OCR | Tesseract, EasyOCR, PaddleOCR |
| Document processing | PyMuPDF, pypdfium2, Pillow, OpenCV |
| PDF generation | FPDF2 |
| Database integration | Supabase client integration |

---

## Project Architecture

```text
EduSahayak/
│
├── README.md
│
├── shikshak-ai/                 # Next.js frontend
│   ├── app/
│   │   ├── api/                 # Frontend API routes
│   │   ├── components/
│   │   ├── views/
│   │   ├── login/
│   │   └── page.tsx
│   ├── components/
│   ├── lib/
│   ├── public/
│   ├── utils/
│   ├── package.json
│   └── .env.example
│
└── ocr-backend/                 # Python FastAPI backend
    ├── main.py
    ├── bloom_analyzer.py
    ├── evaluator_pro.py
    ├── ocr_engine.py
    ├── document_parser.py
    ├── requirements.txt
    ├── Dockerfile
    ├── .env.example
    └── static/
```

---

## Backend API Endpoints

The FastAPI backend currently exposes the following principal endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/evaluate` | POST | Standard answer evaluation |
| `/api/evaluate-single` | POST | Individual question evaluation |
| `/api/evaluate-pro` | POST | Advanced evaluation workflow |
| `/api/compare-ocr` | POST | OCR comparison |
| `/api/enhance-text` | POST | Text enhancement |
| `/api/generate-pdf` | POST | Generate PDF reports |
| `/api/bloom/analyze` | POST | Analyse question paper text |
| `/api/bloom/analyze-upload` | POST | Analyse uploaded question papers |
| `/api/download-report/{filename}` | GET | Download generated reports |

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

---

## Installation and Setup

### Prerequisites

Install the following:

- Node.js and npm
- Python 3.12 (recommended for the current development environment)
- Git
- Required OCR engines and their runtime dependencies
- API credentials for the AI providers used by your selected features

### 1. Clone the repository

```bash
git clone https://github.com/vishwash25/EduSahayak.git
cd EduSahayak
```

### 2. Configure the backend

```bash
cd ocr-backend

python -m venv venv
```

Activate the virtual environment.

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the backend environment file:

```powershell
Copy-Item .env.example .env
```

Add your actual API credentials to `ocr-backend/.env`.

Start the backend:

```bash
python main.py
```

The backend runs on:

`http://127.0.0.1:8000`

### 3. Configure the frontend

Open a second terminal:

```bash
cd shikshak-ai
```

Install dependencies:

```bash
npm ci
```

Create the environment file:

```powershell
Copy-Item .env.example .env.local
```

Configure the required provider keys and backend URL in `.env.local`.

### 4. Start the frontend

```bash
npm run dev
```

Open:

`http://localhost:3000`

Keep both the frontend and backend terminals running during local development.

---

## Environment Variables

### Backend

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API authentication |
| `GEMINI_API_KEY` | Gemini API authentication |
| `GROQ_MODEL` | Optional Groq model override |

### Frontend

| Variable | Description |
|---|---|
| `PYTHON_BACKEND_URL` | FastAPI backend URL |
| `GROQ_API_KEY` | Groq integration for applicable routes |
| `GEMINI_API_KEY` | Gemini integration for applicable routes |
| `ANTHROPIC_API_KEY` | Anthropic integration for applicable routes |
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase public client key |

Never commit real API keys, `.env` files or other credentials to GitHub.

---

## Development Commands

Run these commands from the `shikshak-ai` directory:

```bash
npm run dev
```

Create a production build:

```bash
npm run build
```

Run the production server:

```bash
npm run start
```

Run lint checks:

```bash
npm run lint
```

---

## Academic Project Information

**Institution:** Nitte Meenakshi Institute of Technology, Bengaluru

**Department:** Information Science and Engineering

**Project:** EduSahayak

**Team:** Team 18

| Team Member | USN |
|---|---|
| Prem Piyush | 1NT23IS165 |
| Ayush Sikriwal | 1NT23IS041 |
| Rishi Raj | 1NT23IS180 |
| Vishwash Srivastava | 1NT23IS250 |

**Project Guide:** Dr. Mohan S. G. — Head of Department

**Co-Guide:** Dr. N. Jayanthi — Assistant Professor

---

## Future Scope

Potential areas for further development include:

- More comprehensive academic performance analytics.
- Improved OCR accuracy for handwritten answer scripts.
- Expanded question-paper and curriculum analysis.
- Additional evaluation and reporting capabilities.
- Further integration with academic information systems.

---

## Disclaimer

EduSahayak is an academic assistance and evaluation project. AI-generated assessments should be reviewed by educators before being used for official academic decisions.

---

**EduSahayak — Making academic assessment more intelligent and efficient.**
