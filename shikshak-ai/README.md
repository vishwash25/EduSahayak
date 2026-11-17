# EduSahayak — Frontend

The web application for **EduSahayak**, an AI-assisted academic assessment and evaluation platform.

Built with Next.js, React and TypeScript, the frontend provides the interface for educators to analyse question papers, process student answer scripts, evaluate responses and explore Bloom's Taxonomy alignment.

## Technology Stack

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS
- Supabase client integration
- AI provider integrations through applicable API routes

## Project Structure

```text
shikshak-ai/
├── app/
│   ├── api/           # Frontend API routes
│   ├── components/    # Application components
│   ├── views/         # Application views
│   ├── login/         # Authentication interface
│   └── page.tsx       # Main application entry
├── components/
├── lib/
├── public/
├── utils/
├── package.json
└── .env.example
```

## Getting Started

### Prerequisites

- Node.js and npm
- Python backend configured separately
- Required API credentials

### 1. Install dependencies

```bash
npm ci
```

### 2. Configure environment variables

Create your local environment file:

```powershell
Copy-Item .env.example .env.local
```

Configure the required variables in `.env.local`, including the backend URL, AI provider credentials and Supabase configuration.

Never commit `.env.local` or real API credentials.

### 3. Start the application

Start the Python FastAPI backend separately from the `ocr-backend` directory.

Then run the frontend:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

Keep both servers running while using the application locally.

## Available Commands

| Command | Description |
|---|---|
| `npm run dev` | Start the development server |
| `npm run build` | Create a production build |
| `npm run start` | Start the production server |
| `npm run lint` | Run lint checks |

## Backend Integration

The frontend communicates with the Python FastAPI backend for document processing, OCR workflows, answer evaluation and related services.

The backend API documentation is available at:

`http://127.0.0.1:8000/docs`

For complete project architecture, backend setup, API endpoints and team information, see the [main project README](../README.md).

---

**EduSahayak — AI-Assisted Academic Assessment and Evaluation Platform.**
