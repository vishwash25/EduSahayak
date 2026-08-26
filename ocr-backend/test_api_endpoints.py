import os
import io
import json
import unittest
from unittest.mock import patch, MagicMock
import httpx
from PIL import Image
from fpdf import FPDF

from main import app

class TestApiEndpoints(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    def _mock_groq_success(self, items):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps({"analysis": items})))]
        return mock_resp

    async def test_get_root(self):
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)

    async def test_bloom_analyze_json(self):
        payload = {
            "text": "Q1. Define Polymorphism. (2 Marks) [BL-1]\nQ2. Design a microservices-based e-commerce platform. (10 Marks) [BL-6]"
        }
        mock_items = [
            {"id": "Q1", "aiBloomLevel": 1, "aiBloomName": "Remember", "confidence": 0.95, "reason": "Recall definition."},
            {"id": "Q2", "aiBloomLevel": 6, "aiBloomName": "Create", "confidence": 0.97, "reason": "Design architecture."}
        ]
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"}):
            with patch("bloom_analyzer.OpenAI") as mock_openai_cls:
                mock_client = MagicMock()
                mock_openai_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = self._mock_groq_success(mock_items)

                res = await self.client.post("/api/bloom/analyze", json=payload)
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "success")
                self.assertEqual(data["summary"]["totalQuestions"], 2)
                self.assertEqual(len(data["questions"]), 2)
                self.assertEqual(data["questions"][0]["fullIdentifier"], "Q1")
                self.assertEqual(data["questions"][0]["aiBloomLevel"], 1)
                self.assertEqual(data["questions"][1]["fullIdentifier"], "Q2")
                self.assertEqual(data["questions"][1]["aiBloomLevel"], 6)

    async def test_bloom_analyze_empty_json(self):
        payload = {"text": "   "}
        res = await self.client.post("/api/bloom/analyze", json=payload)
        self.assertEqual(res.status_code, 400)

    async def test_bloom_analyze_upload_form_text(self):
        mock_items = [
            {"id": "Q1", "aiBloomLevel": 2, "aiBloomName": "Understand", "confidence": 0.92, "reason": "Explain concept."}
        ]
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"}):
            with patch("bloom_analyzer.OpenAI") as mock_openai_cls:
                mock_client = MagicMock()
                mock_openai_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = self._mock_groq_success(mock_items)

                res = await self.client.post(
                    "/api/bloom/analyze-upload",
                    data={"text": "Q1. Explain virtual memory management. [BL-2] (5 Marks)"}
                )
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "success")
                self.assertEqual(data["summary"]["totalQuestions"], 1)
                self.assertEqual(data["questions"][0]["aiBloomLevel"], 2)

    async def test_bloom_analyze_upload_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 10, text="OPERATING SYSTEMS FINAL EXAMINATION", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text="Q1. Explain Round Robin CPU scheduling. [BL-2] (5 Marks)", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, text="Q2. Apply Banker's Algorithm to detect deadlock state. [BL-3] (10 Marks)", new_x="LMARGIN", new_y="NEXT")
        
        pdf_bytes = bytes(pdf.output())
        
        mock_items = [
            {"id": "Q1", "aiBloomLevel": 2, "aiBloomName": "Understand", "confidence": 0.93, "reason": "Explain scheduling."},
            {"id": "Q2", "aiBloomLevel": 3, "aiBloomName": "Apply", "confidence": 0.95, "reason": "Apply Banker's algorithm."}
        ]
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"}):
            with patch("bloom_analyzer.OpenAI") as mock_openai_cls:
                mock_client = MagicMock()
                mock_openai_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = self._mock_groq_success(mock_items)

                res = await self.client.post(
                    "/api/bloom/analyze-upload",
                    files={"file": ("exam.pdf", pdf_bytes, "application/pdf")}
                )
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "success")
                self.assertEqual(data["summary"]["totalQuestions"], 2)

    async def test_bloom_missing_api_key_returns_500(self):
        payload = {"text": "Q1. What is OOP?"}
        with patch.dict(os.environ, {}, clear=True):
            res = await self.client.post("/api/bloom/analyze", json=payload)
            self.assertEqual(res.status_code, 500)
            self.assertIn("GROQ_API_KEY is not configured", res.json()["detail"])

    async def test_existing_generate_pdf(self):
        res = await self.client.post(
            "/api/generate-pdf",
            json={"text": "EduSahayak Sample Test Document"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("file", data)

if __name__ == "__main__":
    unittest.main()
