import os
import sys
import io
import json
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
from fpdf import FPDF

from bloom_analyzer import (
    parse_question_paper_structure,
    extract_bloom_and_marks,
    analyze_question_paper_text,
    normalize_question_id,
    parse_groq_json_response,
    assess_questions_with_groq,
    BLOOM_LEVELS
)
from document_parser import DocumentParser

class TestBloomTaxonomyAlignmentChecker(unittest.TestCase):

    def setUp(self):
        self.mock_groq_analysis = [
            {
                "id": "Q1",
                "aiBloomLevel": 2,
                "aiBloomName": "Understand",
                "confidence": 0.94,
                "reason": "Requires conceptual explanation of object-oriented principles.",
                "isAmbiguous": False
            },
            {
                "id": "Q2",
                "aiBloomLevel": 3,
                "aiBloomName": "Apply",
                "confidence": 0.95,
                "reason": "Requires applying the binary search algorithm procedure.",
                "isAmbiguous": False
            },
            {
                "id": "Q3",
                "aiBloomLevel": 6,
                "aiBloomName": "Create",
                "confidence": 0.96,
                "reason": "Requires synthesizing an original software architecture schema.",
                "isAmbiguous": False
            }
        ]

    def test_1_missing_groq_api_key_raises_error(self):
        """
        Verify that missing GROQ_API_KEY raises a clear configuration error
        and NEVER silently fabricates Level 2 results.
        """
        sample_text = "Q1. What is Object Oriented Programming? (2 Marks) [BL-3]"
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_question_paper_text(sample_text, api_key=None)
            self.assertIn("GROQ_API_KEY is not configured", str(ctx.exception))

    def test_2_groq_api_failure_raises_error(self):
        """
        Verify that Groq API exceptions raise a clear service error
        and NEVER silently fabricate Level 2 results.
        """
        sample_text = "Q1. What is Object Oriented Programming? (2 Marks) [BL-3]"
        with patch("bloom_analyzer.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded 429")
            
            with self.assertRaises(RuntimeError) as ctx:
                analyze_question_paper_text(sample_text, api_key="fake-test-key")
            self.assertIn("Groq AI service unavailable", str(ctx.exception))

    def test_3_valid_groq_response_preserves_levels(self):
        """
        Verify that genuine Groq responses across all Bloom levels (L1 to L6)
        are preserved without corruption or default overwrites.
        """
        sample_text = """
        Q1. Define encapsulation. (2 Marks) [BL-1]
        Q2. Explain dynamic binding. (5 Marks) [BL-2]
        Q3. Apply binary search to array. (5 Marks) [BL-3]
        Q4. Analyze algorithm complexity. (8 Marks) [BL-4]
        Q5. Evaluate architectural trade-offs. (10 Marks) [BL-5]
        Q6. Design a distributed database. (15 Marks) [BL-6]
        """
        groq_resp = {
            "analysis": [
                {"id": "Q1", "aiBloomLevel": 1, "aiBloomName": "Remember", "confidence": 0.95, "reason": "Recall."},
                {"id": "Q2", "aiBloomLevel": 2, "aiBloomName": "Understand", "confidence": 0.92, "reason": "Explain."},
                {"id": "Q3", "aiBloomLevel": 3, "aiBloomName": "Apply", "confidence": 0.93, "reason": "Procedure."},
                {"id": "Q4", "aiBloomLevel": 4, "aiBloomName": "Analyze", "confidence": 0.91, "reason": "Deconstruct."},
                {"id": "Q5", "aiBloomLevel": 5, "aiBloomName": "Evaluate", "confidence": 0.94, "reason": "Judge."},
                {"id": "Q6", "aiBloomLevel": 6, "aiBloomName": "Create", "confidence": 0.97, "reason": "Design."}
            ]
        }
        with patch("bloom_analyzer.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(groq_resp)))]
            mock_client.chat.completions.create.return_value = mock_resp

            response = analyze_question_paper_text(sample_text, api_key="fake-test-key")
            self.assertEqual(response.status, "success")
            self.assertEqual(response.summary.totalQuestions, 6)
            self.assertEqual(response.summary.matches, 6)
            
            levels = [q.aiBloomLevel for q in response.questions]
            self.assertEqual(levels, [1, 2, 3, 4, 5, 6])

    def test_4_id_normalization_q1_vs_1(self):
        """
        Verify that questions are matched properly even if Groq returns IDs as
        '1' vs 'Q1', '2(a)' vs 'Q2(a)', or 'q3'.
        """
        sample_text = """
        Q1. Define polymorphsim. (2 Marks) [BL-1]
        Q2.
        a) Explain inheritance. (5 Marks) [BL-2]
        b) Apply factory pattern. (8 Marks) [BL-3]
        """
        # Groq returns IDs without 'Q' prefix
        groq_resp = {
            "analysis": [
                {"id": "1", "aiBloomLevel": 1, "aiBloomName": "Remember", "confidence": 0.95, "reason": "Recall."},
                {"id": "2(a)", "aiBloomLevel": 2, "aiBloomName": "Understand", "confidence": 0.92, "reason": "Explain."},
                {"id": "2(b)", "aiBloomLevel": 3, "aiBloomName": "Apply", "confidence": 0.93, "reason": "Apply pattern."}
            ]
        }
        with patch("bloom_analyzer.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(groq_resp)))]
            mock_client.chat.completions.create.return_value = mock_resp

            response = analyze_question_paper_text(sample_text, api_key="fake-test-key")
            self.assertEqual(response.summary.totalQuestions, 3)
            self.assertEqual(response.questions[0].fullIdentifier, "Q1")
            self.assertEqual(response.questions[0].aiBloomLevel, 1)
            self.assertEqual(response.questions[1].fullIdentifier, "Q2(a)")
            self.assertEqual(response.questions[1].aiBloomLevel, 2)
            self.assertEqual(response.questions[2].fullIdentifier, "Q2(b)")
            self.assertEqual(response.questions[2].aiBloomLevel, 3)

    def test_5_markdown_fenced_json_parsing(self):
        """
        Verify that JSON responses wrapped in markdown code blocks are parsed cleanly.
        """
        raw_markdown = """```json
        {
            "analysis": [
                {
                    "id": "Q1",
                    "aiBloomLevel": 4,
                    "aiBloomName": "Analyze",
                    "confidence": 0.91,
                    "reason": "Analyze complexity.",
                    "isAmbiguous": false
                }
            ]
        }
        ```"""
        parsed = parse_groq_json_response(raw_markdown)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["id"], "Q1")
        self.assertEqual(parsed[0]["aiBloomLevel"], 4)

    def test_6_raw_json_array_parsing(self):
        """
        Verify that direct JSON arrays (without top-level 'analysis' object) are supported.
        """
        raw_array = json.dumps([
            {"id": "Q1", "aiBloomLevel": 5, "aiBloomName": "Evaluate", "confidence": 0.9, "reason": "Trade-offs."}
        ])
        parsed = parse_groq_json_response(raw_array)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["aiBloomLevel"], 5)

    def test_7_missing_question_id_in_groq_response_raises_error(self):
        """
        Verify that if Groq fails to return an item for a question ID,
        an error is raised instead of silently defaulting to L2.
        """
        sample_text = """
        Q1. Define OOP. (2 Marks) [BL-1]
        Q2. Explain inheritance. (5 Marks) [BL-2]
        """
        # Groq returns only Q1, omitting Q2
        groq_resp = {
            "analysis": [
                {"id": "Q1", "aiBloomLevel": 1, "aiBloomName": "Remember", "confidence": 0.95, "reason": "Recall."}
            ]
        }
        with patch("bloom_analyzer.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(groq_resp)))]
            mock_client.chat.completions.create.return_value = mock_resp

            with self.assertRaises(RuntimeError) as ctx:
                analyze_question_paper_text(sample_text, api_key="fake-test-key")
            self.assertIn("did not include a classification for question ID 'Q2'", str(ctx.exception))

    def test_8_regression_user_question_paper_structure(self):
        """
        Regression test: Verify that the 6 expected question nodes are parsed
        without headers, subject titles, instructions, or footers.
        """
        sample_text = """
UNIVERSITY EXAMINATION
B.E. INFORMATION SCIENCE AND ENGINEERING
OBJECT ORIENTED PROGRAMMING

Answer all questions.

Q1. What is Object Oriented Programming? (2 Marks) [BL-3]

Q2.
a) Explain the difference between inheritance and composition. (5 Marks) [BL-2]
b) Apply the Factory Design Pattern to the following problem. (8 Marks) [BL-3]

Q3.
a) Analyze the given algorithm and identify its time complexity. (8 Marks) [BL-4]
b) Evaluate whether inheritance or composition is more appropriate for the given system and justify your decision. (10 Marks) [BL-5]

Q4. Design an object-oriented architecture for an online banking system. (15 Marks) [BL-6]

Page 1 of 2
        """
        parsed_items = parse_question_paper_structure(sample_text)
        self.assertEqual(len(parsed_items), 6)
        
        identifiers = [q["fullIdentifier"] for q in parsed_items]
        expected_identifiers = ["Q1", "Q2(a)", "Q2(b)", "Q3(a)", "Q3(b)", "Q4"]
        self.assertEqual(identifiers, expected_identifiers)

        for q in parsed_items:
            self.assertNotIn("UNIVERSITY EXAMINATION", q["questionText"])
            self.assertNotIn("B.E. INFORMATION SCIENCE AND ENGINEERING", q["questionText"])
            self.assertNotIn("OBJECT ORIENTED PROGRAMMING", q["questionText"])
            self.assertNotIn("Answer all questions.", q["questionText"])
            self.assertNotIn("Page 1 of 2", q["questionText"])

    def test_9_regression_embedded_code_in_question(self):
        """
        Regression test: Verify that embedded code fragments (Java, Python, C++, SQL)
        with statements like 'v.display();' or 'x.foo()' are preserved as part of the
        parent question content and never falsely segmented into sub-questions like Q3(v).
        """
        sample_text = """
Q3. What is the output of the following Java program? (5 Marks) [BL-3]

class Vehicle {
    void display() {
        System.out.println("Vehicle");
    }
}

class Car extends Vehicle {
    @Override
    void display() {
        System.out.println("Car");
    }
}

Vehicle v = new Car();
v.display();
"""
        parsed_items = parse_question_paper_structure(sample_text)
        self.assertEqual(len(parsed_items), 1)
        self.assertEqual(parsed_items[0]["fullIdentifier"], "Q3")
        self.assertIsNone(parsed_items[0]["subQuestionNumber"])
        self.assertEqual(parsed_items[0]["marks"], 5)
        self.assertEqual(parsed_items[0]["assignedBloomLevel"], 3)

        # Assert no false subquestions like Q3(v) were generated
        identifiers = [q["fullIdentifier"] for q in parsed_items]
        self.assertNotIn("Q3(v)", identifiers)
        self.assertNotIn("v", [q.get("subQuestionNumber") for q in parsed_items])

        # Assert code content and parentheses are preserved inside Q3
        q3_text = parsed_items[0]["questionText"]
        self.assertIn("v.display();", q3_text)
        self.assertIn("void display()", q3_text)
        self.assertIn("System.out.println(\"Vehicle\");", q3_text)
        self.assertIn("Vehicle v = new Car();", q3_text)

if __name__ == "__main__":
    unittest.main()

