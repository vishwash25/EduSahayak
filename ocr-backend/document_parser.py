import io
import os
import logging
from typing import List, Optional
from PIL import Image

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

from ocr_engine import OCRComparer

logger = logging.getLogger(__name__)

# Max file size limit: 25 MB
MAX_FILE_SIZE = 25 * 1024 * 1024
SUPPORTED_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}
SUPPORTED_DOC_EXTS = {'.pdf', '.txt', '.md'}

class DocumentParser:
    def __init__(self, ocr_comparer: Optional[OCRComparer] = None):
        self.ocr_comparer = ocr_comparer or OCRComparer()

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extracts text from PDF bytes.
        First tries native digital text extraction via pypdfium2.
        If native text is sparse/scanned, renders pages to images and runs OCR.
        """
        if not pdf_bytes:
            raise ValueError("PDF content is empty.")

        if not pdfium:
            raise RuntimeError("pypdfium2 is not available for PDF processing.")

        pdf = None
        try:
            pdf = pdfium.PdfDocument(pdf_bytes)
            page_texts = []
            total_chars = 0

            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                textpage = page.get_textpage()
                txt = textpage.get_text_range()
                textpage.close()
                page.close()
                if txt and txt.strip():
                    page_texts.append(txt.strip())
                    total_chars += len(txt.strip())

            # If digital text extraction extracted more than 50 characters, use it
            if total_chars >= 50:
                return "\n\n".join(page_texts)

            # Scanned PDF fallback: render each page to image and perform OCR
            logger.info("PDF appears to be scanned or contains minimal digital text. Running OCR on pages...")
            ocr_page_texts = []
            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                # Render at 200 DPI (scale ~2.5) for OCR quality
                pil_image = page.render(scale=2.5).to_pil()
                page.close()

                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                page_img_bytes = img_byte_arr.getvalue()

                page_text = self.extract_text_from_single_image(page_img_bytes)
                if page_text.strip():
                    ocr_page_texts.append(page_text.strip())

            return "\n\n".join(ocr_page_texts) if ocr_page_texts else "\n\n".join(page_texts)

        except Exception as e:
            logger.error(f"Error parsing PDF: {str(e)}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")
        finally:
            if pdf:
                try:
                    pdf.close()
                except Exception:
                    pass

    def extract_text_from_single_image(self, image_bytes: bytes) -> str:
        """
        Runs OCR on a single image and extracts the best text.
        Prefers PaddleOCR or Tesseract based on confidence.
        """
        if not image_bytes:
            return ""

        results = self.ocr_comparer.run_all(image_bytes)
        
        # Check PaddleOCR result first
        paddle_res = results.get("paddleocr", {})
        paddle_text = paddle_res.get("text", "").strip()
        paddle_conf = paddle_res.get("confidence", 0)

        # Check Tesseract result
        tes_res = results.get("tesseract", {})
        tes_text = tes_res.get("text", "").strip()
        tes_conf = tes_res.get("confidence", 0)

        if paddle_text and not paddle_text.startswith("Error:") and not paddle_text.startswith("Dependency Missing"):
            if tes_text and not tes_text.startswith("Error:") and not tes_text.startswith("Dependency Missing"):
                return paddle_text if paddle_conf >= tes_conf else tes_text
            return paddle_text
        
        if tes_text and not tes_text.startswith("Error:") and not tes_text.startswith("Dependency Missing"):
            return tes_text

        return paddle_text or tes_text or ""

    def extract_text_from_multiple_images(self, image_bytes_list: List[bytes]) -> str:
        """
        Processes multiple question paper images in sequence and joins their text.
        """
        all_texts = []
        for idx, img_bytes in enumerate(image_bytes_list, 1):
            txt = self.extract_text_from_single_image(img_bytes)
            if txt:
                all_texts.append(f"--- Page / Image {idx} ---\n" + txt)
        return "\n\n".join(all_texts)
