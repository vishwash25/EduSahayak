import time
import cv2
import numpy as np
import os
import shutil
import io
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

class OCRComparer:
    def __init__(self):
        print("Initializing OCR Engines wrapper...")
        self._paddle_reader = None
        print("Wrapper initialized. Heavy models will load upon first request.")

    def run_all(self, image_bytes: bytes):
        results = {}

        image_np = np.frombuffer(image_bytes, np.uint8)
        img_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        img_pil = Image.open(io.BytesIO(image_bytes))
        width, height = img_pil.size

        # 1. Tesseract - compare original and adaptive-enhanced images.
        start = time.time()
        tes_text = ""
        detections = []
        avg_conf = 0
        selected_psm = None
        selected_preprocessing = None

        try:
            if pytesseract:
                executable = (
                    os.getenv("TESSERACT_CMD")
                    or shutil.which("tesseract")
                )

                if not executable:
                    windows_executable = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                    if os.path.isfile(windows_executable):
                        executable = windows_executable

                if executable:
                    pytesseract.pytesseract.tesseract_cmd = executable

                rgb_array = np.array(img_pil.convert("RGB"))
                gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

                # Adaptive threshold was the most promising preprocessing
                # method in the actual MSE2 PDF diagnostic.
                clahe = cv2.createCLAHE(
                    clipLimit=2.0,
                    tileGridSize=(8, 8)
                )
                enhanced = clahe.apply(gray)

                adaptive = cv2.adaptiveThreshold(
                    enhanced,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    15
                )

                variants = [
                    ("grayscale", gray),
                    ("adaptive", adaptive)
                ]

                best_score = -1

                for preprocessing_name, candidate_image in variants:
                    for psm in (6, 11, 3):
                        config = f"--oem 3 --psm {psm}"

                        data = pytesseract.image_to_data(
                            candidate_image,
                            config=config,
                            output_type=pytesseract.Output.DICT
                        )

                        page_detections = []
                        layout_words = []
                        confidence_values = []
                        useful_word_count = 0

                        for i, raw_token in enumerate(data.get("text", [])):
                            token = str(raw_token).strip()

                            try:
                                confidence = float(data["conf"][i])
                            except (ValueError, TypeError, IndexError):
                                continue

                            if not token or confidence <= 0:
                                continue

                            confidence_values.append(confidence)

                            if len(token) >= 2 and any(ch.isalnum() for ch in token):
                                useful_word_count += 1

                            top_px = float(data["top"][i])
                            left_px = float(data["left"][i])
                            word_height_px = max(1.0, float(data["height"][i]))

                            layout_words.append(
                                (top_px, left_px, word_height_px, token)
                            )

                            page_detections.append({
                                "text": token,
                                "conf": round(confidence),
                                "box": [
                                    round(top_px / height * 100, 2),
                                    round(left_px / width * 100, 2),
                                    round(data["width"][i] / width * 100, 2),
                                    round(word_height_px / height * 100, 2)
                                ]
                            })

                        word_count = len(page_detections)

                        candidate_conf = (
                            round(sum(confidence_values) / len(confidence_values))
                            if confidence_values else 0
                        )

                        # Reconstruct visual reading order using word coordinates,
                        # not Tesseract block identifiers.
                        median_height = (
                            float(np.median([word[2] for word in layout_words]))
                            if layout_words else 1.0
                        )
                        row_tolerance = max(6.0, median_height * 0.65)

                        sorted_words = sorted(
                            layout_words,
                            key=lambda word: (
                                word[0] + word[2] / 2,
                                word[1]
                            )
                        )

                        rows = []
                        for word in sorted_words:
                            center_y = word[0] + word[2] / 2

                            if rows:
                                previous_center = sum(
                                    item[0] + item[2] / 2
                                    for item in rows[-1]
                                ) / len(rows[-1])
                            else:
                                previous_center = None

                            if (
                                rows
                                and abs(center_y - previous_center) <= row_tolerance
                            ):
                                rows[-1].append(word)
                            else:
                                rows.append([word])

                        candidate_lines = []
                        for row in rows:
                            row.sort(key=lambda word: word[1])
                            candidate_lines.append(
                                " ".join(word[3] for word in row)
                            )

                        candidate_text = "\n".join(candidate_lines).strip()

                        meaningful_ratio = (
                            useful_word_count / word_count
                            if word_count else 0
                        )

                        # Prioritize useful text coverage while retaining
                        # a confidence and readable-token quality check.
                        candidate_score = (
                            useful_word_count
                            * (candidate_conf / 100.0) ** 0.5
                            * meaningful_ratio
                            if candidate_text and useful_word_count and candidate_conf >= 40
                            else 0
                        )

                        # Dense examination papers are better reconstructed
                        # with PSM 6 in our real-document benchmark.
                        if psm == 6:
                            candidate_score *= 1.08

                        if candidate_score > best_score:
                            best_score = candidate_score
                            tes_text = candidate_text
                            detections = page_detections
                            avg_conf = candidate_conf
                            selected_psm = psm
                            selected_preprocessing = preprocessing_name

                if not tes_text:
                    avg_conf = 0
                    detections = []

                print(
                    f"[TESSERACT] Selected preprocessing={selected_preprocessing}, "
                    f"PSM={selected_psm}, words={len(detections)}, "
                    f"confidence={avg_conf}",
                    flush=True
                )

            else:
                tes_text = "Dependency Missing: pytesseract not installed."
                detections = []
                avg_conf = 0

        except Exception as e:
            tes_text = f"Error: {str(e)}"
            detections = []
            avg_conf = 0

        results['tesseract'] = {
            "text": tes_text.strip(),
            "time": round((time.time() - start) * 1000),
            "confidence": avg_conf,
            "detections": detections,
            "psm": selected_psm,
            "preprocessing": selected_preprocessing
        }

        # 2. PaddleOCR
        start = time.time()
        try:
            if PaddleOCR:
                if not self._paddle_reader:
                    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
                    # paddleocr>=2.7 does not use 'use_gpu' natively in this wrapper version, use default
                    # disable mkldnn to bypass ConvertPirAttribute2RuntimeAttribute oneDNN crash on Windows CPU
                    self._paddle_reader = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)

                # Use predict() if available (PaddleX v3.0 wrapper), fallback to ocr()
                pad_result = self._paddle_reader.predict(img_cv2) if hasattr(self._paddle_reader, 'predict') else self._paddle_reader.ocr(img_cv2)

                pad_text_lines = []
                detections = []
                conf_sum = 0
                count = 0

                # Support new PaddleX dict structure or classic tuple list
                if pad_result and isinstance(pad_result[0], dict) and 'rec_texts' in pad_result[0]:
                    page = pad_result[0]
                    texts = page.get('rec_texts', [])
                    scores = page.get('rec_scores', [])
                    polys = page.get('rec_polys', [])

                    for i in range(len(texts)):
                        text = texts[i]
                        conf = scores[i]
                        poly = polys[i]

                        pad_text_lines.append(text)
                        conf_val = round(conf * 100)
                        conf_sum += conf_val
                        count += 1

                        top = float(np.min(poly[:, 1])) / height * 100
                        left = float(np.min(poly[:, 0])) / width * 100
                        w = float(np.max(poly[:, 0]) - np.min(poly[:, 0])) / width * 100
                        h = float(np.max(poly[:, 1]) - np.min(poly[:, 1])) / height * 100

                        detections.append({
                            "text": text,
                            "conf": conf_val,
                            "box": [round(top, 2), round(left, 2), round(w, 2), round(h, 2)]
                        })
                elif pad_result and pad_result[0]:
                    for line in pad_result[0]:
                        box, (text, conf) = line
                        pad_text_lines.append(text)
                        conf_val = round(conf * 100)
                        conf_sum += conf_val
                        count += 1

                        left = box[0][0] / width * 100
                        top = box[0][1] / height * 100
                        w = (box[1][0] - box[0][0]) / width * 100
                        h = (box[2][1] - box[0][1]) / height * 100

                        detections.append({
                            "text": text,
                            "conf": conf_val,
                            "box": [round(top, 2), round(left, 2), round(w, 2), round(h, 2)]
                        })

                # Reconstruct visual reading order using OCR bounding boxes.
                valid_heights = [d["box"][3] for d in detections if d.get("box")]
                median_height = float(np.median(valid_heights)) if valid_heights else 1.0
                row_tolerance = max(0.6, median_height * 0.65)

                sorted_detections = sorted(
                    detections,
                    key=lambda d: d["box"][0] + d["box"][3] / 2
                )

                row_groups = []
                for detection in sorted_detections:
                    center_y = detection["box"][0] + detection["box"][3] / 2

                    if row_groups:
                        previous_centers = [
                            item["box"][0] + item["box"][3] / 2
                            for item in row_groups[-1]
                        ]
                        row_center = sum(previous_centers) / len(previous_centers)

                        if abs(center_y - row_center) <= row_tolerance:
                            row_groups[-1].append(detection)
                            continue

                    row_groups.append([detection])

                pad_text_lines = []
                for row in row_groups:
                    row.sort(key=lambda d: d["box"][1])
                    pad_text_lines.extend(d["text"] for d in row)

                pad_text = "\n".join(pad_text_lines)
                avg_conf = round(conf_sum / count) if count > 0 else 0
            else:
                pad_text = "Dependency Missing: paddleocr not installed."
                detections, avg_conf = [], 0
        except Exception as e:
            pad_text = f"Error: {str(e)}"
            detections, avg_conf = [], 0

        results['paddleocr'] = {
            "text": pad_text.strip(),
            "time": round((time.time() - start) * 1000),
            "confidence": avg_conf,
            "detections": detections
        }

        print("\n========== PADDLE OCR COORDINATES ==========", flush=True)
        for item in results.get("paddleocr", {}).get("detections", []):
            print(f"BOX: {item.get('box')} | TEXT: {item.get('text')}", flush=True)
        print("========== END OCR COORDINATES =============\n", flush=True)
        return results
