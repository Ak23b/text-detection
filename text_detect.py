import cv2
import numpy as np
import pytesseract
import os

# -----------------------------
# TESSERACT PATH - SET YOURS
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------
# Load Image
# -----------------------------
img = cv2.imread("ji2frx56.png")

if img is None:
    print("Error loading image.")
    exit()

orig = img.copy()
(H, W) = img.shape[:2]

# -----------------------------
# Load EAST Text Detector
# -----------------------------
east_model = "frozen_east_text_detection.pb"
net = cv2.dnn.readNet(east_model)

blob = cv2.dnn.blobFromImage(img, 1.0, (320, 320),
                             (123.68, 116.78, 103.94),
                             swapRB=True, crop=False)

net.setInput(blob)

(scores, geometry) = net.forward([
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
])

# -----------------------------
# Decode EAST Output
# -----------------------------
def decode(scores, geometry, conf=0.5):
    rects = []
    confidences = []

    H, W = scores.shape[2:4]

    for y in range(H):
        for x in range(W):

            score = scores[0, 0, y, x]
            if score < conf:
                continue

            # geometry data
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]

            endX = int(offsetX + cos * geometry[0, 1, y, x] + sin * geometry[0, 2, y, x])
            endY = int(offsetY - sin * geometry[0, 1, y, x] + cos * geometry[0, 2, y, x])

            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(float(score))

    return rects, confidences


rects, confidences = decode(scores, geometry)

# -----------------------------
# Apply Non-Max Suppression
# -----------------------------
boxes = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)

print("\nDetected regions:", len(boxes))

# -----------------------------
# OCR on Detected Boxes
# -----------------------------
if len(boxes) == 0:
    print("No text found.")
else:
    for b in boxes:

        idx = int(b) if not isinstance(b, (list, tuple)) else b[0]

        (x1, y1, x2, y2) = rects[idx]

        # Fix negative coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        roi = orig[y1:y2, x1:x2]

        # -----------------------------
        # Preprocess for Better OCR
        # -----------------------------
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        text = pytesseract.image_to_string(
            gray,
            config="--psm 7"  # Treat as a single word
        ).strip()

        # Draw results
        color = (0, 255, 0)
        cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)

        if text != "":
            cv2.putText(orig, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        print("Detected:", text)

# -----------------------------
# Show Final Output
# -----------------------------
cv2.imshow("Text Detection + OCR", orig)
cv2.waitKey(0)
