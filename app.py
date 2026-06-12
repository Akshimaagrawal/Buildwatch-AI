from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Create static folder if not present
os.makedirs("static", exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    # Save uploaded files
    before_file = request.files["before"]
    after_file = request.files["after"]

    before_path = "static/before.jpg"
    after_path = "static/after.jpg"

    before_file.save(before_path)
    after_file.save(after_path)

    # Read images
    before_img = cv2.imread(before_path)
    after_img = cv2.imread(after_path)

    # Resize images to same dimensions
    height = min(before_img.shape[0], after_img.shape[0])
    width = min(before_img.shape[1], after_img.shape[1])

    before_img = cv2.resize(before_img, (width, height))
    after_img = cv2.resize(after_img, (width, height))

    # Convert to grayscale
    gray1 = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray1 = cv2.GaussianBlur(gray1, (9, 9), 0)
    gray2 = cv2.GaussianBlur(gray2, (9, 9), 0)

    # Difference
    diff = cv2.absdiff(gray1, gray2)

    # Threshold
    _, thresh = cv2.threshold(
        diff,
        60,
        255,
        cv2.THRESH_BINARY
    )

    # Morphology
    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    thresh = cv2.dilate(
        thresh,
        kernel,
        iterations=2
    )

    # Save mask
    cv2.imwrite(
        "static/mask.jpg",
        thresh
    )

    # Create heatmap
    heatmap = cv2.applyColorMap(
        thresh,
        cv2.COLORMAP_JET
    )

    cv2.imwrite(
        "static/heatmap.jpg",
        heatmap
    )

    # Detect contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected_regions = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 3000:

            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(
                after_img,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                3
            )

            cv2.putText(
                after_img,
                "Potential Change",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            detected_regions += 1

    # YOLO Detection
    try:

        results = model(after_path)

        for r in results:

            boxes = r.boxes

            if boxes is None:
                continue

            for box in boxes:

                confidence = float(box.conf[0])

                if confidence < 0.5:
                    continue

                cls = int(box.cls[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                label = model.names[cls]

                cv2.rectangle(
                    after_img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    after_img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    except Exception as e:
        print("YOLO Error:", e)

    # Confidence Score
    change_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]

    confidence_score = round(
        (change_pixels / total_pixels) * 100,
        2
    )

    # Risk Level
    if confidence_score > 30:
      risk = "High"

    elif confidence_score > 10:
      risk = "Medium"

    else:
      risk = "Low"

    # Save result image
    result_path = "static/result.jpg"

    cv2.imwrite(
        result_path,
        after_img
    )

    return render_template(
        "result.html",
        confidence=confidence_score,
        regions=detected_regions,
        risk=risk,
        image=result_path
    )


if __name__ == "__main__":
    app.run(debug=True)