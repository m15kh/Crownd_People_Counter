import torch
import cv2
from GroundingDINO.groundingdino.util.inference import load_model, predict, annotate, load_image
from PIL import Image
import numpy as np

# Load model
config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
weights_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
model = load_model(config_file, weights_path)

# Settings
TEXT_PROMPT = "person"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
RECT_REGION = (400, 0, 750, None)  # (x1, y1, x2, y2), height will be set dynamically

# Video paths
input_video_path = "input.mp4"
output_video_path = "dino.mp4"

# Open video
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
RECT_REGION = (RECT_REGION[0], RECT_REGION[1], RECT_REGION[2], height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)

    # Apply same preprocessing as load_image()
    from groundingdino.datasets import transforms as T
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)

    # Detect
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # Filter boxes in defined zone
    filtered_boxes, filtered_logits, filtered_phrases = [], [], []
    for box, logit, phrase in zip(boxes, logits, phrases):
        cx, cy, bw, bh = box.tolist()
        x_min = (cx - bw / 2) * width
        x_max = (cx + bw / 2) * width
        if x_min >= RECT_REGION[0] and x_max <= RECT_REGION[2]:
            filtered_boxes.append(box)
            filtered_logits.append(logit)
            filtered_phrases.append(phrase)

    zone_count = len(filtered_boxes)

    # Annotate frame
    if filtered_boxes:
        filtered_boxes_tensor = torch.stack(filtered_boxes)
        filtered_logits_tensor = torch.stack(filtered_logits)
        annotated_frame = annotate(
            image_source=frame_rgb,
            boxes=filtered_boxes_tensor,
            logits=filtered_logits_tensor,
            phrases=filtered_phrases
        )
    else:
        annotated_frame = frame_bgr.copy()

    # Draw rectangle
    cv2.rectangle(
        annotated_frame,
        (RECT_REGION[0], RECT_REGION[1]),
        (RECT_REGION[2], RECT_REGION[3]),
        color=(255, 0, 0),
        thickness=2
    )

    # Draw count text
    cv2.putText(
        annotated_frame,
        f"Count (Zone): {zone_count}",
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    out.write(annotated_frame)
    frame_count += 1
    print(f"Processed frame {frame_count}", end='\r')

cap.release()
out.release()
print(f"\nVideo saved to {output_video_path}")
