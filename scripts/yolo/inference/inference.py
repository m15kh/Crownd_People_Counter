from ultralytics import YOLO
import cv2

# Initialize video capture
cap = cv2.VideoCapture('R:/M/code/Human_Counter/input/input.mp4')

# Define the rectangular region (x_min, y_min, x_max, y_max)
RECT_REGION = (400, 0, 750, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # Full height of the video

# Load the trained YOLOv8 model
model = YOLO('R:/M/code/Human_Counter/scripts/yolo/train/runs/detect/yolo8_people_detection/weights/best.pt')  # Path to the trained model weights

# Run inference on a video
results = model.predict(
    source='R:/M/code/Human_Counter/input/input.mp4',  # Replace with the path to your video file
    save=True,  # Save the output video with predictions
    save_txt=False,  # Optionally save predictions in a text file
    conf=0.25,  # Confidence threshold for predictions
    imgsz=640,  # Image size
    stream=True  # Stream results for processing
)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_count.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

try:
    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # Exit the loop if no more frames are available

        # Get the next result from the YOLO model
        result = next(results)

        # Extract detections
        detections = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        classes = result.boxes.cls.cpu().numpy()  # Class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

        # Filter detections within the rectangular region
        count = 0
        for box, cls, conf in zip(detections, classes, confidences):
            if cls == 0:  # Class 0 is 'person'
                x_min, y_min, x_max, y_max = box
                
                # Check if the center of the bounding box is within the zone
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                if (RECT_REGION[0] <= center_x <= RECT_REGION[2] and 
                    RECT_REGION[1] <= center_y <= RECT_REGION[3]):
                    count += 1
                    # Draw the person's bounding box in green if they're in the zone
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    # Add confidence score
                    cv2.putText(frame, f'{conf:.2f}', (int(x_min), int(y_min) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Draw bounding boxes in red for people outside the zone
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

        # Draw the zone (rectangular region) with a more visible style
        cv2.rectangle(frame, (RECT_REGION[0], RECT_REGION[1]), (RECT_REGION[2], RECT_REGION[3]), (255, 0, 0), 3)
        cv2.putText(frame, "Counting Zone", (RECT_REGION[0], RECT_REGION[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the count more prominently
        # Background for text
        cv2.rectangle(frame, (30, 30), (300, 80), (0, 0, 0), -1)
        # Count text
        cv2.putText(frame, f'People in Zone: {count}', (50, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Write the frame to the output video
        out.write(frame)
except StopIteration:
    pass  # Handle the end of the generator gracefully
finally:
    # Release resources
    cap.release()
    out.release()
    del results  # Explicitly delete the generator to ensure cleanup