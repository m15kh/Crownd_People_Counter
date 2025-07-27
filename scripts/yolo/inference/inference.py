from ultralytics import YOLO
import cv2
import time  # Import time module for FPS calculation
import numpy as np
import psutil  # For memory usage tracking
import os
import json

# Initialize video capture
cap = cv2.VideoCapture('/home/fteam5/m/Crownd_People_Counter/input.mp4')

# Define the rectangular region (x_min, y_min, x_max, y_max)
RECT_REGION = (400, 0, 750, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # Full height of the video

# Load the trained YOLOv8 model
model = YOLO('best.pt')  # Path to the trained model weights

# Run inference on a video
results = model.predict(
    source='/home/fteam5/m/Crownd_People_Counter/input.mp4',  # Replace with the path to your video file
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

# Variables for FPS calculation
prev_time = 0
curr_time = 0
fps = 0

# For benchmarking
benchmark_metrics = {
    'model': 'YOLOv8',
    'fps_values': [],
    'inference_times': [],
    'memory_usage': [],
    'detection_counts': [],
    'total_frames': 0,
    'start_time': time.time(),
}

try:
    while True:
        # Calculate FPS
        curr_time = time.time()
        if prev_time > 0:
            fps = 1 / (curr_time - prev_time)
            benchmark_metrics['fps_values'].append(fps)
        prev_time = curr_time
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info().rss / 1024 / 1024  # in MB
        benchmark_metrics['memory_usage'].append(memory_info)
        
        inference_start = time.time()
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # Exit the loop if no more frames are available

        # Get the next result from the YOLO model
        result = next(results)
        inference_time = time.time() - inference_start
        benchmark_metrics['inference_times'].append(inference_time)
        benchmark_metrics['total_frames'] += 1

        # Extract detections
        detections = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        # Filter detections within the rectangular region
        count = 0
        for box, cls in zip(detections, classes):
            if cls == 0:  # Assuming class 0 is 'person'
                x_min, y_min, x_max, y_max = box
                if RECT_REGION[0] <= x_min <= RECT_REGION[2] and RECT_REGION[1] <= y_min <= RECT_REGION[3]:
                    count += 1
                    # Optionally draw the bounding box
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    
        benchmark_metrics['detection_counts'].append(count)

        # Draw the rectangular region
        cv2.rectangle(frame, (RECT_REGION[0], RECT_REGION[1]), (RECT_REGION[2], RECT_REGION[3]), (255, 0, 0), 2)

        # Display the count on the frame
        cv2.putText(frame, f'Count: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display FPS on the frame
        cv2.putText(frame, f'FPS: {int(fps)}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display inference time
        cv2.putText(frame, f'Inference: {inference_time:.4f}s', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Write the frame to the output video
        out.write(frame)
except StopIteration:
    pass  # Handle the end of the generator gracefully
finally:
    # Release resources
    cap.release()
    out.release()
    del results  # Explicitly delete the generator to ensure cleanup
    
    # Finalize benchmark metrics
    benchmark_metrics['end_time'] = time.time()
    benchmark_metrics['total_time'] = benchmark_metrics['end_time'] - benchmark_metrics['start_time']
    benchmark_metrics['avg_fps'] = np.mean(benchmark_metrics['fps_values'])
    benchmark_metrics['avg_inference_time'] = np.mean(benchmark_metrics['inference_times'])
    benchmark_metrics['avg_memory_usage'] = np.mean(benchmark_metrics['memory_usage'])
    benchmark_metrics['max_memory_usage'] = max(benchmark_metrics['memory_usage'])
    
    # Save metrics to file - save both in current directory and parent directory
    with open('yolo_benchmark_metrics.json', 'w') as f:
        json.dump(benchmark_metrics, f, indent=4)
    
    # Also save to parent directory for easier comparison
    try:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_path = os.path.join(parent_dir, 'yolo_benchmark_metrics.json')
        with open(parent_path, 'w') as f:
            json.dump(benchmark_metrics, f, indent=4)
        print(f"Benchmark metrics also saved to: {parent_path}")
    except Exception as e:
        print(f"Could not save benchmark metrics to parent directory: {e}")
    
    print(f"YOLOv8 Benchmark Summary:")
    print(f"Average FPS: {benchmark_metrics['avg_fps']:.2f}")
    print(f"Average Inference Time: {benchmark_metrics['avg_inference_time']:.4f} seconds")
    print(f"Average Memory Usage: {benchmark_metrics['avg_memory_usage']:.2f} MB")
    print(f"Max Memory Usage: {benchmark_metrics['max_memory_usage']:.2f} MB")
    print(f"Total Processing Time: {benchmark_metrics['total_time']:.2f} seconds")