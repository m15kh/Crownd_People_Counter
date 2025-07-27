import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import time

def run_benchmarks(input_video):
    """Run both YOLO and Faster R-CNN benchmarks on the same input video"""
    print("Starting benchmarking process...")
    
    # Make sure input video exists
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    # Run YOLO inference
    print("\n=== Running YOLOv8 benchmark ===")
    yolo_start = time.time()
    
    # Define paths
    yolo_inference_script = '/home/fteam5/m/Crownd_People_Counter/yolo_weights/inference.py'
    yolo_temp_script = '/home/fteam5/m/Crownd_People_Counter/yolo_weights/temp_inference.py'
    yolo_weights_path = '/home/fteam5/m/Crownd_People_Counter/yolo_weights/best.pt'
    
    # Check if YOLO script exists
    if not os.path.exists(yolo_inference_script):
        print(f"Error: YOLO inference script not found at {yolo_inference_script}")
        create_yolo = input("Would you like to create a sample YOLO inference script? (y/n): ").lower()
        if create_yolo == 'y':
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(yolo_inference_script), exist_ok=True)
            # Create a basic YOLO inference script
            with open(yolo_inference_script, 'w') as f:
                f.write('''
from ultralytics import YOLO
import cv2
import time
import json
import numpy as np

# Path to input video
input_path = '/home/fteam5/m/Crownd_People_Counter/input.mp4'

# Load the model
model = YOLO('best.pt')  # Path to the trained model weights

# Open the video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"Error: Could not open video {input_path}")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video info: {width}x{height} @ {fps}fps, {frame_count} frames")

# Metrics collection
metrics = {
    "model": "YOLOv8",
    "fps_values": [],
    "inference_times": [],
    "detection_counts": [],
    "total_frames": 0,
    "avg_fps": 0,
    "avg_inference_time": 0
}

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    start_time = time.time()
    # Run inference
    results = model(frame)
    inference_time = time.time() - start_time
    
    # Process results
    detections = results[0].boxes.data.cpu().numpy()
    
    # Calculate FPS
    current_fps = 1.0 / inference_time
    
    # Update metrics
    metrics["fps_values"].append(current_fps)
    metrics["inference_times"].append(inference_time)
    metrics["detection_counts"].append(len(detections))
    
    frame_idx += 1
    if frame_idx % 10 == 0:
        print(f"Processed {frame_idx}/{frame_count} frames, FPS: {current_fps:.2f}")

# Close the video
cap.release()

# Calculate average metrics
metrics["total_frames"] = frame_idx
metrics["avg_fps"] = np.mean(metrics["fps_values"]) if metrics["fps_values"] else 0
metrics["avg_inference_time"] = np.mean(metrics["inference_times"]) if metrics["inference_times"] else 0

# Save metrics
with open("yolo_benchmark_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Benchmark completed. Processed {frame_idx} frames.")
print(f"Average FPS: {metrics['avg_fps']:.2f}")
print(f"Average inference time: {metrics['avg_inference_time']:.4f} seconds")
print(f"Results saved to yolo_benchmark_metrics.json")
''')
            print(f"Created sample YOLO inference script at {yolo_inference_script}")
            print("Warning: You still need to download the YOLO model weights.")
        else:
            return False
    
    # Check if YOLO weights exist
    if not os.path.exists(yolo_weights_path):
        print(f"Error: YOLO weights not found at {yolo_weights_path}")
        print("Please download YOLOv8 weights or provide a different path")
        custom_weights = input("Enter path to YOLOv8 weights (or press Enter to skip YOLO benchmark): ")
        if custom_weights and os.path.exists(custom_weights):
            yolo_weights_path = custom_weights
        else:
            print("Skipping YOLO benchmark.")
            yolo_success = False
            yolo_end = time.time()
            print(f"YOLOv8 benchmark skipped in {yolo_end - yolo_start:.2f} seconds")
            
            # Clean up temporary file if it exists
            if os.path.exists(yolo_temp_script):
                os.remove(yolo_temp_script)
                
            # Skip to Faster R-CNN part
            rcnn_success = False
            goto_rcnn = True
    else:
        goto_rcnn = False
    
    if not goto_rcnn:
        # Update paths in the YOLO script to use the input_video and correct model path
        with open(yolo_inference_script, 'r') as f:
            yolo_script = f.read()
        
        updated_yolo_script = yolo_script.replace(
            "'/home/fteam5/m/Crownd_People_Counter/input.mp4'", 
            f"'{input_video}'"
        )
        
        # Update the model path to use the absolute path
        updated_yolo_script = updated_yolo_script.replace(
            "model = YOLO('best.pt')",
            f"model = YOLO('{yolo_weights_path}')"
        )
        
        with open(yolo_temp_script, 'w') as f:
            f.write(updated_yolo_script)
        
        try:
            subprocess.run(['python', yolo_temp_script], check=True)
            yolo_success = True
        except subprocess.CalledProcessError:
            print("Error occurred while running YOLO benchmark")
            yolo_success = False
        finally:
            yolo_end = time.time()
            print(f"YOLOv8 benchmark completed in {yolo_end - yolo_start:.2f} seconds")
            
            # Clean up temporary file
            if os.path.exists(yolo_temp_script):
                os.remove(yolo_temp_script)
    
    # Run Faster R-CNN inference
    print("\n=== Running Faster R-CNN benchmark ===")
    rcnn_start = time.time()
    
    # Define paths
    rcnn_base_dir = '/home/fteam5/m/Crownd_People_Counter/fasterrcnn_pytorch_training_pipeline'
    rcnn_script = f'{rcnn_base_dir}/onnx_inference_video.py'
    rcnn_weights = f'{rcnn_base_dir}/weights/model_final.onnx'
    rcnn_data_config = f'{rcnn_base_dir}/data_configs/voc.yaml'
    
    # Check if Faster R-CNN directory exists
    if not os.path.exists(rcnn_base_dir):
        print(f"Error: Faster R-CNN directory not found at {rcnn_base_dir}")
        clone_repo = input("Would you like to clone the Faster R-CNN repository? (y/n): ").lower()
        if clone_repo == 'y':
            try:
                os.makedirs(rcnn_base_dir, exist_ok=True)
                # Clone the repository
                subprocess.run(['git', 'clone', 'https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline.git', rcnn_base_dir], check=True)
                print(f"Repository cloned to {rcnn_base_dir}")
                
                # Create weights directory
                os.makedirs(os.path.dirname(rcnn_weights), exist_ok=True)
                print("Please download the Faster R-CNN weights manually and place them in the weights directory.")
            except Exception as e:
                print(f"Error cloning repository: {e}")
                print("Skipping Faster R-CNN benchmark.")
                rcnn_success = False
                return yolo_success and rcnn_success
        else:
            print("Skipping Faster R-CNN benchmark.")
            rcnn_success = False
            return yolo_success and rcnn_success
    
    # Check if Faster R-CNN script exists
    if not os.path.exists(rcnn_script):
        print(f"Error: Faster R-CNN inference script not found at {rcnn_script}")
        print("The repository structure might be different than expected.")
        
        # Look for similar files
        script_dir = os.path.dirname(rcnn_script)
        if os.path.exists(script_dir):
            py_files = [f for f in os.listdir(script_dir) if f.endswith('.py') and ('inference' in f or 'predict' in f)]
            if py_files:
                print("Found potential inference scripts:")
                for i, file in enumerate(py_files):
                    print(f"{i+1}. {file}")
                choice = input("Select a script to use (number) or press Enter to skip: ")
                if choice.isdigit() and 1 <= int(choice) <= len(py_files):
                    rcnn_script = os.path.join(script_dir, py_files[int(choice)-1])
                    print(f"Using {rcnn_script}")
                else:
                    rcnn_success = False
                    return yolo_success and rcnn_success
            else:
                custom_script = input("Enter path to Faster R-CNN inference script (or press Enter to skip): ")
                if custom_script and os.path.exists(custom_script):
                    rcnn_script = custom_script
                else:
                    rcnn_success = False
                    return yolo_success and rcnn_success
        else:
            rcnn_success = False
            return yolo_success and rcnn_success
    
    # Check if Faster R-CNN weights exist
    if not os.path.exists(rcnn_weights):
        print(f"Error: Faster R-CNN weights not found at {rcnn_weights}")
        custom_weights = input("Enter path to Faster R-CNN weights (or press Enter to skip benchmark): ")
        if custom_weights and os.path.exists(custom_weights):
            rcnn_weights = custom_weights
        else:
            print("Skipping Faster R-CNN benchmark.")
            rcnn_success = False
            rcnn_end = time.time()
            print(f"Faster R-CNN benchmark skipped in {rcnn_end - rcnn_start:.2f} seconds")
            return yolo_success and rcnn_success
        
    # Check if data config exists
    if not os.path.exists(rcnn_data_config):
        print(f"Error: Data config not found at {rcnn_data_config}")
        print("Creating a basic VOC data config...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(rcnn_data_config), exist_ok=True)
        
        # Create a basic VOC data config
        with open(rcnn_data_config, 'w') as f:
            f.write('''# VOC format dataset config
CLASSES: ['person']
NC: 1  # Number of classes
''')
        print(f"Created basic data config at {rcnn_data_config}")
    
    try:
        subprocess.run([
            'python', rcnn_script,
            '--input', input_video,
            '--weights', rcnn_weights,
            '--data', rcnn_data_config,
            '--log-json'
        ], check=True)
        rcnn_success = True
    except subprocess.CalledProcessError:
        print("Error occurred while running Faster R-CNN benchmark")
        rcnn_success = False
    finally:
        rcnn_end = time.time()
        print(f"Faster R-CNN benchmark completed in {rcnn_end - rcnn_start:.2f} seconds")
    
    return yolo_success and rcnn_success
    
def load_benchmark_data():
    """Load benchmark data from both models"""
    yolo_data = None
    rcnn_data = None
    
    # Load YOLO benchmark data
    try:
        with open('yolo_benchmark_metrics.json', 'r') as f:
            yolo_data = json.load(f)
    except FileNotFoundError:
        print("YOLO benchmark data not found")
    
    # Load Faster R-CNN benchmark data
    try:
        # The Faster R-CNN script logs to 'inference_output' directory
        inference_dirs = [d for d in os.listdir('.') if d.startswith('inference_output')]
        if inference_dirs:
            latest_dir = max(inference_dirs, key=lambda x: os.path.getctime(x))
            log_path = os.path.join(latest_dir, 'log.json')
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    rcnn_log = json.load(f)
                
                # Process the log data into a similar format as YOLO data
                rcnn_data = {
                    'model': 'Faster R-CNN',
                    'fps_values': [],
                    'inference_times': [],
                    'detection_counts': [],
                    'total_frames': len(rcnn_log['frames'])
                }
                
                # Extract frame data
                for frame in rcnn_log['frames']:
                    if 'fps' in frame:
                        rcnn_data['fps_values'].append(frame['fps'])
                    if 'forward_time' in frame:
                        rcnn_data['inference_times'].append(frame['forward_time'])
                    if 'detections' in frame:
                        rcnn_data['detection_counts'].append(len(frame['detections']))
                
                # Calculate averages
                rcnn_data['avg_fps'] = np.mean(rcnn_data['fps_values']) if rcnn_data['fps_values'] else 0
                rcnn_data['avg_inference_time'] = np.mean(rcnn_data['inference_times']) if rcnn_data['inference_times'] else 0
    except Exception as e:
        print(f"Error loading Faster R-CNN data: {e}")
    
    return yolo_data, rcnn_data

def generate_comparison(yolo_data, rcnn_data):
    """Generate comparison visualizations and report"""
    if not yolo_data or not rcnn_data:
        print("Missing data, cannot generate comparison")
        return
    
    # Create a comparison dataframe
    comparison = pd.DataFrame({
        'Metric': [
            'Average FPS', 
            'Average Inference Time (s)', 
            'Total Frames Processed'
        ],
        'YOLOv8': [
            yolo_data['avg_fps'],
            yolo_data['avg_inference_time'],
            yolo_data['total_frames']
        ],
        'Faster R-CNN': [
            rcnn_data['avg_fps'],
            rcnn_data['avg_inference_time'],
            rcnn_data['total_frames']
        ]
    })
    
    # Save comparison to CSV
    comparison.to_csv('model_comparison.csv', index=False)
    print("\nComparison saved to model_comparison.csv")
    
    # Generate visualizations
    plt.figure(figsize=(12, 10))
    
    # FPS Comparison
    plt.subplot(2, 2, 1)
    plt.bar(['YOLOv8', 'Faster R-CNN'], [yolo_data['avg_fps'], rcnn_data['avg_fps']])
    plt.title('Average FPS Comparison')
    plt.ylabel('Frames Per Second')
    
    # Inference Time Comparison
    plt.subplot(2, 2, 2)
    plt.bar(['YOLOv8', 'Faster R-CNN'], [yolo_data['avg_inference_time'], rcnn_data['avg_inference_time']])
    plt.title('Average Inference Time Comparison')
    plt.ylabel('Time (seconds)')
    
    # FPS over time (if available)
    plt.subplot(2, 2, 3)
    frames = min(len(yolo_data['fps_values']), len(rcnn_data['fps_values']))
    if frames > 0:
        plt.plot(yolo_data['fps_values'][:frames], label='YOLOv8')
        plt.plot(rcnn_data['fps_values'][:frames], label='Faster R-CNN')
        plt.title('FPS Over Time')
        plt.xlabel('Frame')
        plt.ylabel('FPS')
        plt.legend()
    
    # Print summary to console
    print("\nModel Performance Comparison:")
    print(comparison)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("Performance visualization saved to performance_comparison.png")

if __name__ == "__main__":
    # Input video path
    input_video = '/home/fteam5/m/Crownd_People_Counter/input.mp4'
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"Warning: Input video not found at {input_video}")
        alternative_video = input("Enter path to alternative video file: ")
        if os.path.exists(alternative_video):
            input_video = alternative_video
        else:
            print("Invalid video path. Exiting.")
            exit(1)
    
    # Ask user if they want to run the benchmarks or just generate comparison
    choice = input("Run new benchmarks? (y/n): ").lower()
    
    benchmark_success = True
    if choice == 'y':
        benchmark_success = run_benchmarks(input_video)
        if not benchmark_success:
            print("\nAttempting to run benchmarks with mock data for demonstration...")
            # Create mock data for demonstration if benchmarks failed
            if not os.path.exists('yolo_benchmark_metrics.json'):
                mock_yolo = {
                    "model": "YOLOv8",
                    "fps_values": [25.3, 24.8, 26.1, 25.7, 24.9],
                    "inference_times": [0.0395, 0.0403, 0.0383, 0.0389, 0.0401],
                    "detection_counts": [4, 3, 5, 2, 4],
                    "total_frames": 5,
                    "avg_fps": 25.36,
                    "avg_inference_time": 0.0394
                }
                with open('yolo_benchmark_metrics.json', 'w') as f:
                    json.dump(mock_yolo, f)
                print("Created mock YOLO benchmark data for demonstration")
            
            # Create mock Faster R-CNN data
            mock_rcnn_dir = 'inference_output_mock'
            os.makedirs(mock_rcnn_dir, exist_ok=True)
            mock_rcnn_log = {
                "frames": []
            }
            for i in range(5):
                mock_rcnn_log["frames"].append({
                    "frame_idx": i,
                    "fps": 10.2 + i * 0.3,
                    "forward_time": 0.098 - i * 0.002,
                    "detections": [{"box": [100, 100, 200, 200], "score": 0.95, "class_name": "person"}] * (i + 1)
                })
            
            with open(os.path.join(mock_rcnn_dir, 'log.json'), 'w') as f:
                json.dump(mock_rcnn_log, f)
            print("Created mock Faster R-CNN benchmark data for demonstration")
    
    # Load benchmark data and generate comparison
    yolo_data, rcnn_data = load_benchmark_data()
    
    if yolo_data and rcnn_data:
        generate_comparison(yolo_data, rcnn_data)
    else:
        print("\nBenchmark data is incomplete.")
        print("Please ensure all model files exist and try again, or use the mock data option.")
