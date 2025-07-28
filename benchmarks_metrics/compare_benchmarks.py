import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import time
import glob

def run_benchmarks(input_video):
    """Run both YOLO and Faster R-CNN benchmarks on the same input video"""
    # ...existing code...

def find_benchmark_file(filename, search_dirs=None):
    """Search for a benchmark file in multiple directories"""
    if search_dirs is None:
        search_dirs = [
            '.',  # Current directory
            './yolo_weights',  # YOLO weights directory
            '../yolo_weights',  # YOLO weights directory (one level up)
            '/home/fteam5/m/Crownd_People_Counter/yolo_weights',  # Absolute path to YOLO weights
        ]
    
    # First check if file exists at given path
    if os.path.exists(filename):
        print(f"Found benchmark file at: {os.path.abspath(filename)}")
        return filename
    
    # Search in provided directories
    for directory in search_dirs:
        path = os.path.join(directory, os.path.basename(filename))
        if os.path.exists(path):
            print(f"Found benchmark file at: {os.path.abspath(path)}")
            return path
    
    # Search recursively in current directory and subdirectories
    for root, dirs, files in os.walk('.'):
        if os.path.basename(filename) in files:
            path = os.path.join(root, os.path.basename(filename))
            print(f"Found benchmark file at: {os.path.abspath(path)}")
            return path
    
    print(f"Could not find benchmark file: {filename}")
    return None

def load_benchmark_data():
    """Load benchmark data from both models"""
    yolo_data = None
    rcnn_data = None
    
    # Load YOLO benchmark data
    yolo_path = find_benchmark_file('yolo_benchmark_metrics.json')
    if yolo_path:
        try:
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
                print(f"Successfully loaded YOLO benchmark data")
        except Exception as e:
            print(f"Error loading YOLO data: {e}")
    
    # Load Faster R-CNN benchmark data
    # First look in inference output directories
    rcnn_paths = glob.glob('**/rcnn_benchmark_metrics.json', recursive=True)
    if rcnn_paths:
        # Sort by modification time to get the latest
        latest_path = max(rcnn_paths, key=os.path.getmtime)
        try:
            with open(latest_path, 'r') as f:
                rcnn_data = json.load(f)
                print(f"Successfully loaded RCNN benchmark data from {latest_path}")
        except Exception as e:
            print(f"Error loading RCNN data from {latest_path}: {e}")
    else:
        # Try to parse from log.json files in inference_output directories
        try:
            # The Faster R-CNN script logs to 'inference_output' directory
            inference_dirs = []
            for pattern in ['inference_output*', '**/inference_output*']:
                inference_dirs.extend(glob.glob(pattern, recursive=True))
            
            if inference_dirs:
                # Sort by creation time to get the latest
                latest_dir = max(inference_dirs, key=os.path.getctime)
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
                        'total_frames': len(rcnn_log['frames']) if 'frames' in rcnn_log else 0
                    }
                    
                    # Extract frame data
                    if 'frames' in rcnn_log:
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
                    print(f"Successfully processed RCNN data from log.json at {log_path}")
        except Exception as e:
            print(f"Error loading or processing RCNN data: {e}")
    
    return yolo_data, rcnn_data

def generate_comparison(yolo_data, rcnn_data):
    """Generate comparison visualizations and report"""
    if not yolo_data and not rcnn_data:
        print("\nNo benchmark data found for either model.")
        print("Please run the benchmarks first with:")
        print("1. YOLO: cd /home/fteam5/m/Crownd_People_Counter/yolo_weights && python3 inference.py")
        print("2. RCNN: cd /home/fteam5/m/Crownd_People_Counter/fasterrcnn-pytorch-training-pipeline && python3 onnx_inference_video.py --input input.mp4 --weights weights/model_final.onnx --data data_configs/p.yaml --imgsz 640 --threshold 0.7 --device cuda --log-json")
        return
    
    if not yolo_data:
        print("\nMissing YOLO data, cannot generate comparison.")
        print("Please run the YOLO benchmark first with:")
        print("cd /home/fteam5/m/Crownd_People_Counter/yolo_weights && python3 inference.py")
        print("\nMake sure the output file yolo_benchmark_metrics.json is created, then run this script again.")
        return
    
    if not rcnn_data:
        print("\nMissing RCNN data, cannot generate comparison.")
        print("Please run the RCNN benchmark first with:")
        print("cd /home/fteam5/m/Crownd_People_Counter/fasterrcnn-pytorch-training-pipeline && python3 onnx_inference_video.py --input input.mp4 --weights weights/model_final.onnx --data data_configs/p.yaml --imgsz 640 --threshold 0.7 --device cuda --log-json")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs('benchmark_comparison', exist_ok=True)
    
    # Extract key metrics
    metrics = {
        'YOLOv8': {
            'avg_fps': yolo_data['avg_fps'],
            'avg_inference_time': yolo_data['avg_inference_time'],
            'avg_memory_usage': yolo_data['avg_memory_usage'],
            'max_memory_usage': yolo_data['max_memory_usage'],
            'total_time': yolo_data['total_time'],
            'total_frames': yolo_data['total_frames']
        },
        'Faster R-CNN': {
            'avg_fps': rcnn_data['avg_fps'],
            'avg_inference_time': rcnn_data['avg_inference_time'],
            'avg_memory_usage': rcnn_data['avg_memory_usage'],
            'max_memory_usage': rcnn_data['max_memory_usage'],
            'total_time': rcnn_data['total_time'],
            'total_frames': rcnn_data['total_frames']
        }
    }
    
    # Create a comparison dataframe
    metrics_df = pd.DataFrame({
        'Metric': [
            'Average FPS', 
            'Average Inference Time (s)',
            'Average Memory Usage (MB)',
            'Maximum Memory Usage (MB)',
            'Total Processing Time (s)',
            'Total Frames'
        ],
        'YOLOv8': [
            metrics['YOLOv8']['avg_fps'],
            metrics['YOLOv8']['avg_inference_time'],
            metrics['YOLOv8']['avg_memory_usage'],
            metrics['YOLOv8']['max_memory_usage'],
            metrics['YOLOv8']['total_time'],
            metrics['YOLOv8']['total_frames']
        ],
        'Faster R-CNN': [
            metrics['Faster R-CNN']['avg_fps'],
            metrics['Faster R-CNN']['avg_inference_time'],
            metrics['Faster R-CNN']['avg_memory_usage'],
            metrics['Faster R-CNN']['max_memory_usage'],
            metrics['Faster R-CNN']['total_time'],
            metrics['Faster R-CNN']['total_frames']
        ]
    })
    
    # Save comparison to CSV
    metrics_df.to_csv('benchmark_comparison/model_comparison.csv', index=False)
    
    # Generate visualizations
    plt.figure(figsize=(15, 12))
    
    # FPS Comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(['YOLOv8', 'Faster R-CNN'], [metrics['YOLOv8']['avg_fps'], metrics['Faster R-CNN']['avg_fps']])
    plt.title('Average FPS Comparison (higher is better)', fontsize=14)
    plt.ylabel('Frames Per Second')
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', fontsize=12)
    
    # Inference Time Comparison
    plt.subplot(2, 2, 2)
    bars = plt.bar(['YOLOv8', 'Faster R-CNN'], [metrics['YOLOv8']['avg_inference_time'], metrics['Faster R-CNN']['avg_inference_time']])
    plt.title('Average Inference Time Comparison (lower is better)', fontsize=14)
    plt.ylabel('Time (seconds)')
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', fontsize=12)
    
    # Memory Usage Comparison
    plt.subplot(2, 2, 3)
    bars = plt.bar(['YOLOv8', 'Faster R-CNN'], [metrics['YOLOv8']['avg_memory_usage'], metrics['Faster R-CNN']['avg_memory_usage']])
    plt.title('Average Memory Usage Comparison (lower is better)', fontsize=14)
    plt.ylabel('Memory (MB)')
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}', ha='center', fontsize=12)
    
    # FPS over time (if available)
    plt.subplot(2, 2, 4)
    max_frames = min(len(yolo_data['fps_values']), len(rcnn_data['fps_values']))
    if max_frames > 0:
        # Take every nth frame to make the plot cleaner if there are many frames
        sampling_rate = max(1, max_frames // 100)
        sampled_indices = range(0, max_frames, sampling_rate)
        
        plt.plot([i for i in sampled_indices], 
                [yolo_data['fps_values'][i] for i in sampled_indices], 
                label='YOLOv8', linewidth=2)
        plt.plot([i for i in sampled_indices], 
                [rcnn_data['fps_values'][i] for i in sampled_indices], 
                label='Faster R-CNN', linewidth=2)
        plt.title('FPS Over Time (sample)', fontsize=14)
        plt.xlabel('Frame Index')
        plt.ylabel('FPS')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison/performance_comparison.png', dpi=300)
    
    # Create a detailed report
    with open('benchmark_comparison/benchmark_report.txt', 'w') as f:
        f.write("======================================================\n")
        f.write("        OBJECT DETECTION MODEL BENCHMARK REPORT        \n")
        f.write("======================================================\n\n")
        
        f.write(f"YOLO v8:\n")
        f.write(f"  - Average FPS: {metrics['YOLOv8']['avg_fps']:.2f}\n")
        f.write(f"  - Average Inference Time: {metrics['YOLOv8']['avg_inference_time']:.4f} seconds\n")
        f.write(f"  - Average Memory Usage: {metrics['YOLOv8']['avg_memory_usage']:.2f} MB\n")
        f.write(f"  - Maximum Memory Usage: {metrics['YOLOv8']['max_memory_usage']:.2f} MB\n")
        f.write(f"  - Total Processing Time: {metrics['YOLOv8']['total_time']:.2f} seconds\n")
        f.write(f"  - Total Frames Processed: {metrics['YOLOv8']['total_frames']}\n\n")
        
        f.write(f"Faster R-CNN:\n")
        f.write(f"  - Average FPS: {metrics['Faster R-CNN']['avg_fps']:.2f}\n")
        f.write(f"  - Average Inference Time: {metrics['Faster R-CNN']['avg_inference_time']:.4f} seconds\n")
        f.write(f"  - Average Memory Usage: {metrics['Faster R-CNN']['avg_memory_usage']:.2f} MB\n")
        f.write(f"  - Maximum Memory Usage: {metrics['Faster R-CNN']['max_memory_usage']:.2f} MB\n")
        f.write(f"  - Total Processing Time: {metrics['Faster R-CNN']['total_time']:.2f} seconds\n")
        f.write(f"  - Total Frames Processed: {metrics['Faster R-CNN']['total_frames']}\n\n")
        
        # Performance comparison
        fps_ratio = metrics['YOLOv8']['avg_fps'] / metrics['Faster R-CNN']['avg_fps']
        inference_ratio = metrics['Faster R-CNN']['avg_inference_time'] / metrics['YOLOv8']['avg_inference_time']
        
        f.write("Performance Comparison:\n")
        f.write(f"  - YOLOv8 is {fps_ratio:.2f}x faster in FPS compared to Faster R-CNN\n")
        f.write(f"  - YOLOv8 inference is {inference_ratio:.2f}x faster than Faster R-CNN\n")
        
        f.write("\nConclusion:\n")
        if fps_ratio > 1.2:
            f.write("  - YOLOv8 significantly outperforms Faster R-CNN in speed\n")
        elif fps_ratio < 0.8:
            f.write("  - Faster R-CNN significantly outperforms YOLOv8 in speed\n")
        else:
            f.write("  - Both models perform similarly in terms of speed\n")
    
    print("\nBenchmark comparison completed!")
    print(f"Results saved to the 'benchmark_comparison' directory")
    print(f"Check 'benchmark_comparison/benchmark_report.txt' for detailed analysis")

if __name__ == "__main__":
    # Input video path
    input_video = '/home/fteam5/m/Crownd_People_Counter/input.mp4'
    
    # Ask user if they want to run the benchmarks or just generate comparison
    choice = input("Run new benchmarks? (y/n): ").lower()
    
    if choice == 'y':
        run_benchmarks(input_video)
    
    # Load benchmark data and generate comparison
    yolo_data, rcnn_data = load_benchmark_data()
    generate_comparison(yolo_data, rcnn_data)
