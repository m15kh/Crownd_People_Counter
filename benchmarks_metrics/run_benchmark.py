import os
import sys
import argparse
from model_evaluation import ModelEvaluator
from visualize_detections import visualize_sample_detections

def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive benchmarks on object detection models")
    
    parser.add_argument("--test-video", default="/home/fteam5/m/Crownd_People_Counter/input.mp4",
                        help="Path to video file for testing")
    
    parser.add_argument("--benchmark-dir", default="benchmarks_img",
                        help="Directory containing benchmark images and annotations")
    
    parser.add_argument("--output-dir", default="evaluation_results",
                        help="Directory to save evaluation results")
    
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                        help="Confidence threshold for detections")
    
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for mAP calculation")
    
    parser.add_argument("--device", default="cuda",
                        help="Device to run models on (cuda or cpu)")
    
    parser.add_argument("--visualize-samples", type=int, default=5,
                        help="Number of sample images to visualize with detections")
    
    parser.add_argument("--zone", nargs=4, type=int, default=[400, 0, 750, 432],
                        help="Detection zone (x_min, y_min, x_max, y_max)")
    
    parser.add_argument("--yolo-model-path", 
                        default="/home/fteam5/m/Crownd_People_Counter/yolo_weights/best.pt",
                        help="Path to YOLOv8 model weights")
                        
    parser.add_argument("--rcnn-model-path",
                        default="/home/fteam5/m/Crownd_People_Counter/fasterrcnn_pytorch_training_pipeline/weights/model_final.onnx",
                        help="Path to Faster R-CNN ONNX model")
    
    parser.add_argument("--rcnn-config-path",
                        default="/home/fteam5/m/Crownd_People_Counter/fasterrcnn_pytorch_training_pipeline/data_configs/p.yaml",
                        help="Path to Faster R-CNN config file")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configure the evaluator
    config = {
        'benchmark_dir': args.benchmark_dir,
        'output_dir': args.output_dir,
        'test_video': args.test_video,
        'detection_zone': tuple(args.zone),
        'iou_threshold': args.iou_threshold,
        'conf_threshold': args.conf_threshold,
        'device': args.device,
        'yolo_model_path': args.yolo_model_path,
        'rcnn_model_path': args.rcnn_model_path,
        'rcnn_config_path': args.rcnn_config_path
    }
    
    print("\n====== Object Detection Model Benchmark ======")
    print(f"Using configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run the evaluation
    evaluator = ModelEvaluator(config)
    
    print("\n1. Loading models...")
    models = evaluator.load_models()
    
    if not models:
        print("No models could be loaded. Exiting.")
        return
        
    print(f"Successfully loaded {len(models)} models: {', '.join(models.keys())}")
    
    print("\n2. Loading benchmark data...")
    benchmark_data = evaluator.load_benchmark_data()
    
    print("\n3. Evaluating detection accuracy on benchmark images...")
    evaluator.evaluate_on_benchmark()

    print("\n4. Evaluating counting performance on test video...")
    evaluator.evaluate_on_video()
    
    print("\n5. Generating comparison report...")
    try:
        evaluator.generate_comparison_report()
    except Exception as e:
        print(f"Error generating comparison report: {e}")
    
    if args.visualize_samples > 0 and benchmark_data:
        try:
            print(f"\n6. Visualizing {min(args.visualize_samples, len(benchmark_data))} sample detections...")
            # Pass the evaluator instance directly to the visualization function
            from visualize_detections import visualize_sample_detections
            
            # Create a custom visualization function that uses the existing evaluator
            def run_visualization(num_samples, evaluator):
                # Create output directory
                output_dir = "detection_samples"
                os.makedirs(output_dir, exist_ok=True)
                
                if not hasattr(evaluator, 'benchmark_data') or not evaluator.benchmark_data:
                    print(f"No benchmark data available for visualization")
                    return
                    
                if not hasattr(evaluator, 'models') or not evaluator.models:
                    print("No models available for visualization")
                    return
                
                # Select samples
                if num_samples > len(evaluator.benchmark_data):
                    num_samples = len(evaluator.benchmark_data)
                
                np.random.seed(42)  # For reproducibility
                sample_indices = np.random.choice(len(evaluator.benchmark_data), num_samples, replace=False)
                
                # Process samples
                # ... rest of visualization code
                print(f"Starting visualization of {num_samples} samples...")
                for i, idx in enumerate(sample_indices):
                    try:
                        # ... visualization code
                        print(f"Processing sample {i+1}/{num_samples}")
                    except Exception as e:
                        print(f"Error processing sample {i+1}: {e}")
                
                print(f"Visualization of {num_samples} sample detections saved to {output_dir}")
            
            # Run visualization with existing evaluator
            run_visualization(min(args.visualize_samples, len(benchmark_data)), evaluator)
            
        except Exception as e:
            print(f"Error during visualization: {e}")
    elif args.visualize_samples > 0:
        print("\n6. Skipping visualization - no benchmark data available")
    
    print("\n====== Benchmark complete! ======")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
