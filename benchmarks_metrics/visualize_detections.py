import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model_evaluation import ModelEvaluator

def visualize_sample_detections(num_samples=5):
    """
    Visualize sample detections from each model on the same benchmark images
    """
    # Create output directory
    output_dir = "detection_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the model evaluator
    evaluator = ModelEvaluator()
    
    # Load models and benchmark data
    models = evaluator.load_models()
    evaluator.models = models  # Make sure models are assigned to the evaluator
    benchmark_data = evaluator.load_benchmark_data()
    
    if not benchmark_data:
        print(f"No benchmark data available for visualization")
        return
        
    if not models:
        print("No models available for visualization")
        return
    
    # Select random samples
    if num_samples > len(benchmark_data):
        num_samples = len(benchmark_data)
    
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(len(benchmark_data), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        try:
            sample = benchmark_data[idx]
            image_path = sample['image_path']
            annotation_path = sample['annotation_path']
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Parse ground truth annotations
            gt_annotations = evaluator.parse_annotations(annotation_path)
            
            # Create a figure with subplots (GT + one for each model)
            fig, axes = plt.subplots(1, len(models) + 1, figsize=(20, 5))
            
            # Plot ground truth
            gt_img = image.copy()
            for ann in gt_annotations:
                bbox = ann['bbox']
                cv2.rectangle(gt_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            axes[0].imshow(gt_img)
            axes[0].set_title(f"Ground Truth\n{len(gt_annotations)} people")
            axes[0].axis('off')
            
            # Plot model detections
            for j, (model_name, model_info) in enumerate(models.items()):
                # Run detection
                try:
                    if model_info['type'] == 'yolov8':
                        detections = evaluator.evaluate_detection_yolov8(image_path, gt_annotations)
                    elif model_info['type'] == 'faster_rcnn':
                        detections = evaluator.evaluate_detection_fasterrcnn(image_path, gt_annotations)
                    else:
                        detections = []
                except Exception as e:
                    print(f"Error detecting with {model_name}: {e}")
                    detections = []
                
                # Draw detections
                det_img = image.copy()
                for det in detections:
                    bbox = det['bbox']
                    conf = det['confidence']
                    cv2.rectangle(det_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    cv2.putText(det_img, f"{conf:.2f}", (int(bbox[0]), int(bbox[1]) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                axes[j+1].imshow(det_img)
                axes[j+1].set_title(f"{model_name}\n{len(detections)} detections")
                axes[j+1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{i+1}_comparison.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
    
    print(f"Visualization of {num_samples} sample detections saved to {output_dir}")

if __name__ == "__main__":
    visualize_sample_detections(5)
