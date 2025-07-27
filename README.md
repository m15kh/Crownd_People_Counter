# Human Counter


### Video Results

- **DINO**:  
  <video src="assets/dino.gif" controls width="600"></video>  

- **Fast R-CNN**:  
  <video src="assets/gif.mp4" controls width="600"></video>  

- **YOLO**:  
  <video src="assets/yolo.gif" controls width="600"></video>  


> **Note**: Inline video playback may not work on GitHub. If the videos do not display, you can download them from the links above or access them on Google Drive:  
[Google Drive Link](https://drive.google.com/drive/u/0/folders/1FGkSAPAb_RJfBTVlrp1HlUTjGZn_Wlpn)

### Methodology

- **DINO**:  
  I used the [DINO repository](https://github.com/facebookresearch/dino) for zero-shot inference, which does not require any labeled data.

- **YOLO**:  
  For training YOLO, I used [Roboflow](https://roboflow.com/) for dataset preparation and [LabelMe](https://github.com/wkentaro/labelme) for annotation.

- **Fast R-CNN**:  
  I followed the tutorial from this [blog post](https://debuggercafe.com/optimizing-faster-rcnn-mobilenetv3-for-real-time-inference-on-cpu/). However, due to time constraints, I did not train the model for our specific purpose, which may affect the results.

======================================================
        OBJECT DETECTION MODEL BENCHMARK REPORT        
======================================================

YOLO v8:
  - Average FPS: 91.07
  - Average Inference Time: 0.0116 seconds
  - Average Memory Usage: 1321.96 MB
  - Maximum Memory Usage: 1326.38 MB
  - Total Processing Time: 4.00 seconds
  - Total Frames Processed: 298

Faster R-CNN:
  - Average FPS: 19.74
  - Average Inference Time: 0.0852 seconds
  - Average Memory Usage: 4837.97 MB
  - Maximum Memory Usage: 5279.17 MB
  - Total Processing Time: 28.40 seconds
  - Total Frames Processed: 298

Performance Comparison:
  - YOLOv8 is 4.61x faster in FPS compared to Faster R-CNN
  - YOLOv8 inference is 7.32x faster than Faster R-CNN

Conclusion:
  - YOLOv8 significantly outperforms Faster R-CNN in speed

> **Note**: This comparison was conducted using the script `benchmark_comparison\benchmark_comparison.py`.  
Additionally, a visual representation of the performance comparison has been added:



