# Human Counter


### Video Results

- **DINO**:  
  <video src="assets/dino.mp4" controls width="600"></video>

- **Fast R-CNN**:  
  <video src="assets/rcnn.mp4" controls width="600"></video>

- **YOLO**:  
  <video src="assets/yolo.mp4" controls width="600"></video>

- **YOLO Counter**:  
  <video src="assets/yolo_counter.mp4" controls width="600"></video>

> **Note**: Inline video playback may not work on GitHub. If the videos do not display, you can download them from the links above or access them on Google Drive:  
[Google Drive Link](https://drive.google.com/drive/u/0/folders/1FGkSAPAb_RJfBTVlrp1HlUTjGZn_Wlpn)

### Methodology

- **DINO**:  
  I used the [DINO repository](https://github.com/facebookresearch/dino) for zero-shot inference, which does not require any labeled data.

- **YOLO**:  
  For training YOLO, I used [Roboflow](https://roboflow.com/) for dataset preparation and [LabelMe](https://github.com/wkentaro/labelme) for annotation.

- **Fast R-CNN**:  
  I followed the tutorial from this [blog post](https://debuggercafe.com/optimizing-faster-rcnn-mobilenetv3-for-real-time-inference-on-cpu/). However, due to time constraints, I did not train the model for our specific purpose, which may affect the results.



