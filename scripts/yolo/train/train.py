from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

# Train the model
model.train(
    data='/home/rteam2/m15kh/Human_Counter/data.yaml',  
    epochs=50,  
    imgsz=640,
    batch=16,  
    name='yolo8_people_detection' 
)