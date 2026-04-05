from ultralytics import YOLO

# 1. Sınıflandırma modelini yükle (yolov8n-cls)
model = YOLO('yolov8n-cls.pt')

# 2. Eğitimi başlat
results = model.train(
    data='C:/Users/90554/OneDrive/Masaüstü/kalem_dataset.1', 
    epochs=50, 
    imgsz=224, 
    device='cpu'
)