from ultralytics import YOLO
import torch


if __name__ == '__main__':
    model = YOLO('yolo11m.pt')  # Load model

    results = model.train(
        data='C:/Users/Admin/Desktop/PYTHON/AIO/Module 6/OCR_Yolo_CNN/yolo_data/data.yml',
        epochs=100,
        imgsz=640, 
        cache = True,
        patience =20,
        plots = True,
        device = 0
    )

    # print(f"PyTorch version: {torch.__version__}")
    # print(f"CUDA available: {torch.cuda.is_available()}")
    # if torch.cuda.is_available():
    #     print(f"Using device: {torch.cuda.get_device_name(0)}")