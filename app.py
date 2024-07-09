from fastapi import FastAPI, File, Request, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from ultralytics import YOLO
from PIL import Image
import uuid
from os.path import abspath
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import timm
import  torch.nn as nn
import requests
import matplotlib.pyplot as plt
from torchvision import transforms

app = FastAPI()

# Đường dẫn đến mô hình YOLO đã train
yolo_model_v8x_path = "runs/detect/train20/weights/best.pt"
model_path = 'checkpoints/best_model.pt'
# excel_file = "convert_data/data_refine.xlsx"


# Thiết lập thư mục chứa tệp tĩnh
app.mount("/static", StaticFiles(directory="static"), name="static")

# Tạo đối tượng Jinja2Templates để render trang HTML
templates = Jinja2Templates(directory="static")

# Endpoint để truy cập trang web
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("frontend_demo.html", {"request": request})

# Endpoint xử lý tải lên ảnh và dự đoán
@app.post("/upload/")
async def upload_file(file: UploadFile):
    # Kiểm tra xem tệp tải lên có phải là hình ảnh không
    if file.content_type.startswith("image"):
        # Lưu file tải lên vào thư mục tạm
        random_filename = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join("static", "uploads", random_filename)
        with open(image_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Chọn model dựa trên giá trị model được gửi từ FE
        yolo_model_path = yolo_model_v8x_path

        # Gọi hàm để dự đoán
        image_path_absolute = abspath(image_path)
        predictions, im = predict_with_model(image_path_absolute, yolo_model_path, model_path)

        im.save(image_path)
        # Xóa file tạm sau khi dự đoán
        # os.remove(file.filename)

        return {
            "predictions": predictions,
            "image_path": image_path
        }
        
    else:
        return {"error": "File is not an image"}


def predict_with_model(image_path, yolo_model_path, model_path):
    label_map = {
        0 : "Normal",
        1 : "Advance edge error",
        2 : "Edge error",
        3 : "Shape error",
        4 : "Baking error",
        5 : "Size error",
        6 : "Topping error",
        7 : "Fermentation condition error"
    }
    yolo = YOLO(model=yolo_model_path)
    img_size = (256, 256)
    model_name = 'efficientnet_b5'
    num_classes = 8
    model = timm.create_model(model_name, pretrained=True, in_chans=3, num_classes=num_classes)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, out_features=1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 8),
        nn.Sigmoid()  # Sử dụng sigmoid cho multi-label classification
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(model_path))
    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    predictions = []
    img = transform_test(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    model.eval().to(device)
    with torch.no_grad():
        outputs = model(img)

    for idx in torch.topk(outputs, k=8).indices.squeeze(0).tolist():
        prob = (outputs)[0, idx].item()
        label = '{label:<10} ({p:.2f}%)'.format(label=label_map[idx], p=prob*100)
        predictions.append(label)
    # Lặp qua từng bounding box và hiển thị class và confidence tương ứng
    results = yolo(image_path)
    bboxes = results[0].boxes
    if bboxes is not None and len(bboxes) > 0:
        for i, box in enumerate(bboxes):
            x, y, w, h = box.xywh.squeeze().tolist()
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
            I = Image.open(image_path)
            draw = ImageDraw.Draw(I)
            draw.rectangle(((x1, y1), (x2, y2)), outline='red', width=3)


    return predictions, I
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)