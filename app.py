from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import uuid
from os.path import abspath
import torch
import timm
import torch.nn as nn
from torchvision import transforms

app = FastAPI()

# Paths to models
YOLO_MODEL_PATH = "runs/detect/train20/weights/best.pt"
MODEL_PATH = 'checkpoints/best_model.pt'

# Set up static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="static")

# Label map for predictions
LABEL_MAP = {
    0: "Normal",
    1: "Advance edge error",
    2: "Edge error",
    3: "Shape error",
    4: "Baking error",
    5: "Size error",
    6: "Topping error",
    7: "Fermentation condition error"
}

# Image transformation
TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load YOLO model once
YOLO_MODEL = YOLO(model=YOLO_MODEL_PATH)

# Load classification model once
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CLASSIFICATION_MODEL = timm.create_model('efficientnet_b5', pretrained=True, in_chans=3, num_classes=8)
CLASSIFICATION_MODEL.classifier = nn.Sequential(
    nn.Linear(CLASSIFICATION_MODEL.classifier.in_features, out_features=1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 8),
    nn.Sigmoid()
)
CLASSIFICATION_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
CLASSIFICATION_MODEL.eval().to(DEVICE)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the main HTML page.
    
    Args:
        request (Request): The request object.
    
    Returns:
        TemplateResponse: The rendered HTML page.
    """
    return templates.TemplateResponse("frontend_demo.html", {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile):
    """
    Handle file upload and make predictions.
    
    Args:
        file (UploadFile): The uploaded file.
    
    Returns:
        dict: A dictionary containing predictions and image path.
    """
    if file.content_type.startswith("image"):
        random_filename = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join("static", "uploads", random_filename)
        
        with open(image_path, "wb") as buffer:
            buffer.write(file.file.read())

        image_path_absolute = abspath(image_path)
        predictions, im = predict_with_model(image_path_absolute)

        im.save(image_path)

        return {
            "predictions": predictions,
            "image_path": image_path
        }
    else:
        return {"error": "File is not an image"}

def predict_with_model(image_path):
    """
    Predict the class and bounding boxes for the given image.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
        tuple: A tuple containing predictions and the image with bounding boxes.
    """
    predictions = []
    try:
        img = TRANSFORM_TEST(Image.open(image_path).convert('RGB')).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = CLASSIFICATION_MODEL(img)
        
        for idx in torch.topk(outputs, k=8).indices.squeeze(0).tolist():
            prob = (outputs)[0, idx].item()
            label = '{label:<10} ({p:.2f}%)'.format(label=LABEL_MAP[idx], p=prob*100)
            predictions.append(label)

        results = YOLO_MODEL(image_path)
        bboxes = results[0].boxes
        if bboxes is not None and len(bboxes) > 0:
            with Image.open(image_path) as im:
                draw = ImageDraw.Draw(im)
                for box in bboxes:
                    x, y, w, h = box.xywh.squeeze().tolist()
                    x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
                    draw.rectangle(((x1, y1), (x2, y2)), outline='red', width=3)
        return predictions, im
    except Exception as e:
        print(f"Error in prediction: {e}")
        return [], None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)