import timm
import torch.nn as nn

def create_model(model_name="efficientnet_b5", num_classes=8):
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
        nn.Linear(128, num_classes),
        nn.Sigmoid() # Sử dụng sigmoid cho multi-label classification
    )
    return model