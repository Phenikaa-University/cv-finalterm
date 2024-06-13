"""Implementation of EfficientNet in PyTorch."""
import os
import cv2
import pandas as pd
import numpy as np

import timm
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import time
from tqdm import tqdm
from data.custom_dataset import CustomDataset


img_size = (256, 256)
n_label = 8
batch_size = 64
train_dir = 'dataset_yolov8x/train'
test_dir = 'dataset_yolov8x/test'
train_label = 'dataset_yolov8x/data_refine_all_errors_train.xlsx'
test_label = 'dataset_yolov8x/data_refine_all_errors_test.xlsx'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform_train = transforms.Compose([
    transforms.Resize(img_size),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



train_dataset = CustomDataset(train_dir, train_label, transform=transform_train)
test_dataset = CustomDataset(test_dir, test_label, transform=transform_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


model_name = 'efficientnet_b5'
num_classes = 8
# model = models.efficientnet_b7(pretrained=True, in_channels=528, num_classes=num_classes)

model = timm.create_model(model_name, pretrained=True, in_chans=3, num_classes=num_classes)
# model = EfficientNet.from_pretrained(model_name)
print(model.classifier.in_features)

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

# In mô hình để kiểm tra
print(model)
model.to(device)

criterion = nn.BCELoss()  # Sử dụng binary cross entropy loss vì bạn đang thực hiện multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training và đánh giá mô hình
num_epochs = 100  # Thay đổi số epoch tùy ý

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    # Sử dụng tqdm để theo dõi tiến trình training
    train_loader_desc = f'Epoch {epoch + 1}/{num_epochs}'
    train_data_loader = tqdm(train_loader, desc=train_loader_desc, leave=False)
    
    for inputs, labels in train_data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}')
    if (epoch + 1) == 1:
        # Lưu trọng số sau epoch đầu tiên
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch + 1}.pt')
    
    if (epoch + 1) % 5 == 0:
        # Lưu trọng số sau mỗi 10 epoch
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch + 1}.pt')
    
    # Đánh giá mô hình trên tập kiểm tra
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs[outputs >= 0.5] = 1  # Áp đặt ngưỡng 0.5 để chuyển đầu ra thành 0 hoặc 1
            outputs[outputs < 0.5] = 0
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy on test data: {accuracy:.2f}%')
    
    # Hiển thị thời gian training của từng epoch
    train_data_loader.set_description_str(f'Epoch {epoch + 1}/{num_epochs} - Time: {time.time() - start_time:.2f}s')

# In kết quả cuối cùng sau khi training
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs[outputs >= 0.5] = 1  # Áp đặt ngưỡng 0.5 để chuyển đầu ra thành 0 hoặc 1
        outputs[outputs < 0.5] = 0
        total += labels.size(0)
        correct += (outputs == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Final accuracy on test data: {accuracy:.2f}%')