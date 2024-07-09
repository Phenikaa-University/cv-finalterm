import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import wandb
from tqdm import tqdm
from models.efficientnet import create_model
from utils import get_data_loaders, plot_loss_curve
import time

def get_config():
    parser = argparse.ArgumentParser(description="Train EfficientNet Model")
    parser.add_argument('--train_dir', type=str, default="./dataset/images/train", help='Path to training images directory')
    parser.add_argument('--test_dir', type=str, default="./dataset/images/test",help='Path to testing images directory')
    parser.add_argument('--train_label', type=str, default="./dataset/labels/train_labels.xlsx",help='Path to training labels file')
    parser.add_argument('--test_label', type=str, default="./dataset/labels/test_labels.xlsx",help='Path to testing labels file')
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='Image size (width, height)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='efficientnet_b5', help='Model name')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--project_name', type=str, default="cv-finalterm", help='WandB project name')
    
    return parser.parse_args()

def train_model(config):
    # Initialize wandb
    wandb.init(project=config.project_name, config=config)
    config = wandb.config
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = get_data_loaders(
        config.train_dir, config.test_dir, 
        config.train_label, config.test_label, 
        config.batch_size, config.img_size
    )
    
    print(len(train_loader), len(test_loader))
    
    # Create model
    model = create_model(config.model_name, config.num_classes)
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    best_loss = float('inf')
    best_accuracy = 0.0  # Initialize best accuracy
    train_losses = []
    val_losses = []
    
    for epoch in range(config.num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        
        train_loader_desc = f'Epoch {epoch + 1}/{config.num_epochs}'
        train_data_loader = tqdm(train_loader, desc=train_loader_desc, leave=False)
        
        for inputs, labels, _ in train_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        train_losses.append(average_loss)
        print(f'Epoch [{epoch + 1}/{config.num_epochs}] - Loss: {average_loss:.4f}')
        wandb.log({"Training Loss": average_loss})
        
        # Validate model
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
        if len(test_loader) > 0:
            average_val_loss = total_val_loss / len(test_loader)
        else:
            average_val_loss = float('inf')  # or some other appropriate value or handling
        val_losses.append(average_val_loss)
        print(f'Validation Loss: {average_val_loss:.4f}')
        wandb.log({"Validation Loss": average_val_loss})
        
        accuracy = correct / total
        print(f'Validation Accuracy: {accuracy:.4f}')
        wandb.log({"Validation Accuracy": accuracy})
        
        # Save the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'checkpoints/best_model_based_acc.pt')
    
        average_val_loss = total_val_loss / len(test_loader)
        val_losses.append(average_val_loss)
        print(f'Validation Loss: {average_val_loss:.4f}')
        wandb.log({"Validation Loss": average_val_loss})
        
        # Save the best model
        if average_val_loss < best_loss:
            best_loss = average_val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model_based_loss.pt')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch + 1}.pt')
        
        elapsed_time = time.time() - start_time
        train_data_loader.set_description_str(f'Epoch {epoch + 1}/{config.num_epochs} - Time: {elapsed_time:.2f}s')
    
    # Plot loss curve
    plot_loss_curve(train_losses, val_losses, 'loss_curve.png')

if __name__ == '__main__':
    config = get_config()
    train_model(config)