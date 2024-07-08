import argparse
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from models.efficientnet import create_model
from utils import get_data_loaders

def get_config():
    parser = argparse.ArgumentParser(description="Test EfficientNet Model")
    parser.add_argument('--train_dir', type=str, default="test/dataset/images/train", help='Path to training images directory')
    parser.add_argument('--test_dir', type=str, default="test/dataset/images/test",help='Path to testing images directory')
    parser.add_argument('--train_label', type=str, default="test/dataset/labels/train_labels.xlsx",help='Path to training labels file')
    parser.add_argument('--test_label', type=str, default="test/dataset/labels/test_labels.xlsx",help='Path to testing labels file')
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='Image size (width, height)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--model_path', type=str, default="checkpoints/best_models.pt", help='Path to saved model')
    parser.add_argument('--model_name', type=str, default='efficientnet_b5', help='Model name')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--project_name', type=str, default="Test CV-finalterm", help='WandB project name')
    
    return parser.parse_args()

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    return y_true, y_pred

def plot_roc_curve(y_true, y_pred, num_classes, save_path):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc_score(y_true[:, i], y_pred[:, i]):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.show()

def main(config):
    wandb.init(project=config.project_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_loader = get_data_loaders(
        config.train_dir, config.test_dir, 
        config.train_label, config.test_label, 
        config.batch_size, config.img_size
    )

    model = create_model(config.model_name, config.num_classes)
    # model.load_state_dict(torch.load(config.model_path))
    model.load_state_dict(torch.load(config.model_path, map_location=torch.device(device)))
    model.to(device)

    y_true, y_pred = evaluate_model(model, test_loader, device)
    # print(y_true)
    # print(y_pred)

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    # print(y_true)
    # print(y_pred)

    accuracy = accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    precision = precision_score(y_true, y_pred.round(), average='weighted')
    recall = recall_score(y_true, y_pred.round(), average='weighted')
    f1 = f1_score(y_true, y_pred.round(), average='weighted')
    # roc_auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    # print(f'ROC AUC Score: {roc_auc:.4f}')

    wandb.log({
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        # "ROC AUC Score": roc_auc
    })

    plot_roc_curve(y_true.numpy(), y_pred.numpy(), config.num_classes, 'roc_curve.png')
    plot_confusion_matrix(y_true.numpy(), y_pred.numpy(), 'confusion_matrix.png')

    wandb.save('roc_curve.png')
    wandb.save('confusion_matrix.png')

if __name__ == '__main__':
    main(get_config())