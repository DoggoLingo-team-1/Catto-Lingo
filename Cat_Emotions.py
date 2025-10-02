import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import mlflow
import mlflow.pytorch

DATA_DIR = r"C:\Users\abdal\Downloads\Compressed\final_data"
CONFIG = {
    "BATCH_SIZE": 16,
    "NUM_EPOCHS": 30,
    "IMG_SIZE": 224,
    "LR": 0.0001,
    "NUM_WORKERS": 4,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(CONFIG["IMG_SIZE"]),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_data():
    """تحميل البيانات وإعداد الـ DataLoaders."""
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=CONFIG["NUM_WORKERS"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])

    classes = train_dataset.classes
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, classes

def train_and_validate(model, criterion, optimizer, train_loader, val_loader, train_dataset, val_dataset, num_epochs):
    """دورة تدريب النموذج والتحقق من صحته."""
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(outputs.argmax(1) == labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(outputs.argmax(1) == labels.data)
        
        val_loss /= len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Log metrics with MLflow
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_acc", epoch_acc.item(), step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc.item(), step=epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best model found with validation accuracy: {best_val_acc:.4f}. Saving model...")
            mlflow.pytorch.log_model(model, "best_model")

def test_model(model, test_loader, classes):
    """تقييم النموذج على مجموعة الاختبار وإنشاء التقارير."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    
    # Log metrics from report to MLflow
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            mlflow.log_metric(f"test_precision_{class_name}", metrics.get("precision", 0))
            mlflow.log_metric(f"test_recall_{class_name}", metrics.get("recall", 0))
            mlflow.log_metric(f"test_f1-score_{class_name}", metrics.get("f1-score", 0))
    mlflow.log_metric("test_accuracy", report["accuracy"])

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    # Log figure to MLflow
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.show()


# ================== 4. Main Function ==================
if __name__ == "__main__":
    # Start a new MLflow run
    with mlflow.start_run():
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        
        # Log all parameters
        mlflow.log_params(CONFIG)
        
        # Load and prepare data
        train_loader, val_loader, test_loader, train_dataset, _, _, classes = load_data()
        print("Classes:", classes)

        # Calculate and log class weights
        y_train = [s[1] for s in train_dataset.samples]
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        print("Class Weights:", class_weights.cpu().numpy())
        for i, weight in enumerate(class_weights):
            mlflow.log_param(f"class_weight_{classes[i]}", weight.item())
        
        # Initialize the model, loss, and optimizer
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(classes))
        model = model.to(DEVICE)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["LR"])
        
        # Train and validate the model
        train_and_validate(model, criterion, optimizer, train_loader, val_loader, train_dataset, _, CONFIG["NUM_EPOCHS"])
        
        # Load the best model from MLflow and test
        print("\nLoading the best model from MLflow for final evaluation...")
        best_model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
        best_model = mlflow.pytorch.load_model(best_model_uri)
        best_model = best_model.to(DEVICE)
        
        test_model(best_model, test_loader, classes)