import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score
import warnings

# Suppress torchvision warnings
warnings.filterwarnings("ignore")

def train_vision_model():
    print("--- Starting Vision Fine-Tuning ---")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vision_dir = os.path.join(base_dir, "data", "vision")
    
    num_flooded = len(os.listdir(os.path.join(vision_dir, "flooded")))
    num_clear = len(os.listdir(os.path.join(vision_dir, "clear")))
    print(f"Dataset Size - Flooded: {num_flooded}, Clear: {num_clear}")
    
    if num_flooded == 0 or num_clear == 0:
        print("Error: Missing images in one of the classes. Cannot train.")
        return
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Skip corrupt images in ImageFolder
    def is_valid_file(path):
        try:
            from PIL import Image
            img = Image.open(path)
            img.verify()
            return True
        except:
            return False

    dataset = datasets.ImageFolder(root=vision_dir, transform=transform, is_valid_file=is_valid_file)
    print(f"Valid images loaded: {len(dataset)}")
    print(f"Classes: {dataset.class_to_idx}")
    
    if len(dataset) < 4:
        print("Too few valid images to train and evaluate.")
        return
        
    # Train/Test Split (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Pre-trained MobileNetV2 (using new weights parameter to avoid warnings)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    
    EPOCHS = 5
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f}")
        
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    class_idx = dataset.class_to_idx
    flooded_idx = class_idx.get("flooded", 1)
    
    prec = precision_score(all_labels, all_preds, pos_label=flooded_idx, zero_division=0)
    rec = recall_score(all_labels, all_preds, pos_label=flooded_idx, zero_division=0)
    
    print("\n--- VISION EVALUATION (Held-Out Test Set) ---")
    print(f"Precision (Flooded): {prec:.4f}")
    print(f"Recall (Flooded):    {rec:.4f}")
    
    # Save Model
    torch.save(model.state_dict(), os.path.join(base_dir, "models", "vision_classifier.pth"))
    
    with open(os.path.join(base_dir, "reports", "vision_evaluation_report.md"), "w") as f:
        f.write("# Computer Vision (Flood Detection) Evaluation\n\n")
        f.write(f"Dataset sourced from real flood imagery (UAV/Street view).\n")
        f.write(f"Total valid dataset size: {len(dataset)} images.\n\n")
        f.write(f"## Metrics (Held-Out Test Set)\n")
        f.write(f"- **Precision:** {prec:.4f}\n")
        f.write(f"- **Recall:** {rec:.4f}\n")

if __name__ == "__main__":
    train_vision_model()
