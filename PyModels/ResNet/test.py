import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from MainNNModel import ResNet18
from MainNNModel import ResNet34
from MainNNModel import ResNet50
from MainNNModel import ResNet101

image_val_path = '/Users/wangsiwei/C++Code/ONNX_Samples/dataset/tiny-imagenet-200/test'  

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(image_val_path, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ResNet50().to(device)
model.load_state_dict(torch.load('best_resnet50.pth', map_location=device))
model.eval()

def main():
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f'Test Accuracy: {acc:.2f}%')

if __name__ == '__main__':
    main() 