# This component is for ResNet ONLY!
import torch
from torch.cuda import is_available
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from MainNNModel import ResNet18
from MainNNModel import ResNet34
from MainNNModel import ResNet50
from MainNNModel import ResNet101

import os

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_train_path = '/Users/wangsiwei/C++Code/ONNX_Samples/dataset/tiny-imagenet-200/train'
image_val_path = '/Users/wangsiwei/C++Code/ONNX_Samples/dataset/tiny-imagenet-200/val'



train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]
)

val_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]
)

train_dataset = datasets.ImageFolder(image_train_path, transform = train_transform)
val_dataset = datasets.ImageFolder(image_val_path, transform = val_transform)

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers= 8, pin_memory= True)
val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = False, num_workers = 8, pin_memory = True)

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



model = ResNet50().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = 0.001, weight_decay= 1e-4)
# No scheduler for AdamW optimizer

def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_index, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        input_device = inputs.to(device)
        target_device = targets.to(device)

        outputs = model(input_device)
        loss = criterion(outputs, target_device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target_device.size(0)
        correct += predicted.eq(target_device).sum().item()

        if batch_index % 20 == 0:
            print(f'Epoch {epoch} [{batch_index}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
            
    print(f'Epoch {epoch} finished. Train Loss:  {running_loss/len(train_loader):.4f} '
           f'Acc: {100.*correct/total:.2f}%')

def validate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs_device = inputs.to(device)
            targets_device = targets.to(device)

            outputs = model(inputs_device)
            _, prediction_index = outputs.max(1)
            total += targets_device.size(0)
            correct += prediction_index.eq(targets_device).sum().item()

    acc = 100. * correct / total
    print(f'Validation Accuracy: {acc:.2f}%')

    return acc

def main(num_epoch):
    best_acc = 0
    for epoch in range(num_epoch):
        train(epoch)
        acc = validate()
        
        if  acc > best_acc:
            best_acc = acc
            pth_path = os.path.join(output_dir, f'best_resnet50_epoch{epoch}.pth')
            torch.save(model.state_dict(), pth_path)
            print(f'Best model saved at {pth_path}!')


if __name__ == '__main__':
    epoch = 30
    main(epoch)

