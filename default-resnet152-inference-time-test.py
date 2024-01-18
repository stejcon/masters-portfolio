import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import time

# Function to measure inference time
def measure_inference_time(model, dataloader, device):
    model.eval()
    total_time = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time

    average_time = total_time / num_batches
    return average_time

# Function to load CIFAR-10 dataset
def load_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    cifar10_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ResNet152 model
resnet18_model = resnet50().to(device)
resnet34_model = resnet50().to(device)
resnet50_model = resnet50().to(device)
resnet101_model = resnet50().to(device)
resnet152_model = resnet152().to(device)

models = [model.to(device) for model in [resnet18(), resnet34(), resnet50(), resnet101(), resnet152()]]

# Specify batch size
batch_size = 64

# Load datasets
cifar10_dataloader = load_cifar10(batch_size)

inferenceTimes = [measure_inference_time(model, cifar10_dataloader, device) for model in models]
for iTime in inferenceTimes:
    print(f'Average Inference Time on CIFAR-10: {iTime:.4f} seconds')
