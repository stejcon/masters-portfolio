import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152
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
resnet152_model = resnet152(pretrained=True).to(device)

# Specify batch size
batch_size = 64

# Load datasets
cifar10_dataloader = load_cifar10(batch_size)

# Measure average inference time for CIFAR-10
cifar10_avg_time = measure_inference_time(resnet152_model, cifar10_dataloader, device)
print(f'Average Inference Time on CIFAR-10: {cifar10_avg_time:.4f} seconds')
