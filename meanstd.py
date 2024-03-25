import torch
import torchvision.transforms as transforms
import torchvision


def calculate_mean_std(dataset):
    channels_sum = torch.zeros(3)  # for RGB images
    channels_squared_sum = torch.zeros(3)  # for RGB images
    num_batches = 0

    # Iterate over the dataset to compute the sum and squared sum
    for data, _ in dataset:
        channels_sum += torch.mean(data, dim=[1, 2])
        channels_squared_sum += torch.mean(data**2, dim=[1, 2])
        num_batches += 1

    # Calculate the mean and std
    mean = channels_sum / num_batches
    std = torch.sqrt(channels_squared_sum / num_batches - mean**2)

    return mean, std


# Example usage:
datasets = ["cifar10", "cifar100", "qmnist", "fashion-mnist"]

for dataset_name in datasets:
    if dataset_name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_name == "cifar100":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_name == "qmnist":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.QMNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_name == "fashion-mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        raise ValueError(
            "Dataset not supported. Please choose 'cifar10', 'cifar100', 'qmnist' or 'fashion-mnist'."
        )

    mean, std = calculate_mean_std(dataset)
    print("Dataset:", dataset_name)
    print("Mean:", mean)
    print("Std:", std)
    print()
