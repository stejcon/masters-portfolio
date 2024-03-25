import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
import generation
import tempfile
import sys
import importlib

writer = SummaryWriter()


def getDevice():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    return device


def Cifar10Splits(batchSize=64):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.49139968, 0.48215827, 0.44653124],
                std=[0.24703233, 0.24348505, 0.26158768],
            ),
        ]
    )

    trainDataset = datasets.CIFAR10(
        train=True, root="./data", transform=train_transform, download=True
    )
    testDataset = datasets.CIFAR10(
        train=False, root="./data", transform=test_transform, download=True
    )

    # Split train dataset to have 10% validation dataset
    valid_size = 0.10
    num_train = len(trainDataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    trainSampler = SubsetRandomSampler(train_idx)
    validSampler = SubsetRandomSampler(valid_idx)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, sampler=trainSampler)
    validLoader = DataLoader(trainDataset, batch_size=batchSize, sampler=validSampler)
    testLoader = DataLoader(testDataset, batch_size=batchSize)

    return trainLoader, validLoader, testLoader


def get_custom_dataloaders(dataset_name, batch_size=64, validation_split=0.1):
    if dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616],
                ),
            ]
        )
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset_name == "cifar100":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4866, 0.4409],
                    std=[0.2673, 0.2564, 0.2761],
                ),
            ]
        )
        train_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset_name == "qmnist":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        train_dataset = torchvision.datasets.QMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.QMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset_name == "fashion-mnist":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        train_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(
            "Dataset not supported. Please choose 'cifar10', 'cifar100', 'qmnist' or 'fashion-mnist'."
        )

    # Split the training dataset into training and validation sets
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(validation_split * num_train)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# JSON Files should be a dictionary, with the index being the current inference number, first = 0, second = 1 etc.
# Each entry will have 3 values, the entropy, the accuracy (correct or not correct), and the time taken, exactly in that order
def graphFromJson(filePath):
    with open(filePath, "r") as file:
        results = json.load(file)

    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 24))

    accuratePredictions = [
        value["entropy"] for value in results if value["correct"] is True
    ]
    totalPredictions = [value["entropy"] for value in results]

    accurateCount, _ = np.histogram(accuratePredictions, bins=200)
    totalCount, totalBins = np.histogram(totalPredictions, bins=200)

    cumulativePercent = 100 * np.cumsum(accurateCount) / np.cumsum(totalCount)
    ax1.plot(totalBins[:-1], cumulativePercent)
    ax1.set_title("Accuracy vs Entropy")
    ax1.set_ylabel("Cumulative Correct (%)")

    ax2.plot(totalBins[:-1], 100 * np.cumsum(totalCount) / np.sum(totalCount))
    ax2.set_title("Dataset Progress vs Entropy")
    ax2.set_ylabel("Progress (%)")

    plt.xlabel("Entropy")
    plt.show()


def trainModel(model, trainLoader, validLoader, testLoader):
    model.train()
    device = getDevice()
    model.to(device)

    epoch = 20
    learning_rate = 0.01

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    for e in range(epoch):
        break
        start = time.time()
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainLoader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            exitNumber, outputs = model(images)

            loss = criterion(outputs, labels)
            writer.add_scalar("loss/batch_loss", loss.item(), e * len(trainLoader) + i)
            running_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {e}: Inference {i} used exit {exitNumber}")

        train_loss = running_loss / len(trainLoader)
        print(f"Epoch {e} took {time.time() - start}, Loss: {train_loss}")

        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = 0.0
            for images, labels in validLoader:
                images = images.to(device)
                labels = labels.to(device)
                exitNumber, outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item()

            val_accuracy = 100 * correct / total
            val_loss /= len(validLoader)
            writer.add_scalar("accuracy/val", val_accuracy, e)
            print(
                f"Epoch {e}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%"
            )

            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, e)

        # Update learning rate
        lr_scheduler.step()

    return

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testLoader:
            images = images.to(device)
            labels = labels.to(device)
            exitNumber, outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        writer.add_scalar("accuracy/test", test_accuracy)
        print(
            f"Accuracy of the network on the {len(testLoader)} test images: {test_accuracy:.2f}%"
        )


# 1. Generate temporary results json
# 2. Read in the data from the results json
# 3. Use the same bin technique from the graph section
# 4. Find the bin for the target accuracy (unsure whether this should be the bin before, after or containing the accuracy)
def getAccDataset(model, testLoader, fileName):
    fileFileName = generateJsonResults(model, fileName, testLoader)

    with open(fileFileName, "r") as file:
        results = json.load(file)

    accuratePredictions = [
        value["entropy"] for value in results if value["correct"] is True
    ]
    totalPredictions = [value["entropy"] for value in results]

    accurateCount, _ = np.histogram(accuratePredictions, bins=200)
    totalCount, totalBins = np.histogram(totalPredictions, bins=200)

    cumulativeAccuracy = 100 * np.cumsum(accurateCount) / np.cumsum(totalCount)
    cumulativeDataset = 100 * np.cumsum(accurateCount) / np.sum(totalCount)

    return cumulativeAccuracy, cumulativeDataset, totalBins


def getEntropyForAccuracy(model, testLoader, target):
    acc, _, bins = getAccDataset(model, testLoader, "temp-results")
    a = np.asarray(acc < target)
    a = np.asarray(a[:-1] != a[1:])
    indices = np.nonzero(a[:-1] != a[1:])
    if len(indices[0]) == 0:
        return 0.0
    return bins[indices[0]][0]


def getAccuracy(model, testLoader):
    return getAccDataset(model, testLoader, "temp-results")[0][-1]


def trainModelWithBranch(model, trainLoader, validLoader, testLoader, test):
    exitTracker = generation.ExitTracker(model, 100)
    exitTracker.saveAst()
    exitTracker.reloadable_model.reload(False)
    trainModel(
        exitTracker.reloadable_model.getModel(), trainLoader, validLoader, testLoader
    )

    # Model now only contains full branch, get total accuracy
    accuracy = getAccuracy(model.getModel(), test)
    exitTracker.targetAccuracy = accuracy
    exitTracker.useNextExit()

    while not exitTracker.lastExitTrained():
        exitTracker.reloadable_model.reload()

        trainModel(
            exitTracker.reloadable_model.getModel(),
            trainLoader,
            validLoader,
            testLoader,
        )

        exitTracker.setCurrentExitCorrectly(test)
        exitTracker.reloadable_model.reload()
        exitTracker.useNextExit()

    exitTracker.removeUnneededExits()
    exitTracker.reloadable_model.getModel().eval()


def generateJsonResults(model, modelName, testLoader):
    device = getDevice()

    results = []
    with torch.no_grad():
        for _, (images, labels) in enumerate(testLoader):
            images = images.to(device)
            labels = labels.to(device)

            start = time.time()
            exitNumber, outputs = model(images)
            end = time.time()

            pk = F.softmax(outputs.data, dim=1)
            entropy = -torch.sum(pk * torch.log(pk + 1e-20))
            _, predicted = torch.max(outputs.data, 1)

            correct = predicted == labels

            result = {
                "entropy": entropy.item(),
                "correct": correct.item(),
                "time_taken": end - start,
                "exit_number": exitNumber,
            }
            results.append(result)

    fileName = f"{modelName}-{int(time.time())}.json"

    with open(fileName, "a") as file:
        json.dump(results, file)

    return fileName


def createModelsFolder(name):
    script_directory = os.path.dirname(os.path.realpath(__file__))
    folder_name = name
    folder_path = os.path.join(script_directory, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


class ReloadableModel:
    def __init__(self, dataset_channels, model_class, *args):
        self.model = model_class(*args)
        self.model.to(getDevice())
        self.model_class = model_class
        self.model_args = args
        self.dc = dataset_channels
        _ = self.model(torch.randn(1, self.dc, 224, 224))

    def reload(self, grad=True):
        with tempfile.TemporaryFile() as file:
            torch.save(self.model.state_dict(), file)
            file.seek(0)
            module_name = self.model_class.__module__
            importlib.reload(sys.modules[module_name])
            reloaded_module = sys.modules[module_name]
            self.model_class = getattr(reloaded_module, self.model_class.__name__)
            self.model = self.model_class(*(self.model_args))
            _ = self.model(torch.randn(1, self.dc, 224, 224))
            saved_state_dict = torch.load(file)
            self.model.load_state_dict(saved_state_dict, strict=False)
            if grad:
                for name, param in self.model.named_parameters():
                    if "exit" not in name:
                        param.requires_grad = False

    def getModel(self):
        return self.model
