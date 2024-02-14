import json
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import generation
import tempfile
import inspect
import importlib

# gpuString lets you define which GPU to use if there are multiple
# Project presumes only one GPU is used
def getDevice():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    return device

def Cifar10Splits(batchSize=64):
    normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])

    trainDataset = datasets.CIFAR10(train=True, root='./data', transform=transform, download=True)
    testDataset = datasets.CIFAR10(train=False, root='./data', transform=transform, download=True)

    # Split train dataset to have 10% validation dataset
    valid_size=0.15
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

# JSON Files should be a dictionary, with the index being the current inference number, first = 0, second = 1 etc.
# Each entry will have 3 values, the entropy, the accuracy (correct or not correct), and the time taken, exactly in that order
def graphFromJson(filePath):
    with open(filePath, 'r') as file:
        results = json.load(file)

    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,24))

    accuratePredictions = [value['entropy'] for value in results if value['correct'] is True]
    totalPredictions = [value['entropy'] for value in results]

    accurateCount, _ = np.histogram(accuratePredictions, bins=200)
    totalCount, totalBins = np.histogram(totalPredictions, bins=200)

    cumulativePercent = 100 * np.cumsum(accurateCount) / np.cumsum(totalCount)
    ax1.plot(totalBins[:-1], cumulativePercent)
    ax1.set_title("Accuracy vs Entropy")
    ax1.set_ylabel("Cumulative Correct (%)")

    ax2.plot(totalBins[:-1], 100*np.cumsum(totalCount)/np.sum(totalCount))
    ax2.set_title("Dataset Progress vs Entropy")
    ax2.set_ylabel("Progress (%)")

    plt.xlabel("Entropy")
    plt.show()

def trainModel(model, trainLoader, validLoader, testLoader):
    model.train()
    device = getDevice()
    
    epoch = 20
    learning_rate = 0.01

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    for e in range(epoch):
        start = time.time()
        for i, (images, labels) in enumerate(trainLoader):
            print(f"Epoch {e}: Inference {i}")
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            exitNumber, outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {e} took {time.time() - start}")
                
        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in validLoader:
                images = images.to(device)
                labels = labels.to(device)
                exitNumber, outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
            print('Accuracy of the network on the {} validation images: {} %'.format(len(validLoader), 100 * correct / total)) 
            
        # Update learning rate
        lr_scheduler.step()

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

        print('Accuracy of the network on the {} test images: {} %'.format(len(testLoader), 100 * correct / total))

# 1. Generate temporary results json
# 2. Read in the data from the results json
# 3. Use the same bin technique from the graph section
# 4. Find the bin for the target accuracy (unsure whether this should be the bin before, after or containing the accuracy)
def getAccDataset(model, testLoader, fileName):
    fileFileName = generateJsonResults(model, fileName, testLoader)

    with open(fileFileName, 'r') as file:
        results = json.load(file)

    accuratePredictions = [value['entropy'] for value in results if value['correct'] is True]
    totalPredictions = [value['entropy'] for value in results]

    accurateCount, _ = np.histogram(accuratePredictions, bins=200)
    totalCount, totalBins = np.histogram(totalPredictions, bins=200)

    cumulativeAccuracy = 100 * np.cumsum(accurateCount) / np.cumsum(totalCount)
    cumulativeDataset = 100 * np.cumsum(accurateCount) / np.sum(totalCount)

    return cumulativeAccuracy, cumulativeDataset, totalBins

def getEntropyForAccuracy(model, testLoader, target):
    acc, _, bins = getAccDataset(model, testLoader, "temp-results")
    return bins[np.where(acc < target)[0][0]]

def getAccuracy(model, testLoader):
    return getAccDataset(model, testLoader, "temp-results")[0][-1]

def trainModelWithBranch(model, trainLoader, validLoader, testLoader):
    trainModel(model, trainLoader, validLoader, testLoader)

    # Model now only contains full branch, get total accuracy
    accuracy = getAccuracy(model, testLoader)

    for param in model.parameters():
        param.requires_grad = False

    exitTracker = generation.ExitTracker(model, accuracy)
    exitTracker.transformFunction()

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

            y_hat = torch.nn.functional.softmax(outputs, dim=1)
            entropy = -torch.sum(y_hat * torch.log2(y_hat), dim=1)            
            _, predicted = torch.max(outputs.data, 1)

            correct = predicted == labels
            
            for e, c in zip(entropy, correct):
                result = {
                    "entropy": e.item(),
                    "correct": c.item(),
                    "time_taken": end - start,
                    "exit_number": exitNumber
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
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

class ReloadableModel():
    def __init__(self, model_class, *args):
        for arg in args:
            print(arg)
        self.model = model_class(*args)
        self.model_class = model_class
        self.model_args = args

    def reload(self):
        with tempfile.TemporaryFile("weights") as file:
            torch.save(self.model[0].state_dict(), file)
            importlib.reload(inspect.getmodule(self.model))
            self.model = self.model_class(*(self.model_args)).load_state_dict(torch.load(file))

        return self.model

    def getModel(self):
        return self.model
