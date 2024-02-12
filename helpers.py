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

# gpuString lets you define which GPU to use if there are multiple
# Project presumes only one GPU is used
def getDevice():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    return device

def Cifar10Splits(batchSize=1):
    normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])

    trainDataset = datasets.CIFAR10(train=True, root='./data', transform=transform, download=True)
    testDataset = datasets.CIFAR10(train=False, root='./data', transform=transform, download=True)

    # Split train dataset to have 10% validation dataset
    valid_size=0.1
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

    accuratePredictions = [value[0] for value in results.values() if value[1] is True]
    inaccuratePredictions = [value[0] for value in results.values() if value[1] is False]

    allPredictions = np.sort(np.concatenate([accuratePredictions, inaccuratePredictions]))
    bins = np.histogram_bin_edges(allPredictions, bins=200)

    _, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, sharex=True, figsize=(16,24))

    accurateBinValues, _, _ = ax1.hist(accuratePredictions, bins=bins, label="Correct")
    ax1.legend()
    ax1.set_title("Histogram of Correct Predictions vs Entropy")
    ax1.set_ylabel("Number of Correct Predictions")

    inaccurateBinValues, _, _ = ax2.hist(inaccuratePredictions, bins=bins, label="Incorrect")
    ax2.legend()
    ax2.set_title("Histogram of Incorrect Predictions vs Entropy")
    ax2.set_ylabel("Number of Incorrect Predictions")

    # Fix for divide by 0
    # Should this be done differently
    inaccurateBinValues = [x if x != 0 else 1 for x in inaccurateBinValues]
    allBinValues = accurateBinValues + inaccurateBinValues

    ratioAccurateInaccurate = accurateBinValues / inaccurateBinValues
    ax3.plot(bins[:-1], ratioAccurateInaccurate)
    ax3.set_title("Point Ratio of Correct to Incorrect")
    ax3.set_ylabel("Correct:Incorrect")

    cumulativeRatio = np.cumsum(accurateBinValues) / np.cumsum(inaccurateBinValues)
    ax4.plot(bins[:-1], cumulativeRatio)
    ax4.set_title("Cumulative Ratio of Correct to Incorrect")
    ax4.set_ylabel("Cumulative Correct:Incorrect")

    percentAccurateInaccurate = 100 * accurateBinValues / allBinValues
    ax5.plot(bins[:-1], percentAccurateInaccurate)
    ax5.set_title("Point Percent of Correct to Total")
    ax5.set_ylabel("Correct (%)")

    cumulativePercent = 100 * np.cumsum(accurateBinValues) / np.cumsum(allBinValues)
    ax6.plot(bins[:-1], cumulativePercent)
    ax6.set_title("Accuracy vs Entropy")
    ax6.set_ylabel("Cumulative Correct (%)")

    ax7.plot(bins[:-1], 100*np.cumsum(allBinValues)/np.sum(allBinValues))
    ax7.set_title("Dataset Progress vs Entropy")
    ax7.set_ylabel("Progress (%)")

    plt.xlabel("Entropy")
    plt.show()

def trainModel(model, trainLoader, validLoader, testLoader):
    model.train()
    device = getDevice()
    
    epoch = 5
    learning_rate = 0.01

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    exitTracker = generation.ExitTracker(model)

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
        if (e == epoch - 1):
            exitTracker.transformFunction()

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

        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))

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
                    "entropy": e,
                    "correct": c,
                    "time_taken": end - start,
                    "exit_number": exitNumber.item()  # Assuming exitNumber is a scalar
                }
                results.append(result)
    
    with open(f"{modelName}-{int(time.time())}.json", "a") as file:
        json.dump(results, file)

def createModelsFolder(name):
    # Get the current directory of the script
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # Define the folder name
    folder_name = name

    # Create the folder path
    folder_path = os.path.join(script_directory, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
