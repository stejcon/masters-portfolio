import helpers
import torch
import models
from models import ResNet50QMNIST, ResNet50Cifar100, ResNet50Cifar10, ResNet50Fashion
from models import (
    ResNet101QMNIST,
    ResNet101Cifar100,
    ResNet101Cifar10,
    ResNet101Fashion,
)
from models import (
    ResNet152QMNIST,
    ResNet152Cifar100,
    ResNet152Cifar10,
    ResNet152Fashion,
)


def main():
    helpers.createModelsFolder("models")

    resnet_names = ["50", "101", "152"]
    resnet_sizes = [[3, 4, 6, 3], [3, 4, 23, 3], [3, 8, 36, 3]]
    datasets = ["cifar10", "cifar100", "qmnist", "fashion-mnist"]
    model_classes = [
        ResNet50Cifar10,
        ResNet50Cifar100,
        ResNet50QMNIST,
        ResNet50Fashion,
        ResNet101Cifar10,
        ResNet101Cifar100,
        ResNet101QMNIST,
        ResNet101Fashion,
        ResNet152Cifar10,
        ResNet152Cifar100,
        ResNet152QMNIST,
        ResNet152Fashion,
    ]

    for i, (name, size) in enumerate(zip(resnet_names, resnet_sizes)):
        for j, dataset in enumerate(datasets):
            print(f"Doing {model_classes[i*len(resnet_names)+j]}")
            trainLoader, validLoader, testLoader = helpers.get_custom_dataloaders(
                dataset
            )
            _, _, test = helpers.get_custom_dataloaders(dataset, 1)
            model = helpers.ReloadableModel(
                model_classes[i * len(resnet_names) + j],
                models.Bottleneck,
                size,
                len(trainLoader.dataset.classes),
            )
            helpers.trainModelWithBranch(
                model, trainLoader, validLoader, testLoader, test
            )
            torch.save(model.getModel().state_dict(), f"models/resnet{name}-{dataset}")
            helpers.generateJsonResults(
                model.getModel(), f"resnet{name}-{dataset}", test
            )


if __name__ == "__main__":
    main()
