import helpers
import torch
import models
import importlib


def main():
    helpers.createModelsFolder("models")

    resnet_names = ["50", "101", "152"]
    resnet_sizes = [[3, 4, 6, 3], [3, 4, 23, 3], [3, 8, 36, 3]]
    datasets = ["cifar10", "cifar100", "qmnist", "fashion-mnist"]
    model_classes = [
        models.ResNet50Cifar10,
        models.ResNet50Cifar100,
        models.ResNet50QMNIST,
        models.ResNet50Fashion,
        models.ResNet101Cifar10,
        models.ResNet101Cifar100,
        models.ResNet101QMNIST,
        models.ResNet101Fashion,
        models.ResNet152Cifar10,
        models.ResNet152Cifar100,
        models.ResNet152QMNIST,
        models.ResNet152Fashion,
    ]

    for i, (name, size) in enumerate(zip(resnet_names, resnet_sizes)):
        for j, dataset in enumerate(datasets):
            print(
                f"Doing {model_classes[i*len(resnet_names)+j]}, should be {name} with {dataset}"
            )
            if model_classes[i * len(resnet_names) + j] in []:
                continue
            trainLoader, validLoader, testLoader = helpers.get_custom_dataloaders(
                dataset
            )
            _, _, test = helpers.get_custom_dataloaders(dataset, 1)
            model = helpers.ReloadableModel(
                next(iter(trainLoader))[0].shape[1],
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
            importlib.reload(models)
            model_classes = [
                models.ResNet50Cifar10,
                models.ResNet50Cifar100,
                models.ResNet50QMNIST,
                models.ResNet50Fashion,
                models.ResNet101Cifar10,
                models.ResNet101Cifar100,
                models.ResNet101QMNIST,
                models.ResNet101Fashion,
                models.ResNet152Cifar10,
                models.ResNet152Cifar100,
                models.ResNet152QMNIST,
                models.ResNet152Fashion,
            ]


if __name__ == "__main__":
    main()
