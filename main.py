import helpers
import torch
import models


def main():
    helpers.createModelsFolder("models")

    resnet_names = ["50", "101", "152"]
    resnet_sizes = [[3, 4, 6, 3], [3, 4, 23, 3], [3, 8, 36, 3]]
    datasets = ["cifar10", "cifar100", "qmnist", "fashion-mnist"]

    for name, size in zip(resnet_names, resnet_sizes):
        for dataset in datasets:
            trainLoader, validLoader, testLoader = helpers.get_custom_dataloaders(
                dataset
            )
            _, _, test = helpers.get_custom_dataloaders(dataset, 1)
            model = helpers.ReloadableModel(
                models.ResNet,
                models.Bottleneck,
                size,
                len(trainLoader.dataset.classes),
            )
            helpers.trainModelWithBranch(
                model, trainLoader, validLoader, testLoader, test
            )
            torch.save(model.getModel().state_dict(), f"models/resnet{name}-cifar10")
            helpers.generateJsonResults(model.getModel(), f"resnet{name}-cifar10", test)


if __name__ == "__main__":
    main()
