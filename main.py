import helpers
import torch
import models


def main():
    helpers.createModelsFolder("models")
    trainLoader, validLoader, testLoader = helpers.Cifar10Splits()

    fullModel = helpers.ReloadableModel(
        models.ResNet, models.ResidualBlock, [3, 4, 6, 3]
    )
    helpers.trainModelWithBranch(fullModel, trainLoader, validLoader, testLoader)
    torch.save(fullModel.getModel().state_dict(), "models/fullModel")

    _, _, test = helpers.Cifar10Splits(1)
    helpers.generateJsonResults(fullModel.getModel(), "fullModel", test)


if __name__ == "__main__":
    main()
