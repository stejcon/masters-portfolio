import helpers
import torch
import models

def main():
    device = helpers.getDevice()
    helpers.createModelsFolder("models")
    trainLoader, validLoader, testLoader = helpers.Cifar10Splits()

    fullModel = models.ResNet(models.ResidualBlock, [3, 4, 6, 3]).to(device).train()
    helpers.trainModelWithBranch(fullModel, trainLoader, validLoader, testLoader)
    fullModel.eval()
    torch.save(fullModel.state_dict(), "models/fullModel")

    halfModel = models.HalfResNet(models.ResidualBlock, [3, 4, 6, 3]).to(device).train()
    helpers.trainModel(halfModel, trainLoader, validLoader, testLoader)
    halfModel.eval()
    torch.save(halfModel.state_dict(), "models/halfModel")

    branchedModel = models.BranchedResNet(models.ResidualBlock, [3, 4, 6, 3]).to(device).train()
    helpers.trainModel(branchedModel, trainLoader, validLoader, testLoader)
    branchedModel.eval()
    torch.save(branchedModel.state_dict(), "models/branchedModel")

    helpers.generateJsonResults(fullModel, "fullModel", testLoader)
    helpers.generateJsonResults(halfModel, "halfModel", testLoader)
    helpers.generateJsonResults(branchedModel, "branchedModel", testLoader)

if __name__ == "__main__":
    main()
