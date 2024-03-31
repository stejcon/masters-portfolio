import helpers
import models
import torch

models = [
    #     (
    #         models.ResNet18Cifar10Stripped(
    #             models.BasicBlock, [2, 2, 2, 2], num_classes=10, channels=3
    #         ),
    #         "resnet18",
    #         "cifar10",
    #     ),
    #     (
    #         models.ResNet18Cifar100Stripped(
    #             models.BasicBlock, [2, 2, 2, 2], num_classes=100, channels=3
    #         ),
    #         "resnet18",
    #         "cifar100",
    #     ),
    #     (
    #         models.ResNet18QMNISTStripped(
    #             models.BasicBlock, [2, 2, 2, 2], num_classes=10, channels=1
    #         ),
    #         "resnet18",
    #         "qmnist",
    #     ),
    #     (
    #         models.ResNet18FashionStripped(
    #             models.BasicBlock, [2, 2, 2, 2], num_classes=10, channels=1
    #         ),
    #         "resnet18",
    #         "fashion-mnist",
    #     ),
    #     (
    #         models.ResNet34Cifar10Stripped(
    #             models.BasicBlock, [3, 4, 6, 3], num_classes=10, channels=3
    #         ),
    #         "resnet34",
    #         "cifar10",
    #     ),
    #     (
    #         models.ResNet34Cifar100Stripped(
    #             models.BasicBlock, [3, 4, 6, 3], num_classes=100, channels=3
    #         ),
    #         "resnet34",
    #         "cifar100",
    #     ),
    #     (
    #         models.ResNet34QMNISTStripped(
    #             models.BasicBlock, [3, 4, 6, 3], num_classes=10, channels=1
    #         ),
    #         "resnet34",
    #         "qmnist",
    #     ),
    #     (
    #         models.ResNet34FashionStripped(
    #             models.BasicBlock, [3, 4, 6, 3], num_classes=10, channels=1
    #         ),
    #         "resnet34",
    #         "fashion-mnist",
    #     ),
    #     (
    #         models.ResNet50Cifar10Stripped(
    #             models.Bottleneck, [3, 4, 6, 3], num_classes=10, channels=3
    #         ),
    #         "resnet50",
    #         "cifar10",
    #     ),
    (
        models.ResNet50Cifar100Stripped(
            models.Bottleneck, [3, 4, 6, 3], num_classes=100, channels=3
        ),
        "resnet50",
        "cifar100",
    ),
    #    (
    #        models.ResNet50QMNISTStripped(
    #            models.Bottleneck, [3, 4, 6, 3], num_classes=10, channels=1
    #        ),
    #        "resnet50",
    #        "qmnist",
    #    ),
    #    (
    #        models.ResNet50FashionStripped(
    #            models.Bottleneck, [3, 4, 6, 3], num_classes=10, channels=1
    #        ),
    #        "resnet50",
    #        "fashion-mnist",
    #    ),
]

for model, name, data in models:
    model.eval()
    model.to(helpers.getDevice())
    _ = model(torch.randn(1, model.channels, 224, 224))
    model.load_state_dict(torch.load(f"./models-safe/{name}-{data}"), strict=False)
    _, _, test = helpers.get_custom_dataloaders(f"{data}", batch_size=1)
    helpers.generateJsonResults(model, f"{name}-{data}-stripped", test)
