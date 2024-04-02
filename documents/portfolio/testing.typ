// #set page(paper: "a4", margin: (x: 41.5pt, top: 80.51pt, bottom: 89.51pt))
// #counter(page).update(0)
// #align(center, text(weight: "bold", size: 24pt, "Appendix C"))
// 
// #align(center, text(weight: "bold", size: 24pt, "Testing and Results"))
// 
// Appendix content follows...
// #pagebreak()
// #set page(footer: align(right, "C-" + counter(page).display("1")))
// #counter(heading).update(0)
#let section-refs = state("section-refs", ())

// add bibliography references to the current section's state
#show ref: it => {
  if it.element != none {
    // citing a document element like a figure, not a bib key
    // so don't update refs
    it
    return
  }
  section-refs.update(old => {
    if it.target not in old {
      old.push(it.target)
    }
    old
  })
  locate(loc => {
    let idx = section-refs.at(loc).position(el => el == it.target)
    "[" + str(idx + 1) + "]"
  })
}

// print the "per-section" bibliography
#let section-bib() = locate(loc => {
  let ref-counter = counter("section-refs")
  ref-counter.update(1)
  show regex("^\[(\d+)\]\s"): it => [
    [#ref-counter.display()]
  ]
  for target in section-refs.at(loc) {
    block(cite(target, form: "full"))
    ref-counter.step()
  }
  section-refs.update(())
})

#set heading(offset: 1)
= Choice of Models and Datasets
ResNet was the only model architecture tested for this project due to various reasons. The model architecture was taken directly from the TorchVision source with a few minor changes as described in Appendix B @TorchVision. Most models implemented in TorchVision follow a pattern similar to the following from the SqueezeNet implementation#footnote(link("https://github.com/pytorch/vision/blob/2c4665ffbb64f03f5d18016d3398af4ac4da5f03/torchvision/models/squeezenet.py#L94")):

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
```

`self.features` is a function which calls all important layers for the model. The methodology described by this project to place exits between the lines of the `forward` function does not function with this pattern. Either `self.features` would need to be recursively analysed to place exits, which would majorly complicate the implementation, or the model would need to be reorganised to unwrap the `self.features` function into the forward function. Both of these changes would take significant time to complete, so it was chosen to avoid implementations which followed this pattern.

ResNet uses the following `forward` function taken from the ResNet implementation in TorchVision#footnote(link("https://github.com/pytorch/vision/blob/2c4665ffbb64f03f5d18016d3398af4ac4da5f03/torchvision/models/resnet.py#L266")):

```python
def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

This `forward` function is far easier to manipulate as exits can be placed between each "layer". This layers are groups of ResNet blocks, so exits aren't truly placed at all possible locations, but it is a step closer to an ideal model. ResNet doesn't use the `self.classifier` function to wrap it's exit which would have been beneficial to avoid the hardcoded number of lines skipped at the end as referred to in Appendix B.

The chosen datasets for testing were CIFAR10, CIFAR100, QMNIST, and Fashion-MNIST. CIFAR10 and CIFAR100 were chosen due to their widespread use in previous literature. Both datasets include the same set of images, but it is important to use both due to the different number of output classes. No analysis was done by any paper on the effect of a higher or lower number of output classes on an exits performance. QMNIST and Fashion-MNIST were chosen for a similar reason. Only three channel (RGB) images had been analysed and one channel images had been ignored.

Ideally, the models ResNet101 and ResNet152 would have been tested as well. However, resource constraints meant there was no way to train these models as they were too large for the available GPUs. Imagenet also was not chosen as a dataset due to it being so large, which would have taken far too long to train on the available hardware.

= Choice of Training Scheme
Split training was utilised for testing, similar to what was used by @individualtraining. Using the weighted loss method as shown in @BranchyNet was not feasible due to the time needed to identify the hyperparameters, with each exit requiring a new hyperparameter. While this would likely be beneficial in terms of accuracy as the exits can effectively teach each other, it was not feasible for this project. The training approach was also inspired by knowledge distillation, as the accuracy of the final exit is used as a hyperparameter, which is akin to training based on labelling done by the backbone model. The hyperparameters chosen from @hyperparameters were kept for all models. Good hyperparameters take a long time to identify and there are many techniques of doing so. Choosing the backbone models hyperparameters is out of scope for this project, so the same hyperparameters were kept as all ResNet models should perform decently with the same hyperparameters. Exits were only enabled after the backbone model was trained, and they were trained in a bottom-up fashion as described in @bottomup.

= Method of Generating Results
In `models.py`, a series of models were added. Almost all models are identical, but needed to be kept as separate classes to allow writing the `forward` and `__init__` function to work. Each model was trained with its respective dataset, and all changes were saved into `models.py`. This resulted in twelve trained models, with each of ResNet18, ResNet34 and ResNet50 trained on CIFAR10, CIFAR100, QMNIST and Fashion-MNIST. The training used a changing script in `main.py`. The script frequently changed as the training was not completed in a single run. Each of the parameters at the beginning of the script was changed depending on the models which were to be trained. Once a model was trained and tested, the `models` module needed to be reloaded due to the rewritten model conflicting with the version stored in memory. The list was reinitialised to allow the references to the class to be correct.

#show figure: set block(breakable: true)
#figure(
```python
def main():
    helpers.createModelsFolder("models")

    resnet_names = ["34"]
    resnet_sizes = [[3, 4, 6, 3]]
    datasets = ["cifar10", "cifar100", "qmnist", "fashion-mnist"]
    model_classes = [
        models.ResNet34Cifar10,
        models.ResNet34Cifar100,
        models.ResNet34QMNIST,
        models.ResNet34Fashion,
    ]

    for i, (name, size) in enumerate(zip(resnet_names, resnet_sizes)):
        for j, dataset in enumerate(datasets):
            print(
                f"Doing {model_classes[i*len(resnet_names)+j]}, should be {name}/{size} with {dataset}"
            )
            _, _, test = helpers.get_custom_dataloaders(dataset, 1)
            model = helpers.ReloadableModel(
                next(iter(trainLoader))[0].shape[1],
                model_classes[i * len(resnet_names) + j],
                models.BasicBlock,
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
                models.ResNet34Cifar10,
                models.ResNet34Cifar100,
                models.ResNet34QMNIST,
                models.ResNet34Fashion,
            ]
```,
caption: "The core script used for training", placement: none)

The test dataset used must have a batch size of 1, as @BranchyNet noted that a batch size greater than this effects the ability of the model to exit early. This is because each entry in the batch will have a different entropy, meaning the threshold condition cannot be met. The results which are saved are the entropy, accuracy and time taken for each inference grouped by exit. These results are shown in @tables for completeness, but only a few parts will need to be noted.

To generate the same results for the backbone model without using any early exits, the parameters saved from the branched models were loaded into a base ResNet model with no modifications to the `forward` function. This cut down on training time as the backbone layers from the branched models were unchanged by the exits, and the inference time was also guaranteed to not be affected by any of the exits.

It is obvious from the tables in @tables that the inference time was majorly effected by the exits. Most exits are also hardly ever used, with most sitting below 1% utilisation despite the efforts to prune exits. A third set of models were created called the `Stripped` models, where these underperforming exits were solely due to a bad pruning mechanism. These stripped models had the exits removed from the `forward` function, and the parameters were loaded from the saved branched models. The same rounds of testing were then carried out.

= Effect of Automatically Generated Early Exits
The average time taken, and average accuracy for all models are shown in the following figure:

#figure(
grid(
  columns: (auto),
    rows: (auto, auto, auto, auto, auto, auto),
    image("./images/resnet18_accuracy.png"),
    image("./images/resnet34_accuracy.png"),
    image("./images/resnet50_accuracy.png"),
    image("./images/resnet18_time_taken.png"),
    image("./images/resnet34_time_taken.png"),
    image("./images/resnet50_time_taken.png"),
  ),
  placement: none,
) <graphs>

= Summary of Results
Without manually stripping exits, the model is severely slowed down. When the model is stripped, performance is comparable in terms of inference time with some datasets performing slightly better. There is a consistent drop in accuracy for branched models across all datasets, with only QMNIST remaining somewhat equal. This is to be expected as MNIST is a reasonably simple dataset to accurately classify. The reasons for this are discussed in @observations

#let table-json(model-name) = {
  let data = json("./results/refined_" + model-name + ".json")
  figure(
    table(
      columns: (auto, ) * (6),
      ..data.at(0).keys(),
      ..(for x in data {
        x.values().map(a=>str(a)).flatten()
      }).flatten(),
    ),
    caption: "Data on exits for model " + model-name,
    placement: none,
  )
}


= All Results Tabulated <tables>
The raw results from each model are printed here for completeness sake. Any outputs of interest has been discussed above or in @observations
#table-json("resnet18-cifar10-base")
#table-json("resnet18-cifar10-branched")
#table-json("resnet18-cifar10-stripped")
#table-json("resnet18-cifar100-base")
#table-json("resnet18-cifar100-branched")
#table-json("resnet18-cifar100-stripped")
#table-json("resnet18-fashion-base")
#table-json("resnet18-fashion-branched")
#table-json("resnet18-fashion-stripped")
#table-json("resnet18-qmnist-base")
#table-json("resnet18-qmnist-branched")
#table-json("resnet18-qmnist-stripped")
#table-json("resnet34-cifar10-base")
#table-json("resnet34-cifar10-branched")
#table-json("resnet34-cifar10-stripped")
#table-json("resnet34-cifar100-base")
#table-json("resnet34-cifar100-branched")
#table-json("resnet34-cifar100-stripped")
#table-json("resnet34-fashion-base")
#table-json("resnet34-fashion-branched")
#table-json("resnet34-fashion-stripped")
#table-json("resnet34-qmnist-base")
#table-json("resnet34-qmnist-branched")
#table-json("resnet34-qmnist-stripped")
#table-json("resnet50-cifar10-base")
#table-json("resnet50-cifar10-branched")
#table-json("resnet50-cifar10-stripped")
#table-json("resnet50-cifar100-base")
#table-json("resnet50-cifar100-branched")
#table-json("resnet50-cifar100-stripped")
#table-json("resnet50-fashion-base")
#table-json("resnet50-fashion-branched")
#table-json("resnet50-fashion-stripped")
#table-json("resnet50-qmnist-base")
#table-json("resnet50-qmnist-branched")
#table-json("resnet50-qmnist-stripped")

#heading(numbering: none, "References")
#section-bib()
