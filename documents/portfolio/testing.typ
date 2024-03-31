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
- Only ResNet
- Models from torchvision only had 4 lines in forward and would take too much time to reimplement to make it workable for exits
- The implementation also hard codes how many lines to ignore at the end of the forward function to ignore the ResNet classifier. This is because unlike other models, ResNet doesnt use `self.classifier`. This again would take time to reimplement in which case the implementation could be improved by removing the hard coded number of lines to remove, and only insert exits before `self.classifier`.

- Datasets of choice were CIFAR10, CIFAR100, QMNIST, Fashion-MNIST
- CIFAR10 and CIFAR100 are popular in literature, making them good for comparison
- Also gives insight into how the number of classes effect exits
- QMNIST and Fashion-MNIST were chosen to analyse how the exits perform when the number of channels is different to that of the CIFAR datasets
- Ideally, Resnet101 and Resnet152 would have been also been used to have a better idea of how depth effects exits, along with Imagenet dataset, but resource constraints in terms of GPU power made these impossible to train.

= Choice of Training Scheme
- Using the weighted loss method as shown in @BranchyNet wasn not feasible due to the time needed to identify the hyperparameters, with each exit requiring a new weight hyperparameter. While this would likely be beneficial, it was not feasible for this project.
- Instead, split training was used, where every exit was trained seperately. It also is inspired by knowledge distillation, as the accuracy of the final exit is used as a hyperparameter, which is akin to training based on labelling done by the backbone model.
- The hyperparameters chosen from @hyperparameters were kept for all models. Good hyperparameters take a long time to identify and there are many techniques of doing so. Chosing the backbone models hyperparameters is out of scope for this project, so the same hyperparameters were kept as all resnet models should perform decently with the same hyperparameters.

= Method of Generating Results
- Train backbone model
- Train exits bottom up as described in @bottomup
- Once all exits are trained, run testing data for the dataset through the model with a batch size of 1, as @BranchyNet notes early exiting makes little sense when used with batches of more than 1
- Record the entropy, the time taken, whether the output was correct, and the exit used
- Generate tables for all of this data for every model
- Generate graphs showing the average time taken and the accuracy of each model across all exits

- After analyse, the branched models did not prune well with many exits being used less than 1% of the time. To allow the analysis of the other used exits fairly, exits which were used less than 1% of the time were manually removed and recorded as `Stripped` models in `models.py`.

= Effect of Automatically Generated Early Exits
- contradicting the statement made by @optimalbranchplacement, simple exits with just a fully connected layer do not work well. As shown in the graphs, the time taken rises dramatically when branches are used, even though the number of layers is small.
- As the number of layers increases, the time difference between the base model and branched model shrinks significantly, although the branched remains slower. This is likely due to the large input dimension on fully connected layers. Fully connected layers scaled quadratically, so as the input size increases, the time taken to complete the operation increases quadratically.

- Accuracy was also always slightly below that of the base model. When looking at the accuracy of indivi

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
    caption: "Data on exits for model " + model-name
  )
}


= All Results Tabulated
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
