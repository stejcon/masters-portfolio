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
})

// clear the previously stored references every time a level 1 heading
// is created.
#show heading.where(level: 1): it => {
  section-refs.update(())
  it
}
// Workaround for the lack of an `std` scope.
#let std-bibliography = bibliography

// This function gets your whole document as its `body` and formats
// it as an article in the style of the IEEE.
#let ieee(
  // The paper's title.
  title: [Paper Title],

  // An array of authors. For each author you can specify a name,
  // department, organization, location, and email. Everything but
  // but the name is optional.
  authors: (),

  // The paper's abstract. Can be omitted if you don't have one.
  abstract: none,

  // A list of index terms to display after the abstract.
  index-terms: (),

  // The article's paper size. Also affects the margins.
  paper-size: "a4",

  // The result of a call to the `bibliography` function or `none`.
  bibliography: none,

  // The paper's content.
  body
) = {
  // Set the body font.
  set text(font: "STIX Two Text", size: 10pt)

  // Configure the page.
  set page(
    paper: paper-size,
    // The margins depend on the paper size.
    margin: if paper-size == "a4" {
      (x: 41.5pt, top: 80.51pt, bottom: 89.51pt)
    } else {
      (
        x: (50pt / 216mm) * 100%,
        top: (55pt / 279mm) * 100%,
        bottom: (64pt / 279mm) * 100%,
      )
    }
  )

  // Configure equation numbering and spacing.
  set math.equation(numbering: "(1)")
  show math.equation: set block(spacing: 0.65em)

  // Configure appearance of equation references
  show ref: it => {
    if it.element != none and it.element.func() == math.equation {
      // Override equation references.
      link(it.element.location(), numbering(
        it.element.numbering,
        ..counter(math.equation).at(it.element.location())
      ))
    } else {
      // Other references as usual.
      it
    }
  }

  // Configure lists.
  set enum(indent: 10pt, body-indent: 9pt)
  set list(indent: 10pt, body-indent: 9pt)

  // Configure headings.
  set heading(numbering: "I.A.1.")
  show heading: it => locate(loc => {
    // Find out the final number of the heading counter.
    let levels = counter(heading).at(loc)
    let deepest = if levels != () {
      levels.last()
    } else {
      1
    }

    set text(10pt, weight: 400)
    if it.level == 1 [
      // First-level headings are centered smallcaps.
      // We don't want to number of the acknowledgment section.
      #let is-ack = it.body in ([Acknowledgment], [Acknowledgement])
      #set align(center)
      #set text(if is-ack { 10pt } else { 12pt })
      #show: smallcaps
      #v(20pt, weak: true)
      #if it.numbering != none and not is-ack {
        numbering("I.", deepest)
        h(7pt, weak: true)
      }
      #it.body
      #v(13.75pt, weak: true)
    ] else if it.level == 2 [
      // Second-level headings are run-ins.
      #set par(first-line-indent: 0pt)
      #set text(style: "italic")
      #v(10pt, weak: true)
      #if it.numbering != none {
        numbering("A.", deepest)
        h(7pt, weak: true)
      }
      #it.body
      #v(10pt, weak: true)
    ] else [
      // Third level headings are run-ins too, but different.
      #if it.level == 3 {
        numbering("1)", deepest)
        [ ]
      }
      _#(it.body):_
    ]
  })

  // Display the paper's title.
  v(3pt, weak: true)
  align(center, text(18pt, title))
  v(8.35mm, weak: true)

  // Display the authors list.
  for i in range(calc.ceil(authors.len() / 3)) {
    let end = calc.min((i + 1) * 3, authors.len())
    let is-last = authors.len() == end
    let slice = authors.slice(i * 3, end)
    grid(
      columns: slice.len() * (1fr,),
      gutter: 12pt,
      ..slice.map(author => align(center, {
        text(12pt, author.name)
        if "department" in author [
          \ #emph(author.department)
        ]
        if "organization" in author [
          \ #emph(author.organization)
        ]
        if "location" in author [
          \ #author.location
        ]
        if "email" in author [
          \ #link("mailto:" + author.email)
        ]
      }))
    )

    if not is-last {
      v(16pt, weak: true)
    }
  }
  v(40pt, weak: true)

  // Start two column mode and configure paragraph properties.
  show: columns.with(2, gutter: 12pt)
  set par(justify: true, first-line-indent: 1em)
  show par: set block(spacing: 0.65em)

  // Display abstract and index terms.
  if abstract != none [
    #set text(weight: 700)
    #h(1em) _Abstract_---#abstract

    #if index-terms != () [
      #h(1em)_Index terms_---#index-terms.join(", ")
    ]
    #v(2pt)
  ]

  // Display the paper's contents.
  body

  // Display bibliography.
  if bibliography != none {
    show std-bibliography: set text(8pt)
    set std-bibliography(title: text(12pt)[References], style: "ieee")
    bibliography
  }
}

#show: ieee.with(
  title: "MEng in Electronic and Computer Engineering",
  abstract: [Early exiting is a technique used to decrease the average time taken for neural networks to complete an inference. However, traditional methods of implementing early exits require a significant amount of manual work. To enable the widespread use of early exits, an approach to automatically add early exits to networks is presented, which relies on code generation techniques to modify a PyTorch model at runtime. The approach presented is a modular method which can reload models at runtime with updated architectures, allow exits with varying architectures to be added simultaneously and allows easily replacing the training techniques used. The approach is tested on various ResNet models without various popular datasets. While there are some performance regressions, the approach easily enables future work to implement suggested fixes. The approach is generalisable to any popular machine learning framework, making it a significant step forward in enabling widespread use of early exiting.],
  authors: (
    (
      name: "Stephen Condon",
      organization: [Dublin City University],
      location: [Dublin, Ireland],
      email: "stephen.condon5@mail.dcu.ie"
    ),
  ),
  paper-size: "a4",
  index-terms: ("Early exiting", "code generation", "machine learning"),
  bibliography: none,
  // bibliography: bibliography("refs.bib"),
)

= Introduction
Early exiting in neural networks is an area that hasn't received much academic interest, with few papers being released on the topic in recent years. The key idea behind early exiting is that a network can already have identified the correct output without running through the entire model, and this can be identified by how converged the model is to a single output. To avoid the need to manually add exits to models, they can be automatically added by modifying a models architecture. This project implements the underlying architecture needed to allow models to be modified while also being trained.

@PriorWork discusses the traditional techniques used to implement early exiting, as well as methods used to automatically modify models to optimise them.
@TechnicalDescription describes the methodologies used for implementing automatic early exits, including a compile-time and training-time solution.
@Results details the performance of the branched models against the base models.
@Analysis discusses the results and identifies strengths and weaknesses of the approach.
@Conclusions summarises the impact of the implementation, and how future work can build on what has been presented.

= Prior Work <PriorWork>
BranchyNet @BranchyNet was one of the first successful methods for adding exits to a model. @BranchyNet proposes a structure which can be used to decide whether the model has converged on an output at defined point in the model. Specifically, the entropy of an exit's output is used to determine whether the model has converged to a specific output. Every exit also needs an entropy threshold, and any output with an entropy below this is considered sufficiently converged and the model can exit. @BranchyNet also says that, generally, exits very early in the model should contain extra layers and later exits should have fewer layers, but exactly how many layers should be added isn't discussed. How the threshold should be set also isn't specified, but scanning over multiple values and observing the accuracy is mentioned. A default value of 0.5 is used throughout the paper. @optimalbranchplacement discusses an algorithm to search for the optimal placement of branches in BranchyNet models. It identifies the cost of each exit, and determines whether this cost is less than the cost of continuing in the model. This search only outputs where a branch should be placed, but doesn't actual add a branch to a model. It also notes that exits with just a fully-connected layer perform almost as well as exits with many layers. @earlyexitmasters shows how exits containing a pooling layer and a fully-connected layer can be used to exit a model early while maintaining the same accuracy as the backbone model.

All of the above papers describe manual methods for adding exits to a model, which adds too much work to the designing of models to allow for widespread adoption of early exiting. Early exits should be added automatically to already-defined models to allow old models to be adapted easily and new models to be created with exits without the need for manual intervention. One method already commonly used for manipulating models is ONNX-MLIR, a compiler designed build shared libraries which contain a specific model @onnxmlir. ONNX is a protobuf file format which stores model architectures as well as model parameters @onnx. LLVM is a group of technologies used for developing compilers @LLVM. It is composed of frontend parses which emit code in an LLVM intermediate representation (IR), and many backends for compiling LLVM IR into binary for a target architecture. MLIR is a technology built on top of LLVM which allows domain-specific extension to be made to the LLVM IR @mlir. Each extension to the IR is known as a dialect. ONNX-MLIR is a frontend LLVM compiler which uses multiple new dialects to convert an ONNX model into a runnable binary. Importantly, it has an `onnx` dialect which wraps all ONNX operations, such as convolutions and batch normalisations, for use in the compiler. It also has a compiler pass that runs on the ONNX model which allows for manipulating the model, such as constant propagation and instruction folding. Creating a set of functions to run during this pass could allow for adding exits in specific locations. The compiled model would then need to somehow be trained.

Early exiting isn't only useful for decreasing the inference time of models. Split computing is useful for models which collect data from edge devices. Typically, models would need to be run in their entirety on a single device. As models are now being designed for use in IoT applications, there is a desire to keep data close to the edge as to not send private data to a central server to be processed. @TowardsEdgeComputing describes a BranchyNet-based architecture which allows edge devices to compute as much of the inference as possible and, if the model has converged on an output, to return locally. If the model hasn't converged to an answer, the remaining section of the model is ran on a central server. Automatically added exits provide good boundaries for the breaks in models which can be used to split the model across devices. 

= Technical Description <TechnicalDescription>
To begin, any attempts at adding early exits would be tested on the ResNet34 model and the CIFAR10 dataset. ResNet is a popular, state-of-the-art convolutional neural network architecture, designed to win the ImageNet dataset challenge @ResNet that is frequently used in literature. CIFAR10 is a smaller dataset that is popular in literature which can classify images with 10 separate labels. These were used not only due to their popularity in literature, but could also be trained in a reasonable time on the available hardware, which included a Nvidia 3070. The default (backbone) ResNet34 model could train to approximately 85% accuracy on CIFAR10 in approximately 1.5 hours. This would allow for quicker development of the automatic addition of early exits.

== Compile Time Solution
A compile-time implementation was the first attempted approach. Using an ONNX-MLIR based solution is preferably to creating a framework-specific solution. The most useful for editting a model in ONNX-MLIR is `ONNXOpTransformPass`. This allows for manipulation of the forward graph of the model. These operations use rules written in LLVM's TableGen domain-specific language. To add the exit structures to the model, TableGen rules could be used to identify blocks within a CNN, as blocks generally contain a convolution, an averaging operation, and possibly another non-linear layer such as the rectifed linear unit (ReLU), and an exit can be placed after a block. 

The exit would need to be generated in the `onnx` MLIR dialect. All exits would be structured be structured as an averaging layer, followed by a fully connected layer. This matches the pooling blocks laid out by @earlyexitmasters as well as the default exit in ResNet models @ResNet. These added exits would not yet be trained, so all weights would need to be initialised. Methods for initialising layers has been discussed for years in literature, with many different initialisation schemes being used. Most major Python frameworks use Kaiming initialisation @kaiming as a default so this would be utilised.

Training the models from this point becomes difficult. Once the model is compiled, there is no feasible method for loading the model into a framework to be trained. For example, PyTorch requires that a model class needs to exist which represents the model, but this defeats the purpose of using ONNX, as the training is still tied to a specific framework. Some layers also have parameters which need to be kept during training but which are lost during compilation, such as batch normalisation, making training difficult. 

There is potential in this method to be useful due to its portability. However there were too many obvious issues that would need to be solved given the timeframe of the project. Utilising TableGen can be difficult with work with since debugging the model after running the compiler passes can lead to long output messages with little important information. Build times were another concern as rebuilding ONNX-MLIR on the available hardware took a significant amount of time, and the model would then need to be trained afterwards. Due to this, a training-time approach was taken instead.

== Training Time Solution
A similar idea to the compile-solution was used, where the AST of a model would be manipulated. This implementation however would focus on manipulating a PyTorch model. The AST for a particular class or method can be gotten using the `ast.parse()` function available in the Python standard library. This would return an AST node representing the method, which would contain a list of AST nodes for all lines in the method as the `body` parameter. In the AST representing the `forward` function, the `body` list was modified by adding the AST representation of the code in @exit in a particular location.

#figure(
```python
y = self.avgpool(x)
y = y.view(x.size(0), -1)
y = torch.nn.Linear(y.size(1), self.num_classes)(y)
y = torch.nn.functional.softmax(y, dim=1)
entropy = -torch.sum(y*torch.log2(y+1e-20), dim=1)
if (entropy < 0.5):
    return (1,x)
```,
caption: "Equivalent code to exit AST") <exit>

Adding just a single exit in a specific location allowed the verification of whether the `forward` method of a ResNet model could be overwritten correctly. The updated AST node was then compiled, and the global memory space was modified, and the updated `forward` function was bound to the ResNet class. The value of 0.5 for the entropy threshold was taken from @BranchyNet were it was used as a default threshold. The `ExitTracker` class was used to track the original and current state of the `forward` function, and would handle the binding of the updated function to the ResNet class. To allow the exit to train correctly, the threshold was initially set to 3,000,000 instead of 0.5, as to have such a high value that the calculated entropy would always be less than the threshold. The exit was then trained using the same hyperparameters as the final exit. After this, the entropy was changed to 0.5 by recompiling the `forward` function again. 

Once the exit was trained, both the backbone model and the early exit only had an accuracy of 10%. This meant, in a dataset of 10 classes like CIFAR10, both exits were effectively guessing. This was due to the early exit updating the layers of the backbone model, negatively effecting it's ability to extract features, and the early exits weights were not being updated in the backpropagation. To solve this, a number of design changes were needed.

It was difficult to debug what updates were made to the `forward` function after compilation, so a different technique needed to be used to verify what was happening to the model. Instead of compiling the updated `forward` function, `ast.unparse()` is used to convert the updated AST back into source code for the `forward` function, and the module containing the model is overwritten. This made it trivial to debug when the AST manipulation was working incorrectly. As noted above, the entropy threshold needs to be updated twice, so to better handle the updating of exits, a new class `EarlyExit` is introduced to handle the tracking and manipulation of an individual exits AST. `ExitTracker` then keeps track of the current AST as a list containing both AST nodes and `EarlyExit`s. The `ExitTracker` class handled unparsing and writing the current AST to the model file.

One of the key parts of `EarlyExit` is it's ability to change the entropy threshold of the exit. This is done by finding the entropy for a target accuracy. As can be seen in @accuracyent, as entropy increases, the accuracy of an exit drops. The target accuracy for every exit can be taken as the overall accuracy of the backbone model. This should ensure that the exits do not cause a major regression in accuracy as they should be approximately as accurate as the base model. The entropy threshold for an exit can then be calculated from it's accuracy-entropy graph. This graph can be calculated by using the exit to classify test data, record the entropy and accuracy of every inference, sort the data points based on entropy, and plot the cumulative accuracy versus entropy.

#figure(
  image("./images/accuracy-entropy-graph.png", width: 100%),
  caption: "Sample graph of an exit's accuracy against entropy"
) <accuracyent>

Once the module is overwritten, it needs to be reloaded with `importlib.reload()`. While this would correctly save the updated architecture of the model, it would save the parameters of the model. This meant reloading during training would remove any updates completed during backpropagation. To solve this, a new class `ReloadableModel` is introduced. This class contains a PyTorch model, and has the ability to recreate the model with the updated architecture as well as saving the parameters during training. It achieves this by storing all parameters usually used in the models constructor and saving them. The stored model is constructed, and whenever the PyTorch model would have been accessed or called, the model stored in a `ReloadableModel` is accessed and called. Once the model needs to reload, all parameters of the model are saved to a temporary file, the module containing the model is reloaded, and the model is recreated using the same parameter used to construct it originally. The parameters are then loaded back into the model and training can continue. `ReloadableModel` also freezes the backbone models layers, which fixes the issue of early exits effecting it's ability to extract features.

The layers for the early exit were not being updated as any layers defined in the `forward` function, as is done in line 3 of @exit, are not registered by PyTorch's autograd engine, which handles backpropagation. To register layers, they must be defined in the model's `__init__` function. `EarlyExit` and `ExitTracker` are modified to track the current AST for both the `forward` and `__init__` functions, and the AST representing @exit is modified to call the saved layers in the updated `__init__` function.

Once exits were trained, some would perform poorly. Initial testing showed that any exit with a low entropy threshold, specifically exit's with thresholds below 0.1, would never be utilised by testing data. `ExitTracker` was modified to remove any `EarlyExit`s which had a threshold less than 0.1. Other pruning methods may produce more refined results but as a beginning assumption, it appeared to work decently.

All classes needed for adding early exits could manage multiple exits with different architectures, however actually training these exists was difficult. Multiple training schemes have been used in literature. @BranchyNet computes the gradients against the weighted losses of all exits at once. This was impractical due to time constraints as finding the correct weights for each loss takes a lot of training. @individualtraining suggest using a very simple method: all exits are trained individually. This method was more feasible as not as much training time was needed when compared to finding the weights for each loss. The choice of which exit to train first is also important. If the exits are trained top-down, that is from the first exit to the last exit, then the entropy thresholds need to be updated four times for each exit. First, to enable the exit the threshold is set extremely high. Next, the correct threshold is calculated for the target accuracy. Then, the exit needs to be disabled for the next exit to train on all available data. Once all exits are trained, every exit would then need to be re-enabled. To simplify this process, exits were trained bottom up, as this only required two changes to the threshold; one to enable the exit, and a second to correctly set it. Once the next exit was enabled, the correctly set deeper exit would never be utilised until all exits were trained.

= Results <Results>
The models tested were ResNet18, ResNet34 and ResNet50, with all implementations taken from the TorchVision repository @TorchVision. ResNet was the only architecture tested, as the implementations for all other models contain many layers of wrappers in their `forward` functions, which gave very few locations for exits to be placed. Due to time restrictions, other models could not be reimplemented, but there isn't a technical limitation stopping other models being utilised. ResNet101 and ResNet152 would have been selected if not for the limited memory on the available hardware, meaning these models could not be trained.

The datasets selected for training were CIFAR10, CIFAR100, QMNIST and Fashion-MNIST. CIFAR10 and CIFAR100 are composed of the same set of images which are labelled different to have 10 and 100 classes respectively. They have widespread usage throughout literature due to their good mixture of images while still not being too large to train in a reasonable amount of time. Both datasets were chosen to identify whether the number of output classes effected the performance of early exits differently. While CIFAR100 does take longer to train, and will have a lower accuracy in the results than models with CIFAR10 due to a cap on the number of epochs to run, the performance of the exits relative to the base model is what is important, not necessarily the absolute performance.

QMNIST and Fashion-MNIST both have 10 classes of inputs, where QMNIST is based on the MNIST dataset which contains digits between 0 and 9, and Fashion-MNIST contains images of clothing items. Both of these datasets are in black and white, mean the inputs are 1-channel. As CIFAR10 and CIFAR100 are both in color, and so have 3-channel inputs, it was important to use datasets with 1-channel input to verify whether the input shape would have an effect on the exits.

For brevity, the graphs and tables displaying the results will not be reprinted here, and can be found in Appendix C, as @graphs and @tables The most important points from these results are detailed here.

The branched models consistently had many exits which were infrequently utilised. This indicates that the pruning heuristic used in the implementation is poor. Due to time constraints, other pruning mechanisms weren't implemented. To get more accurate results, a set of stripped models were created, where the unutilised exits were manually removed. Other pruning methods exist, such as taking the cost of the exit into account @optimalbranchplacement @earlyexitmasters, so the results from the stripped should mimic that of a model trained with better pruning methods.

The branched models consistently were 1ms slower than the base models. This caused a major performance degradation in ResNet18, but had a far smaller effect on ResNet50. The stripped models did not show the same level of degradation, and in some instances performed better than the base models. This indicates that the architecture used in the added exits is not computationally negligible. 

The accuracy of both the branched and stripped models show a consistent drop in accuracy when compared to the base model. The drop in accuracy is measurable, but not as dramatic as the performance lost in inference time. As this effects both branched and stripped models this indicates that there are potential issues with the structure of the exits causing them to perform suboptimally.

= Analysis <Analysis>
Although the target accuracy for all early exits is set to the same accuracy as the base model, the average accuracy across all exits is lower than that of the base model. The cause of this is not exactly clear, but the likely cause is the early exits, while exiting with a high accuracy, are misclassifying a small number of samples that the full model would have accurately classified. One technique which could be employed to reduce this effect is adding a constant offset to the target accuracy. This would force early exits to be more confident when returning, although this would only reduce the effect, and not completely remove it. Adding a constant offset to the target accuracy would also greatly reduce the utilisation of exits, which is already a problem with the current implementation. 

A change in training scheme would likely improve the utilisation issue. @BranchyNet computes the gradients based on the weighted losses of all exits. This causes the exitss to perform better due to better regularisation of the exits which leads to an increase in accuracy. This could greatly improve the utilisation of earlier exits. While this wouldn't solve the accuracy degradation issue, it would significantly help the inference time degradation issue. If the inference time issue was solved, it may be worth keeping the small decrease in accuracy, given the increases in inference time were large enough. However this training scheme, as mentioned earlier, requires the weighting of each loss to be tuned as hyperparameters. This is a major drain on training resources so could not be used in testing. However, if more training resources were available, hyperparameter tuning could be completed automatically, similar to mechanisms used for traditional models.

It is difficult to place exits in locations that would be utilised due to how the TorchVision implementation of ResNet works. Every ResNet model has four layers, which each contain a predefined number of residual blocks. The third layer of the four always contains the majority of blocks, so it is always the case where exits before the third layer perform poorly as there have been very residual blocks completed, but after the third layer exits perform extremely well as most of the model can run. This can be seen since exits with the ID's 6 and 7 are by far the most common exits to be used in @tables. If the ResNet implementation were changed to have all block's individually written in the `forward` function, it is likely many more useful exits would appear.

The regression in inference time was more surprising than the degradation in accuracy, as it contradicts @BranchyNet, @optimalbranchplacement, and @earlyexitmasters. Using profiling techniques, it was found the entropy calculation itself accounted for more than two thirds of the time needed to perform an exit. That is, using `torch.log()` and `torch.sum()` were taking a significant amount of time to complete. The entropy calculation was taking longer than the entirety of `layer1`, `layer1` or `layer3` in the ResNet implementation. This is surprising as the functions appears short and the inputs are small. This is the same formula described in @BranchyNet, although that was implemented in a different framework. It was also implemented almost a decade ago, so it's possible improvements to technologies like CUDA have massively reduced the models runtime by improved feature extracting neural network operations such as convolutions and batch normalisations, without improving the performance of general computing functions such as log and sum calculations. It is also possible that as the inputs are wrapped in PyTorch `Tensor`s that there is a strange memory access pattern occurring, causing the functions to take longer to complete than expected.This could also be fixed by attempting to identify a new metric to use in place of entropy that wouldn't be as complicated to compute, although that is left as an open question for future investigation.

The stripped models occasionally beat the inference time of the backbone model, indicating better pruning methods would also speed up the branched model. One potential pruning method that could be used using only data available within the current implementation is using the percentage of the dataset a specific exit would exit from at the target accuracy as a metric. If the percentage of the dataset that would be used is not greater than the percentage cost of that exit, then that exit should be removed. This is an alternative to using the computational cost of the exit directly as suggested could be implemented above. An example of how that would work can be seen in @datasetent. By using the entropy chosen from the target accuracy, a corresponding percentage of the test dataset that a specific exit would return from can be calculated. If an exit half way through the model is not returning on more than 50% of the model, it is likely this exit will not be worth the loss in inference time.

#figure(
  image("./images/dataset-vs-entropy.png", width: 100%),
  caption: "Sample graph of an exit's dataset utilisation against entropy"
) <datasetent>

= Conclusions <Conclusions>
Early exiting has proven itself to be a useful method of decreasing the average inference time of neural networks. However, they can take a significant amount of engineering work to identify good locations for exits in many separate neural network architectures. To enable the widespread use of early exits, they need to be automatically added to neural networks. The proposed methodology for adding these exits is novel, and works around many of the limitations of popular Python machine learning frameworks, which make modifying the model architecture at runtime incredibly difficult. While there are some performance degradations, they are not due to inherit issues with the approach and stem from time and resource constraints on testing. The implementation of the approach is also very modular. This makes it easy to swap out sections of the code to test new approaches. For example, to test new exit architectures, only the `EarlyExit` class would need to be modified. While the performance of automatically branched models does not yet match that found in manually branched models, the proposed approach lays the groundwork needed to create branched models, and enables future research so implement some of the considerations identified in the analysis.

#heading(numbering: none, "References")
#section-bib()
