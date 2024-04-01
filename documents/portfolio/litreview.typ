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
= Neural Networks
Deep convolutional neural networks (CNNs) are neural networks containing many convolutional layers, and they are commonly used in tasks such as image classification and object identification. The convolutional layers act as filters which can extract certain features from an image. A simple example of this is a convolution which identifies where in an image any horizontal lines are located. The exact filters are learned by the network during the training on a given dataset. One of the state-of-the-art CNNs is ResNet, which stands for Residual Network @ResNet. A ResNet is a type of CNN that avoids a common issue with deep neural networks called the vanishing gradient problem (VGP). The VGP is when the gradient, which is a measure of how the layers should change to lower the overall error of the model, becomes too small to cause a meaningful update to the earlier layers in a network. This is done by using a "skip connection" where the input to a block of layers is also added to it output. This gives the model an additional, shorter path to update the earlier layers, as shown in @resnetskipconnection. The issue with deep neural networks like ResNet is the resource requirements to run the models, with ResNet152 requiring 11.3 giga floating point operations per second (GFLOPS) even though not all inputs require the full model to run before the model has converged on an output.

#figure(
  image("./images/resnet-skip-connection.png"),
  caption: [
    An example on a skip connection, shown as Figure 2 in @ResNet
  ],
) <resnetskipconnection>

= Optimising Neural Networks
Open Neural Network Exchange (ONNX) is both a file format and runtime environment for neural networks which has backing from major technology companies such as Intel and Nvidia @onnx. The file format stores a standard set of operations that neural networks can perform, including common operations such as convolution and batch normalisation, as well as the needed parameters for those operations such as the weights and biases. Having a common file format to store models with industry backing is important as existing frameworks typically just stored a models state in some form of custom file format or as a custom binary file, both of which locked users into using a single framework. 

Low-Level Virtual Machine (LLVM) is a compilation framework that can separate a high-level language from a low-level architecture @LLVM. It uses an intermediate representation called LLVM IR which acts as a high-level assembly language. High-level languages are parsed and converted into this IR using a frontend program, and the IR is compiled into a runnable binary for a particular architecture with a backend program. This IR can then be manipulated to provide low level optimisations, such as constant folding. The backend can also provide architecture-specific optimisations. Multi-Level Intermediate Respresentation (MLIR) was developed as an extendable version of LLVM IR, allowing for domain-specfic compilers to define higher level abstractions in the IR @mlir. These domain-specific IR entries are called dialects. Using dialects can make optimisation easier, and is used in the ONNX-MLIR compiler. ONNX-MLIR is used to compile an ONNX model while applying low-level optimisations @onnxmlir. It defines multiple dialects to perform different operations. For example, the `onnx` dialect is used for operations like shape inference, which computes the output shape of each instruction, and the `krnl` dialect allows for optimising loop instructions. All of the dialects provided by ONNX-MLUR are show in @onnxmlirarchitecture. Implementing optimisations in a compiler like ONNX-MLIR allows models developed in any framework to run on any LLVM-supported architecture without needing to reimplement optimisations many times, whereas training-time solutions would be framework specific.

#figure(
  image("./images/onnx-mlir-architecture.png", width: 55%),
  caption: [
    The architecture of ONNX-MLIR, shown as Figure 2 in @onnxmlir
  ],
) <onnxmlirarchitecture>

The above methods all focused on optimising a model after it has been developed and trained, but other methods exist to optimise models during training. One such method is neural architecture search (NAS) @NAS. NAS is a method by which a neural network designs a new neural network for a given dataset. @NAS utilises a recurrent neural network to provide designs for convolutional neural networks on particular datasets. This method tends to produce more accurate models, but these models were also much larger in terms of parameter count than manually-developed alternatives. Results from trying many handcrafted models as well as the generated models on the CIFAR-10 dataset are shown in @nasresults, which is taken from @NAS.

#figure(
  image("./images/nas-results.png", width: 40%),
  caption: [
    Results showing the improved error rate of generated models, shown as Table 1 in @NAS
  ],
) <nasresults>

This method, while producing more accurate models, also produces models which are more resource-heavy. Many higher level approaches to optimising models tend to optimise for accuracy, without much concern for the resource cost. Low level approaches tend to optimise for the number of instructions needed to run the model, while accepting some small loses to accuracy for the sake of increased inference speed. A better approach would reduce the number of instructions needed without affecting the overall accuracy of the model. 

= Early Exiting
Early exiting in neural networks is an area that hasn't received much acemedic interest, with few papers being released on the topic in recent years. The key idea behind early exiting is that a network can already have identified the correct output without running through the entire model. An easy input can be correctly classified without the need for the entire depth of state-of-the-art models. To solve this, many exits may be placed somewhere along the model. At each of these exits, some condition is used as a measure of how confident the model is that its current classification is the correct one. For example, a CNN doing object identification may be able to correctly classify an image of a dog early in the network, but for an image of a complex machine, the network may need to use all layers to correctly classify the input. If a good condition is used to estimate the likelyhood of the current classification being correct, the average inference time can be greatly lowered. 

Some of the issues to consider with this approach are the time taken to decide on whether to exit (which may grow significant as the number of exits grows large and as an exits complexity increases), trying to analyse different exit architectures, specifically exits including or excluding pooling layers, and the amount of training time needed to search for the best placement for branches. However, the biggest problem for widespread use of early exiting is the need to change the architecture of a neural network to use exits. Currently, model architectures need to be manually changed to make use of exits which may need a significant amount of analysis after already training the main model. Ideally, whatever framework or compiler is used should automatically add exits in optimal locations for the model.

One of the most cited ways to add early exiting is the BranchyNet architecture, described by #cite(form: "normal", <BranchyNet>). This architecture describes how exits should be structured when added to a model. In particular it describes how, for CNNs, an exit should consist of some number of convolution layers (the earlier the exit, the more convolutional layers) and a fully connected layer. This can be seen in @branchynetarchitecture. Each branch can then be trained independently. The entropy of the softmax of the output of the early exit is used as the condition referred to as `e` in this report. That is, `e = entropy(softmax(y))`. This value is a measure of how confident the model is that it has accurately classified the input. If the model is unsure, many of the possible classes will have a high value in the output, resulting in a high entropy. If the model has narrowed in on only a few possible options for the output class, most of the classes in the output will have a tiny value resulting in a smaller entropy. Each exit also has an associated threshold entropy value. If `e` is less than the threshold `T`, the model presumes it is correct, and the softmax of the output is returned. That is, `if e < T then return softmax(y) else continue`. This architecture is the basis of most of the recent research into early exiting. However, @BranchyNet discusses neither the optimal location of branches or the optimal number of branches per model. Ideally, a method for adding early exits automatically to an architecture should be smart enough to find the optimal placement and number of branches to use for a given architecture. The threshold for the exit is also presented as a hyperparameter for each branch to be learned during training-time in the paper, but may be decided by other methods.

#figure(
  image("./images/branchynet-architecture.png", width: 40%),
  caption: [
    An example of a BranchyNet model, shown as Figure 1 in @BranchyNet
  ],
) <branchynetarchitecture>

@optimalbranchplacement discusses an algorithm which, when given a set of different combinations of branches, will analyse which combination of branches reduces the average inference time the most. However, it is impractical to give a list of all possible combinations of exits to the algorithm, as an early exit could in theory be placed after every layer of a network, but that may be computationally expensive. Ideally, a method of automatically adding early exits shouldn't need to exhaustively check every possible combination, but should instead only look at combinations at least better than the current considered batch of early exits. @optimalbranchplacement also noted that, instead of having convolutional layers followed by a fully-connected layer, just having a fully-connected layer performs approximately equally to more complex exits.

@earlyexitmasters looks at different architectures that can be used as an exit, including single fully-connected layers, multiple fully-connected, and fully-connected layers combined with some pooling or batch normalisation layers. All exits shown in this paper have the confidence value of the model bounded between 0 and 1 by using a sigmoid function. Both the confidence value and the actual output from the network are outputted from the exit. A custom loss function is also presented that takes into account the cost of the model in terms of floating point operations (FLOPs), preferring lower cost exits, and also the accuracy and confidence of the model. Exits are also distributed using both self-similar methods and linear methods. This paper shows significant improves in performance without a significant decrease in accuracy. For example, an early exit ResNet152 trained on the SVHN database needs 2% of the relative cost of the original model, while losing 1% accuracy from 95.68% to 94.68%. Most models can keep the same accuracy with 20% of the relative cost. This paper shows how having an exit can have significant savings, but the exit has the confidence as a learnable trait, which differs from the solutions offered by this project.

All of the above show great improvements to the inference time of models without affecting the overall accuracy, and in some cases slightly increasing it. What hasn't been sufficiently researched is the exact effect of different exit structures on the inference time and the accuracy. #cite(form: "normal", <BranchyNet>) suggest using deeper exits is more benefical, while #cite(form: "normal", <optimalbranchplacement>), #cite(form: "normal", <earlyexitmasters>) suggest shallower exits perform decently as well. All of the above methods also rely on handcrafting the model with the exits, which can be a major blocker for implementing exits in pre-existing models. There also hasn't been a major focus on the effect of different training mechanisms. #cite(form: "normal", <BranchyNet>) trains all the exits using a weighted sum of loss functions, which updates all exits at the same time, where some exits have a higher effect on how much to update the model compared to other exits. Knowledge distillation is another mechanism which could be utilised, as described by #cite(form: "normal", <knowledgedistillation>). It uses a larger teacher model, and a smaller student model, where the output of the teacher model is used as the label for the student model. A potential use of this with early exits is treating each exit as a student model, while the original model is used as a teacher model.

= Split Computing
Early exiting is also useful for moving towards edge computing. As neural networks uses more personal data, there is a greater need for privacy, and one of the best ways to do that is to have models running on edge devices, such as mobile phones and personal computers. Some architectures have been proposed to run a subset of the model on a resource constrained edge device, and if that subset of the model is not able to exit confidently, then the rest of the inference is carried out by a centralised server. #cite(form: "normal", <TowardsEdgeComputing>) utilises this split computing approach with BranchyNet models to perform as much of the model as possible on edge devices.

Split computing could benefit from the automatic addition of exits as well. Every exit which can be trained well is a potential exit point, making it easier to split a model after every early exit. It is then easier to have multiple splits, with different edge devices running a different number of subsets depending on their available resources.

#heading(numbering: none, "References")
#section-bib()
