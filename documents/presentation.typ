#import "@preview/polylux:0.3.1": *

#import themes.simple: *

#set text(font: "Inria Sans")
#set cite(form: "normal")

#show: simple-theme

#title-slide[
  = Evaluting the Effects of Early Exits on the Performance of Convolutional Neural Networks: Status Report
  #v(2em)

  Student: Stephen Condon, 19403354

  Supervisor: Dr Robert Sadlier

  #datetime.today().display("[day]/[month]/[year]")
]

#centered-slide[
  = Current State of Neural Networks
]

#slide[
  == Issues
  - Deep models are taking more resources to run
  - Many industries are trying to use models in more applications
  - Many of these applications deal with personal data

  To solve these issues, deep models need to be adapted to be runnable on the edge.

  #figure(
    image("./resnet.png"),
    caption: [
      _ResNet34 Model_ @ResNet
    ],
  )
]

#centered-slide[
  = BranchyNet
]

#slide[
  Adding branches or early exits is the most common solution.
  *BranchyNet* @BranchyNet is an architecture that suggests one method of adding these exits.
  It suggests:

  - Branches should use entropy as an exit condition
  - Each branch has an entropy threshold to exit
  - Branches can have differing layers

  #figure(
    image("./exit.png"),
    caption: [
      _An example of different exits in a single model_ @BranchyNet
    ],
  )
]

#slide[
  == Do exits need to be complicated?
  BranchyNet has somewhat complicated exits. "Optimal Branch Location for Cost-effective Inference on Branchynet" @optimalbranchplacement shows that this isn't always needed.
  
  Instead, a single fully-connected layer seems to suffice. Other papers such as "Early-exit Convolutional Neural Networks" @earlyexitmasters do some analysis with bigger branches including pooling layers.
]

#slide[
  == What other benefits do early exits bring?
  For IoT problems using neural networks, adding branches makes it easy to split networks. This is shown in multiple papers, where networks are split, one half run on embedded devices,
  the rest in the cloud. One is example is "Towards Edge Computing Using Early-Exit Convolutional Neural Networks" @TowardsEdgeComputing, which shows how networks can be split.

  #figure(
    image("./edge.png"),
    caption: [
      _A split network as shown in @TowardsEdgeComputing Figure 2_
    ],
  )
]

#centered-slide[
  = Compile-Time vs Training-Time
]

#slide[
  == Adopting Early Exiting
  Early exiting requires models to be adapted with a new architecture, which for prebuilt models may turn into a lot of work with needing to engineer exits.
  
  For adoption to increase, models should be automatically adapted to have exits either at training-time or at compile-time.
]

#slide[
  == Compile-Time and ONNX-MLIR
  Every model needs to be compiled to a binary to be used. ONNX is one of the industry standard formats
  for storing models. MLIR is a compiler technology developed by LLVM, allowing any compiler frontend to compile 
  to any supported hardware using an intermediate representation.

  One of the proposed solutions included using ONNX-MLIR @onnxmlir and adding compiler passes to add the early exit architecture
  to models. Then during runtime, the exits can be trained against the output of the model until they are sufficiently accurate.

  This solution has some issues, as trying to train a model on edge devices while also running the model is extremely difficult if not
  impossible.

  The biggest benefit of this solution is how portable it is to different hardware architectures thanks to MLIR.
]

#slide[
  == Training-Time and ONNX-MLIR
  Compile-Time solutions still require some kind of training, so adding branches at training-time can make more sense.
  However, training-time solutions would need to be framework specific. No two frameworks have the same way of expressing
  a model and training a model. 

  PyTorch was chosen to be used for a training-time solution due to it's popularity and ease of use. PyTorch is also good
  as it already supports branches.

  To add early exiting at training-time, branches need to be added to the model using code generation. Branches are added based on the layers surrounding it, then the branches are trained.
  If the branch is taking a long time to converge or the loss is too high, it can be removed and a new branch added and trained.

  To decide on whether a branch is good enough, mostly on FLOPs vs accuracy is observed. A metric that has come out from this also includes the percentage of the training database that can be exited from a branch.

  #figure(
    image("./image.png"),
    caption: [
      _Sample Graphs of Accuracy and Dataset Progress vs Entropy_
    ],
  )
]

#centered-slide[
  = What's left to be done?
]

#slide[
  == Currently done:
  - Multiple ResNet models with branches have been trained against the CIFAR10 Dataset
  - The main metrics have been developed
  - Some experiments with doing the code generation have been done

  == Left to do:
  - Use multiple datasets for testing, and potentially multiple models.
  - Code generation needs to be completed
  - Implementing the search for branch locations needs to be completed.


  #figure(
    image("./timeline.png"),
    caption: [
      _Project Plan for Semester 2_
    ],
  )
]

#slide[
  #show bibliography: set heading(numbering: "1.1")
  #bibliography("refs.bib")
]
