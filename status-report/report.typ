#import "template.typ": *
#import "@preview/timeliney:0.0.1"

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "Status Report",
  authors: (
    (name: "Stephen Condon", email: "stephen.condon5@mail.dcu.ie"),
  ),
  date: "January 8, 2024",
)

#show outline.entry.where(
  level: 1
): it => {
  v(12pt, weak: true)
  strong(it)
}

#outline(indent: auto)

= Project Problem
Deep Learning has grown in use throughout many industries in recent years. One of the most common types of network in deep learning is the Convolutional Neural Network (CNN). CNNs are commonly used for image based tasks, such as object identification, getting an objects bounding box and facial recognition, althought they can also be used for timeseries data. One of the biggest issues with using the best models is they may be too large to be useful in realtime. To run inferences at 60fps, inferences need to average #calc.round(1000/60, digits: 2)ms. On consumer hardware, a model such as ResNet152 may take over 100ms to execute, averaging about 8 to 10fps. Using the @inferencetimetest script on an i7-1260P to measure the inference time of a ResNet152 model on the CIFAR10 dataset, the average inference time was 356.3ms, which is #calc.round(1000/356.3, digits: 2)fps. To make these large models usable in cases where a high inference rate is needed, two kinds of optimisations can be used; training-time optimisations and compile-time optimisations. Training-time optimisations are any methods used to lower the final inference time during the training of a model. Compile-time optimisations are any methods used to lower the inference time during the compilation of the model into a usable binary.

Early exiting in neural networks is an area that hasn't received much acemedic interest, with few papers being released on the topic in recent years. The idea behind early exiting is that a network can have many exits. At each of these exits, some condition is used as a measure of how likely the model is to already have the correct output. For example, a CNN doing object identification may be able to correctly classify an image of a dog before the majority of the network is ran, but for an image of a complex machine, the network may need to run fully through to correctly identify it. If a good condition is used to estimate the likelyhood of the current output being the correct output, the average inference time can be greatly lowered. The some of the major issues to consider with this approach are the time taken to consider exiting which can be significant as the number of exits grows large, and how to analyse the condition used to see how effective it is. However, the biggest problem for widespread use of early exiting is the need to change the architecture of a neural network to use exits. Ideally, whatever framework or compiler is being used should automatically add exits where they make sense in a model.

One of the most cited ways to add early exiting is the BranchyNet architecture, described in @BranchyNet. This architecture describes how exits should be structured when used. In particular is describes how for CNNs, an exit should consist of some number of convolutional layers (the earlier the exit, the more convolutional layers) and a fully connected layer. Each branch can then be trained independently. The entropy of the softmax of the output of the early exit is used as the condition. That is, `e = entropy(softmax(y))`. Each exit also has an associated threshold. If the condition value e is less than the threshold T, then the softmax of the output is returned. That is, `if e < T then return softmax(y) else continue`. This architecture is the basis of most of the recent research into early exiting. However, the paper discusses neither the optimal location of branches or the optimal number of branches per model. Ideally, a method of added early exiting automatically to an architecture should be smart enough to find the optimal placement and number of branches to use for a given architecture. The threshold for the exit is also presented as a hyperparameter for each branch to be learned during training-time.

@optimalbranchplacement discusses an algorithm which, when given a list of different locations for branches, will analyse which combination of locations is ideal. However, it is impractical to give a list of all possible combinations of exits to the algorithm, as an early exit could be put after every layer of a network, but that would be computationally expensive. Ideally, a method of automatically adding early exits shouldn't need to exhaustively check every possible combination, but should instead only look at combinations at least better than the current considered batch of early exits. This paper also noted that, instead of having convolutional layers followed by a fully-connected layer, just having a fully-connected layer performs approximately equally to more complex exits.

Early exiting is also useful for moving towards edge computing. As neural networks uses more personal data, there is a greater need for privacy, and one of the best ways to do that is to have models running on edge devices, such as mobile phones and personal computers. Some architectures have been proposed to run a section of the model on an edge device, and if that section of the model is not able to exit confidently, then the rest of the inference is carried out in a central server. Early exiting from BranchyNet is used to assess how confident the edge device is about having an accurate output. One example of this is @TowardsEdgeComputing, which describes how to split the network to function across multiple devices.

= Unsuccessful Solution
A compile-time solution for adding exits was originally considered. The basic idea was that during compilation, the architecture for the exits would be added to the model. Then when the model was being ran, the exits would be trained during runtime based on the output from the final exit of the branch, and after some time the early exits when then be enabled and used. To do this, a compiler for models would need to be used, and the project that was considered for this was ONNX-MLIR @onnxmlir. This compiler is based on LLVM @mlir which is a framework used for designing compilers, and it reads in networks stored in the ONNX binary format @onnx as its source. It then can output a shared library to be used in other programs. There were multiple reasons that ONNX-MLIR was the best base to use for adding any optimisations developed in the project:

+ *ONNX-Based:* ONNX is a binary format used for storing models. It is a protobuf schema with support for many of the most used layers in models. It also has industry backing from companies such as Microsoft, Nvidia, Intel, IBM etc. Due to is wide industry support and strong schema, it makes for a good base to allow models from many frameworks to use a common compiler.
+ *MLIR-Based:* LLVM is a set of tools used to split a compiler into frontends and backends, with frontends being language specific and backends being hardware specific and the frontend sends data to the backend in the form of an intermediate representation (IR). So a programming language can create a frontend for LLVM specific to that language which outputs the LLVM IR, and then the backend for the target architecture can be used to build the binary executable. MLIR is intended to supersede LLVM IR by making it extensible, so tasks not usually found in typical programming languages can still use the target backends to generate binaries. MLIR is split into dialects, with each dialect being an extension that can be defined in a project. LLVM IR is now a dialect automatically available in MLIR. This means that ONNX instructions can be converted into a MLIR dialect and can be built for any platform which supports LLVM.
+ *Builtin Optimisations:* ONNX-MLIR already comes with standard optimisations builtin, such as instruction and constant folding, loop optimisations etc. This means that if any optimisations were added, it would make sure that they were compatible with any optimisations commonly used in industry.

To implement early exiting with ONNX-MLIR, after the ONNX model was converted to MLIR, the exits would be added as MLIR functions, with dimensions generated from the layer the exit was placed at. These would be inserted by editing compiler passes already built into the ONNX-MLIR project. Then during runtime, the branches weights could be updated by training, with the "correct" output being the output of the full model. Even though the full model could be inaccurate, it would ensure that the branches would approach the same accuracy of the full model, as expecting a higher accuracy from an earlier exit is impractical. The biggest theoretical issue to deal with is how to ensure that the branches would correctly have the weights rewritten back into the binary, and how to ensure that everytime the model is loaded, the branches aren't retrained once they are fully retrained, unless it is specifically requested to retrain the branches.

While there may be merit to this approach due to its more portable approach as any framework can output an ONNX model, there is an incredible steep level of learning to be able add code to large compilers such as ONNX-MLIR. Given the time contraints with this project, it is impractical to gain enough knowledge on the architecture of the ONNX-MLIR compiler to effectively add and test any code generation. Branches would also still have to be trained once they are deployed, which on mobile devices would make the model require much more power until the branches were fully trained and would also lower the performance. Due to the inexperience with ONNX-MLIR and the need for training after compiling anyways, a training-time method of adding branches to models was pursued instead.

= Proposed Solution
- How does my project fix some of the failings of the branchynet
- What investigations have been done so far
- How can metrics developed be used to identify good places for branches
- How can this be done with PyTorch
- What models and what datasets will be used

The training-time solution proposed is specific to the PyTorch framework. Any training-time solution is likely to be framework-specific, as the training function must be changed once early exiting is enabled and the training function is framework-specific. To narrow the focus of the project, only PyTorch will be used, but the general approach should still be applicable to other frameworks. If time is available, a sample of converting the method to another framework may be provided, but it is not a primary focus of the project.



= Project Plan for Semester 2
#timeliney.timeline(
  show-grid: true,
  {
    import timeliney: *

    headerline(
      group(..range(13).map(n => strong("W" + str(n+1)))),
    )
  
    taskgroup(title: [*Status Report*], {
      task("Status Report", (0, 1), style: (stroke: 2pt + gray))
      task("Presentation Video", (0, 1), style: (stroke: 2pt + gray))
      task("Interview", (1, 2), style: (stroke: 2pt + gray))
    })

    taskgroup(title: [*Implementation*], {
      task("Consolidate test models", (1, 2), style: (stroke: 2pt + gray))
      task("Implement algorithm", (2, 5), style: (stroke: 2pt + gray))
      task("Train models", (5, 7), style: (stroke: 2pt + gray))
      task("Analyse results", (6, 7), style: (stroke: 2pt + gray))
    })

    taskgroup(title: [*Final Paper*], {
      task("First draft", (6, 8), style: (stroke: 2pt + gray))
      task("Gather extra results", (8, 9), style: (stroke: 2pt + gray))
      task("Finalise paper", (9, 10), style: (stroke: 2pt + gray))
      task("Collect all appendices", (10, 11), style: (stroke: 2pt + gray))
      task("Final interview", (11, 13), style: (stroke: 2pt + gray))
    })

    milestone(
      at: 1,
      style: (stroke: (dash: "dashed")),
      align(center, [
        *Status Report Due*\
        22/01/2024
      ])
    )

    milestone(
      at: 11,
      style: (stroke: (dash: "dashed")),
      align(center, [
        *Paper Submission*\
        25/03/2024
      ])
    )
  }
)

#show bibliography: set heading(numbering: "1.1")
#bibliography("refs.bib")

= Appendix
== ResNet152 CIFAR10 Inference Time Test <inferencetimetest>
```py 
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import time

# Function to measure inference time
def measure_inference_time(model, dataloader, device):
    model.eval()
    total_time = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time

    average_time = total_time / num_batches
    return average_time

# Function to load CIFAR-10 dataset
def load_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    cifar10_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ResNet152 model
resnet152_model = resnet152(pretrained=True).to(device)

# Specify batch size
batch_size = 64

# Load datasets
cifar10_dataloader = load_cifar10(batch_size)

# Measure average inference time for CIFAR-10
cifar10_avg_time = measure_inference_time(resnet152_model, cifar10_dataloader, device)
print(f'Average Inference Time on CIFAR-10: {cifar10_avg_time:.4f} seconds')
```
