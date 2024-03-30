#set page(paper: "a4", margin: (x: 41.5pt, top: 80.51pt, bottom: 89.51pt))
#counter(page).update(0)
#align(center, text(weight: "bold", size: 24pt, "Appendix B"))

#align(center, text(weight: "bold", size: 24pt, "Design and Implementation"))

Appendix content follows...
#pagebreak()
#set page(footer: align(right, "B-" + counter(page).display("1")))
#counter(heading).update(0)
= Adding An Early Exit
To add an early exit to a model, there are a number of steps which would need to be followed.
+ The exits need to be generated in an abstract syntax tree or graph representing the original model
+ The model needs to be saved with its' new structure.
+ The exits then need to be trained

The original model with no exits (the backbone model) can be either trained or untrained when adding the structure of the exits, what is important is that the backbone model is trained before any of the exits. Once the backbone model is trained, the exits can then be trained by knowledge distillation @knowledgedistillation, a weighted sum of losses @BranchyNet, or individual losses @individualtraining. Both training-time and compile-time solutions would need to follow this general approach.

= Compile-Time Solution
A compile-time implementation for the automatic addition of exits was the first approach analysed. As discussed in Appendix A, ONNX-MLIR is a compiler for models which can allow them to run on any LLVM-supported architecture, which is preferable to a framework-specific approach. It exposes multiple passes over an onnx model, but the most useful for editting the model itself is the first pass, `ONNXOpTransformPass`. This also for manipulation of the forward graph of the model for operations such as constant propogation and shape inference. These operations use rules written in LLVM's TableGen domain-specific language. To add the exit structures to the model, TableGen rules could be used to identify blocks within a CNN, as blocks generally contain a convolution, an averaging operation, and possibly another non-linear layer such as the rectifed linear unit (ReLU).

Once the blocks are identified, an exit would need to be generated in the `onnx` MLIR dialect. This becomes tricky, as literature hasn't found a good method for structuring exits, other than the earlier the exit the more layers should be included. Having deeper exits is even disputed by @optimalbranchplacement@earlyexitmasters. To being, all exits will be structured as an averaging layer, followed by a fully connected layer. This matches the pooling blocks laid out by @earlyexitmasters as well as the exits placed in ResNet models @ResNet. These added blocks would not be trained at this point, so all weights would need to be initialised. Methods for initialising layers has been discussed for years in literature, with many different initialisation schemes being used. Most major Python frameworks use Kaiming initialisation @kaiming as a default, so it is likely that this should be used on the exits, but it is still an open question.

After the exits are added, the model would then be compiled to a binary. However, if the model is in binary form, it difficult to train the exits. The onnx runtime library can be used to load the model into memory, but then training is difficult since to train, there needs to exist a PyTorch implementation of the onnx model. Some layers in the model also have parameters that need to be tracked during training, like batch normalisation. So if the model had been trained and then saved, these parameters are lost making accurately training take much longer. Compiling also isn't normally done until a model is being deployed, so this means edge devices would need enough resources to accurately train the exits, but edge devices are typically resource constrained making this process infeasible.

There is potential in this method to be useful due to its portability. However there were too many obvious issues that would need to be solved given the timeframe of the project. Utilising TableGen can be difficult with work with since debugging the model after running the compiler passes can lead to long output messages with little important information. Build times were concerning, as rebuild ONNX-MLIR on the available hardware took a significant amount of time, and the model would then need to be trained afterwards anyways. Attempting to correctly initialise layers would also take significant time away from testing the automatic insertion of layers as well. Due to this, a training-time approach was taken instead.

= Training-Time Solution
- PyTorch was chosen due to its popularity
- Torchvision comes with prebuilt datasets and models that can be copied and changed easily
- Began by attempting to implement early exits for resnet models with the cifar10 dataset. both are common in literature so make for good comparisons

== Initial Implementation
A single exit was added in a particular location in the ResNet `forward()` function. This was done to allow testing the addition and training of an AST based exit before adding the complexity of an algorithm to place useful early exits. The exits were created by compiling a list of AST nodes composed of the AST from the `forward()` function of the model, and the AST corresponding to the following code:

```python
y = self.avgpool(x)
y = y.view(x.size(), -1)
y = torch.nn.Linear(, self.num_classes)
y = torch.nn.functional(y, dim=1)
entropy = -torch.sum(y*torch.log2(y+1e-20), dim=1)
if (entropy < 0.5):
    return (1,x)
```

- Started by compiling to bytecode then replacing forward function
- The exit architecture used was based on the actual exit of a resnet model, and had a pooling layer and a fully connected layer, comparable to what the other masters project did
- Used ast tree manipulation to generate the exists, and then compiled and replaced
- This meant it was difficult to see exactly how exits were being structures and placed, and also made it hard to see if thresholds were being updated
- Generating an exit needed the dimensions of the fully connected layer to be calculated, was just taken from the output of the previous instruction
- An exit tracker class was used to keep track of the original ast of the forward function and the current ast, to allow the ast to be reset if needed
- Exits were added individually, then trained, then the next added
- The threshold was set by getting the entropy that would make the exit be equally as accurate as the final dataset
- This seemed like the most obvious route for allowing a searching algorithm, but tracking where the exits were became difficult
- Instead, all exits were placed in at the beginning to simplify the logic while still trying to get the exits working

== Issues with Initial Implementation
- Ran into issue where exits werent being trained correctly, which stemmed from multiple distinct issues with the approach.
 - The base model was not frozen, meaning the early exit was effecting the early layers to the point where the final exit was effectively guessing
 - The exits layers weren't having weights updated since they weren't initialised in the \_\_init\_\_ function, this meant both the init function and the forward function had to be replaced
 - Replacing the init function meant a model would need to be reinitialised during training
 - reinitialisation meant some weights were reset
 - Replaced functions weren't saved as couldn't correctly load function from .pyc files, doesn't appear to be a technical reason why but proved more difficult than it was worth in the given time
 - Exits couldn't be removed as trying to identify what parts of the current ast were exits was difficult

=== Unparse Instead Of Compile
- This required multiple extra classes
- Firstly, to save replaced function, `ast.unparse()` was used, this could take the updated ast nodes and unparse them into the equivalent python that would have generated them. it has some limitations described in the documentation but none apply
- It was chosen to rewrite the unparsed classes back into the file they were originally loaded from, this meant the code could be simplified as it didnt need to decide which version of the model should be loaded, but has the annoyance on both not formatting the written code to PEP8, and since it rewrites to the same file it means models need to be constantly committed in git
- Both minor annoyances and do not effect the technical working of the code, and has the greatly added benefit of making it exceptionally easy to see how the model is updated

=== Reloading An Overwritten Module
- Once the file was rewritten, the module and the model had to be reloaded. The module would need to be reload in any other module it was loaded in as well
- To do this, the classes `ReloadableModel` was written. It would contain the model type, the arguments used to contruct the model, and the actual instance of the model
- Whenever a model is created, it should be created by this wrapper class, and whenever a function needs the model, it should access it from the wrapper
- The wrapper has a reload function which will save the weights of the instance of the model, use importlib to reload the models module, recreate the model with the updated structure, and then load back in the temporarily saved weights
- This is also where the freezing of the model is placed, this is because it is assumed that a model is trained before it is reloaded

=== Untrainable Layers
- The ast generation needed to be adapted to first set the exit layers in the init function, and then to use those layers in the forward function
- This is because layers declared in the forward function are not part of the named parameters list in the pytorch model, which are the only parameters the autograd engine will update
- LazyLayers were used when creating the instructions in the init function as it was easier than calculating dimensions, the model needs to be run with dummy data once initialised for the dimensions to be calculated though
- This has the added benefit of the approach being more expandable by easily allowing more complicated instructions to be used in the exit architecture if needed, such as a Conv2D

=== Pruning Bad Exits
- What defines a bad exit is unclear, and can be defined multiple ways
- A simple pruning method of removing exits with an entropy of less than 0.1 was chosen as a starting point as it was observed that exits with a threshold below this generally weren't being used
- Removing an exit required a way of tracking which parts of the ast were exits
- To do this, the current ast list in ExitTracker was changed to contain ast nodes from the original ast and objects of EarlyExit type
- The early exit class has functions to manipulate both the threshold and the id of the return point, this allows renumbering of the exits after an exit is removed

== Expanding To Other Models And Datasets
- Something about wanting to compare the depth of models
- Using multiple datasets
- everything had asssumed 3 channel, but then some 1 channel datasets were used so had to track the channels in the dataset
- pytorch implementation hardcodes the number of channels, so slightly modified the code

== Why This Solution Is Significant
- No work had been presented on the infrastructure that would be needed for automatically generating exits
- All of the elements of the code that have potential for interesting future work are also very modular, so it will be easy to expand the implementation to try different search methods and different exit architectures
- None of the code is particularly pytorch specific, and much of the code can be quickly adapted to other frameworks

#bibliography("refs.bib")
