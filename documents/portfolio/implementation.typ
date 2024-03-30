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
PyTorch was the obvious choice as a framework to implement this solution. It has a very structured method for creating models, meaning any AST manipulations created to mimic the compiler passes described above would be easier to create. It also comes with the torchvision library, which includes many implementations of popular models and wrappers around popular datasets to make training far easier. The documentation for PyTorch is also high quality meaning any issues could potentially be solved far quicker.

To begin the implementation, it was chosen to use ResNet34 as the base model, trained on the CIFAR10 dataset. This combination has the benefit of both being relatively quick to train to the available hardward, taking approximately 1.5 hours to train the model for 20 epochs on a Nvidia 3070. The model was trained multiple time with varying hyperparameters to identify the hyperparameters which cause convergence the quickest, which proved to be:

#figure(
table(
  columns: 2,
  align: left,
  table.header(
    [*Parameter*], [*Value*],
  ),
  "Epochs",
  "20",
  "Learning Rate",
  "0.01",
  "Learning Rate Scheduler",
  "Exponential, w/ 0.99",
  "Loss",
  "Cross Entropy Loss",
  "Optimizer",
  "Stochastic Gradient Descent",
  "Weight Decay",
  "0.001",
  "Momentum",
  "0.9"
),
caption: "Chosen Hyperparameters"
)

== Initial Implementation
A single exit was added in a particular location in the ResNet `forward()` function. This was done to allow testing the addition and training of an AST based exit before adding the complexity of an algorithm to place useful early exits. The exits were created by compiling a list of AST nodes composed of the AST from the `forward()` function of the model, and the AST corresponding to the following code:

```python
y = self.avgpool(x)
y = y.view(x.size(0), -1)
y = torch.nn.Linear(y.size(1), self.num_classes)(y)
y = torch.nn.functional(y, dim=1)
entropy = -torch.sum(y*torch.log2(y+1e-20), dim=1)
if (entropy < 0.5):
    return (1,x)
```

This structure was based on the pooling block as described in @earlyexitmasters and the normal exit of a ResNet model as described in @ResNet. The value of 0.5 for the entropy threshold was taken from @BranchyNet were it was used as a default threshold before more suffisticated approachs were presented.

The ExitTracker class was used to track the original and current state of the `forward()` function, and would then override the `forward()` function dynamically. The `exitTransformer` added the above exit structure approximately half way through the ResNet forward graph. The current forward ast was then compiled, and replaced the bound forward function for the class of the stored model, which in this case was ResNet. Replacing the function required accessing the global variables, finding the correct class, and then setting forward function to the updated, compiled forward function. Confusingly, `__get__` is used in this context to bind the new forward function to the model class, and the `forward()` function then needs to be set to the bound updated function. This is an artifact of Python's descriptor API.

```python
class ExitTracker:
    def __init__(self, model):
        self.first_transform_complete = False
        self.model = model
        self.original_forward_ast = getAst(self.model.forward)
        self.current_forward_ast = getAst(self.model.forward)

    def transformFunction(self):
        exitTransformer = AddExitTransformer()
        if not self.first_transform_complete:
            self.current_forward_ast = exitTransformer.visit(self.original_forward_ast)
            self.first_transform_complete = True
        else:
            self.current_forward_ast = exitTransformer.visit(self.current_forward_ast)
        ast.fix_missing_locations(self.current_forward_ast)
        code_object = compile(self.current_forward_ast, '', 'exec')
        exec(code_object, globals())
        bound_method = globals()["forward"].__get__(self.model, self.model.__class__)
        setattr(self.model, 'forward', bound_method)
```

The `exitTransformer` used an AST visitor which passes over the original AST, and performs an operation on every `Assign` node. To specifically place the exit to ensure training would work, a series of checks were done to check if the current assign was for `layer1`, and if this is true, to append the exit.

```python
class AddExitTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "layer1":
            return [node] + exitAst
        return node
```

To allow the exit to train correctly, the threshold was initially set to 3,000,000, as to have such a high value that no inference would result in an entropy above this. The exit was then trained using the same hyperparameters as the final exit. After this, the entropy was changed to 0.5 by recompiling the forward function again. This allowed the exit to train, but a somewhat correct entropy value.

== Issues with Initial Implementation
Once the exit was trained, the performance of the model dramatically fell, with both the final exit and early exit having an accuracy of approximately 10%. This occured irrespective of the chosen hyperparameters. This singular issue stemmed from a multitude of issues with the above naive approach.

+ The layers for the early exit were identical to when they were initialised, indicating that the backwards propogation was not updating these layers.
+ The layers for the backbone model were being updated as they were not frozen, effecting the final exit's accuracy as it could no longer extract the needed features.
+ There was no way to save the forward function, as attempting to save and load from a `.pyc` file caused many runtime errors which proved too difficult to debug given the time limitations of the project.
+ As there was no easy way to visualise the current state of the forward function once it was compiled, it was difficult to confirm whether the entropy threshold was correctly reset after the second compilation.
+ There was no method to remove an exit after it was added without completely resetting the forward function, which would result in every exit being permanent in the forward graph, which is impractical once multiple exits need to be added.

The following five major overhauls to the implementation were made to address all of these shortcomings.

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

=== Setting Threshold Levels
- Needed to edit the threshold for each exit
- Using the test data after a full train of an exit, determine the entropy level where the overall accuracy of the exit would match the final exit
- Set that threshold
- For an exit to be trained, all training data must exit through the if statement
- This meant if an exit was enabled, the data would never go through the rest of the forward graph
- To solve this, train from bottom up
- Would require a more sophisticated approach if a searching algorithm is used, as exits would need to be temporarily disabled but keep track of the correct threshold to reenable

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
