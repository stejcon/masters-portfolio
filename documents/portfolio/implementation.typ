// #set page(paper: "a4", margin: (x: 41.5pt, top: 80.51pt, bottom: 89.51pt))
// #counter(page).update(0)
// #align(center, text(weight: "bold", size: 24pt, "Appendix B"))
// 
// #align(center, text(weight: "bold", size: 24pt, "Design and Implementation"))
// 
// Appendix content follows...
// #pagebreak()
// #set page(footer: align(right, "B-" + counter(page).display("1")))
// #counter(heading).update(0)
#set heading(offset: 1)
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

= Adding An Early Exit
To add an early exit to a model, there are a number of steps which would need to be followed.
+ The exits need to be generated in an abstract syntax tree (AST) or graph representing the original model
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

To begin the implementation, it was chosen to use ResNet34 as the base model, trained on the CIFAR10 dataset. This combination has the benefit of both being relatively quick to train to the available hardward, taking approximately 1.5 hours to train the model for 20 epochs on a Nvidia 3070. The implementation of ResNet34 was taken from the torchvision library included in PyTorch. The model was trained multiple time with varying hyperparameters to identify the hyperparameters which cause convergence the quickest, which proved to be:

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

#figure(
```python
y = self.avgpool(x)
y = y.view(x.size(0), -1)
y = torch.nn.Linear(y.size(1), self.num_classes)(y)
y = torch.nn.functional(y, dim=1)
entropy = -torch.sum(y*torch.log2(y+1e-20), dim=1)
if (entropy < 0.5):
    return (1,x)
```,
caption: "Equivalent code to exit AST") <exitcode>

This structure was based on the pooling block as described in @earlyexitmasters and the normal exit of a ResNet model as described in @ResNet. The value of 0.5 for the entropy threshold was taken from @BranchyNet were it was used as a default threshold before more suffisticated approachs were presented. The return for the exit, as well as all exits for this implementation, also return an exit ID, also referred to as a return ID in the implementation, to indicate what exit was used. 0 is used to indicate the original backbone model, and any other positive integer indicates which exit was used, with a higher number indicating a deeper exit. This would not be strictly needed in a production-ready implementation. However, to compare how often each exit is used for the purposes of testing, it is important to have a return ID.

The ExitTracker class was used to track the original and current state of the `forward()` function, and would then override the `forward()` function dynamically. The `exitTransformer` added the above exit structure approximately half way through the ResNet forward graph. The current forward AST was then compiled, and replaced the bound `forward` function for the class of the stored model, which in this case was ResNet. Replacing the function required accessing the global variables, finding the correct class, and then setting `forward` function to the updated, compiled `forward` function. Confusingly, `__get__` is used in this context to bind the new `forward` function to the model class, and the `forward()` function then needs to be set to the bound updated function. This is an artifact of Python's descriptor API.

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

To allow the exit to train correctly, the threshold was initially set to 3,000,000, as to have such a high value that no inference would result in an entropy above this. The exit was then trained using the same hyperparameters as the final exit. After this, the entropy was changed to 0.5 by recompiling the `forward` function again. This allowed the exit to train, but it would have somewhat incorrect entropy threshold.

== Issues with Initial Implementation
Once the exit was trained, the performance of the model dramatically fell, with both the final exit and early exit having an accuracy of approximately 10%. This occured irrespective of the chosen hyperparameters. This singular issue stemmed from a multitude of issues with the above naive approach.

+ The layers for the early exit were identical to when they were initialised, indicating that the backwards propogation was not updating these layers.
+ The layers for the backbone model were being updated as they were not frozen, effecting the final exit's accuracy as it could no longer extract the needed features.
+ There was no way to save the `forward` function, as attempting to save and load from a `.pyc` file caused many runtime errors which proved too difficult to debug given the time limitations of the project.
+ As there was no easy way to visualise the current state of the `forward` function once it was compiled, it was difficult to confirm whether the entropy threshold was correctly reset after the second compilation.
+ There was no method to remove an exit after it was added without completely resetting the `forward` function, which would result in every exit being permanent in the forward graph, which is impractical once multiple exits need to be added.

The following five major overhauls to the implementation were made to address all of these shortcomings.

== Unparse Instead Of Compile
To enable easier debugging, the Python code equivalent to the AST was saved to a file using the `ast.unparse()` function. This function takes an AST as input, and produces Python code which corresponds to the AST. This code is then written to a file to verify whether the entropy threshold was correctly set. `ast.unparse()` has some limitations, in particular it can fail if the AST being unparsed is too complex in terms of depth, and it also isn't guarenteed to perfectly invert `ast.parse()`. That is, `ast.unparse()` can produce different code to the code which originally produced the AST. Neither of these limitations come into play for this implementation however as exits are kept simple and there is no original code being parsed, exits are produced directly from a prewritten AST.

To keep track of each exit's individual AST to allow manipulation of parameters like the entropy threshold, a new class called `EarlyExit` was used. This class stored the exit's AST, and had multiple methods which manipulated the return ID and the threshold of an exit. This was not done using AST visitors as in the original naive implementation, but as the exact layers for each exit are already known, they were directly manipulated. The `ExitTracker` class was then modified to contain a list of AST nodes and `EarlyExit`s to represent the current `forward` function. The `ExitTracker` then handled creating new exits and saving the `forward` function to a file. This was complicated as the current `forward` function was now a list of multiple different classes, so when saving, a new list was created with the nodes and each exit's AST being added, before unparsing.

It was also chosen to overwrite the original class the model was created from. This was done as otherwise it would become complicated to keep track of newly written modules, and enabling use of the latest model in later runs of the code. This has some annoyances. If the models are committed in a source version control program such as Git, the models will freqently change and need to be recommitted. `ast.unparse()` also does not unparse code to be PEP8 compliant, which describes how Python code should be formatted, meaning external tools need to be used to keep the models module formatted for readability. Neither are technical limitations and have no bearing on the operation of the code, but are some minor nuisances for developers. These do not outweigh the benefits of having easily debuggable models, and the ability to correctly edit exits after they have been added which is essential to allow exit removal during training.

== Reloading An Overwritten Module
Once the model file has been rewritten, the model must be reinitialised, as the model code is compiled and stored in memory the first time it is loaded. It is relatively easy to just reload a module with `importlib.reload()`, however it becomes tricky when needing to ensure the state of the model besides the added instructions is kept identical between reloads. Reloading is further complicated by the fact that reloading with importlib only reloads the import in the current module and not globally. This means if the model is being referenced in multiple modules, as is done in this implementation since the training of models and addition of exits are implemented in different modules, both sets of the model must be reloaded. This becomes almost impossible to scale as keeping track of when to reload is incredibly tricky. To solve this, a new class called `ReloadableModel` was introduced. 

This class holds a reference to the model, and all of the parameters used to create the model. include the actual model class, any arguments passed to the constructor and any extra needed parameters. Whenever code needs to make use of the model, it should be accessed through a `ReloadableModel` instance, as that guarentees that the model being used is the reloaded version. Model creation is changed from 

```python
model = models.ResNet(models.BasicBlock, [3, 4, 6, 3])
```
to
```python
model = helpers.ReloadableModel(models.ResNet, models.BasicBlock, [3, 4, 6, 3])
```

To reload the model itself, the following steps are taken:
+ Save the parameters list for the model to a temporary file, which in PyTorch can be fetched by calling `model.state_dict()`.
+ The module the model is stored in is reloaded.
+ The stored values which were originally used to create the model are then reused when calling the models constructor.
+ The model parameters are readded to the model from the temporary file. It is important that the parameters are loaded with strict loading disabled. This is because the reloaded model will have a few extra potential parameters for the added layers which are not matched in the temporary file.

At this point, the layers from the base model are frozen. If these layers aren't frozen, they will be updated by the training of exits. This will ruin the backbone model's ability to extract necessary features from the input rendering it unusable. To do this, any layers whose name does not begin with "exit" has it's `requires_grad` option set to false. This technically could lead to issues if a model's implementation used layers with a name beginning with exit, but in practice from analysing implementations in the torchvision library, this isn't a practical issue. This will leave all of the exits capable of training without compromising the backbone model. This makes the assumption that the model is trained before being reloaded but this seems to be a reasonable assumption as there is little reason to reload until after the backbone model has been trained.

Once the model had been reinitialised, the model could not be moved to the correct device using the normal `.to(device)` method. This is important as the model and the input data need to on the same device, but the default device is the CPU. For some undiscovered reason, the model became split between both the CPU and GPU and could not be moved. The only fix was to change the default device used by PyTorch to the GPU whenever the device was fetched. This method may be suboptimal as changing the default device does take a minor amount of time (approximately two seconds), but in the overall implementation this time is unimportant. For multi-device training, this may need to be investigated further but dur to resource limitations, multiple GPU's could not be accessed as the same time for testing.

== Untrainable Layers
The weights for the exit layers still do not train correctly with this set up. This is due to a quirk of PyTorch's autograd engine which completes the backpropogation. Only layers included in a models `named_parameters` list are updated in the backpropogation. However no layers defined in the `forward` function, as was done with the linear layers as shown in @exitcode, are included in the `named_parameters` list. Only layers which have declared in the `__init__` function are updated. To solve this, the same methods used to update the `forward` function in `ExitTracker` are used to update the `__init__` function. `EarlyExit` was also modified to keep track of the layers which are needed for it's exit, and could modify the name of the layer from `self.exitX` to `self.exitY` to correspond to an updated ID. This would allow the freezing of the backbone model to continue to work while keeping the layers seperated. This is not limited to single instructions and it is possible to have deeper exits with multiple layers for each exit as `EarlyExit` stores a list of layers which it needs.

When creating the `exitX` layers, the naive solution computed the dimensions needed manually. As every instruction may require slightly different treatment to calculate it's dimensions, this became complicated. To solve this, lazy layers were used instead. These are an experimental feature in PyTorch where the layers can calculate their own input dimensions. This requires a model to be run with a sample inference of the expected input dimensions for runtime samples, and the layers will then be initialised with the correct dimensions. This is also added to the reloading functionality of the `ReloadableModel`, and even if the model doesn't contain lazy layers, the model will still function. While still experimental, lazy layers makes the possbility of deep early exits easy to implement with simply a different list of AST nodes needing to be given to an `EarlyExit` instance.

== Setting Threshold Levels
There is no clear consensus on what method is best for chosing an entropy threshold. @BranchyNet simply uses 0.5 as the threshold for all branches in it's implementation, whereas @earlyexitmasters uses a trainable threshold. The method developed to set the threshold for exits in this implementation is based on one assumption: no exit should be expected to achieve a higher accuracy than that of the backbone model. After running the testing data through an exit, a set of graphs can be created, plotting accuracy against entropy. An example of an exits accuracy against entropy is shown in @accent.

#figure(
  image("./images/accuracy-entropy-graph.png", width: 60%),
  caption: "Sample graph of an exit's accuracy against entropy"
) <accent>

The limit of this graph is the accuracy of an exit when all inferences use that exit. At any point earlier than that, the accuracy of the model for a given threshold is found. This does not include data on how often the exit at this point would be used, just how accurate it would be when used. To set the threshold, the accuracy of the backbone model is created. This is stored in `ExitTracker` as the target accuracy. Once an exit is trained, an epoch of testing data is ran with all inferences using the exit. The entropy which corresponds to the target accuracy is then used as the entropy threshold for that exit. The setting of the threshold is controlled by the `EarlyExit` class.

== Training Multiple Exits
As shown in @exitcode, the exit will only return if the condition of entropy being below the threshold is met. This causes some headaches for training the exits. To correctly control the training of the exits, this condition should always be met until an accurate threshold can be calculated. However, if exits are trained from first exit to last exit, and the first exit has it's correct threshold set, there is no way for further exits to have all inferences flow through them as the first exit will exit at least some of the time. To solve this, all exits are added to all possible locations in the `forward` function once the backbone model is trained, and they are all disabled. That is, the threshold is set to 0, and no inference can result in an entropy of 0 or less. Then, the bottom exit is enabled by setting it's threshold to a high value, as described for the naive solution. Once that exit is trained, it's threshold is correctly set, and the next exit which is less deep has it's threshold set high. This works as the deeper exit will never be used during the training cycle since the next exit has it's threshold so high. 

Training bottom-up makes the implementation far simpler than training top-down. The most complicated section is `ExitTracker` must keep track on the index of the currently training exit. This tracking is not smart, and assumes that exits are present at all possible locations, which is between every line of the `forward` function. This method was chosen due to time constrictions, and would need to be redesigned if a smarter searching method for optimal exit locations was used.

== Pruning Exits
There isn't a solid definition for what defines a bad exit. In general, if an exit requires more computing power than continuing to the next exit, it isn't worth including that exit in the model. Another way of defining a bad exit is whether of not an exit can accurately predicate a percentage of inputs relative to it's depth in the model. Similar to the accuracy against entropy plots, dataset against entropy can be shown as

#figure(
  image("./images/dataset-vs-entropy.png", width: 60%),
  caption: "Sample graph of an exit's dataset utilisation against entropy"
) <dataent>

If exit's are not computationally significant, they could be kept in the model with no consequence. However, if they are computationally significant, then bad exits should be removed. While both above methods could potentially work, time restrictions caused a simpler method to be adopted. From testing, exit's with a threshold below approximately `0.1` appeared to never be utilised, while exit's with thresholds above this did appear to be used, no matter how infrequently. Any exit's with a threshold below 0.1 were removed to allow more accurate analysis of exits which are utilised to prevent effecting the inference time of the model as heavily.

== Expanding To Other Models And Datasets
- Only use ResNet in this implementation
- The implementation of other models in torchvision weren't helpful for testing exits
- ResNet implementation made an assumption that all inputs were 3 channel RGB images
- Needed to rework the ResNet model as well as the `ReloadableModel` to correctly initialise the model with the number of channels needed

One of the questions which needs to be answered about early exits is how the depth of a model effects the performance of early exits. That is, there is no documentation on whether exits near the end of the model are more beneficial to runtime than those at the beginning, as while earlier exits exit quicker, they will exit less frequently. To analyse this effect, ResNet18, ResNet34 and ResNet50 were chosen as the models for testing. While utilising ResNet101 and ResNet152 would have been preferable, the available GPU did not have enough memory to load these models.

Multiple datasets were also chosen. Cifar10, Cifar100, QMNIST and Fashion-MNIST were chosen as the datasets to be tested. Cifar10 was chosen due to its popularity in literature, Cifar100 was chosen both due to popularity and to examine the effect of having a higher number of classes on early exits. ImageNet would also have been an interesting dataset to study but due to resource contraints this was not feasible. QMNIST and Fashion-MNIST were chosen to analyse whether having a lower number of channels in the input images would improve the performance of earlier exits as both datasets are in black-and-white.

The move to models having either 3 channel images (RGB) or 1 channel (B&W) caused some issues. Most parts of the implementation had not considered single channel inputs, including the torchvision implementation of ResNet which was utilised. To solve this, mutliple model definitions were created to be dataset dependent, and the number of input channels was changed from 3 to 1. The `ReloadableModel` class was then adapted to accept a parameter stating the number of input channels to control the dummy data used to initialise the lazy layers.

= Why This Solution Is Significant <b1>
This solution is significant as no literature has dealt with the issues involved in automatically adding exits to a model. Many of the challenges described above have not been discussed or documented, meaning the issues which appeared throughout the implementation of the above features were unexpected and were major blockers as they had been undiscovered. This solution lays the general groundwork for how exits should be automatically added, with most features being isolated and easy to change. To implement deeper exits, a different set of nodes can be sent to instances of `EarlyExit`. To provide searching for exits, all code can be excapsulated in `ExitTracker`. The approach described above, while implemented on PyTorch, should be relatively easy to reimplement on other frameworks.

As the major trappings of automatically adding exits have been covered, extending the solution to account for more potential areas of research should be not only possible, but feasible. There are some improvements still left to be made which will be discussed in @futurework

#heading(numbering: none, "References")
#section-bib()
