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
= Observations <observations>
When exits are not pruned, there is major performance degradation in terms of inference time. As the number of layers in the model increase, this discrepancy becomes smaller proportionally, but stays constant at approximately 1ms. The exit architecture used is extremely simple, so this result would seem to contradict @BranchyNet. Using profiling techniques on the `forward` function, most layers in the exit are performed quickly, with the exception of the line:
```python
entr = -torch.sum(pk * torch.log(pk + 1e-20))
```
This single line takes approximately as long to complete as the entirety of `layer4` in the ResNet implementation and almost entirely accounts for the slowdown in exiting. The cause for this is unclear. @BranchyNet did not use PyTorch to implement their architecture which makes it likely that this calculation could be quickened by avoid calling `torch.sum` and `torch.log`. The runtime for the above line was approximately `15ms`, whereas `layer3` takes approximately `18ms`. Calculating the entropy does not do any feature extraction, so to take this long to run is unacceptable. If this line were to be optimised, it is clear that the exit's would take negligible time, potentially removing the need to prune exit's at all. The times were measured using `time.time()` which does slow down a model, so the runtimes discussed do not represent a production model, but layers still take the same amount of time proportionally. Potentially, it may be worth exploring other measures of how well a model has converged due to the computational intensity of the entropy check.

Looking at the results shown in @tables, it is obvious the vast majority of exits are rarely used, with the majority of exits not being utilised on more than 1% of inferences. This is another cause of the performance degradation noted above. It may be acceptable to have a slow entropy calculation if the quick early exits were utilised more, however when exits placed near the end of the model are the only exits being used the performance will drop. Utilising more intelligent training mechanisms may solve this, such as the weighted losses method as described in @BranchyNet. The weighted losses method allow the exits to achieve better regularisation, which is important to ensure the exits do not attempt to extract the wrong features from the input.

It is also difficult to place exits in locations that would be utilised due to how the TorchVision implementation of ResNet works. Every ResNet model has four layers, which each contain a predefined number of residual blocks. The third layer of the four always contains the majority of blocks, so it is always the case where exits before the third layer perform poorly as there have been very residual blocks completed, but after the third layer exits perform extremely well as most of the model can run. This can be seen since exits with the ID's 6 and 7 are by far the most common exits to be used in @tables. If the ResNet implementation were changed to have all block's individually written in the `forward` function, it is likely many more useful exits would appear.

Early exits which occur before the majority of the blocks in a ResNet also should have a deeper architecture, which would mean including more convolution and non-linear layers. This would also help to improve the utilisation of earlier exits. The implementation of `EarlyExit` already supports this as exits are stored as lists of AST nodes which can be any length. This makes it easy to add more layers. The difficulty with adding more layers is deciding on how many layers should be used. One potential option would be to use neural architecture search as discussed in @NAS, and have the recurrent neural network return a list of layers that should be used, and their structure. For example it could specify how many kernels should be used in a convolution as well as the stride value. Alternatively, a simple metric could used where exits before the mid-way point of a CNN should contain a specific number of extra convolutional layers.

From the results, it is also clear there is a constant drop in accuracy across any model with early exits, no matter the number of exits, as can be seen in @graphs. Even with the target accuracy matching the accuracy of the backbone model, the early exits still misclassify inputs which the full model would have correctly classified. The above described improvements would likely improve this, as to improve utilisation would require more accurate exits, but some exits would still likely need to be pruned. Exit were already being pruned in the implementation, but the simple heuristic of removing exits with a threshold less than 0.1 did not prove sufficient. Early testing indicated that exits above this threshold would be utilised well, but the sheer number of infrequently used exits prove this to not be the case. Potentially, the target accuracy for every exit could be increased by a fixed amount to combat this fixed drop, but that would also decrease the early exits utilisation, which could have negative effects on the inference time of the model.

Improved pruning methods could be implemented to remove more underperforming exits. A good candidate metric is the percentage of the dataset an exit will exit from, as described in @pruningexits. For a given threshold level which is calculated from the target accuracy, a dataset percentage can be found. If this percentage is lower than the cost of completing that exit, then the exit can be considered bad. This would allow exits which are being utilised heavily to be kept while removing exits which have negative effects on the performance of the model.

= Future Work <futurework>
With the groundwork completed to correctly adding exits and allowing them to be trained, the implementation can be extended with some new features to allow further research. Importantly, most parts of the implementation are very modular, which allows many improvements to be implemented quickly. There are five main points of interest in the near future with the implementation.

Firstly, the entropy calculation needs to be optimised. @BranchyNet was originally implemented in a framework known as Chainer, and they did not run into this issue. It is highly likely there something that PyTorch is doing in the background with `torch.sum` and `torch.log` that is causing this strange performance degradation. Another option if there is no way to feasibly optimise this is to try and find a new metric which is more performance friendly to quantify how well the model has converged on an output.

Secondly, the training mechanism used can be improved. Inserting all exits at the beginning and then training individually leads to long training times. The first step would be to move to w weighted losses method and attempting to find the actual weights for each loss during training. If that is solved, then a searching algorithm can be used. For example, exits could be trained for `n` epochs, and after that time analyse the exits. Remove any underperforming exits and then continue for another `n` epochs until there are no underperforming exits.

Thirdly, @NAS could be utilised to find good architectures for each exit. This would potentially allow early exits to be utilised far more often, but architectures developed from a NAS method tend to be very large, so this may prove to be more of a hindrance than a boon.

Fourthly, since the main model file is being manipulated with this implementation, models should be automatically split into multiple classes to allow for easy deployment as shown in @TowardsEdgeComputing. Automatically generating these breaks in models along exit boundaries will make deploying sections of a model to edge devices relatively simple. This change shouldn't be over intensive as all the data needed to complete this is already stored between `EarlyExit`, `ExitTracker` and `ReloadableModel`.

Finally, there are a number of improvements that can be made to the frameworks used for machine learning in Python that would allow a far cleaner implementation of adding early exits to neural networks. PyTorch in particular can be difficult to work with because there is no good way to save the architecture of a model to a file, similar to what ONNX has achieved. If there was a method to save the architecture of a model along with its parameters, it would be very easy to avoid using the `ReloadableModel` class entirely as the source files for the model would not need to be modified. If PyTorch was modified to support importing ONNX models, the approach could be changed to modify ONNX model files which could result in an implementation that doesn't rely on the reloading of modified source code files.

= Conclusions
Early exits have shown themselves to be useful for reducing the inference time of models, but they take a long time to implement manually. Automatically adding exits to neural networks come with some major technical challenges, and has not been documented in the past. Much of the underlying infrastructure needed, such as the ability to reload modules and to update the forward graph of a model easily, was missing. This implementation has put forward the infrastructure needed to enable automatically adding early exits to neural in an expandable, modular fashion. While there are some performance regressions in models using this technique, this is mainly due to the simpler methods of training and simpler exit architectures used to allow the development of the infrastructure. In it's current state, the implementation is already useful for split computing as it automatically provides the potential exit points for edge devices, and even if the average inference is slightly slower, privacy is greatly improved. Now that the underlying framework of classes needed to add exits has been developed, future work can focus on developing better methods for training and constructing exits can be rapidly explored. 

#heading(numbering: none, "References")
#section-bib()
