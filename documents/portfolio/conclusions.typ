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
- Massive losses in performance contradict findings in @BranchyNet. However, this isn't due simply because of a branched network as they also save time in split computing @TowardsEdgeComputing.
- Profiling the forward function with print statements shows the line takes as lone as multiple blocks of the resnet model. That is, calculating the entropy is the main bottleneck for performance. All other layers of the exit take negligble time. If this line was to be optimised, the exits would function far quicker and would result in a speed up to the average inference time.
```python
entr = -torch.sum(pk * torch.log(pk + 1e-20))
```

- Ealier exits are very rarely used, only the last two or three exits are every really used. This could be due to multiple reasons. First, the implementation of ResNet in TorchVision has four "layers", which are defined by a list similar to something like `[3, 4, 6, 3]`, where each number is the number of blocks in the corresponding layer. The second last number contains the majority of the layers, so most of the feature extraction is done here. Earlier exits do not have enough information to exit unless they were made deeper with more convolutional layers. This could be fixed in multiple ways.
- The ResNet implementation could be changed to include all blocks individually in the `forward` function to allow more exits to be placed between exits. This would majorly degrade performance however, unless the entropy calculation was optimised.
- Neural architecture search could be utilised to suggest good locations for branches as well as the architecture for each branch. The output suggestions of the NAS could then be used as an input for the `EarlyExit` AST lists.
- Pruning methods could be improved to remove more underperforming exits. A good candidate metric is the percentage of the dataset an exit will exit from. For a given threshold level which is calculated from the target accuracy, a dataset percentage can be found. If this percentage is lower than the percentage cost of completing that exit, then the exit can be considered bad.

= Conclusions
- Automatically adding early exits to neural networks is a novel method of attempting to speed up models.
- Branches have already had success in applications such as split computing, but the need to manually add them adds a long time to model creation
- The functioning of popular python ml frameworks makes the generation of new parts of a model during training difficult
- However, now that models can be trained and tested with early exits being added, the development of good exits should rapidly speed up.

= Future Work <futurework>
With the groundwork completed to correctly adding exits and allowing them to be trained, the implementation can be extended with some new features to allow further research.
- Optimise the entropy calculation
- Train with weighted method from BranchyNet
- Generate split models for use in split computing, should be trivial
- Use NAS to generate the exit architecture
- Use metrics to "search" for exits

#heading(numbering: none, "References")
#section-bib()
