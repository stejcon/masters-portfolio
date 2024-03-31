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

= Conclusions

= Future Work <futurework>
With the groundwork completed to correctly adding exits and allowing them to be trained, the implementation can be extended with some new features to allow further research.

+ A
Text

+ B
text

+ C
text

+ D
text

- Train with weighted method from BranchyNet
- Generate split models for use in split computing, should be trivial
- Use NAS to generate the exit architecture
- Use metrics to "search" for exits

Can i reference @b1 here?

#heading(numbering: none, "References")
#section-bib()
