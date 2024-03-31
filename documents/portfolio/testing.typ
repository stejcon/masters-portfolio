// #set page(paper: "a4", margin: (x: 41.5pt, top: 80.51pt, bottom: 89.51pt))
// #counter(page).update(0)
// #align(center, text(weight: "bold", size: 24pt, "Appendix C"))
// 
// #align(center, text(weight: "bold", size: 24pt, "Testing and Results"))
// 
// Appendix content follows...
// #pagebreak()
// #set page(footer: align(right, "C-" + counter(page).display("1")))
// #counter(heading).update(0)
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
= Choice of Models and Datasets

= Choice of Training Scheme

= Method of Generating Results

= Effect of Automatically Generated Early Exits on Runtime

= Effect on Automatically Generated Early Exits on Accuracy

#heading(numbering: none, "References")
#section-bib()
