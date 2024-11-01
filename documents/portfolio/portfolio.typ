#show bibliography: none
#bibliography("refs.bib")
#set figure(placement: auto)
#align(center, text(weight: "bold", size: 24pt, "Evaluting the Effects of Automatically Added Early Exits on the Performance of Convolutional Neural Networks"))

#align(center, text(weight: "bold", size: 18pt, "Project Portfolio"))
#pagebreak()
#heading(outlined: false, [Acknowledgements])
I would like to extend my sincere thanks to Dr. Robert Sadlier for his constant guidance throughout the process of finding a research topic and his help in working around technical blockers with the project. His advice proved essential to finishing and presenting a solid implementation for this project. He ensured I was consistently using my time for the most important features of the implementation.

I would also like to acknowledge the late Dr. Kevin McGuinness who inspired the project and presented me with many ways a research project could explore early exiting. Without his guidance this project would not have progressed in the direction it has, and would not have been as interesting or important.

#pagebreak()
#let clean_numbering(..schemes) = {
  (..nums) => {
    let (section, ..subsections) = nums.pos()
    let (section_scheme, ..subschemes) = schemes.pos()

    if subsections.len() == 0 {
      numbering(section_scheme, section)
    } else {
      clean_numbering(..subschemes)(..subsections)
    }
  }
}
#show outline.entry.where(
  level: 1
): it => {
  v(12pt, weak: true)
  strong(it)
}

#outline(indent: auto)

#set page(paper: "a4", margin: (x: 41.5pt, top: 80.51pt, bottom: 89.51pt))
#set page(numbering: "1", number-align: right + bottom)
#counter(page).update(1)
#set text(font: "STIX Two Text", size: 10pt)
#include "./paper.typ"
#pagebreak()

#set page(number-align: right + bottom, numbering: (..nums) => {
  let sequence = nums.pos()
  // discard first entry (chapter number)
  if sequence.len() > 0 {
    "A-" + str(sequence.first())
  }
}, header: none, footer: none)
#set heading(numbering: (..nums) => {
  let sequence = nums.pos()
  // discard first entry (chapter number)
  let _ = sequence.remove(0)
  if sequence.len() > 0 {
    "A." + numbering("1.", ..sequence)
  }
})
#counter(page).update(1)
#counter(heading).update(1)
#align(center, text(weight: "bold", size: 24pt, "Appendix A"))
#hide(heading(numbering: none, "Appendix A"))
#align(center, text(weight: "bold", size: 24pt, "Literature Review"))

Appendix content follows...
#pagebreak()
#include "./litreview.typ"
#pagebreak()

#set page(number-align: right + bottom, numbering: (..nums) => {
  let sequence = nums.pos()
  // discard first entry (chapter number)
  if sequence.len() > 0 {
    "B-" + str(sequence.first())
  }
}, header: none, footer: none)
#set heading(numbering: (..nums) => {
  let sequence = nums.pos()
  // discard first entry (chapter number)
  let _ = sequence.remove(0)
  if sequence.len() > 0 {
    "B." + numbering("1.", ..sequence)
  }
})
#counter(page).update(1)
#counter(heading).update(1)
#align(center, text(weight: "bold", size: 24pt, "Appendix B"))
#hide(heading(numbering: none, "Appendix B"))
#align(center, text(weight: "bold", size: 24pt, "Implementation and Design"))

Appendix content follows...
#pagebreak()
#include "./implementation.typ"
#pagebreak()

#set page(number-align: right + bottom, numbering: (..nums) => {
  let sequence = nums.pos()
  // discard first entry (chapter number)
  if sequence.len() > 0 {
    "C-" + str(sequence.first())
  }
}, header: none, footer: none)
#set heading(numbering: (..nums) => {
  let sequence = nums.pos()
  // discard first entry (chapter number)
  let _ = sequence.remove(0)
  if sequence.len() > 0 {
    "C." + numbering("1.", ..sequence)
  }
})
#counter(page).update(1)
#counter(heading).update(1)
#align(center, text(weight: "bold", size: 24pt, "Appendix C"))
#hide(heading(numbering: none, "Appendix C"))
#align(center, text(weight: "bold", size: 24pt, "Testing and Results"))

Appendix content follows...
#pagebreak()
#include "./testing.typ"
#pagebreak()

#set page(number-align: right + bottom, numbering: (..nums) => {
  let sequence = nums.pos()
  // discard first entry (chapter number)
  if sequence.len() > 0 {
    "D-" + str(sequence.first())
  }
}, header: none, footer: none)
#set heading(numbering: (..nums) => {
  let sequence = nums.pos()
  // discard first entry (chapter number)
  let _ = sequence.remove(0)
  if sequence.len() > 0 {
    "D." + numbering("1.", ..sequence)
  }
})
#counter(page).update(1)
#counter(heading).update(1)
#align(center, text(weight: "bold", size: 24pt, "Appendix D"))
#hide(heading(numbering: none, "Appendix D"))
#align(center, text(weight: "bold", size: 24pt, "Observations and Conclusions"))

Appendix content follows...
#pagebreak()
#include "./conclusions.typ"
