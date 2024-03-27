#align(center, text(weight: "bold", size: 24pt, "Evaluting the Effects of Automatically Added Early Exits on the Performance of Convolutional Neural Networks"))

#align(center, text(weight: "bold", size: 18pt, "Project Portfolio"))
#pagebreak()
#heading(outlined: false, [Acknowledgements])

#pagebreak()
#show outline.entry.where(
  level: 1
): it => {
  v(12pt, weak: true)
  strong(it)
}

#outline(indent: auto)

#set page(paper: "a4", margin: (x: 41.5pt, top: 80.51pt, bottom: 89.51pt))
#set page(footer: align(right, counter(page).display("1")))
#counter(page).update(1)
#set heading(numbering: "1.1)")
#set text(font: "STIX Two Text", size: 10pt)
#include "./paper.typ"
#pagebreak()

#set page(footer:"")
#counter(page).update(0)
#align(center, text(weight: "bold", size: 24pt, "Appendix A"))

#align(center, text(weight: "bold", size: 24pt, "Literature Review"))

Appendix content follows...
#pagebreak()
#set page(footer: align(right, "A-" + counter(page).display("1")))
#counter(heading).update(0)
#include "./litreview.typ"

#set page(footer:"")
#counter(page).update(0)
#align(center, text(weight: "bold", size: 24pt, "Appendix B"))

#align(center, text(weight: "bold", size: 24pt, "Implementation and Design"))

Appendix content follows...
#pagebreak()
#set page(footer: align(right, "B-" + counter(page).display("1")))
#counter(heading).update(0)
#include "./implementation.typ"

#pagebreak()
#set page(footer:"")
#counter(page).update(0)
#align(center, text(weight: "bold", size: 24pt, "Appendix C"))

#align(center, text(weight: "bold", size: 24pt, "Testing and Results"))

Appendix content follows...
#set page(footer: align(right, "C-" + counter(page).display("1")))
#counter(heading).update(0)
#include "./testing.typ"

#pagebreak()
#set page(footer:"")
#counter(page).update(0)
#align(center, text(weight: "bold", size: 24pt, "Appendix D"))

#align(center, text(weight: "bold", size: 24pt, "Observations and Conclusions"))

Appendix content follows...
#pagebreak()
#set page(footer: align(right, "D-" + counter(page).display("1")))
#counter(heading).update(0)
#include "./conclusions.typ"
