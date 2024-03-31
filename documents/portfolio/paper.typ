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
})

// clear the previously stored references every time a level 1 heading
// is created.
#show heading.where(level: 1): it => {
  section-refs.update(())
  it
}
// Workaround for the lack of an `std` scope.
#let std-bibliography = bibliography

// This function gets your whole document as its `body` and formats
// it as an article in the style of the IEEE.
#let ieee(
  // The paper's title.
  title: [Paper Title],

  // An array of authors. For each author you can specify a name,
  // department, organization, location, and email. Everything but
  // but the name is optional.
  authors: (),

  // The paper's abstract. Can be omitted if you don't have one.
  abstract: none,

  // A list of index terms to display after the abstract.
  index-terms: (),

  // The article's paper size. Also affects the margins.
  paper-size: "a4",

  // The result of a call to the `bibliography` function or `none`.
  bibliography: none,

  // The paper's content.
  body
) = {
  // Set the body font.
  set text(font: "STIX Two Text", size: 10pt)

  // Configure the page.
  set page(
    paper: paper-size,
    // The margins depend on the paper size.
    margin: if paper-size == "a4" {
      (x: 41.5pt, top: 80.51pt, bottom: 89.51pt)
    } else {
      (
        x: (50pt / 216mm) * 100%,
        top: (55pt / 279mm) * 100%,
        bottom: (64pt / 279mm) * 100%,
      )
    }
  )

  // Configure equation numbering and spacing.
  set math.equation(numbering: "(1)")
  show math.equation: set block(spacing: 0.65em)

  // Configure appearance of equation references
  show ref: it => {
    if it.element != none and it.element.func() == math.equation {
      // Override equation references.
      link(it.element.location(), numbering(
        it.element.numbering,
        ..counter(math.equation).at(it.element.location())
      ))
    } else {
      // Other references as usual.
      it
    }
  }

  // Configure lists.
  set enum(indent: 10pt, body-indent: 9pt)
  set list(indent: 10pt, body-indent: 9pt)

  // Configure headings.
  set heading(numbering: "I.A.1.")
  show heading: it => locate(loc => {
    // Find out the final number of the heading counter.
    let levels = counter(heading).at(loc)
    let deepest = if levels != () {
      levels.last()
    } else {
      1
    }

    set text(10pt, weight: 400)
    if it.level == 1 [
      // First-level headings are centered smallcaps.
      // We don't want to number of the acknowledgment section.
      #let is-ack = it.body in ([Acknowledgment], [Acknowledgement])
      #set align(center)
      #set text(if is-ack { 10pt } else { 12pt })
      #show: smallcaps
      #v(20pt, weak: true)
      #if it.numbering != none and not is-ack {
        numbering("I.", deepest)
        h(7pt, weak: true)
      }
      #it.body
      #v(13.75pt, weak: true)
    ] else if it.level == 2 [
      // Second-level headings are run-ins.
      #set par(first-line-indent: 0pt)
      #set text(style: "italic")
      #v(10pt, weak: true)
      #if it.numbering != none {
        numbering("A.", deepest)
        h(7pt, weak: true)
      }
      #it.body
      #v(10pt, weak: true)
    ] else [
      // Third level headings are run-ins too, but different.
      #if it.level == 3 {
        numbering("1)", deepest)
        [ ]
      }
      _#(it.body):_
    ]
  })

  // Display the paper's title.
  v(3pt, weak: true)
  align(center, text(18pt, title))
  v(8.35mm, weak: true)

  // Display the authors list.
  for i in range(calc.ceil(authors.len() / 3)) {
    let end = calc.min((i + 1) * 3, authors.len())
    let is-last = authors.len() == end
    let slice = authors.slice(i * 3, end)
    grid(
      columns: slice.len() * (1fr,),
      gutter: 12pt,
      ..slice.map(author => align(center, {
        text(12pt, author.name)
        if "department" in author [
          \ #emph(author.department)
        ]
        if "organization" in author [
          \ #emph(author.organization)
        ]
        if "location" in author [
          \ #author.location
        ]
        if "email" in author [
          \ #link("mailto:" + author.email)
        ]
      }))
    )

    if not is-last {
      v(16pt, weak: true)
    }
  }
  v(40pt, weak: true)

  // Start two column mode and configure paragraph properties.
  show: columns.with(2, gutter: 12pt)
  set par(justify: true, first-line-indent: 1em)
  show par: set block(spacing: 0.65em)

  // Display abstract and index terms.
  if abstract != none [
    #set text(weight: 700)
    #h(1em) _Abstract_---#abstract

    #if index-terms != () [
      #h(1em)_Index terms_---#index-terms.join(", ")
    ]
    #v(2pt)
  ]

  // Display the paper's contents.
  body

  // Display bibliography.
  if bibliography != none {
    show std-bibliography: set text(8pt)
    set std-bibliography(title: text(12pt)[References], style: "ieee")
    bibliography
  }
}

#show: ieee.with(
  title: "MEng in Electronic and Computer Engineering",
  abstract: [#lorem(150)],
  authors: (
    (
      name: "Stephen Condon",
      organization: [Dublin City University],
      location: [Dublin, Ireland],
      email: "stephen.condon5@mail.dcu.ie"
    ),
  ),
  paper-size: "a4",
  index-terms: (),
  bibliography: none,
  // bibliography: bibliography("refs.bib"),
)

= Introduction

= Prior Work
Early exiting in neural networks is an area that hasn't received much acemedic interest, with few papers being released on the topic in recent years. The key idea behind early exiting is that a network can already have identified the correct output without running through the entire model, and this can be identified by how converged the model is to a single output.

BranchyNet @BranchyNet is the most cited method for adding exits to a model. They propose the structure for any early exit. Specifically, the entropy of an exits' output is used to determine how well converged that output is. Every exit also needs a threshold, and any output with an entropy below this is considered sufficiently converged and the model can exit. They also say generally, exits very early in the model should contain extra layers and later exits should have fewer layers to exit well, but exactly how many layers should be added isn't discussed. How the threshold should be set also isn't specified, but scanning over multiple values and observing the accuracy is mentioned. @optimalbranchplacement discusses an algorithm to search for the optimal placement of branches in BranchyNet models. It identifies the cost of each exit, and determines whether this cost is less than the cost of continuing in the model. This search only outputs where a branch should be placed, but doesn't actual add a branch to a model. It also notes that exits with just a fully-connected layer perform almost as well as exits with many layers. @earlyexitmasters shows how exits containing a pooling layer and a fully-connected layer can be used to exit a model early, 

Both of these papers describe manual methods of adding exits to a network, which adds too much work to the designing of models to allow for widespread adoption of early exiting in models. Early exits should be added automatically to already-defined models to allow both old models to be adapted easily and new models to be created with exits without the need for manual intervention. Exits could be inserted by compiler transformation passes and then trained by some machine learning framework. The threshold for each 

Some of the issues to consider with this approach are the time taken to decide on whether to exit (which may grow significant as the number of exits grows large and as an exits complexity increases), trying to analyse different exit architectures, specifically exits including or excluding pooling layers, and the amount of training time needed to search for the best placement for branches. However, the biggest problem for widespread use of early exiting is the need to change the architecture of a neural network to use exits. Currently, model architectures need to be manually changed to make use of exits which may need a significant amount of analysis after already training the main model. Ideally, whatever framework or compiler is used should automatically add exits in optimal locations for the model.

One of the most cited ways to add early exiting is the BranchyNet architecture, described in @BranchyNet. This architecture describes how exits should be structured when added to a model. In particular it describes how, for CNNs, an exit should consist of some number of convolution layers (the earlier the exit, the more convolutional layers) and a fully connected layer. Each branch can then be trained independently. The entropy of the softmax of the output of the early exit is used as the condition referred to as `e` in this report. That is, `e = entropy(softmax(y))`. This value is a measure of how confident the model is that it has accurately classified the input. If the model is unsure, many of the possible classes will have a high value in the output, resulting in a high entropy. If the model has narrowed in on only a few possible options for the output class, most of the classes in the output will have a tiny value resulting in a smaller entropy. Each exit also has an associated threshold entropy value. If `e` is less than the threshold `T`, the model presumes it is correct, and the softmax of the output is returned. That is, `if e < T then return softmax(y) else continue`. This architecture is the basis of most of the recent research into early exiting. However, @BranchyNet discusses neither the optimal location of branches or the optimal number of branches per model. Ideally, a method for adding early exits automatically to an architecture should be smart enough to find the optimal placement and number of branches to use for a given architecture. The threshold for the exit is also presented as a hyperparameter for each branch to be learned during training-time in the paper, but may be decided by other methods.

@optimalbranchplacement discusses an algorithm which, when given a set of different combinations of branches, will analyse which combination of branches reduces the average inference time the most. However, it is impractical to give a list of all possible combinations of exits to the algorithm, as an early exit could in theory be placed after every layer of a network, but that may be computationally expensive. Ideally, a method of automatically adding early exits shouldn't need to exhaustively check every possible combination, but should instead only look at combinations at least better than the current considered batch of early exits. @optimalbranchplacement also noted that, instead of having convolutional layers followed by a fully-connected layer, just having a fully-connected layer performs approximately equally to more complex exits.

@earlyexitmasters looks at different architectures that can be used as an exit, including single fully-connected layers, multiple fully-connected, and fully-connected layers combined with some pooling or batch normalisation layers. All exits shown in this paper have the confidence value of the model bounded between 0 and 1 by using a sigmoid function. Both the confidence value and the actual output from the network are outputted from the exit. A custom loss function is also presented that takes into account the cost of the model in terms of floating point operations (FLOPs), preferring lower cost exits, and also the accuracy and confidence of the model. Exits are also distributed using both self-similar methods and linear methods. This paper shows significant improves in performance without a significant decrease in accuracy. For example, an early exit ResNet152 trained on the SVHN database needs 2% of the relative cost of the original model, while losing 1% accuracy from 95.68% to 94.68%. Most models can keep the same accuracy with 20% of the relative cost. This paper shows how having an exit can have significant savings, but the exit has the confidence as a learnable trait, which differs from the solutions offered by this project.

Early exiting is also useful for moving towards edge computing. As neural networks uses more personal data, there is a greater need for privacy, and one of the best ways to do that is to have models running on edge devices, such as mobile phones and personal computers. Some architectures have been proposed to run a section of the model on an edge device, and if that section of the model is not able to exit confidently, then the rest of the inference is carried out in a central server. Early exiting from BranchyNet is used to assess how confident the edge device is about having an accurate output. One example of this is @TowardsEdgeComputing, which describes how to split the network to function across multiple devices.


= Technical Description

= Results

= Analysis

= Conclusions

#heading(numbering: none, "References")
#section-bib()
