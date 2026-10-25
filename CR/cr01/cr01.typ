#import "@preview/touying:0.7.4": *
#import themes.simple: *

#let presentation-info = (
  title: [Projet IMAGE — Hiéroglyphes],
  subtitle: [Compte rendu 1],
  author: [Léo HAFDANE · Mathys CHAMPENOIS · Yaqi ZHANG],
  institution: [Université de Montpellier],
)

#show: simple-theme.with(
  aspect-ratio: "16-9",
  header: presentation-info.title,
  config-info(..presentation-info),
)

#set text(lang: "fr")


#title-slide[
  #text(size: 32pt, weight: "bold")[#presentation-info.title]

  #v(0.5em)
  #text(size: 23pt)[#presentation-info.subtitle]

  #v(0.8em)
  #text(size: 18pt)[#presentation-info.author]

  #text(size: 16pt)[#presentation-info.institution]

]


== Présentation du projet

*Identification d’un hiéroglyphe égyptien _ka_*
#[
  #show figure.caption: set text(size: 15pt, fill: rgb("#555555"))

  #cols[
    - Classifier des imagettes du signe _ka_ déjà extraites.

    #v(0.5em)
    
    - Localiser les occurrences du signe _ka_ dans une image plus large, en extraire des imagettes, puis les classer selon les mêmes critères.
  ][
    #align(center)[
      #figure(
        image("../common/images/ka_exemple.png", height: 40%),
        caption: "Variantes graphiques du signe ka.",
      ) <fig-ka>
    ]
  ]
]


== Données et difficultés potentielles

#[
  #set text(size: 22pt)
  #set par(leading: 0.5em)
  #set list(tight: false, spacing: 1.0em)

  #v(1.0em)

  - Jeu de données non encore examiné : taille, catégories et répartition à confirmer.

  #v(1.5em)

  - Variations possibles de forme, de texture et de qualité des images.

  #v(1.5em)

  - Nombre limité d’exemples ou déséquilibre des classes : points à vérifier.

  #v(1.5em)

  - Risque de surapprentissage si les données disponibles sont peu nombreuses.
]


== Méthodes étudiées

#[
  #set text(size: 21pt)
  #set par(leading: 0.4em)
  #set list(tight: false, spacing: 0.8em)
  #set figure(gap: 0.2em)
  #show figure.caption: set text(size: 12pt, fill: rgb("#555555"))

  #cols()[
    #text(size: 23pt, weight: "bold")[Classification directe par CNN]

    - Glyphnet @barucci2021glyphnet.
    - ConvNeXt avec CVV @fuentesferrer2025cvv.

    #v(0.3em)
    #figure(
      image(
        "../common/images/end_to_end_CNN.png",
        width: 100%,
        height: 4cm,
        fit: "contain",
      ),
      caption: [Classification par CNN : MobileNet @fuentesjimenez2025ocrptct.],
    ) <fig-cnn>
  ][
    #text(size: 23pt, weight: "bold")[Apprentissage métrique @fuentesjimenez2025ocrptct]

   - Apprendre un espace de représentation où les signes similaires sont proches.
  - Classifier par similarité avec les prototypes des classes.

    #figure(
      image(
        "../common/images/apprentissage_métrique.png",
        width: 100%,
        height: 5cm,
        fit: "contain",
      ),
      caption: [Classification par similarité @fuentesjimenez2025ocrptct.],
    ) <fig-metric>
  ]
]


== Travail réalisé et prochaines étapes

#[
  #set text(size: 22pt)
  #set list(tight: false, spacing: 0.5em)

  *Travail réalisé*

  - Mise en place du dépôt Git.
  - Lecture et comparaison de trois articles.

  #v(0.4em)

  *Prochaines étapes*

  + Obtenir le jeu de données.

  + Réaliser une analyse descriptive : nombre total d’images, format, effectifs et proportions par classe, etc.

  + Effectuer de premiers essais de classification par CNN, en reproduisant les approches étudiées : Glyphnet @barucci2021glyphnet ou ConvNeXt avec CVV @fuentesferrer2025cvv.
]

== Références

#set text(size: 12pt)

#bibliography(
  "../common/references.bib",
  title: none,
  style: "ieee",
)
