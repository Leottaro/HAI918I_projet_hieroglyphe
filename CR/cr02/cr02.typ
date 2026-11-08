#import "@preview/touying:0.7.4": *
#import themes.simple: *

#let presentation-info = (
  title: [Projet IMAGE — Hiéroglyphes],
  subtitle: [Compte rendu 2],
  author: [Léo HAFDANE · Mathys CHAMPENOIS · Yaqi ZHANG],
  institution: [Université de Montpellier],
)

// Une section par diapositive, avec détection des débordements.
#show: simple-theme.with(
  aspect-ratio: "16-9",
  header: presentation-info.title,
  header-right: presentation-info.subtitle,
  subslide-preamble: block(
    below: 0.65em,
    text(size: 28pt, weight: "bold", utils.display-current-heading(level: 2)),
  ),
  config-info(..presentation-info),
  config-page(margin: (left: 36pt, right: 36pt, top: 42pt, bottom: 32pt)),
  config-common(breakable: false, detect-overflow: true),
)

#set text(size: 20pt, lang: "fr")
#set par(leading: 0.45em, spacing: 0.65em)
#set list(tight: false, spacing: 0.45em)
#set enum(tight: false, spacing: 0.65em)
#set figure(gap: 0.3em)
#show figure.caption: set text(size: 12pt, fill: rgb("#555555"))

#title-slide[
  #text(size: 34pt, weight: "bold")[#presentation-info.title]

  #v(0.5em)
  #text(size: 25pt)[#presentation-info.subtitle]

  #v(1em)
  #text(size: 19pt)[#presentation-info.author]

  #text(size: 17pt)[#presentation-info.institution]
]

== Travail réalisé cette semaine

#[
#set text(size: 23pt)
#set par(leading: 0.5em)
#set enum(spacing: 24pt)
#v(10pt)

*Reproduction de l’architecture de Glyphnet* @barucci2021glyphnet

+ Récupération et analyse de la partie publique du jeu de données utilisé dans l’article.
+ Implémentation de l’architecture avec PyTorch.
+ Entraînement du modèle sur les données disponibles.
+ Évaluation des résultats de classification.

#v(20pt)
*Réception du jeu de données initial de notre projet*, fourni par le Dr Maxime Brun.
]

== Données utilisées pour entraîner Glyphnet

#[
#set text(size: 22pt)
#set list(spacing: 10pt)
#v(8pt)
#cols(columns: (1.4fr, 1fr), gutter: 24pt)[
  *Caractéristiques générales*

  - 4 210 images en niveaux de gris.
  - 172 étiquettes : 171 classes de signes et une étiquette `UNKNOWN`.
  - Résolution : 75 × 50 pixels (hauteur × largeur).

  #v(12pt)
  *Répartition des classes*

  - Fort déséquilibre : de 1 à 448 images par classe.
  - Sélection des *40 classes les plus représentées* pour les premiers essais de classification.
][
  #align(center)[
    #image("../common/images/exemple_image_glyphnet_01.png", width: 100%, height: 125pt, fit: "contain")
    #v(20pt)
    #figure(
      image("../common/images/exemple_image_glyphnet_02.png", width: 100%, height: 125pt, fit: "contain"),
      caption: [Exemples d’images du jeu de données public.],
    ) <fig-exemples-donnees>
  ]
]
]

== Architecture de Glyphnet

#[
  #set text(size: 17pt)
  #set par(leading: 0.55em, spacing: 1em)
  #set list(spacing: 10pt)

  #cols(columns: (2.3fr, 1fr), gutter: 22pt)[
    *Architecture reproduite avec PyTorch*

    - *Entrée :* 100 × 100 pixels dans l’article, qui réunit deux jeux de données ; 75 × 50 pixels dans notre implémentation.
    - *Bloc 1 :* convolutions classiques, normalisation par lots (BN), max-pooling et activation ReLU.
    - *Blocs 2 à 5 :* convolutions séparables en profondeur, pour réduire le nombre de paramètres.
    - *Bloc 6 :* convolution séparable, BN et ReLU, moyenne globale, dropout et couche entièrement connectée ; softmax pour obtenir les probabilités des classes.

    #v(10pt)
    *Convolution séparable en profondeur*

    - *Étape spatiale :* une convolution indépendante sur chaque canal d’entrée.
    - *Étape ponctuelle :* une convolution 1 × 1 combine les canaux et fixe le nombre de canaux de sortie.
  ][
    #align(center)[
      #figure(
        image("../common/images/structure_glyphnet.png", height: 302pt, fit: "contain"),
        caption: [Architecture de l’article @barucci2021glyphnet.],
      ) <fig-architecture-glyphnet>
    ]
  ]
]

== Entraînement du modèle

#[
  #set text(size: 18pt)
  #set par(leading: 0.55em, spacing: 1em)
  #set list(spacing: 12pt)
  #cols(columns: (1fr, 2fr), gutter: 30pt)[
    *Partition stratifiée*

    - Entraînement : *70 %*.
    - Validation : *15 %*.
    - Test : *15 %*.
  ][
    *Suivi de l’entraînement*

    - Perte et exactitude sur les ensembles d’entraînement et de validation.
    - Fonction de perte : entropie croisée.
  ]
  #v(12pt)
  #cols(columns: (1fr, 1fr), gutter: 24pt)[
    #figure(
      image("../common/images/glyphnet_trained_30_epochs_loss.png", width: 100%, height: 205pt, fit: "contain"),
      caption: [Évolution de la perte sur 30 époques.],
    ) <fig-perte-entrainement>
  ][
    #figure(
      image("../common/images/glyphnet_trained_30_epochs_accuracy.png", width: 100%, height: 205pt, fit: "contain"),
      caption: [Évolution de l’exactitude sur 30 époques.],
    ) <fig-exactitude-entrainement>
  ]
]

== Évaluation du modèle entraîné

#[
  #set text(size: 18pt)
  #set par(leading: 0.55em, spacing: 0.65em)
  #set list(spacing: 16pt)
  #cols(columns: (1.15fr, 1fr), gutter: 24pt)[
    *Sélection du modèle*

    - Analyse d’un entraînement de 30 époques à partir des courbes de perte et d’exactitude.
    - Signes de léger surapprentissage après environ 15 époques, avec des fluctuations en validation.
    - Modèle retenu à *15 époques*, avec une faible perte et une bonne exactitude en validation, puis évalué sur le test.

    #v(10pt)
    *Résultats sur l’ensemble de test*

    - Exactitude globale : *96,32 %*.
    - F1-score macro : *0,9063*.
    - La plupart des signes sont correctement classés ; certaines classes restent confondues.
  ][
    #align(center)[
      #figure(
        image("../common/images/glyphnet_epochs_15_confusion_matrix.png", width: 100%, height: 290pt, fit: "contain"),
        caption: [Matrice de confusion du modèle à 15 époques.],
      ) <fig-matrice-confusion>
    ]
  ]
]

== Jeu de données fourni par le Dr Maxime Brun

#[
  #set text(size: 22pt)
  #set list(spacing: 20pt)
  #set par(leading: 0.5em)
  #v(12pt)
  #cols(columns: (1fr, 1.35fr), gutter: 28pt)[
    *Premières observations*

    - 150 images au total.
    - Dimensions variables.
    - Certains signes sont difficiles à distinguer.
    - De nombreuses images d’un même type de signe présentent un fond similaire.
  ][
    *Difficultés et points à discuter*

    - Apprendre avec peu d’exemples.
    - Définir une résolution d’entrée adaptée.
    - Localiser et reconnaître les signes peu lisibles.
    - Éviter que le modèle s’appuie surtout sur le fond, au détriment de sa capacité à généraliser.
    - Traiter les images contenant plusieurs signes.
  ]
]

== Prochaines étapes

#[
  #set text(size: 24pt)
  #set enum(spacing: 30pt)
  #set par(leading: 0.5em)
  #v(12pt)

  + Reproduire les opérations d’augmentation et d’équilibrage des données décrites dans l’article sur Glyphnet @barucci2021glyphnet.

  + Appliquer le vote par validation croisée (*Cross-Validation Voting*, CVV) à Glyphnet, puis comparer les résultats @fuentesferrer2025cvv.

  + Effectuer un premier prétraitement des images fournies par le Dr Maxime Brun.

  + Étudier les méthodes d’apprentissage adaptées à un faible nombre d’exemples.
]

== Références

#[
  #set text(size: 16pt)
  #set par(leading: 0.65em, spacing: 1.5em)
  #v(40pt)
  #bibliography(
    "../common/references.bib",
    title: none,
    style: "ieee",
  )
]
