#import "@preview/touying:0.7.4": *
#import themes.simple: *

#let presentation-info = (
  title: [Projet IMAGE — Hiéroglyphes],
  subtitle: [Compte rendu 3],
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

  *Augmentation de données*

  + Implémentation d'une augmentation de données déterministe basée sur la librairie #link("https://github.com/albumentations-team/AlbumentationsX")[Albumentations]@albumentationsx
  + Test de l'effet de l'augmentation sur Glyphnet @barucci2021glyphnet

  #v(20pt)
  *Travail sur le jeu de données*

  + Récupération des données fournies pour le projet
  + Début du développement d’un outil de prétraitement des imagettes basé sur la méthode proposée dans l’article@fuentesferrer2025cvv.
]


== Augmentations de données du papier

#[
  + Vise à équilibrer les différentes classes
    - Classes de plus de 2× la moyenne : sous-échantillonnées
    - Classes rares : complétées par des images transformées
  + Augmenter = Translation d'amplitude max 10 px, Rotation entre -10° et 10°, zoom entre 95% et 105%
  + Symétrie horizontale appliquée à toutes les images d'entraînement
]

== Avec et sans augmentation : la perte

#[
  #set text(size: 18pt)
  #figure(
    grid(
      columns: (1fr, 1fr),
      column-gutter: 24pt,
      stack(
        spacing: 8pt,
        [*Sans augmentation*],
        image("../common/images/glyphnet_trained_30_epochs_loss.png", width: 100%, height: 260pt, fit: "contain"),
      ),
      stack(
        spacing: 8pt,
        [*Avec augmentation*],
        image(
          "../common/images/glyphnet_trained_128_epochs_loss-augmented.png",
          width: 100%,
          height: 260pt,
          fit: "contain",
        ),
      ),
    ),
    caption: [Comparaison de la perte sur 30 époques.],
  ) <fig-perte-entrainement>
]

== Avec et sans augmentation : l'exactitude

#[
  #set text(size: 18pt)
  #figure(
    grid(
      columns: (1fr, 1fr),
      column-gutter: 24pt,
      stack(
        spacing: 8pt,
        [*Sans augmentation*],
        image("../common/images/glyphnet_trained_30_epochs_accuracy.png", width: 100%, height: 260pt, fit: "contain"),
      ),
      stack(
        spacing: 8pt,
        [*Avec augmentation*],
        image(
          "../common/images/glyphnet_trained_128_epochs_accuracy-augmented.png",
          width: 100%,
          height: 260pt,
          fit: "contain",
        ),
      ),
    ),
    caption: [Comparaison de l'exactitude sur 30 époques.],
  ) <fig-exactitude-entrainement>
]

== Pourquoi si peu d'effet ?

#[
  #set text(size: 18pt)
  #set par(leading: 0.55em, spacing: 1em)

  L'augmentation n'a pas beaucoup d'effet sur la perte et l'exactitude, pour trois raisons :

  + *Données homogènes* : toutes les images (pyramide d'Unas) ont le même support et les mêmes conditions de prise de vue, et la validation suit la même distribution. Décalages, rotations et zooms apportent donc peu de variabilité utile.
  + *Peu de surapprentissage au départ* : Glyphnet n'a qu'environ 500 000 paramètres et il est déjà régularisé (BatchNorm, dropout).
  + *Métrique globale* : l'exactitude est dominée par les classes fréquentes, alors que le rééquilibrage aide surtout les classes rares. Son effet se verrait plutôt sur le rappel ou le F1 par classe.
]

== Problèmes avec les données fournies

+ Tailles d’images très variables.

+ Taille et position des signes variables dans l’image.

+ Plusieurs signes peuvent apparaître dans une même image.

+ Risque que le modèle apprenne le contexte ou le fond plutôt que le signe.

== Début d'un outil pour préparer le jeu de données
#[
  #set text(size: 18pt)
  #set par(leading: 0.5em)
  #set enum(spacing: 24pt)
  #v(10pt)
  *Objectif*

  Développer une interface graphique basée sur Qt, permettant de:
  + effectuer un découpage automatique préliminaire à l’aide d’une méthode de segmentation automatique.
  + Les résultats seront ensuite vérifiés et ajustés manuellement, avant de procéder à l’annotation manuelle des données.

  #v(0.5cm)
  *Auto segementation*
  #text(size: 16pt)[(Voir un exemple dans @exemple-auto-segmentation)]
  + Binarisation par la méthode d’Otsu puis nettoyage de l’image binaire.
  + Recherche des composantes connexes.
  + Échantillonnage aléatoire d’un nombre fixe de points dans chaque composante.
  + Utiliser ces points pour effectuer une segmentation automatique à l’aide du modèle SAM@sam
  + Obtenir la boîte englobante de l’objet cible, puis l’ajuster à des dimensions fixes.



]

== Début d'un outil pour préparer le jeu de données
#[

  #v(10pt)

  #show figure.caption: set text(size: 15pt, fill: rgb("#555555"))

  #cols(align: horizon)[
    #set text(size: 24pt)
    #set par(leading: 1em)
    #set enum(spacing: 24pt)
    *État d’avancement actuel*

    - Chargement des images terminé.
    - Ajout de points de guidage positifs et négatifs sur les images implémenté.
    - Ajout de boîtes de sélection sur les images implémenté.
    - Déplacement et suppression des points de guidage et des boîtes de sélection implémentés.
  ][
    #align(center)[
      #figure(
        image("../common/images/exemple_auto_segmentation.png", width: 100%),
        caption: [Exemple d'auto segmentation.],
      ) <exemple-auto-segmentation>
    ]
  ]
]

== Prochaines étapes

#[
  #set text(size: 24pt)
  #set enum(spacing: 30pt)
  #set par(leading: 0.5em)
  #v(12pt)

  + Finir l'outil de prétraitement des données

  + Effectuer le prétraitement des images fournies par le Dr Maxime Brun.

  + Étudier les méthodes d’apprentissage adaptées à un faible nombre d’exemples.

  + Considérer les solutions de détection de formes généralistes

  + Poursuivre le développement de l'outil de préparation du jeu de données, puis entraîner le modèle sur ces données.
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
