# Apprentissage de représentations spatiales de scènes par apprentissage contrastif sur graphes

## Présentation du projet

Ce projet a été réalisé dans le cadre de l’UE **Modélisation de Systèmes Intelligents** du **Master 2 Informatique - Parcours Vision et Machine Intelligente (VMI)** à **l’Université Paris Cité**.

L’objectif principal est de modéliser et d’apprendre la **spatialité des scènes visuelles**, c’est-à-dire l’organisation relative des objets dans une image, **indépendamment de leur apparence visuelle**.
Pour cela, les scènes sont représentées sous la forme de **graphes de scène**, permettant une description explicite et structurée des relations spatiales entre objets.

Chaque scène est modélisée comme un graphe :

* les **noeuds** représentent les objets,
* les **arêtes** représentent les relations spatiales entre objets.

Nous adoptons une approche **auto-supervisée** basée sur l’**apprentissage contrastif sur graphes (Graph Contrastive Learning, GraphCL)**.

---

## Données utilisées

Le projet repose sur le dataset **Panoptic Scene Graph (PSG)**, qui fournit :

* des images naturelles issues de COCO,
* des masques de segmentation panoptiques,
* des graphes de scène associés à chaque image.

### Points forts du dataset

* représentation explicite de la structure des scènes,
* cohérence géométrique entre objets et relations,
* cadre réaliste pour l’étude de la spatialité des scènes naturelles.

### Limites du dataset

* fort déséquilibre des annotations de relations,
* nombreuses relations de type **action** ou **sémantique** (*holding*, *wearing*, etc.),
* désalignement partiel entre la nature **spatiale** des descripteurs et les labels disponibles.

Ces limitations influencent fortement les choix méthodologiques et les stratégies d’évaluation.

---

## Représentation des scènes

Chaque image est transformée en un **graphe de scène** combinant des descripteurs.

### Noeuds

* Les noeuds correspondent aux objets de la scène.
* Chaque objet est décrit à l’aide des **moments de Hu**, permettant de capturer des informations de forme invariantes à l’échelle et à la rotation.

### Arêtes

* Les arêtes correspondent aux relations spatiales entre paires d’objets.
* Elles sont décrites à l’aide de **bandeaux de force** (Relative Position Descriptors, RPD), qui encodent des **relations directionnelles et spatiales continues**.

Cette représentation permet de séparer explicitement :

* les propriétés intrinsèques des objets,
* les informations relationnelles et géométriques.

---

## Méthodologie

### Apprentissage contrastif sur graphes

La stratégie d’apprentissage est inspirée de **SimCLR**, adaptée aux graphes de scène :

* deux vues augmentées d’un même graphe sont générées,
* les augmentations sont appliquées principalement aux **attributs d’arêtes** (bruit, décalage angulaire, lissage), afin de rester cohérentes avec la nature spatiale des descripteurs,
* un **Graph Neural Network (GNN)** encode les graphes et produit des représentations latentes,
* une fonction de perte contrastive rapproche les embeddings issus d’un même graphe et éloigne ceux provenant de graphes différents.

Les représentations apprises peuvent être exploitées :

* au **niveau des relations** (embeddings d’arêtes),
* au **niveau global du graphe** (pooling).

<p align="center">
  <img src="figures/architecture_contrastive_learning.png" width="650">
</p>

<p align="center">
  <em>
  Architecture de l’apprentissage contrastif.
  </em>
</p>

