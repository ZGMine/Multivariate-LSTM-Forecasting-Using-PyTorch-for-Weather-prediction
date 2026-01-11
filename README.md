# Multivariate-LSTM-Forecasting-Using-PyTorch-for-Weather-prediction

Ce projet propose une approche de Deep Learning pour prédire la température future à partir de séries temporelles multivariées de variables atmosphériques. L'objectif est de générer des prévisions précises pour près de 10 000 locations distinctes.

## Contexte du Projet

La prévision météorologique repose traditionnellement sur des modèles physiques complexes. Ce projet explore une alternative basée sur les données (Data-Driven) en utilisant des réseaux de neurones récurrents.

Le jeu de données contient des relevés quotidiens (précipitations, humidité, vent, etc.) pour des milliers de points géographiques. Le défi technique consistait à gérer un volume de données important (> 14 millions de lignes) tout en concevant un modèle capable de capturer les dynamiques temporelles et saisonnières spécifiques à chaque lieu.

## Architecture et Méthode

Nous avons utilisé un modèle **LSTM (Long Short-Term Memory)** implémenté avec **PyTorch**, reconnu pour sa capacité à retenir des informations sur de longues séquences temporelles, crucial pour modéliser l'inertie thermique.

### Pré-traitement des données
*   **Encodage cyclique des dates :** Transformation des dates en sinus/cosinus pour que le modèle comprenne la cyclicité des saisons (le 31 décembre est proche du 1er janvier).
*   **Normalisation :** Utilisation d'un `StandardScaler` pour ramener toutes les variables à une échelle commune, évitant ainsi que le modèle ne "sature" ou ne prédise une moyenne constante et surtout l'explosion du gradient.
*   **Fenêtrage (Sliding Window) :** Séquences de 7 jours passés pour prédire le jour suivant (J+1).

### Configuration du Modèle
*   **Entrée :** Vecteur de dimension 29 (variables météo + dates encodées).
*   **Réseau :** 1 couche LSTM (Hidden Size = 64) suivie d'une couche Linéaire.
*   **Sortie :** 1 valeur prédite (Température).
*   **Prédiction récursive :** Pour prédire à J+30, le modèle réinjecte ses propres prédictions de J+1, J+2, etc., en entrée.

### Dataset
lien du dataset : https://meteo.data.gouv.fr/datasets/6569b27598256cc583c917a7 original.
Lien du dataset utilisé pour le training et le test : https://kaggle.com/datasets/60c178d17ed4e8f86627fe8762ccfa080bee24a1daf9180e622be3911802e48a .
Le dataset contient des attributs concernant : la position, les précipitations, le vent, l'humidité, le rayonnement, la pluie, la neige, les températures min et max, ...

## Résultats

*   **Convergence :** Le modèle apprend efficacement avec une perte (MSE) très faible (~0.02 sur les données scalées).
*   **Prédiction à J+30 :** Le modèle ne parvient pas vraiment à générer des courbes de température cohérentes pour les 30 jours futurs, le manque d'attributs nuit grandements aux prédictions.

*   **Améliorations possible :** Nous avons très bien vu par nos méthodes d'évaluations que le modèle est très robuste pour des prédictions à J+1 mais ne l'est pas du tout pour des prédictions sur de plus longues période. Un principal axe d'amélioration de l'implémentation serait de non pas générer la température uniquement du jour suivant mais tous ses (29) attributs, de ce fait les prévisions postérieures auraient bien plus d'informations et seraient surement bien plus précises. Ici les variables distinctes (multivariate) correspondent aux 10 000 locations où nous faisons une prédiction, l'amélioration constituerait donc un modèle, qui pour chaque prédiction : prédise 10 000 (locations) * 29 (attributs), ce modèle serait alors très large est nécessiterait une énorme puissance de calcul. D'autant plus que dans notre implémentation, seules des données datant d'il y a 4 ont été utilisés, implémenter cette amélioration avec des données remontant à plus de 30 ans (car les données sont proposées sur le site data.gouv) nécessiterait une quantité phénoménale de RAM. Les résultats seraient sans doute bien meilleurs.

## Matériel utilisé

Les expérimentations ont été réalisées sur la configuration suivante :
*   **RAM :** 32 Go
*   **GPU :** Nvidia P100 (16 Go)
*   **Environnement :** Kaggle Notebook / Jupyter

## Auteur

**ZG**
*Date : Janvier 2026*
