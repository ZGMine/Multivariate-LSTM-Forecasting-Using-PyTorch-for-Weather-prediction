# Multivariate-LSTM-Forecasting-Using-PyTorch-for-Weather-prediction

Ce projet propose une approche de Deep Learning pour prédire la météo et la température future à partir de séries temporelles multivariées de variables atmosphériques. L'objectif est de générer des prévisions précises et cohérentes sur le moyen/long terme pour des milliers de localisations distinctes.

## Contexte du Projet

La prévision météorologique repose traditionnellement sur des modèles physiques complexes. Ce projet explore une alternative basée sur les données (Data-Driven) en utilisant des réseaux de neurones récurrents profonds.

Le jeu de données contient des relevés quotidiens (précipitations, humidité, vent, etc.) pour des milliers de points géographiques. Le défi technique consistait à gérer un volume de données important tout en concevant une architecture capable d'éviter la "dérive autorégressive" (accumulation d'erreurs) typique des prédictions sur de longues périodes (J+50).

## Architecture et Méthode

Nous avons utilisé un modèle **LSTM (Long Short-Term Memory)** implémenté avec **PyTorch**, reconnu pour sa capacité à retenir des informations sur de longues séquences temporelles. Pour maximiser les performances (HPC), le jeu de données est entièrement pré-chargé en mémoire vidéo (VRAM Preloading), éliminant ainsi les goulets d'étranglement CPU/GPU.

### Pré-traitement des données
*   **Encodage cyclique des dates :** Transformation des dates en sinus/cosinus pour que le modèle comprenne la cyclicité des saisons.
*   **Séparation des variables (Features vs Targets) :** Les variables statiques (coordonnées LAMBX, LAMBY) et déterministes (temps) sont utilisées en entrée mais exclues des cibles à prédire pour ne pas bruiter l'apprentissage.
*   **Double Normalisation :** Utilisation de deux `StandardScaler` distincts (un pour les entrées, un pour les sorties) pour garantir une dénormalisation mathématiquement stable lors de l'inférence.
*   **Fenêtrage (Sliding Window) :** Séquences longues de 60 jours passés pour capturer une macro-tendance avant de prédire le jour suivant.

### Configuration du Modèle
*   **Entrée :** Vecteur de dimension 30 (26 variables météo dynamiques + 2 coordonnées + 2 variables temporelles).
*   **Réseau :** Architecture LSTM profonde (16 couches, Hidden Size = 64) suivie d'une couche Linéaire.
*   **Sortie :** Vecteur de dimension 26 (Prédiction simultanée de *toutes* les variables météorologiques dynamiques, et non plus seulement de la température).
*   **Prédiction récursive intelligente :** Pour prédire jusqu'à J+50, le modèle réinjecte ses 26 prédictions dynamiques, auxquelles il concatène les coordonnées statiques de la ville et les nouvelles variables temporelles (sin/cos) recalculées pour le jour cible.

### Dataset
Lien du dataset original : https://meteo.data.gouv.fr/datasets/6569b27598256cc583c917a7
Lien du dataset utilisé pour le training et le test : https://kaggle.com/datasets/f1420d24aef99ee792ec69f601de96e74accd71e7300f79f2b9ea5d7a4afd757
Le dataset contient des attributs concernant : la position, les précipitations, le vent, l'humidité, le rayonnement, la pluie, la neige, les températures min et max, l'écoulement, etc.

## Résultats

*   **Convergence & Optimisation :** Grâce au pré-chargement en VRAM et à l'augmentation des *batch sizes*, le modèle exploite pleinement le GPU. Le modèle apprend efficacement avec une perte (MSE/MAE) très faible sur les données de test.
*   **Prédiction à J+50 :** Contrairement aux architectures naïves qui prédisent uniquement la température (entraînant une saturation rapide ou des valeurs physiquement impossibles), **notre modèle génère des courbes de température hautement cohérentes sur 50 jours**. En forçant le modèle à prédire l'état météorologique *complet* (26 variables) à chaque étape, il conserve le contexte physique nécessaire pour ne pas diverger, respectant ainsi les tendances saisonnières et les variations journalières.

*   **Améliorations possibles :** Bien que la prédiction récursive multivariée ait résolu le problème des valeurs aberrantes, les courbes ont tendance à se lisser légèrement sur le très long terme (convergence vers la moyenne). L'étape suivante consisterait à abandonner la récursion au profit d'une architecture **Seq2Seq (Sequence-to-Sequence)** ou **Transformer**, capable de cracher directement les 50 prochains jours en une seule passe (Direct Multi-step Forecasting). De plus, l'intégration de l'historique complet de Météo France (plus de 30 ans de données) permettrait d'ancrer encore mieux les prédictions, bien que cela nécessiterait des ressources distribuées (Multi-GPU) en raison de la quantité phénoménale de RAM/VRAM requise.

## Matériel utilisé

Les expérimentations ont été réalisées sur la configuration suivante :
*   **RAM :** 32 Go
*   **GPU :** Nvidia P100 (16 Go)
*   **Environnement :** Kaggle Notebook / Jupyter

## Auteur

**ZG**
*Date : Mars 2026*
