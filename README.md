# ConvLst 

Ce code sert à prédire des actions à partir d'une séquence vidéo.

## Fonctions

1- *Extraction des frames* : La fonction frames_extraction lit des frames d'une vidéo, les redimensionne, les normalise, et les prépare pour la prédiction.

2- *Prédiction de l'action* : La fonction predict_single_action charge un modèle pré-entraîné, effectue des prédictions sur les frames extraites, et retourne la classe prédite (normal ou anormal).

3- *Fonction principale* : La fonction ar utilise le modèle pour prédire le comportement dans la vidéo spécifiée et retourne False si le comportement est "anormal" et True si le comportement est "normal".

## Utilisation

Il suffit de modifier le chemin de la data si besoin, ainsi que celui de la vidéo.

Puis exécuter par la commande python3 ar.py




