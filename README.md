# Objectif

Ce projet a pour but de classifier des images issues du dataset SVHN sans avoir accès aux labels. 
Ainsi, nous avons à notre disposition le dataset MNIST avec les labels associés. L'objectif sera donc d'utiliser une technique d'adaptation de domaine non-supervisé à partir de MNIST vers SVHN.

Notre solution reprend la méthode présentée au sein de l'article 'Generative Pseudo-label Refinement for Usupervised Domain Adaptation.'
Ainsi, l'approche se découpe en plusieurs parties :
- pré-entrainement d'un classifieur sur le dataset MNIST, permettant ainsi d'avoir des pseudo-labels pour SVHN
- pré-entraînement d'un cGAN sur des images SVHN à partir des pseudo-labels du classifieur
- boucle d'entraînement où le classifier et le cGAN seront mis-à-jour alternativement

Cette méthode repose sur le constat qu'un classifieur seul overfit le 'shift-noise' issu du changement de domaine (erreur de classification non uniforme sur l'ensemble des classes). Les performances du classifieur serait améliorées dans le cas où le 'shift-noise' soit plus uniformes. C'est pour cela qu'un cGAN est intégré dans le but de filter un tel bruit et ainsi de générer des images 'plus propres'.

Le code a donc pour vocation d'implémenter cette approche à partir de l'architecture décrites au sein du papier (classifier convNet, generateur/discriminateur cGAN avec architecture suivant DCGAN).
La méthode d'entrainment permettant d'améliorer itérativement un classifieur pré-entrainer sur MNIST, nous avons investiguer comment améliorer le plus possible ce classifieur initiale. En remarquant que le dataset SVHN est relativement plus complexe / diversifié que celui du MNIST (expliquant l'écart de performance entre SVHN->MNIST et MNIST->SVHN), nous avons ajouter une étape de data-augmentation permettant d'améliorer les performances du classifieur initial.

# Répertoire

Le répertoire comprends un ensemble de fichier .py représentant la solution finale :
- ```main_v3.py``` : fichier principal reprenant la définition des paramètres, bouble de pré-entrainement /entrainement et appel de fonction
- ```code/models_paper_v2.py``` : définition l'architecture du classifieur, du générateur et du discriminateur
- ```code/classif_training.py``` : définition les boucles de pré-entrainement / entrainement du classifieur
- ```data/dataset.py``` : définition des dataloader pour MNIST / SVHN, sans data_augmentation pour reproduire les performances du papier
- ```data/dataset_v2.py``` : intégration de la data_augmentation pour améliorer les performances de la méthode globale
- ```_MNIST_SVHN_100epochs_results``` : dossier contenant les résultats sans data-augmentation - accuracy 55.99% (papier = 63,4%)
- ```_MNIST_SVHN_100ep_DataAug_results``` : dossier contenant les résultats avec data-augmentation - accuracy 78.73% (papier = 63,4%)
