
# Prédiction de Survie sur le Titanic  

## Table des Matières  
- [À Propos du Projet](#à-propos-du-projet)  
- [Fonctionnalités](#fonctionnalités)  
- [Technologies Utilisées](#technologies-utilisées)  
- [Prise en Main](#prise-en-main)  
  - [Prérequis](#prérequis)  
  - [Installation](#installation)  
  - [Configuration](#configuration)  
- [Utilisation](#utilisation)  
- [Exemples de Résultats](#exemples-de-résultats)  
- [Perspectives du Projet](#perspectives-du-projet)  
- [Contribuer](#contribuer)  
- [Remerciements](#remerciements)  

---

## À Propos du Projet  

Ce projet consiste à utiliser un modèle d'apprentissage automatique pour prédire les chances de survie des passagers du Titanic en utilisant le jeu de données bien connu de Kaggle : *Titanic - Machine Learning from Disaster*.  
En se basant sur des caractéristiques telles que l'âge, le sexe, la classe du billet, etc., le modèle entraîne un réseau de neurones pour estimer la probabilité de survie de chaque passager.  

---

## Fonctionnalités  

- Traitement des données manquantes et encodage des variables catégorielles.  
- Standardisation des données pour améliorer l'entraînement du modèle.  
- Construction d'un modèle de réseau de neurones avec TensorFlow/Keras.  
- Utilisation de la validation croisée et de l'arrêt anticipé pour éviter le surapprentissage (*overfitting*).  
- Génération d'un fichier de soumission CSV pour Kaggle.  

---

## Technologies Utilisées  

- **Langages :** Python  
- **Bibliothèques :**  
  - *Pandas* pour la manipulation des données  
  - *Numpy* pour les calculs numériques  
  - *TensorFlow/Keras* pour la création et l'entraînement du modèle  
  - *Scikit-learn* pour la normalisation et la division des données  

---

## Prise en Main  

### Prérequis  

- Python 3.x installé  
- *pip* pour la gestion des paquets  

### Installation  

1. Cloner le dépôt :  

\`\`\`sh  
git clone https://github.com/callbak/Titanic-MachineLearning-FromDisaster.git  
cd Titanic-MachineLearning-FromDisaster  
\`\`\`  

2. Installer les dépendances :  

\`\`\`sh  
pip install -r requirements.txt  
\`\`\`  

### Configuration  

- Assurez-vous que les fichiers \`train.csv\` et \`test.csv\` se trouvent dans le répertoire racine du projet.  

---

## Utilisation  

1. Exécuter le script principal :  

\`\`\`sh  
python main.py  
\`\`\`  

2. Vérifier les résultats dans le fichier \`submission.csv\` généré.  

---

## Exemples de Résultats  

On constate que, lors des 10 premiers et derniers 'epochs', la précision ('accuracy') et la fonction de perte ('loss') se sont continuellement améliorées, aussi bien pendant l'entraînement que durant la phase de validation ('validation split'). Cela indique une convergence progressive du modèle vers des performances optimales.
---
![first-10-epochs](https://github.com/user-attachments/assets/5d09904e-5f32-4051-8e5c-d79cc51ff068)

![last-10-epochs](https://github.com/user-attachments/assets/c3c6f0aa-e533-401e-aba6-68ab7e13af67)

## Remerciements  

Merci à Kaggle pour le jeu de données **Titanic - Machine Learning from Disaster**
