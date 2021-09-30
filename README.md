# Human to Emoji

Détection de gestes transformés en emojis en temps réel à la webcam 
Il s'agit d'une amélioration de mon précédent projet "TraductionEmojiWebcam"

![Alt Text](https://github.com/DaoudKAD/HumanToEmoji/blob/master/resultat.gif)

### Installation

- Cloner le repos : 
  - Via SSH : ``git clone git@github.com:DaoudKAD/HumanToEmoji.git``
  - Via HTTPS : ``git clone https://github.com/DaoudKAD/HumanToEmoji.git``


- Installer les différentes librairies via les commandes suivantes :
  - ``cd HumanToEmoji`` 
  - ``pip install -r requirements.txt``  

## Démarrage

- ``cd WebCamDetection``


- Changer le path RACINE ``detection.py`` afin qu'il corresponde au chemin menant au repos sur votre machine

- Démarrez l'application avec ``python3 detection.py``


## Training & ajout de nouveaux emojis

Étapes à suivre afin de réentraîner le modèle sur de nouveaux emojis :

- Modification du fichier ``acquisition/images_acquisition.py``
  - ``cd acquisition``
  - Changez ``path_to_save_images`` et ``labels_list`` selon vos besoins
  - Lancez l'acquisition des images ``python3 detection.py``
  - Labelisez ces images acquises avec un outils tel que LabelImg
  - Créez un dossier avec deux repertoirs train/test contenant les images acquises ainsi que le fichier XML généré précédemment, au format : 
    - ``dataset/``
      - ``train/img1.png``
      - ``train/img5.png``
      - ``train/img1.xml``
      - ``train/img5.xml``
      - ...
      - ``test/img1.png``
      - ``test/img5.png``
      - ``test/img1.xml``
      - ``test/img5.xml``
      - ...
  

- Entraînement du modèle MobileNet-SSD : 
  - Modifiez les variables globales du fichier ``training/training_script.py`` selon l'emplacement des données train test
  - Modifiez les chemins du fichier tf_records.sh selon votre repertoire
  - ``cd training``
  - ``python3 training_script && sh tf_records.sh && training_script.py``
  - Executez la commande retournée permettant le training du modèle
  

- Une fois l'entraînement terminé, démarrez l'application avec ``python3 detection.py``

## Outils utilisés

* [Python 3.9](https://www.python.org/) 
* [TensorFlow 2](https://www.tensorflow.org/) 
* [OpenCV](https://opencv.org/)

## Auteurs
* **Daoud Kadoch** _alias_ [@daoud.kadoch](https://github.com/DaoudKAD)
