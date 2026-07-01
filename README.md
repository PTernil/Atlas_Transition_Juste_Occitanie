Module python servant à la création de cartes de la région Occitanie.

>[!IMPORTANT]
Le module est prévu pour une disposition des dossiers comme suit :
Racine du projet :
- main.py
- grille_base.py
- traitement_preliminaire.py
- (off_main.py)
- atlas_modules:
  - carto.py
  - import_données.py
- Cartes
- Données brutes
- Données traitées
- (Données utilisées)

>[!NOTE]
Après exécution, des dossiers _pycache_ seront créés dans le dossier racine et le dossier atlas_modules. Ils peuvent être supprimés sans problème si le programme n'est pas en cours d'utilisation. 

>[!IMPORTANT]
En cas de traitement de nouvelles données en traitement_preliminaire.py, si elles sont exportées au format .csv, on veillera à ajouter le suffixe "_data" aux données numériques. En cas d'oubli, elles seront interprétées comme du texte.

>[!TIP]
En cas de traitement de nouvelles données en traitement_preliminaire.py, si elles sont exportées au format .csv, on pourra ajouter au nom du fichier le suffixe '_IRIS', '_commune', '_EPCI', '_dept', '_maille_safran' ou '_maille_drias' en fonction de la maille utilisée, afin de permettre au programme de détecter automatiquement l'échelle à l'importation des données.
Dans ce cas, il faudra que l'identifiant géographique des données soit l'une des valeurs spécifiées dans atlas_modules.import_donnees.geom_columns_dict et de préférence : 'code iris', 'code_insee', 'code_epci', 'code_dep', 'maille_safran' et 'maille_drias'

>[!TIP]
Le dossier Données utilisées peut être mobilisé pour sauvegarder des données obtenues par traitement dans la console de main.py. Le fichier off_main.py sert à réutiliser ces données.

>[!NOTE]
La fonctionnalité d'off_main.py n'est pas garantie
