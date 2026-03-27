# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:04:50 2026

@author: ternilp
"""

# Importation des librairies
import import_donnees as imp

# Importation des données
n_data = int(input("Nombre de jeux de données à importer :"))
datasets = []
for n in range(n_data):
    print(f"\nJeu de données n°{n+1}")
    filepath = imp.ask_filepath()
    filepath = imp.check_filepath(filepath)[1]
    datasets.append(imp.import_progress(filepath, compact=False))
    # Raccordement à la géométrie cadre
    geom_validation = False
    while geom_validation==False:
        geom_grid = imp.ask_geom()
        geom_data = imp.search_geom(geom_grid, datasets[-1])
        if not geom_data is None:
            geom_validation=True


# Importation des grilles
print("Importation des références géographiques...\n", flush=True)
reg = imp.import_progress(r"Données traitées\Région.gpkg")
dep = imp.import_progress(r"Données traitées\Départements.gpkg")
com = imp.import_progress(r"Données traitées\Communes.gpkg")
iris = imp.import_progress(r"Données traitées\IRIS.gpkg")
pays_v = imp.import_progress(r"Données traitées\Pays limitrophes.gpkg")
reg_v = imp.import_progress(r"Données traitées\Régions limitrophes.gpkg")
dep_v = imp.import_progress(r"Données traitées\Départements limitrophes.gpkg")
print("Importation terminée.", flush=True)