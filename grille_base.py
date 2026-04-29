# -*- coding: utf-8 -*-
"""
@author=Paul Ternil
"""

"""Note : La gestion du cas de Conques-en-Rouergue, commune ayant deux codes Insee suivant l'année,
engendre la nécessité d'avoir un index de type int pour iris_occ."""

# Importation des librairies
import numpy as np
import pandas as pd
import geopandas as gpd

""""GRILLE"""
    
# Récupération des contours administratifs occitans
liste_dep_occ=["09","11","12","30","31","32","34","46","48","65","66","81","82"]

# Récupération des IRIS de la région Occitanie (EPSG:2154)
    # Import des IRIS français
iris_occ=gpd.read_file(r"Données brutes\CONTOURS-IRIS\1_DONNEES_LIVRAISON_2025-06-00117\CONTOURS-IRIS_3-0_GPKG_LAMB93_FXX-ED2025-01-01\iris.gpkg")
    # Filtrage sur le département
iris_occ=iris_occ.loc[np.isin([a[:2] for a in iris_occ["code_insee"]],liste_dep_occ)]
    # Ajout de la surface de chaque unité du maillage pour les opérations de cartographie
iris_occ["surface"]=iris_occ["geometry"].area/1e6
    # Export des IRIS occitans
iris_occ=iris_occ.set_crs("EPSG:2154")
iris_occ[["code_iris","surface","geometry"]].to_file(r"Données traitées\IRIS.gpkg")

# Récupération des contours communaux
    # Fusion des IRIS par commune
com_occ=iris_occ[["code_insee","nom_commune","geometry"]].dissolve(by="code_insee")
    # Identification des préfectures et sous-préfectures
prefs=["09122","11069","12202","30189","31555","32013","34172","46042","48095","65440","66136","81004","82121"]
sous_prefs=["09225","09261","11206","11262","12145","12300","30007","30350","31395","31483","32107","32256",\
            "34032","34142","46102","46127","48061","65025","65059","66049","66149","81065","82033"]
com_occ["prefecture"]=np.isin(com_occ.index,prefs)
com_occ["sous-prefecture"]=np.isin(com_occ.index,sous_prefs)
    # Export des communes occitanes
com_occ.to_file(r"Données traitées\Communes.gpkg")

# Récupération des contours départementaux (EPSG:4326)
    # Import des départements français
dep=gpd.read_file("Données brutes\contours_dep.txt")
    # Filtrage sur le code de département
dep_occ=dep.loc[np.isin(dep["code"],liste_dep_occ)]
    # Reprojection en Lambert 93 (EPSG:2154)
dep_occ=dep_occ.set_crs("EPSG:4326")
dep_occ=dep_occ.to_crs("EPSG:2154")
    # Export des départements occitans
dep_occ.to_file(r"Données traitées\Départements.gpkg")

# Récupération des contours régionaux
    # Fusion des contours départementaux occitans
reg_occ=gpd.GeoDataFrame({"nom":"Occitanie",
                          "code":"76",
                          "geometry":dep_occ.geometry.union_all()},
                         index=[0],
                         crs="EPSG:2154")
    # Export du contour régional occitan
reg_occ.to_file(r"Données traitées\Région.gpkg")
    # Identification des départements des régions limitrophes
dep=dep.loc[np.isin(dep["region"],["75","84","93"])]
    # Reprojection en Lambert 93 (EPSG:2154)
dep=dep.set_crs("EPSG:4326")
dep=dep.to_crs("EPSG:2154")
# Fusion des contours départementaux des régions limitrophes
reg_fr=dep.dissolve(by="region")[["nom","geometry"]]
reg_fr["nom"]=["Nouvelle-Aquitaine","Auvergne-Rhône-Alpes","Provence-Alpes-Côte d'Azur"]
    # Export des contours départementaux et régionaux des régions limitrophes
dep.to_file(r"Données traitées\Départements limitrophes.gpkg")
reg_fr.to_file(r"Données traitées\Régions limitrophes.gpkg")

# Récupération des contours espagnols et andorrais (EPSG:3035)
    # Import des contours nationaux
eu_countries=gpd.read_file(r"Données brutes\eu_countries.gpkg")
    # Sélection des pays limitrophes
eu_countries=eu_countries.loc[np.isin(eu_countries["CNTR_ID"],["AD","ES"]),["NAME_FREN","geometry"]]
    # Reprojection en Lambert 93
eu_countries=eu_countries.set_crs("EPSG:3035")
eu_countries=eu_countries.to_crs("EPSG:2154")
    # Export des pays limitrophes
eu_countries.to_file(r"Données traitées\Pays limitrophes.gpkg")

# Maille des serices météorologiques (SAFRAN et Drias)
maille_safran = gpd.read_file("Données brutes\safran.gpkg")
# Reprojection en Lambert 93
maille_safran = maille_safran.set_crs('EPSG:4326')
maille_safran = maille_safran.to_crs('EPSG:2154')
# Filtrage sur la région
maille_safran = maille_safran.clip(reg_occ.geometry.iloc[0])
# Correspondance SAFRAN / Drias
maille_safran = maille_safran[['cell','geometry']]
corr_safran_drias = pd.read_csv("Données brutes\maille_safran_drias-20250306.csv")
maille_drias = maille_safran.join(corr_safran_drias.set_index('maille_safran'),on='cell',how='inner')
maille_drias = maille_drias[['maille_drias','geometry']]
maille_drias['maille_drias'] = maille_drias['maille_drias'].astype('str')
maille_safran.to_file(r"Données traitées\maille safran.gpkg")
maille_drias.to_file(r"Données traitées\maille drias.gpkg")

"""Table de correspondance grilles adminisatratives"""
table_corr = pd.read_excel(r"Données brutes\EPCI_au_01-01-2026.xlsx", sheet_name=1, header=5)
table_corr = table_corr.loc[table_corr['REG']==76]
table_corr = iris_occ.join(table_corr.set_index('CODGEO'), on='code_insee')
table_corr = table_corr[['code_insee','code_iris','EPCI','DEP']].set_index('code_iris')
table_corr.to_csv(r"Données traitées\Correspondance_echelle_admin.csv")
