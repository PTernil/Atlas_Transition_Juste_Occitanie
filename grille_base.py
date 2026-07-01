# -*- coding: utf-8 -*-
"""
Creation date circa Tue Mar 10, 2026
Last authenticated version : Wed Jul 01 11:15 2026
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
liste_dep_occ=['09','11','12','30','31','32','34','46','48','65','66','81','82']

# Récupération des IRIS de la région Occitanie (EPSG:2154)
    # Import des IRIS français
iris=gpd.read_file(r"Données brutes\CONTOURS-IRIS\1_DONNEES_LIVRAISON_2025-06-00117\CONTOURS-IRIS_3-0_GPKG_LAMB93_FXX-ED2025-01-01\iris.gpkg")
    # Ajout de la surface de chaque unité du maillage pour les opérations de cartographie
iris.loc[:,'surface']=iris['geometry'].area/1e6
    # Filtrage sur le département
iris_occ=iris.loc[np.isin([a[:2] for a in iris["code_insee"]],liste_dep_occ)]
    # Export des IRIS occitans
iris_occ[['code_iris','surface','geometry']].to_file(r'Données traitées\IRIS.gpkg')

# Récupération des contours communaux
    # Fusion des IRIS par commune
com = iris[['code_insee','nom_commune','surface','geometry']].dissolve(by='code_insee',aggfunc={'nom_commune':'first','surface':'sum'})
com['centroids'] = com.geometry.centroid
com_occ=iris_occ[['code_insee','nom_commune','surface','geometry']].dissolve(by='code_insee',aggfunc={'nom_commune':'first','surface':'sum'})
    # Identification des préfectures et sous-préfectures
prefs=['09122','11069','12202','30189','31555','32013','34172','46042','48095','65440','66136','81004','82121']
sous_prefs=['09225','09261','11206','11262','12145','12300','30007','30350','31395','31483','32107','32256',\
            '34032','34142','46102','46127','48061','65025','65059','66049','66149','81065','82033']
com_occ['prefecture']=np.isin(com_occ.index,prefs)
com_occ['sous-prefecture']=np.isin(com_occ.index,sous_prefs)
    # Export des communes occitanes
com_occ.to_file(r'Données traitées\Communes.gpkg')

# Récupération des contours départementaux
dep = iris[['code_insee','surface','geometry']]
    # Création de la colonne de département
dep.loc[:,'dep']=dep.loc[:,'code_insee'].str[:2]
    # Filtrage des départements occitans
dep_occ = dep.loc[np.isin([a[:2] for a in iris['code_insee']],liste_dep_occ),['dep','surface','geometry']]
    # Union des géométries
dep_occ = dep_occ.dissolve(by='dep',aggfunc='sum')
    # Export
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
liste_dep_na = ['16','17','19','23','24','33','40','47','64','79','86','87']
liste_dep_aura = ['01','03','07','15','26','38','42','43','63','69','73','74']
liste_dep_paca = ['04','05','06','13','83','84']
corr_dep_reg = {
    **dict(zip(liste_dep_occ, ['76'] * len(liste_dep_occ))),
    **dict(zip(liste_dep_na, ['75'] * len(liste_dep_na))),
    **dict(zip(liste_dep_aura, ['84'] * len(liste_dep_aura))),
    **dict(zip(liste_dep_paca, ['93'] * len(liste_dep_paca))),
}
dep = dep.loc[np.isin([a for a in dep['dep']],list(corr_dep_reg.keys())),['geometry','dep']]
dep.loc[:,'reg'] = [corr_dep_reg[d] for d in dep.loc[:,'dep']]
dep = dep.loc[np.isin(dep['reg'],['75','84','93'])]
# Fusion des contours régions limitrophes
reg_fr = dep[['reg','geometry']].dissolve(by='reg')
reg_fr.loc[:,'nom'] = reg_fr.index.map({'75':"Nouvelle-Aquitaine",'84':"Auvergne-Rhône-Alpes",'93':"Provence-Alpes-Côte d'Azur"})
# Fusion des contours des départements des régions limitrophes
dep = dep.dissolve(by='dep')
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
# Importation des données commune, EPCI, département
table_corr = pd.read_excel(r"Données brutes\EPCI_au_01-01-2026.xlsx", sheet_name=1, header=5)
# Filtrage des données occitanes
table_corr = table_corr.loc[table_corr['REG']==76]
# Ajout des IRIS
table_corr = iris_occ.join(table_corr.set_index('CODGEO'), on='code_insee')
# Export des géométries des EPCI
epci = table_corr[['EPCI','LIBEPCI','surface','geometry']].dissolve(by='EPCI',aggfunc={'LIBEPCI':'first','surface':'sum'})
epci.to_file(r"Données traitées\EPCI.gpkg")
# Export des correspondances entre mailles administratives
table_corr = table_corr[['code_insee','code_iris','EPCI','DEP']].rename(columns={'DEP':'dep'}).set_index('code_iris')
table_corr.to_csv(r"Données traitées\Correspondance_echelle_admin.csv")


