# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:36:24 2026
Last authenticated version : Wed Jul 01 11:15 2026
@author: ternilp
"""

from affine import Affine
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pathlib import Path
from rasterio.features import rasterize, shapes
from shapely.geometry import shape
from sklearn.decomposition import PCA
import atlas_modules.import_donnees as imp
from grille_base import com

# Conseil : pour clarté d'utilisation, on mentionnera l'échelle de raccordement des données
# dans le nom du fichier, sans rien préciser pour une géométrie propre aux données

"""Grilles pour sélection des données"""

iris_occ = gpd.read_file(r"Données traitées\IRIS.gpkg")
iris_occitanie = iris_occ.loc[:,"code_iris"]

com_occ = gpd.read_file(r"Données traitées\Communes.gpkg")
com_occitanie = com_occ.loc[:,"code_insee"]

liste_dep_occ=["09","11","12","30","31","32","34","46","48","65","66","81","82"]

reg = gpd.read_file(r"Données traitées\Région.gpkg")

"""Couples - Familles - Ménages, 2021 (Insee)"""

# Extraction des données de l'enquête Couples-Familles-Ménages (Insee)
cfm_2021=pd.read_csv(r"Données brutes\base-ic-couples-familles-menages-2021_csv\base-ic-couples-familles-menages-2021.csv",\
                     sep=";", dtype='str')
# Rebaptême des IRIS périmés
cfm_2021['IRIS'] = cfm_2021['IRIS'].replace(imp.deprecated_codes)
    # Identification des IRIS Occitans
cfm_2021 = cfm_2021.loc[np.isin(cfm_2021["IRIS"],iris_occitanie)]
    # Extraction des variables de population et de répartition par PCS
cfm_2021 = cfm_2021.loc[:,["IRIS","COM","C21_PMEN","P21_POP15P","P21_POP5579","P21_POP80P",\
                          "C21_PMEN_CS1","C21_PMEN_CS2","C21_PMEN_CS3","C21_PMEN_CS4",\
                          "C21_PMEN_CS5","C21_PMEN_CS6","C21_PMEN_CS7","C21_PMEN_CS8",\
                          "P21_POP5579_PSEUL","P21_POP80P_PSEUL","C21_PMEN_MENFAMMONO"]]
cfm_2021[cfm_2021.columns[2:]] = cfm_2021.iloc[:,2:].apply(pd.to_numeric)

# Construction d'un indice de répartition des PCS
    # Pondération des PCS
poids_pcs = {"C21_PMEN_CS3":2,
             "C21_PMEN_CS4":1,
             "C21_PMEN_CS2":1,
             "C21_PMEN_CS1":0,
             "C21_PMEN_CS5":0,
             "C21_PMEN_CS6":0,
             "C21_PMEN_CS7":-1,
             "C21_PMEN_CS8":-2}
cfm_2021["Score_CSP"] = sum([poids_pcs[PCS]*cfm_2021[PCS]/cfm_2021["C21_PMEN"] for PCS in poids_pcs.keys()])
cfm_2021["Score_CSP"] = cfm_2021["Score_CSP"].fillna(cfm_2021["Score_CSP"].mean())
    # Normalisation par la méthode des écarts-types
cfm_2021["Score_CSP"] = (cfm_2021["Score_CSP"]-cfm_2021["Score_CSP"].mean())/cfm_2021["Score_CSP"].std()
    # Suppression des colonnes de base PCS pour simplicité d'utilisation, par regroupement en deux catégories
cfm_2021["C21_PMEN_CSP+"] = cfm_2021["C21_PMEN_CS3"]+cfm_2021["C21_PMEN_CS4"]
cfm_2021["C21_PMEN_CSP-"] = cfm_2021["C21_PMEN_CS2"]+cfm_2021["C21_PMEN_CS1"]+cfm_2021["C21_PMEN_CS5"]+\
                            cfm_2021["C21_PMEN_CS6"]+cfm_2021["C21_PMEN_CS7"]+cfm_2021["C21_PMEN_CS8"]
cfm_2021.drop(columns=list(poids_pcs.keys()))

# Populations vulnérables à la chaleur : plus de 75 ans (moins de 4 ans non calculable)
cfm_2021["P_21_PVUL_CHAL"]=cfm_2021["P21_POP5579"]*5/25\
                           +cfm_2021["P21_POP80P"]

# Rebaptême des colonnes d'indicateurs géographiques
cfm_2021 = cfm_2021.rename(columns={'IRIS':'code_iris','COM':'code_insee'})
# Rebaptême des colonnes pour inférence du type à la lecture
cfm_2021 = cfm_2021.rename(columns=dict(zip(cfm_2021.columns[2:],[cfm_2021.columns[n]+"_data"for n in range(2,cfm_2021.shape[1])])))

# Écriture dans un fichier
cfm_2021.to_csv(r"Données traitées\Couples_Familles_Ménages_2021_IRIS.csv", index=False)

"""Population (Insee)"""
pop_insee=pd.read_csv(r"Données brutes\base-ic-evol-struct-pop-2021_csv\base-ic-evol-struct-pop-2021.csv",
                      sep=';',dtype='str')
# Rebaptême des IRIS périmés
pop_insee['IRIS'] = pop_insee['IRIS'].replace(imp.deprecated_codes)
# Identification des IRIS Occitans
pop_insee = pop_insee.loc[np.isin(pop_insee["IRIS"],iris_occitanie)]
#Populations sensibles à la chaleur : moins de 4 ans, plus de 75 ans
pop_insee = pop_insee.loc[:,['IRIS','COM','P21_POP','P21_POP0002','P21_POP0305',
                             'P21_POP0610','P21_POP1117','P21_POP1824','P21_POP2539',
                             'P21_POP4054','P21_POP5564','P21_POP6074','P21_POP75P']]
pop_insee[pop_insee.columns[2:]]=pop_insee.iloc[:,2:].apply(pd.to_numeric)
pop_insee['P21_etudes'] = pop_insee['P21_POP0305']+pop_insee['P21_POP0610']+pop_insee['P21_POP1117']+pop_insee['P21_POP1824']*.62
pop_insee['P21_travail'] = pop_insee['P21_POP1824']*.42+pop_insee['P21_POP2539']+pop_insee['P21_POP4054']+pop_insee['P21_POP5564']
pop_insee["P21_PVUL_CHAL"]=pop_insee["P21_POP0002"]+pop_insee["P21_POP0305"]*2/3+pop_insee["P21_POP75P"]
# Rebaptême des colonnes d'indicateurs géographiques
pop_insee = pop_insee.rename(columns={'IRIS':'code_iris','COM':'code_insee'})
# Rebaptême des colonnes pour inférence du type à la lecture
pop_insee = pop_insee.rename(columns=dict(zip(pop_insee.columns[2:],[pop_insee.columns[n]+"_data"for n in range(2,pop_insee.shape[1])])))
pop_insee.to_csv(r"Données traitées\Population_2021_IRIS.csv", index=False)

"""Zonage exposition à la chaleur (Insee)"""

chaleur_insee=pd.read_excel("Données brutes\chaleurs_insee.xlsx",sheet_name=1)
chaleur_insee=chaleur_insee.iloc[6:] # Les 6 premières lignes de la sources sont des indications de lecture
# Construction des noms de colonnes à partir des deux premières lignes du tableau
new_columns=[chaleur_insee.iloc[0,0]]
for n_column in range(1,chaleur_insee.shape[1]):
    if not pd.isna(chaleur_insee.iloc[0,n_column]):
        new_columns.append(chaleur_insee.iloc[0,n_column]+' : '+chaleur_insee.iloc[1,n_column]+'_data')
    else:
        new_columns.append(chaleur_insee.iloc[0,n_column-1]+' : '+chaleur_insee.iloc[1,n_column]+'_data')
# Utilisastion des nouvelles colonnes
chaleur_insee.columns=new_columns
chaleur_insee = chaleur_insee.iloc[2:]
# Conversion des données vers un format numérique
chaleur_insee = chaleur_insee.apply(pd.to_numeric)
# Rebaptême des colonnes d'indicateurs géographiques
chaleur_insee = chaleur_insee.rename(columns={'ID Safran':'maille_drias'})
chaleur_insee.to_csv(r"Données traitées\Zonage_chaleur_maille_drias.csv", index=False)

"""Revenu (Insee)"""

# Documentation du code : cf enquête Couples - Familles - Ménages
revenus_insee = pd.read_csv(r"Données brutes\BASE_TD_FILO_IRIS_2021_DISP_CSV\BASE_TD_FILO_IRIS_2021_DISP.csv",
                            sep=";", dtype='str')
revenus_insee['IRIS'] = revenus_insee['IRIS'].replace(imp.deprecated_codes)
revenus_insee = revenus_insee.loc[np.isin(revenus_insee["IRIS"],iris_occitanie)]
revenus_insee = revenus_insee[['IRIS','DISP_TP6021','DISP_MED21','DISP_PPSOC21','DISP_S80S2021']]
revenus_insee = revenus_insee.rename(columns={'IRIS':'code_iris'}).set_index('code_iris')
revenus_insee = revenus_insee.replace(',','.',regex=True)
revenus_insee = revenus_insee.apply(pd.to_numeric,errors='coerce')
revenus_insee = revenus_insee.rename(
    columns=dict(zip(revenus_insee.columns,
                     [revenus_insee.columns[n]+"_data"for n in range(revenus_insee.shape[1])])))
revenus_insee.to_csv(r"Données traitées\Revenus_filosofi_2021_IRIS.csv",index=False)

# Lecture et sélection des variables
revenus_insee_france = gpd.read_file(r"Données brutes\carreaux_nivNaturel_met.gpkg")
revenus_insee_france = revenus_insee_france[['ind','men','men_pauv','ind_snv','geometry']]
# Conservation des données de la région, avec interpolation linéaire sur la surface pour les carreaux frontaliers
revenus_insee = revenus_insee_france.clip(reg.geometry.iloc[0])
area_change = revenus_insee.geometry.area/revenus_insee_france.loc[revenus_insee.index].geometry.area
revenus_insee.loc[:,['ind','men','men_pauv','ind_snv']] = revenus_insee.loc[:,['ind','men','men_pauv','ind_snv']].apply(pd.to_numeric)
revenus_insee.loc[:,['ind','men','men_pauv','ind_snv']] = revenus_insee[['ind','men','men_pauv','ind_snv']].apply(lambda x:x*area_change)
# Calcul de variables supplémentaires
revenus_insee["taux_pauv"] = revenus_insee["men_pauv"]/revenus_insee["men"]
revenus_insee["moy_winsor_niv_vie"] = revenus_insee["ind_snv"]/revenus_insee["ind"]
# Réagencement des colonnes pour une lecture plus aisée
revenus_insee = revenus_insee.loc[:,['ind','men','men_pauv','ind_snv','taux_pauv','moy_winsor_niv_vie','geometry']]
# Assignation d'un  indice pour agrégation éventuelle
revenus_insee = revenus_insee.reset_index()
revenus_insee = revenus_insee.rename(columns={'index':'id_car'})
# Écriture
revenus_insee.to_file(r"Données traitées\Revenus_filosofi_2021.gpkg")

"""Pathologies par département (CNAM)"""

# Lecture
pathologies = pd.read_parquet(r"Données brutes\pathologies_CNAM.parquet")
# Sélection de l'année, de la région, des données sans distinction de sexe ou d'âge
# Suppression de dept=999 : somme sur la région
pathologies = pathologies.loc[(pathologies['annee']=='2021') & (pathologies['region']=='76')
                              & (pathologies['sexe']==9) & (pathologies['cla_age_5']=='tsage')
                              & (pathologies['dept']!='999')]
# Conservation des colonnes d'intérêt
pathologies = pathologies[['patho_niv1','patho_niv2','patho_niv3','top','dept','Ntop','Npop']]
# Rebaptême des colonnes d'indicateurs géographiques
pathologies = pathologies.rename(columns={'dept':'dep'})
# Suppression des sous-totaux par catégories de pathologie
pathologies = pathologies.loc[~pd.isna(pathologies['patho_niv3'])]
# Conservation des pathologies rendant vulnérable à la chaleur
pathologies_chaleur = pathologies.loc[pathologies['patho_niv1'].isin(["Insuffisance rénale chronique terminale",
                                                              "Maladies cardioneurovasculaires",
                                                              "Maladies neurologiques",
                                                              "Maladies psychiatriques",
                                                              "Maladies respiratoires chroniques (hors mucoviscidose)"])]
pathologies_pollution = pathologies.loc[pathologies['patho_niv1'].isin(["Maladies cardioneurovasculaires",
                                                              "Maladies respiratoires chroniques (hors mucoviscidose)"])]
# Passage du tableau dans une forme plus exploitable
pathologies_chaleur = pathologies_chaleur.pivot(index='dep',columns=['patho_niv3'],values=['Ntop','Npop']).fillna(0)
    # Compte des malades
pathologies_chaleur['Vulnérables_chaleur'] = pathologies_chaleur['Ntop'].sum(axis=1)
pathologies_chaleur['pop'] = pathologies_chaleur['Npop'].iloc[:,0]
pathologies_chaleur['Part_vul_chaleur'] = pathologies_chaleur['Vulnérables_chaleur']/pathologies_chaleur['pop']
    # Filtrage des colonnes
pathologies_chaleur = pathologies_chaleur[['Vulnérables_chaleur','Part_vul_chaleur']]
pathologies_chaleur.columns = ['Vulnérables_chaleur_data','Part_vul_chaleur_data']
# Idem pour la pollution
pathologies_pollution = pathologies_pollution.pivot(index='dep',columns=['patho_niv3'],values=['Ntop','Npop']).fillna(0)
pathologies_pollution['Vulnérables_pollution'] = pathologies_pollution['Ntop'].sum(axis=1)
pathologies_pollution['pop'] = pathologies_pollution['Npop'].iloc[:,0]
pathologies_pollution['Part_vul_pollution'] = pathologies_pollution['Vulnérables_pollution']/pathologies_pollution['pop']
pathologies_pollution = pathologies_pollution[['Vulnérables_pollution','Part_vul_pollution']]
pathologies_pollution.columns = ['Vulnérables_pollution_data','Part_vul_pollution_data']
# Fusion des deux jeux
pathologies = pathologies_chaleur.join(pathologies_pollution)
# Exportation
pathologies.to_csv(r"Données traitées\Pathologies_dept.csv")

"""Travailleurs exposés chaleur extérieure"""
#Correspondance FAP2009-PCS-ESE pour combinaison données Dares/Insee
    # PCS de troisième niveau 429 PCS Insee, sans détail agriculture, artisans, commerçants
    # 541 PCS pour la Dares avec détail, 225 FAP (niveau 3 aussi)
    # agrégation de niveau 2 pour utilisation à 87 FAP, 29 PCS-ESE
    # En absence de données précises sur la répartition des professionnels suivant les tables croisées,
    # On estime que le nombre de catégories de niveau 3 est le meilleur estimateur du nombre d'individus

with open(r"Données brutes\PCS-ESE_vers_FAP-2009.txt", encoding="utf-8") as f:
    pcs_fap = f.read().replace(' ', '')
# Groupage de l'agriculture pour l'Insee
group_agri_pcs2 = {'11':'10','12':'10','13':'10'}

# Comptage des appariements FAP3/PCS2
pair_counts_3 = {}
for line in pcs_fap.split('\n')[:-1]:
    pcs, fap = line.replace('"', '').split('=')
    pcs = pcs.split(',')
    temp_pair_counts={}
    for pcs3 in pcs:
        pcs2 = group_agri_pcs2.get(pcs3[:2], pcs3[:2])
        temp_pair_counts[pcs2] = temp_pair_counts.get(pcs2,0)+1
    pair_counts_3[fap] = temp_pair_counts
    
# Passage au niveau FAP2
pair_counts = {}
fap2 = 0
for fap3 in pair_counts_3.keys():
    temp_pair_counts = pair_counts_3[fap3]
    if fap3[:3]!=fap2:
        fap2 = fap3[:3]
        pair_counts[fap2] = temp_pair_counts
    else:
        temp_sum_pair_counts = pair_counts[fap2]
        for pcs2 in temp_pair_counts:
            temp_sum_pair_counts[pcs2] = temp_sum_pair_counts.get(pcs2,0) + temp_pair_counts[pcs2]
        pair_counts[fap2] = temp_sum_pair_counts
            
corr_pcs2_fap2 = pd.DataFrame(pair_counts)
corr_pcs2_fap2 = corr_pcs2_fap2/corr_pcs2_fap2.sum()

# Correspondance libellé/code FAP2
with open(r"Données brutes\Libellés_FAP-2009_niv2.txt", encoding="utf-8") as f:
    lib_fap = f.read()
fap2_codes={}
for line in lib_fap.split('\n')[:-1]:
    code,label = line.replace('"','').split('=')
    label=label[6:]
    fap2_codes[label]=code

# Données exposition chaleur
trav_expo_chaleur = pd.read_csv(r"Données brutes\dares_sumer_expo_chaleur.csv")
trav_expo_chaleur = trav_expo_chaleur[~pd.isna(trav_expo_chaleur['part_travail_exterieur'])].iloc[1:,[0,3]]
# Correction des abbréviations
trav_expo_chaleur['Libelle_metier'] = trav_expo_chaleur['Libelle_metier'].str.replace('OQ','Ouvriers qualifiés')
trav_expo_chaleur['Libelle_metier'] = trav_expo_chaleur['Libelle_metier'].str.replace('OPQ','Ouvriers non qualifiés')
trav_expo_chaleur['Libelle_metier'] = trav_expo_chaleur['Libelle_metier'].str.replace('BTP','bâtiment, des travaux publics')
# Passage aux codes FAP2
trav_expo_chaleur = trav_expo_chaleur.replace({'Libelle_metier':fap2_codes}).set_index('Libelle_metier')
# Conversion vers les catégories PCS-ESE
corr_pcs2_fap2 = corr_pcs2_fap2[trav_expo_chaleur.index].fillna(0)
# Normalisation à 1 (montre que l'estimation du nombre d'individus n'est à minima pas parfaite)
trav_expo_chaleur = corr_pcs2_fap2.dot(trav_expo_chaleur).map(lambda x:min(1,x))

# Données de travail par code PCS2, par IRIS
trav_pcs = pd.read_csv(r"Données brutes\Insee_emploi_activité_en_2021.csv", sep=';',dtype='str')
trav_pcs = trav_pcs[['CODGEO','CS3_29','NB']]
trav_pcs = trav_pcs.replace(imp.deprecated_codes)
# Rebaptême des colonnes d'indicateurs géographiques
trav_pcs = trav_pcs.rename(columns={'CODGEO':'code_insee'})
trav_pcs['NB'] = pd.to_numeric(trav_pcs['NB'])
    # Passage à une forme utilisable, restriction géographique à l'occitanie
trav_pcs = trav_pcs.pivot_table(index='code_insee', columns='CS3_29', values='NB', aggfunc='sum').loc[com_occitanie].fillna(0)
    # agrégation selon l'exposition à la chaleur et rebaptême pour inférence à la lecture
trav_pcs = trav_pcs[trav_expo_chaleur.index].dot(trav_expo_chaleur).rename({'part_travail_exterieur':'Nombre_travail_exterieur_data'},axis=1)
trav_pcs.to_csv(r"Données traitées\Travail_chaleur_commune.csv")

"""Risque incendie"""
# Lecture avec suppression de l'en-tête
risque_incendie_1 = pd.read_csv(r"Données brutes\DRIAS_feu_meteo_rcp_4.5.txt", sep=';', skiprows=[i for i in range(24)]+[25])
risque_incendie_1 = risque_incendie_1.rename(columns={'# Point':'maille_drias','PÃ©riode':'Période',
                                                      'NORIFM20':'n_jours_IFM_sup_20_data',
                                                      'NORIFM40':'n_jours_IFM_sup_40_data',
                                                      'NORIFM60':'n_jours_IFM_sup_60_data',
                                                      'NORIFMxAV':'IFMx_moy_data',
                                                      'NORIFMxq80':'IFMx_centile80_data'})
# Suppression d'une colonne vie dûe au mauvais formatage du fichier d'entrée
risque_incendie_1 = risque_incendie_1.iloc[:,:-1]
# Conservation de l'horizon 2021-2050
risque_incendie_1 = risque_incendie_1.loc[risque_incendie_1['Période']=='H1']
# Suppression de colonnes
risque_incendie_1 = risque_incendie_1.drop(columns=['Latitude','Longitude','Contexte','Période'])
risque_incendie_1 = risque_incendie_1.set_index('maille_drias')
risque_incendie_2 = pd.read_csv(r"Données brutes\DRIAS_feu_meteo_rcp_4.5_2.txt", sep=';', skiprows=[i for i in range(19)]+[20])
risque_incendie_2 = risque_incendie_2.rename(columns={'# Point':'maille_drias','PÃ©riode':'Période',
                                                      'NORIFMx50':'n_jours_IFMx_sup_50_data',
                                                      'NORIFMx80':'n_jours_IFMx_sup_80_data'})
# Suppression d'une colonne vie dûe au mauvais formatage du fichier d'entrée
risque_incendie_2 = risque_incendie_2.iloc[:,:-1]
# Suppression de colonnes
risque_incendie_2 = risque_incendie_2.drop(columns=['Latitude','Longitude','Contexte','Période'])
risque_incendie_2 = risque_incendie_2.set_index('maille_drias')
risque_incendie = risque_incendie_1.join(risque_incendie_2,how='outer')
risque_incendie.to_csv(r"Données traitées\Risque_incendie_maille_drias.csv")

"""Risque inondation"""

def correct_geometry(gdf):
    """
    Corrige les géométries invalides du jeu de données en entrée

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Données géographiques.

    Returns
    -------
    gdf : gpd.GeoDataFrame
        Données à géométrie corrigée.

    """
    # Reprojection
    if gdf.crs!='EPSG:2154':
        gdf = gdf.to_crs('EPSG:2154')
    # Correction et remplacement des geométries invalides
    correctness = gdf.geometry.is_valid
    if (correctness!=[True for n in range(len(correctness))]).any(): # Au moins une géométrie invalide
        correct_geom = gdf.geometry.loc[~correctness].make_valid()
        correct_geom = correct_geom.buffer(0) # Opération triviale permettant de corriger des polygones dégénérés
        # Remplacement des géométries invalides
            # is_empty permet de vérifier si la géométrie corrigée est triviale. Les géométries triviales ne sont pas ajoutées.
        gdf = gdf.drop(correct_geom.index[correct_geom.geometry.is_empty])
        correct_geom = correct_geom.loc[~correct_geom.geometry.is_empty]
        gdf.loc[correct_geom.index,'geometry']=correct_geom
    return gdf

def get_max_value_tiling(gdf, value_name):
    """
    À partir d'un GeoDataFrame contenant des géométries pouvant s'intersecter,
    et dont une colonne peut donc contenir plusieurs valeurs pour un même point spatial,
    Calcule un GeoDataFrame dont la valeur en chaque point de l'espace est le maximum des valeurs
    du GeoDataframe en entrée, pour ce point de l'espace.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Jeu de données d'entrée, dont le crs est le Lambert-93
    value_name : str
        nom de la colonne pour laquelle on prend les données maximum.
        Ces données doivent être strictement positives

    Returns
    -------
    gpd.GeoDataFrame
        Jeu de données pavant le plan avec les valeurs de gdf[value_name].

    """
    # Rastérisation pour améliorer la vitesse de calcul
        # Taille de la carte qui sera produite à partir des données, pour déterminer la résolution du raster
    minx, maxx, miny, maxy = 0.4e6,0.92e6,6.12e6,6.5e6 # Assume gdf.crs = 'EPSG:2154'
    map_pixels = 6e3*4e3
    area = (maxx - minx) * (maxy - miny)
    pixel_area = area / map_pixels
    resolution = np.sqrt(pixel_area)
    width = max(1, int((maxx - minx) / resolution))
    height = max(1, int((maxy - miny) / resolution))
    transform = Affine.translation(minx, miny) * Affine.scale((maxx - minx)/width, (maxy - miny)/height)
        # Collecte des données
    geometries = []
    values = []
    for idx, row in gdf.iterrows():
        geometries.append(row.geometry)
        values.append(row[value_name])
        # Préparation de la rastérisation
    max_raster = np.zeros((height, width), dtype=np.float32) # Raster vide
            # Groupage des géométries par valeur
    value_groups = {}
    for geom, val in zip(geometries, values):
        if val not in value_groups:
            value_groups[val] = []
        value_groups[val].append(geom)
        # Rastérisation
    for val, geoms in value_groups.items():
        geom_gen = ((geom, 1) for geom in geoms)
        raster = rasterize(
            geom_gen,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        mask = (raster == 1) # Vérifie quelles zones sont couvertes par le raster associé à val
        max_raster[mask] = np.maximum(max_raster[mask], val) # Vaut val si une des géométries couvre la zone, 0 sinon
    results = []
    for val in value_groups.keys():
        mask = (max_raster == val).astype(np.uint8)
        if mask.sum() == 0: # Aucun point n'a val comme valeur max
            continue
        for geom, val_out in shapes(mask, transform=transform): # Vectorisation (un vecteur par valeur)
            if val_out > 0: # Exclusion des zones couvertes par aucun des géométries d'entrée
                geom_shapely = shape(geom)
                if geom_shapely.geom_type == 'Polygon' and geom_shapely.area > pixel_area * 4: # Uniquement polygones de taille non négligeable
                    results.append({
                        'geometry': geom_shapely,
                        value_name: val
                    })
    return gpd.GeoDataFrame(results, crs=gdf.crs)

# TODO: Le code ci-dessous est très lourd et est commenté pour ne pas être exécuté à chaque exécution du fichier. Il doit être décommenté lors de la première exécution du fichier
# Note : le suralea ne correspond pas aux territoires plus exposés du fait de la présence d'un ouvrage de protection
# Il correspond aux territoires pour lesquels la rupture d'un ouvrage engendre un aléa plus grand que l'absence de l'ouvrage
# Note : les inondations par remontées de nappe ne sont pas fournies
# inondation_names=['n_carte_inond_s.shp',
#              'n_inondable_01_01for_s.shp','n_inondable_01_02moy_s.shp','n_inondable_01_04fai_s.shp',
#              'n_inondable_02_01for_s.shp','n_inondable_02_02moy_s.shp','n_inondable_02_04fai_s.shp',
#              'n_inondable_03_01for_s.shp','n_inondable_03_02moy_s.shp','n_inondable_03_04fai_s.shp',
#              'n_soust_inond_s.shp','n_suralea_s.shp'
#              ] # Le premier chiffre correspond au type d'inondation, le deuxième à la probabilité d'occurence (cf. https://files.georisques.fr/di_2020/COVADIS_standard_DI_v2.1.pdf)
# inondation_paths = [os.path.join(r"Données brutes\tri_2020_sig_di",name) for name in inondation_names]
# # Importation
# inondation_gdfs = [gpd.read_file(path) for path in inondation_paths]
# # Correction
# inondation_gdfs = [correct_geometry(gdf) for gdf in inondation_gdfs]
# # Filtrage géographique sur la région Occitanie
# inondation_gdfs = [gdf.clip(reg) for gdf in inondation_gdfs]
# # Fusion des aléas
# inondation_gdfs.append(pd.concat(inondation_gdfs[1:10]))
# # Écriture (aucune zone occitane n'est exposée au ruissellement dans ces données)
# # Conservation des scénarios moyens et de la superposition des scénarios
# for n in {0}:
#     inondation_gdfs[n] = inondation_gdfs[n].loc[:,['typ_inond1','scenario','geometry']]
# for n in {2,8,10,11}:
#     inondation_gdfs[n] = inondation_gdfs[n].loc[:,['scenario','geometry']]
# for n in {12}:
#     inondation_gdfs[n] = inondation_gdfs[n].loc[:,['typ_inond','scenario','geometry']]
# for n in {0,2,8,10,11,12}:
#     inondation_gdfs[n] = inondation_gdfs[n].loc[inondation_gdfs[n]['scenario']!='03Mcc'] # Risque à court terme : exclusion du scénario prenant compte du changement climatique
#     inondation_gdfs[n]['niv_alea'] = inondation_gdfs[n]['scenario'].map({'04Fai':1,'02Moy':2,'01For':3}) # Probabilités d'occurence interprétées en niveau d'aléa
#     inondation_gdfs[n] = inondation_gdfs[n].loc[inondation_gdfs[n].geometry.area>1] # Suppression des géométries de taille négligeable
# inondation_gdfs_ = [inondation_gdfs[i] for i in[0,2,8,10,11,12]]
# inondation_gdfs_ = [pd.concat([df,reg])[['geometry','niv_risque']].fillna(0) for df in inondation_gdfs_] # Ajout de la région à valeur d'aléa 0
# inondation_gdfs_ = [get_max_value_tiling(df,'niv_risque') for df in inondation_gdfs_]
# inondation_gdfs_ = [df.clip(reg) for df in inondation_gdfs_]
# inondation_gdfs_[0].to_file(r"Données traitées\Inond_surf.gpkg")
# inondation_gdfs_[1].to_file(r"Données traitées\Inond_surf_debordement.gpkg")
# inondation_gdfs_[2].to_file(r"Données traitées\Inond_surf_submar.gpkg")
# inondation_gdfs_[3].to_file(r"Données traitées\Inond_surf_soust_inondation.gpkg")
# inondation_gdfs_[4].to_file(r"Données traitées\Inond_surf_suralea.gpkg")
# inondation_gdfs_[5].to_file(r"Données traitées\Inond_surf_total.gpkg")

"""Retrait Gonflement des Argiles"""
# Lecture des fichiers département par département
rga = [gpd.read_file(rf"Données brutes\zonage_RGA\AleaRG_2025_{dep}_L93\AleaRG_2025_{dep}_L93.shp")
       for dep in ['09','11','12','30','31','32','34','46','48','65','66','81','82']]
# Concaténation pour obtention des données régionales
rga = pd.concat(rga)
# Tri des colonnes et ajout de la région à valeur d'aléa 0
rga = pd.concat([rga,reg])
rga = rga.drop(columns=['gid','insee_dep','surf_m2','nom','code']).fillna(0)
# Goupement des géométries contigues par valeur (aux frontières départementales seulement, si les données sont bien structurées)
rga = rga.dissolve(by='niveau').reset_index()
rga = rga.explode() # Passage à des shapely.Polygon à la place des shapely.MultiPolygon pour amélioration de la vitesse de calcul
rga = get_max_value_tiling(rga,'niveau')
rga = rga.clip(reg)
rga.to_file(r"Données traitées\Zonage_RGA.gpkg")

"""Localisation des ICPE"""
# Lecture
icpe = pd.read_csv(r"Données brutes\installations_industrielles_georisques.csv",sep=';',
                   dtype={'num_dep':'str','cd_postal':'str','cd_insee':'str','code_naf':'str'})
# Filtrage des colonnes et des communes
icpe = icpe[['cd_insee','lib_regime','lib_seveso','ied']]
icpe = icpe.rename(columns={'cd_insee':'code_insee'})
icpe = icpe.loc[icpe['code_insee'].isin(com_occitanie)]
icpe = icpe.loc[icpe['lib_regime'].isin(['Autorisation','Enregistrement'])] # Conservation des ICPE uniquement
icpe.iloc[:,5:] = icpe.iloc[:,5:].apply(pd.to_numeric)
 # Codage d'un niveau d'aléa à patir des seuils réglementaires
icpe['id_regime'] = icpe['lib_regime'].map({'Autorisation':2,'Enregistrement':1})
icpe['id_seveso'] = icpe['lib_seveso'].map({'Non Seveso':0, np.nan:0, 'Seveso seuil haut':2, 'Seveso seuil bas':1})
icpe['id_risque_self'] = icpe['id_regime'] + icpe['id_seveso'] + icpe['ied']
icpe = icpe.loc[:,['code_insee','id_risque_self']]
icpe = icpe.groupby('code_insee').sum() # Niveau d'aléa par commune et non par installation
icpe = icpe.reindex(com_occitanie).fillna(0)
icpe = gpd.GeoDataFrame(data=icpe['id_risque_self'],index=icpe.index,geometry=com_occ.set_index('code_insee')['geometry'],crs=2154)
# Percolation du risque vers les voisins (On prend 1km comme rayon caractéristique (rayon des effets de Seveso))
# On pourrait utiliser une exponentielle plutôt que ces seuils discrets, pour un certain coût de calcul
# Pour chaque commune, on trouve les commune proches et on leur attribue une part du niveau d'aléa de la commune source
icpe['centroids'] = icpe.geometry.centroid
icpe['buffer_1km'] = icpe.geometry.buffer(1000)
icpe['buffer_2km'] = icpe.geometry.buffer(2000)
within_2km = gpd.sjoin(
    icpe[['buffer_2km']].set_geometry('buffer_2km'),
    icpe[['centroids']].set_geometry('centroids'),
    how='inner',
    predicate='contains'
)
within_2km['weighted_2km'] = within_2km.index.map(icpe['id_risque_self'])
weighted_2km = within_2km.groupby('code_insee_right')['weighted_2km'].sum()
icpe['pts_2km'] = weighted_2km
within_1km = gpd.sjoin(
    icpe[['buffer_1km']].set_geometry('buffer_1km'),
    icpe[['centroids']].set_geometry('centroids'),
    how='inner',
    predicate='contains'
)
within_1km['weighted_1km'] = within_1km.index.map(icpe['id_risque_self'])
weighted_1km = within_1km.groupby('code_insee_right')['weighted_1km'].sum()
icpe['pts_1km'] = weighted_1km
icpe['id_risque_total'] = icpe['pts_2km']\
                        + icpe['pts_1km']\
                        + icpe['id_risque_self']
icpe = icpe.loc[:,['id_risque_self','id_risque_total']]
icpe = icpe.rename(
    columns=dict(zip(icpe.columns,
                     [icpe.columns[n]+"_data"for n in range(icpe.shape[1])])))
icpe.to_csv(r"Données traitées\ICPE_commune.csv")

"""Sites et sols pollués"""
# Calcul aux échelles IRIS et commune
# Même méthode que pour les ICPE
for groupvars in [(iris_occitanie,iris_occ,'code_iris','IRIS'),
                  (com_occitanie,com_occ,'code_insee','commune')]:
    ssp = pd.read_excel(r"Données brutes\SSP_georisques.xlsx",sheet_name=0)
    ssp = gpd.GeoDataFrame(ssp,
                           geometry=gpd.points_from_xy(ssp['Coordonnée X'],ssp['Coordonnée y']),
                           crs=4326)
    ssp = ssp.to_crs(2154)
    ssp = ssp.sjoin(groupvars[1])[[groupvars[2],'geometry']].groupby(groupvars[2]).count()
    ssp.columns=['nb_poll_sol_data']
    ssp = ssp.reindex(groupvars[0]).join(groupvars[1].set_index(groupvars[2]))\
        .fillna(0)[['nb_poll_sol_data','geometry']].reset_index()
    ssp = gpd.GeoDataFrame(ssp)
    ssp['centroids'] = ssp.geometry.centroid
    ssp['buffer_1km'] = ssp.geometry.buffer(1000)
    ssp['buffer_2km'] = ssp.geometry.buffer(2000)
    ssp['pts_2km'] = ssp[groupvars[2]]\
        .map(ssp[['buffer_2km']].set_geometry('buffer_2km')\
        .sjoin(ssp[[groupvars[2],'centroids']].set_geometry('centroids'))\
        .groupby(groupvars[2]).size())\
        .fillna(0)
    ssp['pts_1km'] = ssp[groupvars[2]]\
        .map(ssp[['buffer_1km']].set_geometry('buffer_1km')\
        .sjoin(ssp[[groupvars[2],'centroids']].set_geometry('centroids'))\
        .groupby(groupvars[2]).size())\
        .fillna(0)
    ssp['poll_score_data'] = ssp['pts_2km']\
                            + 2*ssp['pts_1km']\
                            + 3*ssp['nb_poll_sol_data']
    ssp = ssp[[groupvars[2],'nb_poll_sol_data','poll_score_data']]
    ssp.to_csv(rf"Données traitées\Poll_sol_{groupvars[3]}.csv")

"""Registre Parcellaire Graphique"""
rpg_parc = gpd.read_file(r"Données brutes\RPG_2024_lamb93\RPG_Parcelles.gpkg")
rpg_bio = gpd.read_file(r"Données brutes\RPG_2024_lamb93\RPG_BIO.gpkg")
rpgs = {'Surfaces_agri':rpg_parc,'Surfaces_agri_bio':rpg_bio}
for rpg_name in ['Surfaces_agri','Surfaces_agri_bio']:
    rpg = gpd.overlay(rpgs[rpg_name], iris_occ) # Passage des données à l'iris pour calcul des surfaces cultivées
    rpg['surface_agri'] = rpg.geometry.area/1e6
    rpg = rpg.dissolve(by='code_iris',
                       aggfunc={'surface':'first','surface_agri':'sum'})
    rpg['part_surface_agri'] = rpg['surface_agri']/rpg['surface']
    rpg = iris_occ.set_index('code_iris').join(rpg[['surface_agri','part_surface_agri']]).fillna(0)
    rpg = rpg[['surface_agri','part_surface_agri','surface']]
    rpg.columns = ['surface_agri_data','part_surface_agri_data','surface_data']
    rpg.to_csv(rf"Données traitées\{rpg_name}.csv")
    
"""Pollution de l'air et de l'eau"""
# Même méthode que pour les ICPE et la pollution des sols, avec un rayon caractéristique de 10 km
air_pol = pd.read_csv(r"Données brutes\Registre Français des émissions polluantes\2024\emissions.csv", sep=';')
air_pol = air_pol.loc[air_pol['code_region']==76]
eau_pol = air_pol.loc[air_pol['milieu'].str.contains('Eau')]
air_pol = air_pol.loc[air_pol['milieu']=='Air']
air_pol = air_pol.drop_duplicates(subset='identifiant')
air_pol = air_pol[['code_insee','code_postal']].groupby('code_insee').count()
air_pol.columns=['nb_poll_air_data']
air_pol = air_pol.reindex(com_occitanie).join(com_occ.set_index('code_insee'))\
    .fillna(0)[['nb_poll_air_data','geometry']].reset_index()
air_pol = gpd.GeoDataFrame(air_pol)
air_pol['centroids'] = air_pol.geometry.centroid
air_pol['buffer_10km'] = air_pol.geometry.buffer(10000)
air_pol['buffer_20km'] = air_pol.geometry.buffer(20000)
air_pol['pts_20km'] = air_pol['code_insee']\
    .map(air_pol[['code_insee', 'buffer_20km']].set_index('code_insee').set_geometry('buffer_20km')\
    .sjoin(air_pol[['code_insee', 'centroids', 'nb_poll_air_data']].set_geometry('centroids'))\
    .groupby(level=0)['nb_poll_air_data'].sum())\
    .fillna(0)
air_pol['pts_10km'] = air_pol['code_insee']\
    .map(air_pol[['code_insee', 'buffer_10km']].set_index('code_insee').set_geometry('buffer_10km')\
    .sjoin(air_pol[['code_insee', 'centroids', 'nb_poll_air_data']].set_geometry('centroids'))\
    .groupby(level=0)['nb_poll_air_data'].sum())\
    .fillna(0)
air_pol['poll_score_data'] = air_pol['pts_20km']\
                         + air_pol['pts_10km']\
                         + air_pol['nb_poll_air_data']
air_pol = air_pol[['code_insee','nb_poll_air_data','poll_score_data']]
air_pol.to_csv(r"Données traitées\Poll_air_commune.csv")

eau_pol = eau_pol.drop_duplicates(subset='identifiant')
eau_pol = eau_pol[['code_insee','code_postal']].groupby('code_insee').count()
eau_pol.columns=['nb_poll_eau_data']
eau_pol = eau_pol.reindex(com_occitanie).join(com_occ.set_index('code_insee'))\
    .fillna(0)[['nb_poll_eau_data','geometry']].reset_index()
eau_pol = gpd.GeoDataFrame(eau_pol)
eau_pol['centroids'] = eau_pol.geometry.centroid
eau_pol['buffer_10km'] = eau_pol.geometry.buffer(10000)
eau_pol['buffer_20km'] = eau_pol.geometry.buffer(20000)
eau_pol['pts_20km'] = eau_pol['code_insee']\
    .map(eau_pol[['code_insee', 'buffer_20km']].set_index('code_insee').set_geometry('buffer_20km')\
    .sjoin(eau_pol[['code_insee', 'centroids', 'nb_poll_eau_data']].set_geometry('centroids'))\
    .groupby(level=0)['nb_poll_eau_data'].sum())\
    .fillna(0)
eau_pol['pts_10km'] = eau_pol['code_insee']\
    .map(eau_pol[['code_insee', 'buffer_10km']].set_index('code_insee').set_geometry('buffer_10km')\
    .sjoin(eau_pol[['code_insee', 'centroids', 'nb_poll_eau_data']].set_geometry('centroids'))\
    .groupby(level=0)['nb_poll_eau_data'].sum())\
    .fillna(0)
eau_pol['poll_score_data'] = eau_pol['pts_20km']\
                        + 2*eau_pol['pts_10km']\
                        + 3*eau_pol['nb_poll_eau_data']
eau_pol = eau_pol[['code_insee','nb_poll_eau_data','poll_score_data']]
eau_pol.to_csv(r"Données traitées\Poll_eau_commune.csv")

"""Bruit"""
bruit_paths = [str(f) for f in Path(r"Données brutes\mesure bruit Noise Capture").iterdir() 
               if f.is_file() and f.name.endswith('areas.geojson')]
# Importation
bruit_gdfs = [gpd.read_file(path,engine='fiona') for path in bruit_paths]
bruit_gdfs = pd.concat(bruit_gdfs)[['laeq','geometry']].to_crs('EPSG:2154')
# Pour chaque IRIS, on prend la moyenne des mesures
bruit_gdfs = bruit_gdfs.sjoin(iris_occ)
bruit_gdfs = bruit_gdfs.loc[bruit_gdfs['laeq']>0] # niveau sonore équivalent
bruit_gdfs = bruit_gdfs.drop(columns=['geometry','index_right','surface'])
bruit_gdfs = bruit_gdfs.groupby('code_iris').mean()
bruit_gdfs.columns = ['laeq_data']
bruit_gdfs.to_csv(r"Données traitées\Bruit_IRIS.csv")

"""Annuaire de l'éducation"""
# Importation
annu_educ = pd.read_csv(r"Données brutes\fr-en-annuaire-education.csv", sep=';')
# Filtrage de la région et des établissements
annu_educ = annu_educ.loc[annu_educ['Code_region']==76]
annu_educ = annu_educ.loc[np.isin(annu_educ['Type_etablissement'],['Ecole','Collège','Lycée'])]
annu_educ = annu_educ.loc[annu_educ['etat']=='OUVERT']
# Regroupement à l'IRIS du nombre d'établissements à partir des coordonnées fournies
annu_educ = gpd.GeoDataFrame(annu_educ,
                             geometry=gpd.points_from_xy(annu_educ['coordX_origine'],annu_educ['coordY_origine']),
                             crs=2154)
annu_educ = annu_educ[['geometry']] #Pas de distinction entre les différents niveaux scolaires
annu_educ = annu_educ.sjoin(iris_occ)
annu_educ = annu_educ.groupby('code_iris').count()
annu_educ = annu_educ[['geometry']].rename(columns={'geometry':'nb_etab_educ_data'})
annu_educ = annu_educ.reindex(iris_occitanie).fillna(0)
annu_educ.to_csv(r"Données traitées\Annuaire_education_IRIS.csv")

"""Zones de répartition des eaux"""
# Importation
zre = gpd.read_file(r"Données brutes\ZRE.gpkg")
# Filtrage des coordonnées régionales
zre = zre.clip(reg)
# Observation de la présence ou non d'une ZRE
zre['tag']=zre['gid']/zre['gid']
zre = zre[['tag','geometry']]
zre.to_file(r"Données traitées\Zones_de_répartition_des_eaux.gpkg")

"""Logement (Insee)"""
# Importation
logement_insee = pd.read_csv(r"Données brutes\base-ic-logement-2022_csv\base-ic-logement-2022.csv",
                              sep=';',dtype='str')
# Mise à jour des codes IRIS ayant possiblement changé
logement_insee['IRIS'] = logement_insee['IRIS'].replace(imp.deprecated_codes)
# Filtrage sur la région et les colonnes
logement_insee = logement_insee.loc[np.isin(logement_insee["IRIS"],iris_occitanie)]
logement_insee = logement_insee.rename(columns={'IRIS':'code_iris'})\
                               .set_index('code_iris')\
                               .drop(columns=['COM','TYP_IRIS','LAB_IRIS'])
logement_insee = logement_insee.apply(pd.to_numeric)
# Estimation de la surface moyenne des logements
logement_insee['Surf_moy_RP_infer'] = (logement_insee['P22_RP_M30M2']*20
                                       + logement_insee['P22_RP_3040M2']*35
                                       + logement_insee['P22_RP_4060M2']*50
                                       + logement_insee['P22_RP_6080M2']*70
                                       + logement_insee['P22_RP_80100M2']*90
                                       + logement_insee['P22_RP_100120M2']*110
                                       + logement_insee['P22_RP_120M2P']*150)\
                                       /logement_insee['P22_RP'].replace(0,1)
# Filtrage des colonnes
logement_insee = logement_insee[['P22_RP','P22_RPMAISON','P22_RPAPPART',
                                 'P22_RP_PROP','P22_RP_LOC','Surf_moy_RP_infer',
                                 'P22_RP_CGAZV','P22_RP_CFIOUL','P22_RP_CELEC',
                                 'P22_RP_CGAZB','P22_RP_CAUT']].reset_index()
logement_insee = logement_insee.rename(
    columns=dict(zip(logement_insee.columns[1:],
                     [logement_insee.columns[n]+"_data"for n in range(1,logement_insee.shape[1])])))
logement_insee.to_csv(r"Données traitées\Logement_2022_IRIS.csv", index=False)
# Données de 2021, privilégiées à 2022, mais avec moins d'indicateurs, même méthode
logement_insee = pd.read_csv(r"Données brutes\base-ic-logement-2021_csv\base-ic-logement-2021.csv",
                              sep=';',dtype='str')
logement_insee['IRIS'] = logement_insee['IRIS'].replace(imp.deprecated_codes)
logement_insee = logement_insee.loc[np.isin(logement_insee["IRIS"],iris_occitanie)]
logement_insee = logement_insee.rename(columns={'IRIS':'code_iris'})\
                               .set_index('code_iris')\
                               .drop(columns=['COM','TYP_IRIS','LAB_IRIS'])
logement_insee = logement_insee.apply(pd.to_numeric)
logement_insee['Surf_moy_RP_infer'] = (logement_insee['P21_RP_M30M2']*20
                                       + logement_insee['P21_RP_3040M2']*35
                                       + logement_insee['P21_RP_4060M2']*50
                                       + logement_insee['P21_RP_6080M2']*70
                                       + logement_insee['P21_RP_80100M2']*90
                                       + logement_insee['P21_RP_100120M2']*110
                                       + logement_insee['P21_RP_120M2P']*150)\
                                       /logement_insee['P21_RP'].replace(0,1)
logement_insee = logement_insee[['P21_RP','P21_RPMAISON','P21_RPAPPART',
                                 'P21_RP_PROP','P21_RP_LOC','Surf_moy_RP_infer']].reset_index()
logement_insee = logement_insee.rename(
    columns=dict(zip(logement_insee.columns[1:],
                     [logement_insee.columns[n]+"_data"for n in range(1,logement_insee.shape[1])])))
logement_insee.to_csv(r"Données traitées\Logement_2021_IRIS.csv", index=False)

"""DPE (Données Performance Energétique du Parc Résidentiel)"""
# Documentation du code : cf enquête Couples - Familles - Ménages
dpe_iris = pd.read_csv(r"Données brutes\classe-dpe-fg-de-par-iris.csv")
dpe_iris['Code IRIS'] = dpe_iris['Code IRIS'].replace(imp.deprecated_codes)
dpe_iris = dpe_iris.loc[np.isin(dpe_iris["Code IRIS"],iris_occitanie)]
dpe_iris = dpe_iris.fillna(0) # Marquage de 0 comme donnée manquante sur la carte
dpe_iris['Part_de_D_et_E'] = dpe_iris['Résidences Principales classe D ou E']/dpe_iris['Résidences Principales']
dpe_iris['Part_de_F_et_G'] = dpe_iris['Résidences Principales classe F ou G']/dpe_iris['Résidences Principales']
dpe_iris = dpe_iris.rename(columns={'Code IRIS':'code_iris'})
dpe_iris = dpe_iris.rename(
    columns=dict(zip(dpe_iris.columns[1:],
                     [dpe_iris.columns[n]+"_data"for n in range(1,dpe_iris.shape[1])])))
dpe_iris.to_csv(r"Données traitées\DPE_IRIS.csv",index=False)

"""DPE par département (SDES,2022)"""
# Importation
dpe_dep = pd.read_excel(r"Données brutes\parc_logements_dpe_2022.xlsx",sheet_name=4).iloc[:-4]
# Filtrage de la région
dpe_dep = dpe_dep.loc[np.isin(dpe_dep['Département'],liste_dep_occ)]
dpe_iris.columns = dpe_iris.columns.str.removesuffix('_data')
# Jointure avec les données à l'IRIS
dpe_iris['dep'] = dpe_iris['code_iris'].str[:2]
dpe_iris = dpe_iris.join(dpe_dep.set_index('Département'), on='dep')
# Estimation de la répartition dans chaque classe à l'IRIS
dpe_iris['abc'] = dpe_iris['Résidences Principales']\
                - dpe_iris['Résidences Principales classe D ou E']\
                - dpe_iris['Résidences Principales classe F ou G']
dpe_iris['Résidences_principales_A_infer'] = dpe_iris['abc']*dpe_iris['A']/(dpe_iris['A']+dpe_iris['B']+dpe_iris['C'])
dpe_iris['Résidences_principales_B_infer'] = dpe_iris['abc']*dpe_iris['B']/(dpe_iris['A']+dpe_iris['B']+dpe_iris['C'])
dpe_iris['Résidences_principales_C_infer'] = dpe_iris['abc']*dpe_iris['C']/(dpe_iris['A']+dpe_iris['B']+dpe_iris['C'])
dpe_iris['Résidences_principales_D_infer'] = dpe_iris['Résidences Principales classe D ou E']*dpe_iris['D']/(dpe_iris['D']+dpe_iris['E'])
dpe_iris['Résidences_principales_E_infer'] = dpe_iris['Résidences Principales classe D ou E']*dpe_iris['E']/(dpe_iris['D']+dpe_iris['E'])
dpe_iris['Résidences_principales_F_infer'] = dpe_iris['Résidences Principales classe F ou G']*dpe_iris['F']/(dpe_iris['F']+dpe_iris['G'])
dpe_iris['Résidences_principales_G_infer'] = dpe_iris['Résidences Principales classe F ou G']*dpe_iris['G']/(dpe_iris['F']+dpe_iris['G'])
# Filtrage des colonnes
dpe_iris = dpe_iris[['code_iris','Résidences Principales',
                     'Résidences_principales_A_infer',
                     'Résidences_principales_B_infer',
                     'Résidences_principales_C_infer',
                     'Résidences_principales_D_infer',
                     'Résidences_principales_E_infer',
                     'Résidences_principales_F_infer',
                     'Résidences_principales_G_infer']]
for lettre in ['A','B','C','D','E','F','G']:
    dpe_iris[f"Part_RP_{lettre}"]=dpe_iris[f'Résidences_principales_{lettre}_infer']/dpe_iris['Résidences Principales']
consos_par_dpe = pd.DataFrame(index=dpe_iris.columns[2:],data=[83,83,83*1.8,83*2.59,83*3.48,83*4.51,83*6.6]*2) # https://cae-eco.fr/static/pdf/focus-103-dpe-230110.pdf
dpe_iris['conso_m2_nb_log'] = dpe_iris.iloc[:,2:9].dot(consos_par_dpe.iloc[:7]) # Consommation au m²*nombre de logements (kWh)
dpe_iris['conso_m2'] = dpe_iris.iloc[:,9:-1].dot(consos_par_dpe.iloc[7:]) # Consommation au m² moyenne (kWh)
dpe_iris = dpe_iris.join(logement_insee[['Surf_moy_RP_infer_data','code_iris']].set_index('code_iris'),on='code_iris')
dpe_iris['conso_DPE_GWhAn_infer'] = dpe_iris['conso_m2_nb_log']*dpe_iris['Surf_moy_RP_infer_data']/1e6 # Consommation totale par IRIS
dpe_iris['conso_DPE_MWhAnlog_infer'] = dpe_iris['conso_m2']*dpe_iris['Surf_moy_RP_infer_data']/1e3 # Consommation moyenne par logement
dpe_iris = dpe_iris.drop(columns='Surf_moy_RP_infer_data')
dpe_iris = dpe_iris.rename(
    columns=dict(zip(dpe_iris.columns[1:],
                     [dpe_iris.columns[n]+"_data"for n in range(1,dpe_iris.shape[1])])))
dpe_iris.to_csv(r"Données traitées\DPE_infer_IRIS.csv",index=False)

"""IRVE"""
# Importation
loc_irve = gpd.read_file("Données brutes\consolidation-etalab-schema-irve-statique-v-2.3.1-20260506.geojson")
loc_irve = loc_irve.to_crs('EPSG:2154')
# Filtrage sur la région
loc_irve = loc_irve.clip(reg)
# Appariement à l'IRIS et groupement pour compte du nombre de points de charge par IRIS et condition d'accès
loc_irve = loc_irve.sjoin(iris_occ)
loc_irve['condition_acces'] = loc_irve['condition_acces'].str.replace({'č':'è','':'è'}) # Caractères corrompus
loc_irve['nbre_pdc'] = loc_irve['nbre_pdc'].apply(pd.to_numeric)
loc_irve = loc_irve[['condition_acces','code_iris','nbre_pdc']].groupby(by=['condition_acces','code_iris']).sum()
loc_irve.columns=['nb_bornes']
loc_irve = loc_irve.pivot_table(values='nb_bornes',index='code_iris',columns=['condition_acces'],aggfunc='sum').fillna(0)
loc_irve['Total'] = loc_irve.loc[:,'Accès libre']+loc_irve.loc[:,'Accès réservé']
loc_irve = loc_irve.reindex(iris_occitanie).fillna(0)
loc_irve = loc_irve.rename(
    columns=dict(zip(loc_irve.columns,
                     [loc_irve.columns[n]+"_data"for n in range(loc_irve.shape[1])])))
loc_irve.to_csv(r"Données traitées\IRVE_IRIS.csv")

"""Part de VE immatriculés entre 2019 et 2024"""
# Importation
part_ve = pd.read_csv(r"Données brutes\part-marche-VE-commune.csv")
part_ve = part_ve[['geocode_commune','type_vehicule','numerateur','denominateur']].rename(columns={'geocode_commune':'code_insee'})
part_ve = part_ve[np.isin(part_ve['type_vehicule'],['DEUX-ROUES MOTORISES','VP','VUL','PL'])][['code_insee','numerateur','denominateur']]
# Somme par commune
part_ve = part_ve.groupby(by='code_insee').sum()
part_ve = part_ve.loc[np.isin(part_ve.index,com_occ)] # Filtrage sur la région
part_ve['part_ve'] = part_ve['numerateur']/part_ve['denominateur']
part_ve = part_ve.rename(
    columns=dict(zip(part_ve.columns,
                     [part_ve.columns[n]+"_data"for n in range(part_ve.shape[1])])))
part_ve.to_csv(r"Données traitées\part_ve_immat_commune.csv")
"""Part de VE en circulation (2021,2025)"""
# Importation
part_ve = pd.read_csv(r"Données brutes\Donnees-sur-le-parc-de-vehicules-au-niveau-communal.2025-09.csv",sep=';',
                      dtype={'COMMUNE_CODE':'str'}).rename(columns={'COMMUNE_CODE':'code_insee'})
# Filtrage sur la région et les colonnes
part_ve = part_ve.loc[part_ve['code_insee'].isin(com_occitanie)]
part_ve['CARBURANT'] = part_ve['CARBURANT'].replace({'Diesel':'thermique','Essence':'thermique','Gaz':'thermique',
                                                     'Diesel HNR':'hybride','Essence HNR':'hybride','Gaz HNR':'hybride',
                                                     'Diesel HR':'hybride','Essence HR':'hybride','Gaz HR':'hybride'})
# Somme par commune, motorisation et statut (particulier ou professionnel) de l'utilisateur
part_ve = part_ve[['code_insee','CARBURANT','STATUT_UTILISATEUR','PARC_2021','PARC_2025']]\
        .groupby(by=['code_insee','CARBURANT','STATUT_UTILISATEUR']).sum().reset_index()
part_ve = part_ve.loc[part_ve['CARBURANT']!='Inconnu']
# Passage à une forme facilement manipulable
part_ve = part_ve.melt(
    id_vars=['code_insee', 'CARBURANT', 'STATUT_UTILISATEUR'],
    value_vars=['PARC_2021','PARC_2025'],
    var_name='year',
    value_name='value'
)
part_ve = part_ve.pivot_table(
    index='code_insee',
    columns=['year', 'CARBURANT', 'STATUT_UTILISATEUR'],
    values='value',
    aggfunc='sum'
).fillna(0)
part_ve.columns = ['_'.join(col).rstrip('_') for col in part_ve.columns.values]
# Calcul des sommes
for year in ['2021','2025']:
    for carb in ['thermique','hybride','Electrique','Hydrogène et autre ZE']:
        part_ve[f"{carb}_{year}"] = part_ve[[col for col in part_ve.columns if f'PARC_{year}_{carb}' in col]].sum(axis=1)
part_ve['PARC_2021'] = part_ve.iloc[:,-8:-4].sum(axis=1)
part_ve['PARC_2025'] = part_ve.iloc[:,-5:-1].sum(axis=1)
for year in ['2021','2025']:
    for carb in ['thermique','hybride','Electrique','Hydrogène et autre ZE']:
        part_ve[f"part_{carb}_{year}"] = part_ve[f"{carb}_{year}"]/part_ve[f"PARC_{year}"]

part_ve = part_ve.rename(
    columns=dict(zip(part_ve.columns,
                     [part_ve.columns[n]+"_data"for n in range(part_ve.shape[1])])))
part_ve.to_csv(r"Données traitées\part_ve_circul_commune.csv")

"""Mobilités pendulaires"""
# Importation
mob_etude = pd.read_csv(r"Données brutes\base-flux-mobilite-domicile-lieu-etude-2022.csv", sep=';',
                        dtype={'CODGEO':'string'})
# Filtrage sur la région (lieu de vie et d'études)
mob_etude = mob_etude.loc[mob_etude['CODGEO'].isin(com_occitanie)]
mob_etude = mob_etude.loc[mob_etude['DCETU'].isin(com.index)]
mob_etude = mob_etude.reset_index().drop(columns='index')
# Distance domicile-études
mob_etude['distance'] = com.loc[mob_etude['CODGEO'],'centroids'].distance(com.loc[mob_etude['DCETU'],'centroids'], align=False).reset_index().drop(columns='code_insee')
mob_etude['distance'] = mob_etude['distance']/1000 # en km
mob_etude['NB_tot'] = mob_etude['CODGEO'].map(mob_etude[['CODGEO','NBFLUX_C22_SCOL02P']].groupby(by='CODGEO').sum().iloc[:,0])
mob_etude['dist_pond'] = mob_etude['distance']*mob_etude['NBFLUX_C22_SCOL02P']/mob_etude['NB_tot'] # Distance moyenne pour les habitants d'une commune
mob_etude_ext = mob_etude[['CODGEO','NBFLUX_C22_SCOL02P']].groupby(by='CODGEO').sum().rename(columns={'NBFLUX_C22_SCOL02P':'NB_tot'})
mob_etude_ext['NB_int_dep'] = mob_etude.loc[mob_etude['DCETU'].str[:2]==mob_etude['CODGEO'].str[:2]]\
                              .groupby(by='CODGEO').sum()['NBFLUX_C22_SCOL02P'] # Personnes étudiant dans leur département
mob_etude_ext['NB_int_com'] = mob_etude.loc[mob_etude['DCETU']==mob_etude['CODGEO']]\
                              .groupby(by='CODGEO').sum()['NBFLUX_C22_SCOL02P'] # Personnes étudiant dans leur commune
mob_etude_ext = mob_etude_ext.fillna(0)
mob_etude_ext['NB_ext_com'] = mob_etude_ext['NB_tot'] - mob_etude_ext['NB_int_com']
mob_etude_ext['NB_ext_dep'] = mob_etude_ext['NB_tot'] - mob_etude_ext['NB_int_dep']
mob_etude_ext['Part_ext_com'] = mob_etude_ext['NB_ext_com']/mob_etude_ext['NB_tot']
mob_etude_ext['Part_ext_dep'] = mob_etude_ext['NB_ext_dep']/mob_etude_ext['NB_tot']
mob_etude_ext['Distance_moyenne'] = mob_etude[['CODGEO','dist_pond']].groupby(by='CODGEO').sum()
mob_etude_ext = mob_etude_ext.rename(
    columns=dict(zip(mob_etude_ext.columns,
                     [mob_etude_ext.columns[n]+"_data"for n in range(mob_etude_ext.shape[1])])))
mob_etude_ext.to_csv(r"Données traitées\mobilité_études_commune.csv")
# Même méthode qu'au-dessus
mob_travail = pd.read_csv(r"Données brutes\base-flux-mobilite-domicile-lieu-travail-2022.csv", sep=';',
                        dtype={'CODGEO':'string'})
mob_travail = mob_travail.loc[mob_travail['CODGEO'].isin(com_occitanie)]
mob_travail = mob_travail.loc[mob_travail['DCLT'].isin(com.index)]
mob_travail = mob_travail.reset_index().drop(columns='index')
mob_travail['distance'] = com.loc[mob_travail['CODGEO'],'centroids'].distance(com.loc[mob_travail['DCLT'],'centroids'], align=False).reset_index().drop(columns='code_insee')
mob_travail['distance'] = mob_travail['distance']/1000
mob_travail['NB_tot'] = mob_travail['CODGEO'].map(mob_travail[['CODGEO','NBFLUX_C22_ACTOCC15P']].groupby(by='CODGEO').sum().iloc[:,0])
mob_travail['dist_pond'] = mob_travail['distance']*mob_travail['NBFLUX_C22_ACTOCC15P']/mob_travail['NB_tot']
mob_travail_ext = mob_travail[['CODGEO','NBFLUX_C22_ACTOCC15P']].groupby(by='CODGEO').sum().rename(columns={'NBFLUX_C22_ACTOCC15P':'NB_tot'})
mob_travail_ext['NB_int_dep'] = mob_travail.loc[mob_travail['DCLT'].str[:2]==mob_travail['CODGEO'].str[:2]]\
                              .groupby(by='CODGEO').sum()['NBFLUX_C22_ACTOCC15P']
mob_travail_ext['NB_int_com'] = mob_travail.loc[mob_travail['DCLT']==mob_travail['CODGEO']]\
                              .groupby(by='CODGEO').sum()['NBFLUX_C22_ACTOCC15P']
mob_travail_ext = mob_travail_ext.fillna(0)
mob_travail_ext['NB_ext_com'] = mob_travail_ext['NB_tot'] - mob_travail_ext['NB_int_com']
mob_travail_ext['NB_ext_dep'] = mob_travail_ext['NB_tot'] - mob_travail_ext['NB_int_dep']
mob_travail_ext['Part_ext_com'] = mob_travail_ext['NB_ext_com']/mob_travail_ext['NB_tot']
mob_travail_ext['Part_ext_dep'] = mob_travail_ext['NB_ext_dep']/mob_travail_ext['NB_tot']
mob_travail_ext['Distance_moyenne'] = mob_travail[['CODGEO','dist_pond']].groupby(by='CODGEO').sum()
mob_travail_ext = mob_travail_ext.rename(
    columns=dict(zip(mob_travail_ext.columns,
                     [mob_travail_ext.columns[n]+"_data"for n in range(mob_travail_ext.shape[1])])))
mob_travail_ext.to_csv(r"Données traitées\mobilité_travail_commune.csv")

"""Accès aux services"""
# L'estimation des besoins liés à chaque service est peu fiable
# Importation et filtrage sur la région et les colonnes
acces_services = pd.read_parquet(r"Données brutes\BPE24.parquet")
acces_services = acces_services.loc[acces_services['REG']=='76']
acces_services = acces_services[['DEPCOM','SDOM']]\
                 .rename(columns={'DEPCOM':'code_insee'})\
                 .groupby(by='code_insee').value_counts()\
                 .reset_index()\
                 .pivot_table(index='code_insee',columns='SDOM', values='count')\
                 .fillna(0)
# Besoins liés à chaque service (nomenclature dans la BPE)
services_weight = pd.DataFrame(columns=['weight'],
    index=['A1','A2','A3','A4','A5','B1','B2','B3','C1','C2','C3','C4','C5','C6','C7','D1','D2','D3','D4','D5','D6','D7','E1','F1','F2','F3','G1'],
    data=[1/60,1/30,1/365,1/365,1/60,1/30,1/3,1/30, 1/1, 1/2, 1/3,1/30,1/30,1/30,1/30,1/90,1/60,1/60,1/120,1/120,1/10,1/30,0/1,1/7,1/14,1/14,0/1])
# Produit matriciel : synthèse des accès aux services
acces_services['indice_acces_data'] = acces_services.dot(services_weight)
acces_services = acces_services[['indice_acces_data']]
acces_services.to_csv(r"Données traitées\acces_services_commune.csv")

"""Réseaux TEC"""
# Lecture de tous les fichiers de réseaux de bus dans les données brutes
bus_dirs = next(os.walk(r"Données brutes\gtfs_bus"))[1]
# Importation et concaténation
bus_stops = pd.concat([pd.read_csv(rf"Données brutes\gtfs_bus\{bus_dir}\stops.txt") for bus_dir in bus_dirs])
# Filtrage des colonnes et passage à un GeoDataFrame
bus_stops = bus_stops[['stop_id','stop_lat','stop_lon']]
bus_stops = gpd.GeoDataFrame(bus_stops,
                             geometry=gpd.points_from_xy(
                                 bus_stops['stop_lon'],
                                 bus_stops['stop_lat']),
                             crs=4326)
bus_stops = bus_stops.to_crs(2154)[['stop_id','geometry']]
bus_stops = gpd.sjoin(bus_stops,iris_occ)[['code_iris','geometry']]
# Nombre d'arrêts par commune comme estimateur de la densité de service (on pourrait y ajoute rla fréquence, contenue dans les données)
bus_stops = bus_stops.groupby(by='code_iris').count()
bus_stops.columns=['nb_arrets_data']
bus_stops = bus_stops.reindex(iris_occ['code_iris'],fill_value=0)
bus_stops.to_csv(r"Données traitées\arrets_bus_IRIS.csv")

"""Construction de l'indice de vulnérabilité sociale"""

pop_insee=pd.read_csv(r"Données brutes\base-ic-evol-struct-pop-2021_csv\base-ic-evol-struct-pop-2021.csv",
                      sep=';',dtype='str')
# Rebaptême des IRIS périmés
pop_insee['IRIS'] = pop_insee['IRIS'].replace(imp.deprecated_codes)
# Identification des IRIS occitans
pop_insee = pop_insee.loc[np.isin(pop_insee["IRIS"],iris_occitanie)]
# Récupération des variables d'intérêt
deprivation = pop_insee[['IRIS','P21_POP','P21_POP_ETR']]
deprivation[['P21_POP','P21_POP_ETR']] = deprivation.loc[:,['P21_POP','P21_POP_ETR']].apply(pd.to_numeric)
deprivation['part_etrangers'] = deprivation['P21_POP_ETR']/deprivation['P21_POP']
deprivation = deprivation.loc[:,['IRIS','part_etrangers']].set_index('IRIS')

# Extraction des données de l'enquête Couples-Familles-Ménages (Insee)
cfm_2021=pd.read_csv(r"Données brutes\base-ic-couples-familles-menages-2021_csv\base-ic-couples-familles-menages-2021.csv",
                     sep=";", dtype='str')
# Rebaptême des IRIS périmés
cfm_2021['IRIS'] = cfm_2021['IRIS'].replace(imp.deprecated_codes)
# Identification des IRIS occitans
cfm_2021 = cfm_2021.loc[np.isin(cfm_2021["IRIS"],iris_occitanie)]
# Récupération des variables d'intérêt
cfm_2021 = cfm_2021[['IRIS','C21_PMEN','C21_PMEN_CS6','C21_PMEN_CS8',
                     'C21_PMEN_MENFAMMONO','C21_PMEN_MENHSEUL','C21_PMEN_MENFSEUL',
                     'C21_FAM','C21_NE24F4P']].set_index('IRIS')
cfm_2021 = cfm_2021.apply(pd.to_numeric)
cfm_2021['C21_PARTCSPM'] = cfm_2021['C21_PMEN_CS6']/cfm_2021['C21_PMEN'] # Part d'ouvriers
cfm_2021['C21_PARTCHOM'] = cfm_2021['C21_PMEN_CS8']/cfm_2021['C21_PMEN'] # Part de chomeurs
cfm_2021['C21_PARTMEN_SEUL'] = (cfm_2021['C21_PMEN_MENHSEUL']+cfm_2021['C21_PMEN_MENFSEUL'])/cfm_2021['C21_PMEN'] # Part de personnes dans des ménages d'une seule personne
cfm_2021['C21_PARTFMONO'] = cfm_2021['C21_PMEN_MENFAMMONO']/cfm_2021['C21_PMEN'] # Part de personnes dans des familles monoparentales
cfm_2021['C21_PARTMEN_6P'] = cfm_2021['C21_NE24F4P']/cfm_2021['C21_FAM'] # Part de personnes dnas des familles nombreuses
cfm_2021 = cfm_2021.loc[:,['C21_PARTCSPM','C21_PARTCHOM','C21_PARTMEN_SEUL','C21_PARTFMONO','C21_PARTMEN_6P']]
deprivation = deprivation.join(cfm_2021)

logement_insee = pd.read_csv(r"Données brutes\base-ic-logement-2021_csv\base-ic-logement-2021.csv",
                              sep=';',dtype='str')
logement_insee['IRIS'] = logement_insee['IRIS'].replace(imp.deprecated_codes)
logement_insee = logement_insee.loc[np.isin(logement_insee["IRIS"],iris_occitanie)]
logement_insee = logement_insee[['IRIS','P21_RP','P21_RP_PROP','C21_RP_HSTU1P','C21_RP_HSTU1P_SUROCC',
                                 'P21_RP_VOIT1P']].set_index('IRIS')
logement_insee = logement_insee.apply(pd.to_numeric)
logement_insee['P21_PARTNONPROP'] = 1 - logement_insee['P21_RP_PROP']/logement_insee['P21_RP'] # Part des logements non détenus par leurs résidents
logement_insee['C21_PARTSUROCC'] = logement_insee['C21_RP_HSTU1P_SUROCC']/logement_insee['C21_RP_HSTU1P'] # Part des logements en suroccupation
logement_insee['P21_PARTSVOIT'] = 1 - logement_insee['P21_RP_VOIT1P']/logement_insee['P21_RP'] # Part des ménages sans accès à une voiture
logement_insee['P21_PARTSVOIT'] = logement_insee['P21_PARTSVOIT'].clip(lower=0) # Correction d'une erreur de flottant
logement_insee = logement_insee.loc[:,['P21_PARTNONPROP','C21_PARTSUROCC','P21_PARTSVOIT']]
deprivation = deprivation.join(logement_insee)

education_insee = pd.read_csv(r"Données brutes\base-ic-diplomes-formation-2021.csv",
                              sep=';', dtype='str')
education_insee['IRIS'] = education_insee['IRIS'].replace(imp.deprecated_codes)
education_insee = education_insee.loc[np.isin(education_insee["IRIS"],iris_occitanie)]
education_insee = education_insee.loc[:,['IRIS','P21_NSCOL15P','P21_NSCOL15P_BAC']].set_index('IRIS')
education_insee = education_insee.apply(pd.to_numeric)
education_insee['P21_PART_EDUCNBAC'] = (education_insee['P21_NSCOL15P'] - education_insee['P21_NSCOL15P_BAC'])/education_insee['P21_NSCOL15P'] # Part des personnes avec un niveau d'éducation inférieur au bac
education_insee = education_insee.loc[:,'P21_PART_EDUCNBAC']
deprivation = deprivation.join(education_insee)

liste_dep_occ = ['09','11','12','30','31','32','34','46','48','65','66','81','82']
# Récupération des IRIS de la région Occitanie (EPSG:2154)
    # Import des IRIS français
iris = gpd.read_file(r"Données brutes\CONTOURS-IRIS\1_DONNEES_LIVRAISON_2025-06-00117\CONTOURS-IRIS_3-0_GPKG_LAMB93_FXX-ED2025-01-01\iris.gpkg")
    # Ajout de la surface de chaque unité du maillage pour les opérations de cartographie
iris.loc[:,'surface'] = iris['geometry'].area/1e6
    # Filtrage sur le département
iris_occ = iris.loc[np.isin([a[:2] for a in iris["code_insee"]],liste_dep_occ)]
iris_occ = iris_occ.loc[:,['code_iris','geometry']].set_index('code_iris')

deprivation = gpd.GeoDataFrame(deprivation.join(iris_occ),geometry='geometry',crs='EPSG:2154')
deprivation = gpd.overlay(deprivation,revenus_insee.loc[:,['taux_pauv','moy_winsor_niv_vie','geometry']],keep_geom_type=True) # Taux de pauvreté et niveau de vie
deprivation.to_file(r"Données traitées\deprivation.gpkg")
deprivation = deprivation.dropna()


# Préparation de l'ACP
deprivation[deprivation.columns[:-1]] = (deprivation.loc[:,deprivation.columns[:-1]]-deprivation.loc[:,deprivation.columns[:-1]].mean())/\
                                        deprivation.loc[:,deprivation.columns[:-1]].std()
pca = PCA(n_components=1)
# ACP
deprivation_index = pca.fit_transform(deprivation[deprivation.columns[:-1]])
# Stockage des données par IRIS et exportation
deprivation_index = gpd.GeoDataFrame(deprivation_index,geometry=deprivation.geometry, columns=['deprivation_index'])
deprivation_index.to_file(r"Données traitées\deprivation_index.gpkg")
