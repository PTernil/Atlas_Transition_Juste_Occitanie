# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:36:24 2026

@author: ternilp
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import atlas_modules.import_donnees as imp

# Conseil : pour clarté d'utilisatio, on mentionnera l'échelle de raccordement des données
# dans le nom du fichier, sans rien préciser pour une géométrie propore aux données

"""Grilles pour sélection des données"""

iris_occ = gpd.read_file(r"Données traitées\IRIS.gpkg")
iris_occitanie = iris_occ.loc[:,"code_iris"]

com_occ = gpd.read_file(r"Données traitées\Communes.gpkg")
com_occitanie = com_occ.loc[:,"code_insee"]

reg = gpd.read_file(r"Données traitées\Région.gpkg")

"""Couples - Familles - Ménages, 2021 (Insee)"""

# Extraction des données de l'enquête Couples-Familles-Ménages (Insee)
cfm_2021=pd.read_csv(r"Données brutes\base-ic-couples-familles-menages-2021_csv\base-ic-couples-familles-menages-2021.csv",\
                     sep=";", dtype='str')
# Rebaptême des IRIS périmés
cfm_2021['IRIS'] = cfm_2021['IRIS'].replace(imp.deprecated_codes)
    # Identification des IRIS Occitans
cfm_2021=cfm_2021.loc[np.isin(cfm_2021["IRIS"],iris_occitanie)]
    # Extraction des variables de population et de répartition par PCS
cfm_2021=cfm_2021.loc[:,["IRIS","COM","C21_PMEN","P21_POP15P","P21_POP5579","P21_POP80P",\
                         "C21_PMEN_CS1","C21_PMEN_CS2","C21_PMEN_CS3","C21_PMEN_CS4",\
                         "C21_PMEN_CS5","C21_PMEN_CS6","C21_PMEN_CS7","C21_PMEN_CS8",\
                         "P21_POP5579_PSEUL","P21_POP80P_PSEUL","C21_PMEN_MENFAMMONO"]]
cfm_2021[cfm_2021.columns[2:]]=cfm_2021.iloc[:,2:].apply(pd.to_numeric)

# Construction d'un indice de répartition des PCS
    # Pondération des PCS
poids_pcs={"C21_PMEN_CS3":2,
           "C21_PMEN_CS4":1,
           "C21_PMEN_CS2":1,
           "C21_PMEN_CS1":0,
           "C21_PMEN_CS5":0,
           "C21_PMEN_CS6":0,
           "C21_PMEN_CS7":-1,
           "C21_PMEN_CS8":-2}
cfm_2021["Score_CSP"]=sum([poids_pcs[PCS]*cfm_2021[PCS]/cfm_2021["C21_PMEN"] for PCS in poids_pcs.keys()])
cfm_2021["Score_CSP"]=cfm_2021["Score_CSP"].fillna(cfm_2021["Score_CSP"].mean())
    # Normalisation par la méthode des écarts-types
cfm_2021["Score_CSP"] = (cfm_2021["Score_CSP"]-cfm_2021["Score_CSP"].mean())/cfm_2021["Score_CSP"].std()
    # Suppression des colonnes de base PCS pour simplicité d'utilisation, par regroupement en deux catégories
cfm_2021["C21_PMEN_CSP+"]=cfm_2021["C21_PMEN_CS3"]+cfm_2021["C21_PMEN_CS4"]
cfm_2021["C21_PMEN_CSP-"]=cfm_2021["C21_PMEN_CS2"]+cfm_2021["C21_PMEN_CS1"]+cfm_2021["C21_PMEN_CS5"]+\
                          cfm_2021["C21_PMEN_CS6"]+cfm_2021["C21_PMEN_CS7"]+cfm_2021["C21_PMEN_CS8"]
cfm_2021.drop(columns=list(poids_pcs.keys()))

# Populations vulnérables à la chaleur : plus de 75 ans (moins de 4 ans non calculable)
cfm_2021["P_21_PVUL_CHAL"]=cfm_2021["P21_POP5579"]*5/25\
                           +cfm_2021["P21_POP80P"]

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
pop_insee = pop_insee.loc[:,['IRIS','COM','P21_POP','P21_POP0002','P21_POP0305','P21_POP6074','P21_POP75P']]
pop_insee[pop_insee.columns[2:]]=pop_insee.iloc[:,2:].apply(pd.to_numeric)
pop_insee["P_21_PVUL_CHAL"]=pop_insee["P21_POP0002"]+pop_insee["P21_POP0002"]*2/3+pop_insee["P21_POP75P"]
# Rebaptême des colonnes pour inférence du type à la lecture
pop_insee = pop_insee.rename(columns=dict(zip(pop_insee.columns[2:],[pop_insee.columns[n]+"_data"for n in range(2,pop_insee.shape[1])])))
pop_insee.to_csv(r"Données traitées\Population_2021_IRIS.csv", index=False)

"""Zonage exposition à la chaleur (Insee)"""

chaleur_insee=pd.read_excel("Données brutes\chaleurs_insee.xlsx",sheet_name=1)
chaleur_insee=chaleur_insee.iloc[6:]
new_columns=[chaleur_insee.iloc[0,0]]
for n_column in range(1,chaleur_insee.shape[1]):
    if not pd.isna(chaleur_insee.iloc[0,n_column]):
        new_columns.append(chaleur_insee.iloc[0,n_column]+' : '+chaleur_insee.iloc[1,n_column]+'_data')
    else:
        new_columns.append(chaleur_insee.iloc[0,n_column-1]+' : '+chaleur_insee.iloc[1,n_column]+'_data')
chaleur_insee.columns=new_columns
chaleur_insee = chaleur_insee.iloc[2:]
chaleur_insee = chaleur_insee.apply(pd.to_numeric)
chaleur_insee.to_csv(r"Données traitées\Zonage_chaleur_maille_drias.csv", index=False)

"""Revenu (Insee)"""

# Documentation du code : cf enquête Couples - Familles - Ménages
revenus_insee = pd.read_csv(r"Données brutes\BASE_TD_FILO_IRIS_2021_DISP_CSV\BASE_TD_FILO_IRIS_2021_DISP.csv",
                            sep=";", dtype='str')
revenus_insee['IRIS'] = revenus_insee['IRIS'].replace(imp.deprecated_codes)
revenus_insee = revenus_insee.loc[np.isin(revenus_insee["IRIS"],iris_occitanie)]
revenus_insee = revenus_insee[['IRIS','DISP_TP6021','DISP_MED21','DISP_PPSOC21','DISP_S80S2021']]
revenus_insee.iloc[:,1:]=revenus_insee.iloc[:,1:].replace(',','.',regex=True)
revenus_insee.iloc[:,1:]=revenus_insee.iloc[:,1:].apply(pd.to_numeric,errors='coerce')
revenus_insee = revenus_insee.rename(
    columns=dict(zip(revenus_insee.columns[1:],
                     [revenus_insee.columns[n]+"_data"for n in range(1,revenus_insee.shape[1])])))
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
# Écriture
revenus_insee.to_file(r"Données traitées\Revenus_filosofi_2021.gpkg")

"""DPE (Données Performance Energétique du Parc Résidentiel)"""

# Documentation du code : cf enquête Couples - Familles - Ménages
dpe_iris = pd.read_csv(r"Données brutes\classe-dpe-fg-de-par-iris.csv")
dpe_iris['Code IRIS'] = dpe_iris['Code IRIS'].replace(imp.deprecated_codes)
dpe_iris = dpe_iris.loc[np.isin(dpe_iris["Code IRIS"],iris_occitanie)]
dpe_iris = dpe_iris.fillna(0) # Marquage de 0 comme donnée manquante sur la carte
dpe_iris['Part de D et E'] = dpe_iris['Résidences Principales classe D ou E']/dpe_iris['Résidences Principales']
dpe_iris['Part de F et G'] = dpe_iris['Résidences Principales classe F ou G']/dpe_iris['Résidences Principales']
dpe_iris = dpe_iris.rename(
    columns=dict(zip(dpe_iris.columns[1:],
                     [dpe_iris.columns[n]+"_data"for n in range(1,dpe_iris.shape[1])])))
dpe_iris.to_csv(r"Données traitées\DPE_IRIS.csv",index=False)

"""Pathologies par département (CNAM)"""

# Lecture
pathologies = pd.read_parquet(r"Données brutes\pathologies_CNAM.parquet")
# Sélection de l'année, de la région, des données sans distinction de sexe ou d'âge
# Suppression de dept=999 : somme sur la région
pathologies = pathologies.loc[(pathologies['annee']=='2021') & (pathologies['region']=='76')
                              & (pathologies['sexe']==9) & (pathologies['cla_age_5']=='tsage')
                              & (pathologies['dept']!='999')]
# Conservation des colonnes d'intérêt
pathologies = pathologies[['patho_niv1','patho_niv2','patho_niv3','top','dept','Ntop']]
# pathologies = pathologies.loc[pathologies['patho_niv1'].isin(["Insuffisance rénale chronique terminale",
#                                                               "Maladies cardioneurovasculaires",
#                                                               "Maladies neurologiques",
#                                                               "Maladies psychiatriques",
#                                                               "Maladies respiratoires chroniques (hors mucoviscidose)"])]
# Suppression des sous-totaux par catégories de pathologie
pathologies = pathologies.loc[~pd.isna(pathologies['patho_niv3'])]
# Conservation des pathologies rendant vulnérable à la chaleur
pathologies_chaleur = pathologies.loc[pathologies['patho_niv1'].isin(["Insuffisance rénale chronique terminale",
                                                              "Maladies cardioneurovasculaires",
                                                              "Maladies neurologiques",
                                                              "Maladies psychiatriques",
                                                              "Maladies respiratoires chroniques (hors mucoviscidose)"])]
# Passage du tableau dans une forme plus exploitable
pathologies_chaleur = pathologies_chaleur.pivot(index='dept',columns=['patho_niv3'],values='Ntop')
pathologies_chaleur['Vulnérables_chaleur']=pathologies_chaleur.sum(axis=1)
pathologies_chaleur = pathologies_chaleur['Vulnérables_chaleur']
pathologies_chaleur.columns = ['Vulnérables_chaleur_data']
pathologies_chaleur.to_csv(r"Données traitées\Pathologies_chaleur_dept.csv")

"""Travailleurs exposés chaleur extérieure"""
#Correspondance FAP2009-PCS-ESE pour combinaison données Dares/Insee
    # PCS de troisième niveau 429 PCS Insee, sans détail agriculture, artisans, commerçants
    # 541 PCS pour la Dares avec détail, 225 FAP (niveau 3 aussi)
    # Aggrégation de niveau 2 pour utilisation à 87 FAP, 29 PCS-ESE
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
trav_pcs['NB'] = pd.to_numeric(trav_pcs['NB'])
    # Passage à une forme utilisable, restriction géographique à l'occitanie
trav_pcs = trav_pcs.pivot_table(index='CODGEO', columns='CS3_29', values='NB', aggfunc='sum').loc[com_occitanie].fillna(0)
    # Aggrégation selon l'exposition à la chaleur et rebaptême pour inférence à la lecture
trav_pcs = trav_pcs[trav_expo_chaleur.index].dot(trav_expo_chaleur).rename({'part_travail_exterieur':'Nombre_travail_exterieur_data'},axis=1)
trav_pcs.to_csv(r"Données traitées\Travail_chaleur_commune.csv")
