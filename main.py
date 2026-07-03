# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:04:50 2026
Last authenticated version : Wed Jul 01 11:15 2026
@author: ternilp
"""

# Importation des librairies
import atlas_modules.import_donnees as imp
from atlas_modules import carto
from tabulate import tabulate
import matplotlib.pyplot as plt
import os

# Optimisation des importations
if not os.getcwd().startswith('C:\\'):
    print("Attention : Vous ne travaillez pas sur un disque local.\n\
          \rLa fonctionnalité du code n'est pas compromise, mais l'importation des données peut être lente.")
    import_function = imp.import_progress
else:
    import_function = imp.import_fast

# Importation des données
n_data = imp.safe_ask("Nombre de jeux de données à importer :\n",int)

datasets = {}
geoms = []
for n in range(n_data):
    print(f"\nJeu de données n°{n+1}")
    filepath = imp.ask_filepath()
    filepath = imp.check_filepath(filepath)[1]
    treat = imp.safe_ask("Appliquer un traitement aux données ? y/[n]", str)
    if treat=='y':
        treatment = imp.select_list(imp.treatments,query=f"Sélectionnez un traitement parmi :\n\
        \r\t- {'\n\r\t- '.join(imp.treatments)}\n")
        datasets[filepath.stem] = import_function(filepath, treatment=treatment, compact=False)
    else:
        datasets[filepath.stem] = import_function(filepath, compact=False)
    # Définition de la géométrie cadre pour chaque jeu de données
    geom_validation = False
    while geom_validation is False:
        geom_grid = imp.ask_geom(datasets[filepath.stem])
        geom_data = imp.search_geom(geom_grid, datasets[filepath.stem])
        if not geom_data is None:
            geom_validation = True
    geoms.append((geom_grid,geom_data))

print("\nRésumé :")
print(tabulate([[f'Jeu n°{n+1}',dataset,geoms[n]] for n,dataset in enumerate(datasets)],
               headers=("Jeu de données","Raccordement")))

dataset_list = list(datasets.keys())
for n in range(len(datasets)):
    datasets[dataset_list[n]].attrs['scale']=geoms[n][0]

# Importation des grilles
print("\nImportation des références géographiques...", flush=True)
grids={
'reg' : import_function(r"Données traitées\Région.gpkg"),
'dep' : import_function(r"Données traitées\Départements.gpkg"),
'epci' : import_function(r"Données traitées\EPCI.gpkg"),
'com' : import_function(r"Données traitées\Communes.gpkg"),
'iris' : import_function(r"Données traitées\IRIS.gpkg"),
'pays_v' : import_function(r"Données traitées\Pays limitrophes.gpkg"),
'reg_v' : import_function(r"Données traitées\Régions limitrophes.gpkg"),
'dep_v' : import_function(r"Données traitées\Départements limitrophes.gpkg"),
'maille_safran' : import_function(r"Données traitées\maille safran.gpkg"),
'maille_drias' : import_function(r"Données traitées\maille drias.gpkg")
}
for grid in grids:
    grids[grid].attrs['scale']=grid
# Importation des données de répartition de la population, pour traitement des autres données
pop_dataset = import_function(r"Données traitées\Population_2021_iris.csv", compact=True)
pop_dataset = grids['iris'].join(pop_dataset.set_index('code_iris'),
                                 on='code_iris', how="left", lsuffix='_grid').convert_dtypes()
pop_dataset.attrs = {'name':'Base_Population','scale':'iris'}
pop_variable = 'P21_POP'
# Importation des données de revenu, pour traitement des autres données
rev_dataset = import_function(r"Données traitées\Revenus_filosofi_2021.gpkg",compact=True).set_index('id_car')
rev_dataset.attrs = {'name':'Base_Revenus','scale':None}
print("Importation terminée.", flush=True)
print("Extraction des préfectures et sous-préfectures...", flush=True)
prefs={
'pref' : grids['com'].loc[grids['com']['prefecture']==True,['geometry','code_insee','nom_commune']],
's_pref' : grids['com'].loc[grids['com']['sous-prefecture']==True,['geometry','code_insee','nom_commune']]
}
print("Extraction terminée.", flush=True)

if not [geoms[n][0]for n in range(len(geoms))] == [None for n in range(len(geoms))]:
    print("Raccordement des données aux références géographiques...")
# Raccordement des données à la grille géographique de référence
for n in range(len(datasets)):
    if geoms[n][0] is not None:
        attrs = datasets[dataset_list[n]].attrs
        datasets[dataset_list[n]]=grids[geoms[n][0]].\
            join(datasets[dataset_list[n]].set_index(geoms[n][1]),\
            on=imp.geom_grid_dict[geoms[n][0]], how="inner", lsuffix='_grid')\
            .set_index(imp.geom_grid_dict[geoms[n][0]])\
            .convert_dtypes()
        datasets[dataset_list[n]].attrs = attrs
# Pour les données à géographie interne, vérification du système de projection
    else:
        if datasets[dataset_list[n]].crs.to_epsg()!=2154:
            print(f"{dataset_list[n]} n'utilise pas la projection Lambert-93.\n\
                  \rReprojection en Lambert-93...", flush=True)
            datasets[dataset_list[n]].to_crs('EPSG:2154')
            print("Terminé.")
if not [geoms[n][0]for n in range(len(geoms))] == [None for n in range(len(geoms))]:
    print("Raccordement terminé.")

corr_admin = import_function(r"Données traitées\Correspondance_echelle_admin.csv")
corr_admin = [corr_admin,{'département':grids['dep'][['dep','geometry']],
                          'epci':grids['epci'][['EPCI','geometry']],
                          'commune':grids['com'][['code_insee','geometry']]}]
# %% Création des variables

# Définition des variables à calculer
datasets = imp.build_variables(datasets,pop_dataset,pop_variable,corr_admin)

# %% Paramétrage de l'affichage

titre = input("Titre de la carte :\n").replace('\\n','\n')
source = input("Source(s) des données :\n").replace('\\n','\n')
print("Sélection des variables à afficher\n\
      \r----------------------------------")
variables = imp.ask_carto(datasets,pop_dataset)
lecture = input("Indications de lecture :\n").replace('\\n','\n')
# %% Agrégation
# TODO: Permettre un normalisation par n'importe quelle variable importée et non seulement les variables de pop_dataset

# Agrégation à un niveau supérieur pour clarté d'affichage, si nécessaire
aggregation = input("Agrégation géographique à l'affichage ? y/[n]")
if aggregation == 'y':
    # Sélection de l'échelle d'agrégation
    min_admin_scale = imp.select_list(list(carto.admin_scales),
                                      query = f"Échelle d'affichage parmi :\n\
                                      \r\t- {'\n\r\t- '.join(list(carto.admin_scales))}\n",
                                      catch_dict=carto.admin_scales_entries)
    # Dictionnaire des jeux à agréger
    plot_datasets = [variables[n][0][0] for n in range(len(variables))]
    datasets_agg = dict([(key, datasets[key]) for key in plot_datasets])
    # Agrégation totale : agrégation de TOUTES les données à l'échelle demandées (sauf données à un échelon supérieur)
    # Agrégation partielle : agrégation des données sur une maille administrative uniquement
    full_agg = imp.safe_ask("agrégation totale : ",bool)
    # Variable de poids utilisée pour les moyennes pondérées
    pop_variable = imp.select_list(list(pop_dataset.columns[4:])+['surface'],
                                   query=f"Nom de la variable d'agrégation parmi :\n\
                                   \r\t- {'\n\r\t- '.join(list(pop_dataset.columns[4:])+['surface'])}\n")
    var_names = [variables[n][0][1] for n in range(len(variables))]
    datasets_agg = imp.aggregate(datasets_agg,min_admin_scale,corr_admin,
                                 pop_dataset,pop_variable,geoms,var_names,full_agg,grids)
else:
    datasets_agg = datasets
    min_admin_scale = None

# TODO: Carte
    # TODO: datatype hors cmap
    # TODO: offer possibility to normalise by area
# %% Affichage
if variables[0][1]['classification'] is not None:
    stats = True
    graphs = False
    # stats = input("Calculs statistiques supplémentaires ? y/[n] ")=='y'
    # graphs = input("Graphiques statistiques supplémentaires ? y/[n] ")=='y'
    if graphs:
        graphs={}
        print("Graphique Population")
        print('-'*20)
        graphs['pop_title'] = input("Titre du graphique : ")
        graphs['pop_xlabel'] = input("Étiquette de l'axe x : ")
        graphs['pop_ylabel'] = input("Étiquette de l'axe y : ")
        graphs['pop_arrow'] = imp.safe_ask("Remplacer les étiquettes de données par une flèche ? ", bool)
        if graphs['pop_arrow']:
            graphs['pop_arrow_text'] = input("Sens de l'augmentation sur l'axe x :\n")
        print("\nGraphique Revenu")
        print('-'*16)
        graphs['rev_title'] = input("Titre du graphique : ")
        graphs['rev_xlabel'] = input("Étiquette de l'axe x : ")
        graphs['rev_ylabel'] = input("Étiquette de l'axe y : ")
        graphs['rev_arrow'] = imp.safe_ask("Remplacer les étiquettes de données par une flèche ? ", bool)
        if graphs['rev_arrow']:
            graphs['rev_arrow_text'] = input("Sens de l'augmentation sur l'axe x :\n")
    else:
        graphs = None
else:
    stats = False
    graphs = None

# Création de la figure
print("Création de la carte...")
fig, ax = plt.subplots(figsize=(60, 40))
ax.set_xlim((0.4e6,0.92e6))
ax.set_ylim((6.12e6,6.5e6))
print("Fonds de carte...")
# Affichage des préfectures
texts = carto.draw_prefs(ax, prefs)
# Arrière-plan
carto.draw_background(fig,ax,texts,grids,min_admin_scale)
# Ajout des étiquettes des territoires voisins
carto.neighbours(ax)
print("Ajout des données...")
carto.plot_data(ax,datasets_agg,variables,corr_admin=corr_admin,
                pop_dataset=pop_dataset,rev_dataset=rev_dataset,
                titre=titre,source=source,lecture=lecture,stats=stats,
                graphs=graphs)
print("Terminé.")
