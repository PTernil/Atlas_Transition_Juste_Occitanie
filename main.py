# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:04:50 2026

@author: ternilp
"""

# Importation des librairies
import atlas_modules.import_donnees as imp
import atlas_modules.carto as carto
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
    # Raccordement à la géométrie cadre
    geom_validation = False
    while geom_validation==False:
        geom_grid = imp.ask_geom(datasets[filepath.stem])
        geom_data = imp.search_geom(geom_grid, datasets[filepath.stem])
        if not geom_data is None:
            geom_validation=True
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
# Importation des données de répartition de la population, pour traitement des autres données
pop_dataset = import_function(r"Données traitées\Population_2021_iris.csv", compact=True)
pop_dataset = grids['iris'].join(pop_dataset.set_index('code_iris'),
                                 on='code_iris', how="left", lsuffix='_grid').convert_dtypes()
pop_dataset.attrs = {'name':'Base_Population','scale':'iris'}
print("Importation terminée.", flush=True)
print("Extraction des préfectures et sous-préfectures...", flush=True)
prefs={
'pref' : grids['com'].loc[grids['com']['prefecture']==True,['geometry','code_insee','nom_commune']],
's_pref' : grids['com'].loc[grids['com']['sous-prefecture']==True,['geometry','code_insee','nom_commune']]
}
print("Extraction terminée.", flush=True)

# Raccordement des données à la grille géographique de référence
# Pour les données à géographie interne, vérification du système de projection
if not [geoms[n][0]for n in range(len(geoms))] == [None for n in range(len(geoms))]:
    print("Raccordement des données aux références géographiques...")

for n in range(len(datasets)):
    if geoms[n][0]!=None:
        attrs = datasets[dataset_list[n]].attrs
        datasets[dataset_list[n]]=grids[geoms[n][0]].\
            join(datasets[dataset_list[n]].set_index(geoms[n][1]),\
            on=imp.geom_grid_dict[geoms[n][0]], how="inner", lsuffix='_grid').convert_dtypes()
        datasets[dataset_list[n]].attrs = attrs

    else:
        if datasets[dataset_list[n]].crs.to_epsg()!=2154:
            print(f"{dataset_list[n]} n'utilise pas la projection Lambert-93.\n\
                  \rReprojection en Lambert-93...", flush=True)
            datasets[dataset_list[n]].to_crs('EPSG:2154')
            print("Terminé.")
# %% variables

# TODO: aggregate après calcul variables, qui agrège sur inter(niveau admin,autre) si autre
# Définition des variables à calculer
datasets = imp.build_variables(datasets,pop_dataset)

# Aggrégation à un niveau supérieur pour clarté d'affichage, si nécessaire
corr_admin = import_function(r"Données traitées\Correspondance_echelle_admin.csv")
corr_admin = [corr_admin,{'département':grids['dep'][['code','geometry']],
                          'epci':grids['epci'][['EPCI','geometry']],
                          'commune':grids['com'][['code_insee','geometry']]}]
min_admin_scale = imp.select_list(list(imp.admin_scales),
                                  query = f"Échelle d'affichage parmi :\n\
                                  \r\t- {'\n\r\t- '.join(list(imp.admin_scales))}\n",
                                  catch_dict=imp.admin_scales_entries)
datasets = imp.aggregate(datasets,min_admin_scale,corr_admin,pop_dataset,geoms)
for n in range(len(geoms)):
    if geoms[n][0] in imp.admin_scales[min_admin_scale]:
        geoms[n] = (imp.geom_dict[min_admin_scale],imp.geom_grid_dict[imp.geom_dict[min_admin_scale]])
# %% Paramétrage de l'affichage

print("Sélection des variables à afficher\n\
      \r----------------------------------")
variables = imp.ask_carto(datasets,pop_dataset)

print("\nRésumé :")
print(tabulate([[f"Jeu n°{i+1}",var["nom"],dataset,var["type"],var["couleur"]]
        for i, (dataset, (n_vars, vars_list)) in enumerate(variables.items())
        for var in vars_list],
    headers=("Indicateur", "Nom du jeu source", "Type", "Couleur")
    )+'\n')

titre = input("Titre de la carte :\n").replace('\\n','\n')
source = "                "+input("Source(s) des données :\n")

# TODO: Carte
    # TODO: datatype hors cmap
    # TODO: encadrés de lecture dans la carte à la main de l'utilisateur
    # TODO: offer possibility to normalise by area
# %% Affichage

# Création de la figure
print("Création de la carte...")
fig, ax = plt.subplots(figsize=(60, 40))
ax.set_xlim((0.4e6,0.92e6))
ax.set_ylim((6.12e6,6.5e6))
x_length = 0.52e6
y_length = 0.38e6
bbox = ax.get_window_extent()
x_length_px = bbox.width
y_length_px = bbox.height

print("Fonds de carte...")
# Affichage des préfectures
texts = carto.draw_prefs(ax, prefs)
# Arrière-plan
carto.draw_background(fig,ax,texts,grids)
# Ajout des étiquettes des territoires voisins
carto.neighbours(ax)
print("Ajout des données...")
ax=carto.plot_data(ax,datasets,variables,
                   pop_dataset=pop_dataset,
                   titre=titre,source=source)

ax.set_axis_off()
ax.add_artist(ax.patch)
ax.patch.set_zorder(-1)
plt.savefig('Cartes\\'+titre.replace('\n',' '), bbox_inches='tight')
plt.show()
print("Terminé.")
