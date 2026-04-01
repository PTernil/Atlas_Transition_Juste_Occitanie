# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:04:50 2026

@author: ternilp
"""

# Importation des librairies
import import_donnees as imp
import maprefs as mr
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.transforms import IdentityTransform
from shapely.geometry import Polygon
from shapely.ops import unary_union
from adjustText import adjust_text


# Importation des données
n_data = int(input("Nombre de jeux de données à importer :\n"))
datasets = {}
geoms = []
for n in range(n_data):
    print(f"\nJeu de données n°{n+1}")
    filepath = imp.ask_filepath()
    filepath = imp.check_filepath(filepath)[1]
    datasets[filepath.stem] = imp.import_progress(filepath, compact=False)
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

# Importation des grilles
print("\nImportation des références géographiques...", flush=True)
grids={
'reg' : imp.import_progress(r"Données traitées\Région.gpkg"),
'dep' : imp.import_progress(r"Données traitées\Départements.gpkg"),
'com' : imp.import_progress(r"Données traitées\Communes.gpkg"),
'iris' : imp.import_progress(r"Données traitées\IRIS.gpkg"),
'pays_v' : imp.import_progress(r"Données traitées\Pays limitrophes.gpkg"),
'reg_v' : imp.import_progress(r"Données traitées\Régions limitrophes.gpkg"),
'dep_v' : imp.import_progress(r"Données traitées\Départements limitrophes.gpkg")
}
print("Importation terminée.", flush=True)
print("Extraction des préfectures et sous-préfectures...", flush=True)
prefs={
'pref' : grids['com'].loc[grids['com']['prefecture']==True,['geometry','code_insee','nom_commune']],
's_pref' : grids['com'].loc[grids['com']['sous-prefecture']==True,['geometry','code_insee','nom_commune']]
}
print("Extraction terminée.", flush=True)

# Raccordement des données à la grille géographique de référence
# Pour les données à géographie interne, vérification du système de projection
for n in range(len(datasets)):
    if geoms[n][0]!=None:
        datasets[dataset_list[n]]=grids[geoms[n][0]].\
            join(datasets[dataset_list[n]].set_index(geoms[n][1]),\
            on=imp.geom_grid_dict[geoms[n][0]], how="inner", lsuffix='_grid').convert_dtypes()
    else:
        if datasets[dataset_list[n]].crs.to_epsg()!=2154:
            print(f"{dataset_list[n]} n'utilise pas la projection Lambert-93.\n\
                  \rReprojection en Lambert-93...", flush=True)
            datasets[dataset_list[n]].to_crs('EPSG:2154')
            print("Terminé.")

# TODO: Carte
    # TODO: Affichage des grilles cf. cartes QGIS
    # TODO: Choix des variables
    # TODO: Création des fourchettes
    # TODO: Création des palettes
    # TODO: Possibilité de modification des constantes
    # TODO: Titre : Indicateur(échelle,Année), Légende, Échelle
    # TODO: encadrés de lecture dans la carte à la main de l'utilisateur
# %%

# Création de la figure, définition de constantes
fig, ax = plt.subplots(figsize=(60, 40))
ax.set_xlim((0.4e6,0.92e6))
ax.set_ylim((6.12e6,6.5e6))
x_length = 0.52e6
y_length = 0.38e6
bbox = ax.get_window_extent()
x_length_px = bbox.width
y_length_px = bbox.height
# Affichage des préfectures
masks=[]
for prefset in prefs.keys():
    # Marqueurs
    prefs[prefset]['centroid']=prefs[prefset].geometry.centroid
    prefs[prefset].set_geometry("centroid").plot(
        ax=ax,
        color=mr.grids[prefset]['color'],
        marker=mr.grids[prefset]['marker'],
        markersize=mr.grids[prefset]['markersize']
    )
    # Textes
    texts = []
    for _, row in prefs[prefset].iterrows():
        x, y = row["centroid"].x, row["centroid"].y
        txt = ax.text(x, y, row["nom_commune"],
                      fontsize=mr.grids[prefset]['fontsize'], ha='center', va='bottom')
        texts.append(txt)
    adjust_text(texts, ax=ax, expand=(1.5,2), force_text=3, force_pull=0.05, force_static=0.1)
    text_polygons = []

    renderer = fig.canvas.get_renderer()

    text_polygons = []
    # Masques pour lisibilité des textes
    for txt in texts:
        label = txt.get_text()
        fontsize = txt.get_fontsize()
    
        tp = TextPath((0, 0), label, size=fontsize)
        bb = tp.get_extents()
    
        # Scaling et alignement
        if txt.get_ha() == 'center':
            dx = -bb.width / 2
        elif txt.get_ha() == 'right':
            dx = -bb.width
        else:
            dx = 0
    
        if txt.get_va() == 'bottom':
            dy = -bb.ymin
        elif txt.get_va() == 'center':
            dy = -bb.height / 2
        elif txt.get_va() == 'top':
            dy = -bb.ymax
        else:
            dy = 0
        x_disp, y_disp = ax.transData.transform(txt.get_position())
        scale = fig.dpi / 72.0
        bbox = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        x_disp = (bbox.x0 + bbox.x1)/2 -3 # Attention : réglage manuel, à changer si affichage d'une autre échelle (non prévu pour ce projet)
        y_disp = (bbox.y0+bbox.y1)/2 +8 # idem
        transform = (
            Affine2D()
            .scale(scale)
            .translate(dx * scale, dy * scale)
            .translate(x_disp, y_disp)
        )
    
        tp_disp = transform.transform_path(tp)
        # Passage en coordonnées matplotlib
        tp_data = ax.transData.inverted().transform_path(tp_disp)
        tp_data = tp_data.interpolated(80)
        
        # Calcul des formes des masques
        current = []
        for (vx, vy), code in zip(tp_data.vertices, tp_data.codes):
            if code == Path.MOVETO:
                current = [(vx, vy)]
            elif code == Path.LINETO:
                current.append((vx, vy))
            elif code == Path.CLOSEPOLY:
                # Vérifie que le polygone n'est pas dégénéré
                if len(current) >= 3:
                    if current[0] != current[-1]:
                        current.append(current[0])
                    poly = Polygon(current)
                    if poly.is_valid and not poly.is_empty:
                        text_polygons.append(poly)
                current = []
    # Taille du masque proportionnelle à la taille des étiquettes
    mask_buffer_size = np.mean([p.bounds[2] - p.bounds[0] for p in text_polygons])*0.2
    buffer = [p.buffer(mask_buffer_size) for p in text_polygons]
    masks.extend(buffer)

mask_union = unary_union(masks)
grids_masked = {}
for grid in grids.keys():
    gdf = grids[grid].copy()
    
    gdf["geometry"] = gdf.geometry.boundary.apply(
        lambda line: line.difference(mask_union) if not line.is_empty else line
    )
    
    grids_masked[grid] = gdf

for grid in grids_masked.keys():
    grids_masked[grid].plot(
        ax=ax,
        edgecolor=mr.grids[grid]['edgecolor'],
        linewidth=mr.grids[grid]['linewidth'],
        zorder=1
    )

ax.set_axis_off()
plt.legend()
plt.title("test")
plt.show()
