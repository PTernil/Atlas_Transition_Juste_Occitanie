# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 08:46:05 2026
Last authenticated version : Wed Jul 01 11:15 2026
@author: ternilp
"""
from adjustText import adjust_text
from copy import deepcopy
import geopandas as gpd
from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap,\
    BoundaryNorm, BivarColormapFromImage, to_rgb, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.font_manager import FontProperties
from matplotlib_map_utils.core.scale_bar import scale_bar
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib import patheffects
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import mapclassify
import numpy as np
import pandas as pd
import re
from shapely.geometry import box
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tabulate import tabulate
# Types d'affichages utilisant une colourmap
cmap_data = ['Densité','Score','Part']
# Création de colourmaps personnalisées
base = cm.get_cmap("RdYlGn") # Modification pour meilleure vue par les daltoniens
colors = base(np.linspace(0, 1, 256))
luminance = np.dot(colors[:, :3], [0.2126, 0.7152, 0.0722])
target = np.linspace(luminance.min(), luminance.max(), 256)
blended_lum = 0.6*luminance + 0.4*target
new_colors = colors.copy()
for i in range(256):
    if luminance[i] > 0:
        scale = blended_lum[i] / luminance[i]
        new_colors[i, :3] = np.clip(colors[i, :3] * scale, 0, 1)
RdYlGn_corr = ListedColormap(new_colors)
GnYlRd_corr = RdYlGn_corr.reversed()
custom_cmaps = {'RdYlGn_corr':RdYlGn_corr,'GnYlRd_corr':GnYlRd_corr}
# Colourmaps à 2 dimensions
def bivar_cmap(lowlow='#ffffff',highlow='#ff0000',lowhigh='#00ffff',name='bivar_cmap'):
    lut = np.zeros((256, 256, 3))
    lowlow = np.array(to_rgb(lowlow))
    highlow = np.array(to_rgb(highlow))
    lowhigh = np.array(to_rgb(lowhigh))
    # Interpolation linéaire sur les axes
    # Mélange multiplicatif pour le reste
    for i in range(256):
        for j in range(256):
            x = i / (256 - 1)
            y = j / (256 - 1)
            ax1_color = (1 - x) * lowlow + x * highlow
            ax2_color = (1 - y) * lowlow + y * lowhigh
            color = [min(1,a*b) for a,b in zip(ax1_color,ax2_color)]
            lut[i, j, :] = color
    return BivarColormapFromImage(lut, shape='square', name=name)
bivariate_cmaps = {'BluePink':bivar_cmap('#e8e8e8','#40dba7','#f73593','BluePink'),
                   'BlueRed':bivar_cmap('#e8e8e8','#4885c1','#ee6a6e','BlueRed'),
                   'PurpleGold':bivar_cmap('#e8e8e8','#9972af','#c8b35a','PurpleGold'),
                   'RedGold':bivar_cmap('#e8e8e8','#ee5a6e','#c8b35a','RedGold')}
bivariate_cmaps.update({'RedBlue':bivariate_cmaps['BlueRed'].transposed()})
bivariate_cmaps.update({'RedBlue_0_r':bivariate_cmaps['RedBlue'].reversed(False,True)})
bivariate_cmaps.update({'RedBlue_r_0':bivariate_cmaps['RedBlue'].reversed(True,False)})
bivariate_cmaps.update({'RedBlue_r_r':bivariate_cmaps['RedBlue'].reversed(True,True)})
bivariate_cmaps.update({'BlueRed_0_r':bivariate_cmaps['BlueRed'].reversed(True,False)})
bivariate_cmaps.update({'BlueRed_r_0':bivariate_cmaps['BlueRed'].reversed(False,True)})

reg = gpd.read_file(r"Données traitées\Région.gpkg")
# Règles d'affichage du fonds de carte
# "_v" signifie voisins
grids={
'reg' : {'edgecolor':'#232323', 'linewidth':3.9},
'dep' : {'edgecolor':'#232323', 'linewidth':1.95},
'epci' : {'edgecolor':'#232323', 'linewidth':0.65},
'com' : {'edgecolor':'#232323', 'linewidth':0.1625},
'iris' : {'edgecolor':'#232323', 'linewidth':0}, 
'pays_v' : {'edgecolor':'#232323', 'linewidth':2.6, 'facecolor':'#dddddd'},
'reg_v' : {'edgecolor':'#232323', 'linewidth':2.6, 'facecolor':'#eeeeee'},
'dep_v' : {'edgecolor':'#232323', 'linewidth':0.325},
'pref' : {'color':'#444444', 'marker':'s', 'markersize':200, 'fontsize':40},
's_pref' : {'color':'#666666', 'marker':'o', 'markersize':100, 'fontsize':30}
}
# Fonds de carte masqué par les noms de ville
grids_masked = ['reg','dep','epci','pays_v','reg_v','dep_v']
grids_no_mask = set(grids.keys())-set(grids_masked)-set(('pref','s_pref'))
# Comparaison des mailles administratives
admin_scales = {'département':['epci','com','iris'],'epci':['com','iris'],'commune':['iris'],'iris':None}
admin_scales_entries = {'dep':'département',
                        'epci':'epci',
                        'com':'commune',
                        'iris':'iris'}
admin_scales_grids = {'département':'dep','epci':'epci','commune':'com','iris':'iris'}
admin_scales_names = {'département':'dep','epci':'EPCI','commune':'code_insee','iris':'code_iris'}
admin_scales_size = {'dep':4,'epci':3,'com':2,'iris':1}

def masks(texts, fig, ax):
    """
    Crée et renvoie des masques autour des objets texts fournis.
    Les masques peuvent ensuite être utilisés pour masquer d'autres objets autour des texts.

    Parameters
    ----------
    texts : list(matplotlib.text.Text)
        Textes autour desquels créer les masques.
    fig : matplotlib.figure.Figure
        Figure sur laquelle les textes sont affichés.
    ax : matplotlib.axes._axes.Axes
        Axes de la figure utilisée.

    Returns
    -------
    shapely.geometry.collection.GeometryCollection
        Contour des masques.

    """
    text_polygons = []
    mask_list=[]
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
                if len(current) >= 3: # Vérifie que le polygone n'est pas dégénéré
                    if current[0] != current[-1]:
                        current.append(current[0])
                    poly = Polygon(current)
                    if poly.is_valid and not poly.is_empty:
                        text_polygons.append(poly)
                current = []

    mask_buffer_size = np.mean([p.bounds[2] - p.bounds[0] for p in text_polygons])*0.3 # Taille du masque proportionnelle à la taille des étiquettes
    buffer = [p.buffer(mask_buffer_size) for p in text_polygons]
    mask_list.extend(buffer)

    return unary_union(mask_list)

def shapely_to_path(geom):
    """
    Convertit un objet shapely.Geometry en objet matplotlib.path.Path

    Parameters
    ----------
    geom : shapely.Geometry
        Géométrie à convertir.

    Returns
    -------
    matplotlib.path.Path
        Géométrie convertie.

    """
    vertices = []
    codes = []
    for polygon in getattr(geom, "geoms", [geom]): # Récupération des arêtes de tous les polygones
        exterior = polygon.exterior.coords
        vertices.extend(exterior)
        codes.extend([Path.MOVETO] + [Path.LINETO]*(len(exterior)-2) + [Path.CLOSEPOLY])
        for interior in polygon.interiors:
            coords = interior.coords
            vertices.extend(coords)
            codes.extend([Path.MOVETO] + [Path.LINETO]*(len(coords)-2) + [Path.CLOSEPOLY])
    return Path(vertices, codes)

def infer_pop_by_geom(geometry, pop_dataset, pop_variable):
    """
    Interpole la population d'un découpage géographique.
    Suppose une densité de population uniforme au sein des IRIS

    Parameters
    ----------
    geometry : gpd.GeoDataFrame
        Géométrie du découpage à interpoler.
        L'index est utilisé
    pop_dataset : gpd.GeoDataFrame
        Contient les données de population.
    pop_variable : str
        Nom de la variable de population à inférer.

    Returns
    -------
    pd.Series.
        Population inférée, indexée comme geometry

    """
    geometry.index = geometry.index.rename('geom_index')
    intersections = gpd.overlay(pop_dataset[[pop_variable,'surface','geometry']], geometry.reset_index(), keep_geom_type=True)
    intersections['intersect_area'] = intersections.geometry.area
    intersections['population'] = intersections[pop_variable]*intersections['intersect_area'] / intersections['surface']/1e6
    return intersections.groupby('geom_index')['population'].sum()

def list_overlay(df_list, proportional=False, pop_dataset=None, pop_variable=None, corr_admin=None):
    """
    Réalise un overlay des GeoDataFrame passés en argument.
    Gère la répétition des géométries par suppression des doublons,
    avant l'appel de gpd.overlay

    Parameters
    ----------
    df_list : list(geopandas.GeoDataFrame)
        Liste des données sur lesquelles opérer l'overlay.
    proportional : list(bool) ou bool, par défaut False
        Si True, répartit proportionnellement les valeurs numériques des colonnes
        en fonction du ratio de population lors des découpages géométriques.
            - Si bool : appliqué à tous les datasets
            - Si list(bool) : un par dataset, indique si ses colonnes doivent être réparties
              proportionnellement lors des découpages ultérieurs
        Fait l'hypothèse d'une population homogène au sein des IRIS
    pop_dataset : gpd.GeoDataFrame, par défaut, None
        Contient les données de population. Indexation par identifiant spatial
    pop_variable : str, par défaut, None
        Nom de la colonne de pop_dataset contenant la donnée de population
    corr_admin : list(pd.DataFrame, dict(gpd.GeoDataFrame)), par défaut, None
        Correspondance entre les différentes échelles administratives
        et géométries des échelons administratifs
    
    Returns
    -------
    result
        Overlay des GeoDataFrame passés en argument.
    
    """
    # Application de proportional à tous les datasets
    if isinstance(proportional, bool):
        proportional = [proportional] * len(df_list)
    # En cas de proportionalité, adapte les données de population aux données à croiser
    if proportional!=[False for n in range(len(df_list))]:
        join_need=False
        for df in df_list:
            if df.attrs['scale'] in admin_scales_size and df.attrs['scale']!='iris':
                join_need=True
        if join_need:
            pop_dataset = pop_dataset.drop(columns='code_insee').join(corr_admin[0].set_index('code_iris'), on='code_iris')
    # Suppression des géométries en double, ajout de suffixes aux colonnes
    # pour tracer les données et éviter les erreurs dans gpd.overlay
    source_dfs = []
    admin_size = 'dep'
    admin_mixed = False
    for idx, gdf in enumerate(df_list):
        unique_gdf = gdf.drop_duplicates(subset='geometry').reset_index()
        rename_dict = {col: f"{col}_{idx}" for col in unique_gdf.columns if col != 'geometry'}
        renamed = unique_gdf.rename(rename_dict, axis=1)
        if gdf.attrs['scale'] in admin_scales_size:
            if admin_scales_size[gdf.attrs['scale']] < admin_scales_size[admin_size]:
                admin_size = gdf.attrs['scale']
        else:
            admin_mixed = True
        source_dfs.append(renamed)
    # Overlay source par source
    result = source_dfs[0]
    inferred=False
    for i in range(len(source_dfs)):
        # Stockage des populations originales pour les sources qui nécessitent
        # une répartition proportionnelle
        if proportional[i]:
            if source_dfs[i].attrs['scale'] in admin_scales_size:
                geo_var_name = admin_scales_names[admin_scales_entries[source_dfs[i].attrs['scale']]]
                source_dfs[i][f'__pop_original_{i}'] = source_dfs[i][f"{geo_var_name}_{i}"]\
                                                       .map(pop_dataset[[pop_variable,geo_var_name]]\
                                                       .groupby(geo_var_name).sum()[pop_variable])
            else:
                source_dfs[i][f'__pop_original_{i}'] = infer_pop_by_geom(source_dfs[i][['geometry']],pop_dataset,pop_variable)
                inferred=True
        if i>0:
            # Overlay
            result = gpd.overlay(result, source_dfs[i],keep_geom_type=True)
            # Suppression des artéfacts (très petits polygones)
            if len(result) > 0:
                result = result[result.geometry.area > 1] # 1m², négligeable à l'échelle régionale
                result = result.reset_index(drop=True)
    # Application de la répartition proportionnelle
    small_var_name = admin_scales_names[admin_scales_entries[admin_size]]
    for i in range(len(source_dfs)):
        pop_col = f'__pop_original_{i}'
        if proportional[i] and pop_col in result.columns:
            mask = result[pop_col] > 0
            if mask.any():
                if inferred or admin_mixed: # Population estimée en fonction de la géométrie
                    result.loc[mask, '__ratio'] = infer_pop_by_geom(result[['geometry']], pop_dataset, pop_variable).loc[mask] / result.loc[mask, pop_col]
                else: # Population exacte calculée 
                    result_pop = pop_dataset[[pop_variable,small_var_name]].groupby(small_var_name).sum()
                    result['__pop_current'] = result[f'{small_var_name}_{len(source_dfs)-1}'].map(result_pop[pop_variable])
                    result.loc[mask, '__ratio'] = result.loc[mask, '__pop_current'] / result.loc[mask, pop_col]
                source_cols = [col for col in result.columns if col.endswith(f'_{i}')]
                for col in source_cols: # Ratio par la population. Note : plusieurs colonne d'un même jeu sont interprétées comme de jeux différents
                # Il est donc possible d'avoir répartition proportionnelle et non-répartition pour un même jeu
                    if pd.api.types.is_numeric_dtype(result[col]):
                        result[col] = result[col].astype(pd.Float64Dtype())
                        result.loc[mask, col] = result.loc[mask, col] * result.loc[mask, '__ratio']
                        result[col] = result[col].astype(float)
                result = result.drop(columns=['__ratio'])
            result = result.drop(columns=[pop_col])
    # Nettoyage des colonnes temporaires
    result = result.drop([col for col in result.columns if col.startswith('__')], axis=1)
    result.attrs = {'scale':admin_size,'admin_mixed':admin_mixed}
    return result

def draw_prefs(ax, prefs):
    """
    Dessine les marqueurs et noms des préfectures et sous-préfecture,
    en s'assurant de l'absence de chevauchement

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axes de la figure sur laquelle tracer les textes.
    prefs : dict(gpd.GeoDataFrame)
        Jeux contenant les position et noms des préfectures et sous-préfectures.

    Returns
    -------
    texts : list(matplotlib.text.Text)
        Objets contenant le nom et la position des textes écrits pour représenter
        préfectures et sous-préfecture.

    """
    texts = []
    for prefset in prefs.keys():
        # Marqueurs
        prefs[prefset]['centroid']=prefs[prefset].geometry.centroid
        prefs[prefset].set_geometry("centroid").plot(ax=ax,
            color=grids[prefset]['color'],
            marker=grids[prefset]['marker'],
            markersize=grids[prefset]['markersize'],
            zorder=3
        )
        # Textes
        for _, row in prefs[prefset].iterrows():
            x, y = row["centroid"].x, row["centroid"].y
            txt = ax.text(x, y, row["nom_commune"],
                          fontsize=grids[prefset]['fontsize'],
                          ha='center', va='bottom',zorder=3,
                          backgroundcolor='#FFFFFF33')
            texts.append(txt)
    # Relocalisation des textes pour éviter le chevauchement
    adjust_text(texts, ax=ax, expand=(1.5,2), explode_radius=1,
                force_text=0.15, force_pull=0.1,force_static=0.01)
    return texts

def draw_background(fig, ax, texts, admingrids,min_admin_scale):
    """
    Dessine le fonds de carte sur lequel les données peuvent ensuite être représentées.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure sur laquelle la carte est dessinée.
    ax : matplotlib.axes._axes.Axes
        Axes de la figure utilisée.
    texts : list(matplotlib.text.Text)
        Noms et positions des textes écrits pour représenter
        préfectures et sous-préfecture.
    admingrids : dict(gpd.GeoDataFrame)
        Contours administratifs du fonds de carte.
    min_admin_scale : str
        Nom de la maille d'affichage du fonds de carte.

    Returns
    -------
    None.

    """
    geogrids = admingrids.copy()
    # Masques pour lisibilité des textes
    text_masks = masks(texts,fig,ax)
    # Fonds de couleur
    ax.patch.set_facecolor('#a6cee3')
    for grid in ['reg_v','pays_v']:
        geogrids[grid].boundary.plot(ax=ax,edgecolor=None,
                                  facecolor=grids[grid]['facecolor'],
                                  zorder=0.8)
    geogrids_masked = {}
    # Suppression du tracé des contours administratifs inférieurs à l'échelle d'agrégation
    if min_admin_scale in admin_scales:
        for scale in admin_scales[min_admin_scale]:
            geogrids.pop(scale)
    else:
        geogrids.pop('epci')
    # Soustraction des masques aux géométries
    for grid in geogrids:
        if grid in grids_masked:
            grid_ = geogrids[grid].copy()
            grid_["geometry"] = grid_.geometry.boundary.apply(lambda line: line.difference(text_masks))
            geogrids_masked[grid] = grid_
    # Tracé des géométries masquées
    for grid in geogrids_masked.keys():
        geogrids_masked[grid].plot(ax=ax,edgecolor=grids[grid]['edgecolor'],
                                linewidth=grids[grid]['linewidth'],zorder=2.1)
    # Tracé des géométries non masquées
    for grid in geogrids:
        if grid in grids_no_mask:
            geogrids[grid].boundary.plot(ax=ax,edgecolor=grids[grid]['edgecolor'],
                             linewidth=grids[grid]['linewidth'],zorder=2.1)
    # Tracé ombré du contour régional
    reg_shadow = []
    # Création de l'ombre
    for i in range(10):
        lw = grids['reg']['linewidth']*2*(1 + i)
        alpha = 0.05*(1-i/10)
        reg_shadow.append(patheffects.Stroke(linewidth=lw,
            foreground=grids['reg']['edgecolor'],alpha=alpha))
    # Création du masque spécifique et application à l'ombre
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    clip_geom = box(xmin, ymin, xmax, ymax).difference(text_masks)
    clip_path = PathPatch(shapely_to_path(clip_geom), transform=ax.transData)
    old_colls = set(ax.collections)
    geogrids['reg'].boundary.plot(ax=ax,edgecolor=grids['reg']['edgecolor'],
                               linewidth=grids['reg']['linewidth'],
                               path_effects=reg_shadow,zorder=0.9)
    new_colls = set(ax.collections)
    shadow_colls = new_colls.difference(old_colls)
    for coll in shadow_colls:
        coll.set_clip_path(clip_path)
    # Tracé de l'ombre
    geogrids_masked['reg'].plot(ax=ax,edgecolor=grids['reg']['edgecolor'],
                             linewidth=grids['reg']['linewidth'],zorder=1)
    # Tracé du fond représentant les données manquantes
    geogrids['reg'].plot(ax=ax, edgecolor=None,
                         facecolor='lightgrey',hatch='///'
                         )

def neighbours(ax):
    """
    Trace les noms des pays et régions voisines.
    Assume un système de coordonnée Lambert-93 (EPSG:2154)

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axes sur lesquels tracer les noms.

    Returns
    -------
    None.

    """
    ax.text(0.47e6,6.15e6,"Espagne", fontsize=48, weight='demi', zorder=3)
    andorre=ax.text(0.556e6,6.16e6,"Andorre", fontsize=48, weight='demi', zorder=3)
    andorre.set_path_effects([patheffects.withStroke(linewidth=5, foreground='#dddddd')])
    ax.text(0.43e6,6.4e6,"Nouvelle-Aquitaine", fontsize=42, weight='demi', zorder=3)
    ax.text(0.765e6,6.44e6,"Auvergne-Rhône-Alpes", fontsize=42, weight='demi', zorder=3)
    ax.text(0.85e6,6.3e6,"Provence-Alpes-\nCôte-d'Azur", fontsize=42, weight='demi', zorder=3)

def format_bin_labels(labels,datatype):
    """
    Passe les entrées fournies au format utilisé dans l'Atlas

    Parameters
    ----------
    labels : list(str)
        Entrées de légende, telles que créées par mapclassify.
    datatype : str
        Nom du type de données.
        Si 'Part', les étiquettes seront en %.

    Returns
    -------
    formatted_labels : list(str)
        Entrées de légende au format utilisé.

    """
    formatted_labels = []
    # évalue l'ordre de grandeur pour effectuer l'arrondi
    try:
        odg = int(np.floor(np.log10(float(re.findall(r'[\d\.\-]+', labels[-1])[1]))))
    except IndexError:
        odg = int(np.floor(np.log10(float(re.findall(r'[\d\.\-]+', labels[-2])[1]))))
    if datatype == 'Part': # Affichage en pourcentage
        odg+=2
    rounding = max(0,2-odg) # Arrondi à l'entier pour des valeurs dont le max est >=100
    for n in range(len(labels)):
        # Extrait les nombres
        numbers = re.findall(r'[\d\.\-]+', labels[n])
        if datatype == 'Part':
            numbers = [float(n)*100 for n in numbers]
        if rounding==0:
            lower = int(np.floor(float(numbers[0])))
        else:
            lower = np.round(float(numbers[0]),rounding)
        if n<len(labels)-1: # pas la dernière étiquette
            if rounding==0:
                upper = int(np.floor(float(numbers[1])))
            else:
                upper = np.round(float(numbers[1]),rounding)
        else: # dernière étiquette
            try:
                if rounding==0: # Arrondi à la valeur supérieure pour inclure toutes les valeurs
                    upper = int(np.ceil(float(numbers[1])))
                else:
                    upper = np.round(float(numbers[1]),rounding)
            except IndexError: # Valeurs infinies
                upper = '∞'
        if datatype == 'Part':
            if lower != upper:
                formatted_labels.append(f"{lower}% - {upper}%")
            else:
                formatted_labels.append(f"{lower}%")
        else:
            if lower != upper:
                formatted_labels.append(f"{lower} - {upper}")
            else:
                formatted_labels.append(f"{lower}")
    return formatted_labels

def wrap_text_to_width(ax, text, required_width, fontsize=12, **text_kwargs):
    """
    Découpe un texte en lignes pour ne pas dépasser la longueur spécifiée.
    Les découpes se font uniquement au niveau des espaces
    Si un mot est plus long que la longueur spécifiée, il sera tracé seul sur sa ligne, en entier

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axes sur lesquel tracer les textes.
        Les texte ne seront pas tracés par cette fonction
    text : str
        texte à découper.
    required_width : float
        Longueur maximale d'une ligne.
    fontsize : int, optional
        Taille de la police d'écriture. Par défaut, 12.
    **text_kwargs : matplotlib.text.Text properties
        Arguments optionnels décrivant le style de text.

    Returns
    -------
    str
        Texte découpés en lignes de longueur maximale required_width.

    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Conversion des tailles en pixels
    x0_px = ax.transAxes.transform((0, 0))[0]
    x1_px = ax.transAxes.transform((required_width, 0))[0]
    max_width_px = x1_px - x0_px
    # Découpage en mots
    tokens = re.findall(r'\S+|\s+', text)
    lines = []
    current_line = ""
    for token in tokens:
        # Regarde si la ligne devient trop longue, change de ligne si oui
        test_line = current_line + token
        t = ax.text(
            0, 0, test_line,
            fontsize=fontsize,
            transform=ax.transAxes,
            **text_kwargs
        )
        bbox = t.get_window_extent(renderer=renderer)
        t.remove()
        if bbox.width <= max_width_px or current_line == "":
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = token.lstrip()
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)

def plot_data(ax, datasets, variables, corr_admin=None, pop_dataset=None, pop_variable='P21_POP',
              rev_dataset=None, rev_variable='moy_winsor_niv_vie',
              titre='', source='', lecture='', stats=False, graphs=None):
    """
    Dessine les données sur ax comme spécifié dans variables.
    N'affiche qu'une variable à la fois, sauf si un affichage bivarié est demandé
    
    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axes sur lesquels dessiner la carte.
    datasets : dict(gpd.GeoDataFrame)
        Jeux contenant les données à représenter.
    variables : list(list(list(str),dict(str,dict(int,str,float))))
        Consignes d'affichage des données, construites en atlas_modules.imp.ask_carto.
    corr_admin : list(pd.DataFrame, dict(gpd.GeoDataFrame)), optional
        Correspondances entre les échelons administratifs (corr_admin[0])
        et géométries administratives supérieures à l'IRIS (corr_admin[1])
        Par défaut, None
    pop_dataset : gpd.GeoDataFrame, optional
        Répartition de la population, pour calculs de statistiques
        Par défaut, None
    pop_variable : str, optional
        Nom de la colonne de pop_dataset contenant la population à utiliser.
        Par défaut, 'P21_POP'.
    rev_dataset : gpd.GeoDataFrame, optional
        Répartition du revenu, pour calcul de statistiques.
        Par défaut, None.
    rev_variable : str, optional
        Nom de la colonne de rev_dataset contenant le revenu à utiliser.
        Par défaut, 'moy_winsor_niv_vie'.
    titre : str, optional
        Titre de la carte. Par défaut, ''.
    source : str, optional
        Source des données utilisées dans la carte. Par défaut, ''.
    lecture : str, optional
        Indications de lecture à écrire sous la carte. Par défaut, ''.
    stats : bool, optional
        Si True, calcul de statistiques supplémentaires sur la population et le revenu.
        Par défaut, False.
    graphs : dict, optional
        Consignes de mise en forme pour des graphiques sur les statistiques supplémentaires.
        Si None, aucun graphique supplémentaire ne sera produit.
        Par défaut, None.
    
    Returns
    -------
    None.
    
    """
    if len(variables)==1: # Une seule variable à représenter
        dataset = variables[0][0][0]
        if variables[0][1]['type'] in cmap_data:
            # Traitement des données, si spécifié
            treatment = variables[0][1].get('treatment','Aucun')
            if treatment=='Logscaling': # Logarithme
                datasets = deepcopy(datasets)
                datasets[dataset][variables[0][0][1]]=\
                    np.log1p(datasets[dataset][variables[0][0][1]]
                           -datasets[dataset][variables[0][0][1]].min())
            if treatment=='loglog': # Logarithme du logarithme, pour les données très étalées
                datasets = deepcopy(datasets)
                datasets[dataset][variables[0][0][1]]=\
                    np.log1p(np.log1p(datasets[dataset][variables[0][0][1]]
                                      -datasets[dataset][variables[0][0][1]].min()))
            # Récupération des paramètres d'affichage
            cmap = variables[0][1]['couleur']
            if cmap in set(custom_cmaps.keys()):
                cmap = custom_cmaps[cmap]
            if variables[0][1]['classification'] is None:
                classification_kwds=None
            elif variables[0][1]['classification']=='HeadTailBreaks':
                classification_kwds=None
            else:
                classification_kwds=variables[0][1]['bins'].copy()
            # Affichage
            plot_obj = datasets[dataset].plot(
                ax=ax,
                column=variables[0][0][1],
                cmap=cmap,
                scheme=variables[0][1]['classification'],
                classification_kwds=classification_kwds,
                missing_kwds={
                    "color": "lightgrey",
                    "edgecolor": "k",
                    "hatch": "///",
                    "label": "Pas de données",
                },
                zorder=2,
                label=variables[0][1]['nom_legende'],
                legend=False
            )
            # Récupération des etiquettes de légende
            if variables[0][1]['classification']=='UserDefined':
                classifier = getattr(mapclassify,variables[0][1]['classification'])(
                    datasets[dataset][variables[0][0][1]],
                    **variables[0][1]['bins'])
                legend_labels = format_bin_labels(classifier.get_legend_classes(fmt='{:.4f}'), variables[0][1]['type'])
            elif variables[0][1]['classification']=='HeadTailBreaks':
                classifier = getattr(mapclassify,variables[0][1]['classification'])(
                    datasets[dataset][variables[0][0][1]].dropna().to_numpy())
                legend_labels = format_bin_labels(classifier.get_legend_classes(fmt='{:.4f}'), variables[0][1]['type'])
            elif variables[0][1]['classification'] in ['FisherJenks'] : # En prévision d'un ajout de classifications possibles
                classifier = getattr(mapclassify,variables[0][1]['classification'])(
                    datasets[dataset][variables[0][0][1]].dropna().to_numpy(),
                    **variables[0][1]['bins'])
                legend_labels = format_bin_labels(classifier.get_legend_classes(fmt='{:.4f}'), variables[0][1]['type'])
            else : # Cas sans classification
                legend_labels = variables[0][1]['labels']
            if hasattr(plot_obj, 'collections') and len(plot_obj.collections) > 0:
                norm = plot_obj.collections[0].norm
                label = variables[0][1]['nom_legende']
                labels = legend_labels
                scheme = variables[0][1]['classification']
                var_type = variables[0][1]['type']
        if variables[0][1]['type'] == 'Localisation de points':
            #todo: Implémentation avec calcul des isochrones OpenRouteServices
            raise NotImplementedError(f"Type de données '{variables[0][1]['type']}'")
        elif variables[0][1]['type'] == 'Compte':
            raise NotImplementedError(f"Type de données '{variables[0][1]['type']}'")
            #todo: Implémentation (Cercles de taille variable)
        elif variables[0][1]['type'] == 'Tracés':
            cmap = variables[0][1]['couleur']
            if cmap in set(custom_cmaps.keys()):
                cmap = custom_cmaps[cmap]
            datasets[dataset][variables[0][0][1]] =\
                datasets[dataset][variables[0][0][1]].astype("float64")
            plot_obj = datasets[dataset].geometry.plot(
                ax=ax,
                cmap=cmap,
                zorder=2,
                label=variables[0][1]['nom_legende'],
                legend=False
            )
            #todo:Vérifier la correction de la fonction (non testée jusqu'à présent)
        # Ajout du titre
        # Écrit le titre dans le vide pour mesurer la place nécessaire
        fp_title = FontProperties(weight='bold', size=48)
        temp_fig = plt.figure()
        temp_ax = temp_fig.add_subplot(111)
        title_text = temp_ax.text(0, 0, label, fontproperties=fp_title, ha='left', va='center')
        temp_fig.canvas.draw()
        title_bbox = title_text.get_window_extent(renderer=temp_fig.canvas.get_renderer())
        title_width = title_bbox.width
        title_height = title_bbox.height
        # Écrit la plus longue des étiquettes de légende dans le vide pour mesurer la place nécessaire
        longest_label = ''
        for legend_label in labels:
            if len(legend_label)>len(longest_label):
                longest_label = legend_label
        fp_labels = FontProperties(size=42)
        label_text = temp_ax.text(0, 0, longest_label, fontproperties=fp_labels, ha='left', va='center')
        temp_fig.canvas.draw()
        label_bbox = label_text.get_window_extent(renderer=temp_fig.canvas.get_renderer())
        label_width = label_bbox.width
        plt.close(temp_fig)
        # Convertit les tailles de point à proportion de la figure
        fig = ax.figure
        fig_dpi = fig.dpi
            # Taille en pouces
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax_width_inches = bbox.width
        ax_height_inches = bbox.height
            # Taille en proportion de la figure
        title_width_axes = (title_width / fig_dpi) / ax_width_inches
        title_height_axes = (title_height / fig_dpi) / ax_height_inches
        label_width_axes = (label_width / fig_dpi) / ax_width_inches
        # Dimensions de la légende, en proportion de la figure (1=figure entière)
        colorbar_width = 0.02
        title_margin_top = 0.008
        title_to_cbar_margin = 0.005
        if scheme in [None,'Légende continue']:
            cbar_height = 0.1
        else:
            cbar_height = 0.03*len(labels)
        cbar_to_labels_margin = 0.005
        cbar_margin_bottom = title_margin_top
        # Dimension de la boîte contenant la légende
        required_width = (cbar_margin_bottom 
                          + max(colorbar_width+cbar_to_labels_margin+label_width_axes, title_width_axes)
                          + cbar_margin_bottom)
        required_height = (title_margin_top + title_height_axes
                           + title_to_cbar_margin + cbar_height
                           + cbar_margin_bottom)
        # Positionnement depuis le coin inférieur droit
        x0 = max(0.65,0.9 - required_width*0.34/(0.34+0.09))
        y0 = 0.055
        # Création du support de la légende
        final_size = [x0, y0, required_width, required_height]
        legend_box = ax.inset_axes(
            final_size,
            transform=ax.transAxes,
            zorder=5,
            xticks=[], yticks=[]
        )
        legend_box.patch.set_facecolor('#FFFFFFCC')
        legend_box.patch.set_edgecolor('k')
        legend_box.patch.set_linewidth(4)
        # Affichage du nom de la variable
        legend_box.text(
            title_margin_top/required_width, 1-title_margin_top/required_height,
            label,
            ha='left', va='top',
            fontsize=48, fontweight='bold'
        )
        # Création de la légende
        cbar_ax = legend_box.inset_axes([cbar_margin_bottom/required_width,
                                         cbar_margin_bottom/required_height,
                                         colorbar_width/required_width,
                                         cbar_height/required_height])
        if not scheme is None: # Données classées
            # Discrétisation de la colormap
            if isinstance(cmap,ListedColormap):
                legend_cmap = cmap.resampled(len(labels))
            else:
                legend_cmap = plt.get_cmap(cmap, len(labels))
            legend_cmap = [legend_cmap(i) for i in range(0,legend_cmap.N)]
            # Ajout de patchs transparents pour séparer les classes de légende
            legend_cmap = [[legend_cmap[i],legend_cmap[i],legend_cmap[i],(0,0,0,0),(0,0,0,0)]
                           for i in range(len(legend_cmap))]
            legend_cmap=[c for c_list in legend_cmap for c in c_list] # Unpacking
            legend_cmap = legend_cmap[:-2] # Suppression des patchs transparents en extrémité de légende
            legend_cmap_ = legend_cmap.copy()
            legend_cmap = ListedColormap(legend_cmap)
            n_colors = len(legend_cmap_)
            bounds = np.linspace(0, 1, n_colors + 1)
            norm = BoundaryNorm(bounds, n_colors)
            # Création de la colorbar (avec espaces entre les entrées de légende)
            cbar = ColorbarBase(
                cbar_ax,
                cmap=legend_cmap,
                norm=norm,
                orientation='vertical',
                boundaries=bounds
            )
            # Espaces rendus transparents (à la main, matplotlib.Colorbar ne gère pas correctement la transparence)
            for i, patch in enumerate(cbar_ax.patches):
                if legend_cmap_[i][3] == 0:
                    patch.set_facecolor('none')
                    patch.set_edgecolor('none')
            cbar.outline.set_visible(False)
            # Étiquettes
            cbar.set_ticks([(1.5+5*i)/(5*len(labels)-2) for i in range(len(labels))],
                           labels=labels, fontsize=42)
            cbar_ax.tick_params(size=0, pad=cbar_to_labels_margin*fig_dpi*ax_width_inches)
        else: # Données non classées
            cbar = ColorbarBase(
                cbar_ax,
                cmap=cmap,
                norm=norm,
                orientation='vertical',
            )
            # Étiquettes
            cbar.set_ticks(
                [norm.vmin*0.9+norm.vmax*0.1, norm.vmin*0.1+norm.vmax*0.9],
                labels=labels, fontsize=42
            )
        if var_type=='Densité':
            cbar.ax.invert_yaxis() # Affichage des valeurs les plus élevées en bas de la colorbar
        cbar_ax.set_facecolor('none')
        cbar_ax.patch.set_alpha(0) # Transparence du fond de la colorbar
    elif len(variables)==2 and variables[0][1].get('bivariate_cmap',False): # 2 variables avec colormap en 2 dimensions
        if variables[0][1]['classification'] is None:
            raise NotImplementedError("Affichage de 2 variables avec une légende non continue")
        else:
            classifiers=[]
            # Récupération des étiquettes de légende
            for n in range(2):
                if variables[n][1]['classification']=='UserDefined':
                    classifiers.append(getattr(mapclassify,variables[n][1]['classification'])(
                        datasets[variables[n][0][0]][variables[n][0][1]],
                        **variables[n][1]['bins']))
                elif variables[n][1]['classification']=='HeadTailBreaks':
                    classifiers.append(getattr(mapclassify,variables[n][1]['classification'])(
                        datasets[variables[n][0][0]][variables[n][0][1]].dropna().to_numpy()))
                elif variables[n][1]['classification'] in ['FisherJenks'] : # En prévision d'un ajout de classifications possibles
                    classifiers.append(getattr(mapclassify,variables[n][1]['classification'])(
                        datasets[variables[n][0][0]][variables[n][0][1]].dropna().to_numpy(),
                        **variables[n][1]['bins']))
            # Récupération des classes de chaque maille pour les deux variables
            dataset_0 = gpd.GeoDataFrame(data=classifiers[0].yb,columns=['class_0'],
                                         index=datasets[variables[0][0][0]].index,
                                         geometry=datasets[variables[0][0][0]].geometry,
                                         crs='EPSG:2154')
            dataset_0.attrs=datasets[variables[0][0][0]].attrs
            dataset_1 = gpd.GeoDataFrame(data=classifiers[1].yb,columns=['class_1'],
                                         index=datasets[variables[1][0][0]].index,
                                         geometry=datasets[variables[1][0][0]].geometry,
                                         crs='EPSG:2154')
            dataset_1.attrs=datasets[variables[1][0][0]].attrs
            # Attribution d'une classe globale en fonction des classes pour chaque variable
            dataset = gpd.overlay(dataset_0,dataset_1,keep_geom_type=True) # Fusion géographique # TODO : optimiser en ne réalisant l'overlay que si les géométries sont difféntes et non basées sur une maille administrative. (sjoin ou assimilé sinon)
            dataset.attrs = {'name':'plot_dataset',
                             'scale':dataset_0.attrs['scale']
                                 if admin_scales_size.get(dataset_0.attrs['scale'],5)<admin_scales_size.get(dataset_1.attrs['scale'],5)
                                 else dataset_1.attrs['scale'],
                             'admin_mixed':(dataset_0.attrs.get('admin_mixed',False) or dataset_1.attrs.get('admin_mixed',False))
                                           or (dataset_0.attrs.get('scale',None)==None or dataset_1.attrs.get('scale',None)==None)
                             }
            dataset['global_class'] = dataset['class_0']*(classifiers[1].k)+dataset['class_1'] # Classe globale pour utilisation de la colomap
            cmap = bivariate_cmaps[variables[1][1]['couleur']].resampled((len(classifiers[0].counts),len(classifiers[1].counts)))
            cmap_flattened = ListedColormap(cmap.lut.reshape(-1,cmap.lut.shape[-1])) # colormap aplatie pour lecture de global_class
            # Affichage des données
            dataset.plot(
                ax = ax,
                column = 'global_class',
                cmap = cmap_flattened,
                norm=Normalize(vmin=0, vmax=cmap_flattened.N - 1),
                missing_kwds={
                    "color": "lightgrey",
                    "edgecolor": "k",
                    "hatch": "///",
                    "label": "Pas de données",
                },
                zorder=2,
                label=variables[0][1]['nom_legende'],
                legend=False)
            # Calcul de la taille de la légende en proportion de la taille de l'image
            title_margin_top = 0.008
            labels_to_cmap_margin = 0.005
            cmap_margin_bottom = 0.008
            title_to_labels_margin = 0.008
            n_classes_ax0 = len(classifiers[0].counts)
            n_classes_ax1 = len(classifiers[1].counts)
            cmap_height = 0.05*n_classes_ax0
            cmap_width = 0.05*n_classes_ax1
            # Écrit les noms des variables dans le vide pour mesurer la place nécessaire
            fp_title = FontProperties(weight='bold', size=44)
            temp_fig = plt.figure()
            temp_ax = temp_fig.add_subplot(111)
            title_text_ax0 = temp_ax.text(0, 0, variables[0][1]['nom_legende'], fontproperties=fp_title,
                                          ha='left', va='center',rotation=90)
            title_text_ax1 = temp_ax.text(0, 0, variables[1][1]['nom_legende'], fontproperties=fp_title,
                                          ha='left', va='center')
            temp_fig.canvas.draw()
            # Mesure des tailles des noms de variables
            title_bbox_ax0 = title_text_ax0.get_window_extent(renderer=temp_fig.canvas.get_renderer())
            title_bbox_ax1 = title_text_ax1.get_window_extent(renderer=temp_fig.canvas.get_renderer())
            title_width_ax0 = title_bbox_ax0.width
            title_height_ax0 = title_bbox_ax0.height
            title_width_ax1 = title_bbox_ax1.width
            title_height_ax1 = title_bbox_ax1.height
            # Écrit les étiquettes de données les plus longues pour mesurer la place nécessaire
            fp_labels = FontProperties(size=36)
            if variables[0][1]['default_labels']: # Étiquette représentant les plages de données
                # Récupération de l'étiquette la plus longue
                labels_ax0 = format_bin_labels(classifiers[0].get_legend_classes(fmt='{:.4f}'), variables[0][1]['type'])
                longest_label_ax0 = ''
                for legend_label in labels_ax0:
                    if len(legend_label)>len(longest_label_ax0):
                        longest_label_ax0 = legend_label
                label_text_ax0 = temp_ax.text(0, 0, longest_label_ax0, fontproperties=fp_labels) # Tracé de l'étiquette
            else: # Étiquette unique représentant la signification d'un changement de classe
                label_text_ax0 = temp_ax.text(0, 0, variables[0][1]['label'], rotation=90, fontproperties=fp_labels) # Tracé
            if variables[1][1]['default_labels']: # Étiquettes représentant les plages de données
                # Récupération de l'étiquette la plus longue
                labels_ax1 = format_bin_labels(classifiers[1].get_legend_classes(fmt='{:.4f}'), variables[0][1]['type'])
                longest_label_ax1 = ''
                for legend_label in labels_ax1:
                    if len(legend_label)>len(longest_label_ax1):
                        longest_label_ax1 = legend_label
                label_text_ax1 = temp_ax.text(0, 0, longest_label_ax1, fontproperties=fp_labels, rotation=45) # Tracé
            else: # Étiquette unique représentant la signification d'un changement de classe
                label_text_ax1 = temp_ax.text(0, 0, variables[1][1]['label'], fontproperties=fp_labels) # Tracé
            temp_fig.canvas.draw()
            # Mesure des tailles des étiquettes
            label_bbox_ax0 = label_text_ax0.get_window_extent(renderer=temp_fig.canvas.get_renderer())
            label_bbox_ax1 = label_text_ax1.get_window_extent(renderer=temp_fig.canvas.get_renderer())
            label_height_ax0 = label_bbox_ax0.height
            label_width_ax0 = label_bbox_ax0.width
            label_height_ax1 = label_bbox_ax1.height
            label_width_ax1 = label_bbox_ax1.width
            plt.close(temp_fig)
            # Convertit les tailles de point à proportion de la figure
            fig = ax.figure
            fig_dpi = fig.dpi
                # Taille en pouces
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            ax_width_inches = bbox.width
            ax_height_inches = bbox.height
                # Taille en proportion de la figure
            title_width_ax0 = (title_width_ax0 / fig_dpi) / ax_width_inches
            title_height_ax0 = (title_height_ax0 / fig_dpi) / ax_height_inches
            title_width_ax1 = (title_width_ax1 / fig_dpi) / ax_width_inches
            title_height_ax1 = (title_height_ax1 / fig_dpi) / ax_height_inches
            label_width_ax0 = (label_width_ax0 / fig_dpi) / ax_width_inches
            label_height_ax0 = (label_height_ax0 / fig_dpi) / ax_height_inches
            label_width_ax1 = (label_width_ax1 / fig_dpi) / ax_width_inches
            label_height_ax1 = (label_height_ax1 / fig_dpi) / ax_height_inches
            # Prise en compte des flèches (non tracées) par ajout d'une marge entre les étiquettes et la colormap
            if not variables[0][1]['default_labels']:
                label_width_ax0 += 2*labels_to_cmap_margin
            if not variables[1][1]['default_labels']:
                label_height_ax1 += 2*labels_to_cmap_margin
                label_width_ax1 = 0
            # Dimensions de la boîte contenant la légende
            required_width = (title_margin_top
                              + max(title_width_ax0 + title_to_labels_margin
                                    + label_width_ax0 + labels_to_cmap_margin + cmap_width
                                    + max(0,label_width_ax1-cmap_width/n_classes_ax1*0.95),
                                    title_width_ax1, label_width_ax1)
                              + cmap_margin_bottom)
            required_height = (title_margin_top
                               + max(title_height_ax1 + title_to_labels_margin
                                   + label_height_ax1 + labels_to_cmap_margin + cmap_height,
                                   title_height_ax0, label_height_ax0)
                               + cmap_margin_bottom)
            # Positionnement depuis le coin inférieur droit, sans pouvoir aller trop à gauche
            x0 = max(0.65,0.9 - required_width*0.34/(0.34+0.09))
            y0 = 0.055
            # Création du support de la légende
            final_size = [x0, y0, required_width, required_height]
            legend_box = ax.inset_axes(
                final_size,
                transform=ax.transAxes,
                zorder=5,
                xticks=[], yticks=[]
            )
            legend_box.patch.set_facecolor('#FFFFFFCC')
            legend_box.patch.set_edgecolor('k')
            legend_box.patch.set_linewidth(4)
            
            # Création de la grille de légende bivariée
                # Positionnement de la grille
            grid_x_start = (title_margin_top + title_width_ax0 + title_to_labels_margin
                            + label_width_ax0 + labels_to_cmap_margin) / required_width
            if variables[1][1]['default_labels']: # Grille sous les étiquette de l'axe horizontal
                grid_y_start = cmap_margin_bottom / required_height
            else: # Grille au-dessus de l'étiquette de l'axe horizontal
                grid_y_start = (title_margin_top + title_height_ax1 + title_to_labels_margin
                                + label_height_ax1 + labels_to_cmap_margin) / required_height
                # Dimensions de la grille
            grid_width = cmap_width / required_width
            grid_height = cmap_height / required_height
            # Création de la grille
            grid_ax = legend_box.inset_axes([grid_x_start, grid_y_start, grid_width, grid_height])
            # Coloriage
            for i in range(n_classes_ax0):
                for j in range(n_classes_ax1):
                    color_idx = i * n_classes_ax1 + j
                    rect = Rectangle((j / n_classes_ax1, i / n_classes_ax0),
                                   1 / n_classes_ax1, 1 / n_classes_ax0,
                                   facecolor=cmap_flattened.colors[color_idx],
                                   edgecolor='black', linewidth=0.5)
                    grid_ax.add_patch(rect)
            grid_ax.axis('off')
            
            # Nom de la variable 0, centrée (verticalement) si possible
            if title_height_ax0/required_height/2 < grid_height / 2: # Pas de dépassement si la variable est centrée
                legend_box.text(title_margin_top / required_width,grid_y_start + grid_height / 2,
                                variables[0][1]['nom_legende'],ha='left', va='center',
                                fontsize=44, fontweight='bold',rotation=90) # Tracé centré
            else: # En cas de dépassement, tracé à partir d'une extrémité de la colormap
                if variables[1][1]['default_labels']: # cmap en bas de la légende, tracé depuis le bas de la cmap
                    legend_box.text(title_margin_top / required_width,grid_y_start,
                                    variables[0][1]['nom_legende'],ha='left', va='bottom',
                                    fontsize=44, fontweight='bold',rotation=90)
                else: # cmap en haut de la légende, tracé depuis le haut de la cmap
                    legend_box.text(title_margin_top / required_width,grid_y_start + grid_height,
                                    variables[0][1]['nom_legende'],ha='left', va='top',
                                    fontsize=44, fontweight='bold',rotation=90)
            # Nom de la variable 1, centrée (horizontalement) si possible
            if variables[1][1]['default_labels']: # cmap en bas de la légende
                name_y = 1 - (title_margin_top + title_height_ax1) / required_height
            else: # cmap en haut de la légende
                name_y = title_margin_top / required_height / 2 # title_margin_top est la marge en bas dans ce cas
            if title_width_ax1/required_width/2 < 1-(grid_x_start + grid_width / 2): # Pas de dépassement si la variable est centrée
                legend_box.text(grid_x_start + grid_width / 2, name_y,
                                variables[1][1]['nom_legende'],ha='center', va='bottom',
                                fontsize=44, fontweight='bold') # Tracé centré
            else: # En cas de dépassement, tracé depuis la droite de la cmap
                legend_box.text(1 - cmap_margin_bottom / required_width, name_y,
                                variables[1][1]['nom_legende'], ha='right', va='bottom',
                                fontsize=44, fontweight='bold')
            # Étiquettes de la variable 0
            if variables[0][1]['default_labels']: # Étiquettes multiples représentant les limites des classes
                for i in range(n_classes_ax0):
                    label_y = grid_y_start + (i + 0.5) / n_classes_ax0 * grid_height
                    legend_box.text(grid_x_start - labels_to_cmap_margin/required_width,label_y,
                                    labels_ax0[i],ha='right', va='center',
                                    fontsize=36)
            else: # Étiquette unique représentant la signification d'un changement de classe
                legend_box.annotate(
                    '', xy=(grid_x_start - 2*labels_to_cmap_margin/required_width,
                            grid_y_start + grid_height),
                    xytext=(grid_x_start - 2*labels_to_cmap_margin/required_width,
                            grid_y_start),
                    arrowprops=dict(arrowstyle='->, head_length=2, head_width=1.25', color='black', lw=5),
                    transform=legend_box.transAxes
                    ) # Tracé de la flèche
                if grid_y_start + grid_height/2 + label_height_ax0/2/required_height < 1-cmap_margin_bottom/required_height: # Variable centrée sans dépassement
                    legend_box.text(grid_x_start - 3*labels_to_cmap_margin/required_width,
                                    grid_y_start + grid_height/2,
                                    variables[0][1]['label'],ha='right',va='center',
                                    fontsize=36, rotation=90
                        ) # Tracé centré (verticalement)
                else: # Dépassement si la variable est centrée, tracé depuis le haut de la cmap
                    legend_box.text(grid_x_start - 3*labels_to_cmap_margin/required_width,
                                    1-cmap_margin_bottom/required_height,
                                    variables[0][1]['label'],ha='right',va='top',
                                    fontsize=36, rotation=90
                        )
            # Étiquettes de la variable 1
            if variables[1][1]['default_labels']: # Étiquettes multiples représentant les limites des classes
                for i in range(n_classes_ax1):
                    label_x = grid_x_start + (i + 0.05) / n_classes_ax1 * grid_width
                    legend_box.text(label_x,grid_y_start + grid_height + labels_to_cmap_margin/required_height,
                                    labels_ax1[i],ha='left', va='bottom',
                                    fontsize=36,rotation=45)
            else: # Étiquette unique représentant la signification d'un changement de classe
                legend_box.annotate(
                    '', xy=(grid_x_start + grid_width,
                            grid_y_start - 2*labels_to_cmap_margin/required_height),
                    xytext=(grid_x_start,
                            grid_y_start - 2*labels_to_cmap_margin/required_height),
                    arrowprops=dict(arrowstyle='->, head_length=2, head_width=1.25', color='black', lw=5),
                    transform=legend_box.transAxes
                    ) # Tracé de la flèche
                legend_box.text(grid_x_start + grid_width/2,
                                grid_y_start - 3.5*labels_to_cmap_margin/required_height,
                                variables[1][1]['label'],ha='center',va='top',
                                fontsize=36
                    ) # Tracé centré (horizontalement)
                #TODO : Implémenter la vérification de dépassement pour la variable 1 (horizontale)
    # Titre
    ax.text(0.41e6,6.495e6,titre,fontsize=100, fontweight='bold', va='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
    )
    # Source
    ax.text(x0,y0-0.005,"Source :", fontweight='bold',fontsize=32,va='top',transform=ax.transAxes)
    wrapped = wrap_text_to_width(ax," "*16+source,required_width,fontsize=32) # Écriture sous la légende uniquement
    ax.text(x0, y0-0.005,wrapped,fontsize=32,va='top',transform = ax.transAxes)
    # Barre d'échelle
    if type(dataset)==str:
        scale_bar(ax, location="lower left", style="ticks",
                  bar={"projection": datasets[dataset].crs,
                       'height':0.5, 'tickwidth':5,
                       'minor_div':5,'minor_type':'first'},
                  labels={'fontsize':42},
                  units={'fontsize':48})
    else:
        scale_bar(ax, location="lower left", style="ticks",
                  bar={"projection": dataset.crs,
                       'height':0.5, 'tickwidth':5,
                       'minor_div':5,'minor_type':'first'},
                  labels={'fontsize':42},
                  units={'fontsize':48})
    ax.set_axis_off()
    ax.add_artist(ax.patch)
    ax.patch.set_zorder(-1)
    # Indications de lecture
    if lecture!='':
        ax.text(0.005, -0.005, 'Lecture :', ha='left', va = 'top',
                fontsize=32, fontweight='bold', transform = ax.transAxes)
        wrapped = wrap_text_to_width(ax," "*17+lecture,0.99,fontsize=32)
        ax.text(0.005, -0.005,wrapped,fontsize=32,va='top',transform = ax.transAxes)
    plt.savefig('Cartes\\'+titre.replace('\n',' '), bbox_inches='tight')
    plt.show()
    if stats:
        print("Calcul des statistiques supplémentaires...")
        if isinstance(dataset, str):
            dataset = datasets[dataset]
        if variables[0][1].get('bivariate_cmap',False): # Légende bivariée
            dataset['global_class'] = list(zip(dataset['class_0'],dataset['class_1']))
        else: # Une seule variable
            dataset['global_class'] = classifier.yb
        if not pop_dataset is None:
            plot_dataset_pop = list_overlay([dataset[['global_class','geometry']],pop_dataset[[pop_variable,'code_iris','geometry']]],
                                            pop_dataset=pop_dataset, pop_variable=pop_variable,proportional=[False,True],
                                            corr_admin=corr_admin) # Fusion avec les données de population
            plot_dataset_pop.columns = plot_dataset_pop.columns.str.removesuffix('_0')
            plot_dataset_pop.columns = plot_dataset_pop.columns.str.removesuffix('_1')
            plot_dataset_pop['surface'] = plot_dataset_pop.geometry.area/1e6
            plot_dataset_pop = plot_dataset_pop.drop(columns=['geometry']).groupby('global_class').sum() # Calcul des populations par classe
            if plot_dataset_pop[pop_variable].max()/plot_dataset_pop[pop_variable].min()>10: # Données dans une grande plage
                logscale = True
            else:
                logscale = False
            print("Personnes vivant dans chaque classe : ")
            if len(variables)==2 and variables[0][1].get('bivariate_cmap',False):
                labels=plot_dataset_pop.index # Indexation du tableau par les deux classes
            print(tabulate(zip(labels,
                               plot_dataset_pop[pop_variable],
                               plot_dataset_pop[pop_variable]/plot_dataset_pop[pop_variable].sum()*100,
                               plot_dataset_pop['surface'],
                               plot_dataset_pop['surface']/plot_dataset_pop['surface'].sum()*100),
                           headers=['Classe','Population','Part de la population (%)','Surface','Part de la surface (%)'],
                           colalign=['left','right','right','right','right']))
            if not graphs is None:
                # Création des graphiques
                fig_pop, ax_pop = plt.subplots(figsize=(14, 8))
                plt.grid(axis='y',zorder=0)
                ax_pop.bar(x=plot_dataset_pop.index,height=plot_dataset_pop[pop_variable], log=logscale)
                ax_pop.set_title(graphs['pop_title'],fontsize=16, fontweight='bold')
                pop_labelpad = 4
                if graphs['pop_arrow']: # Remplacement des valeurs de la classe par une flèche indiquant le sens de variation
                    ax_pop.set_xticks([])
                    ax_pop.annotate('',xy=(0.95, -0.02),xytext=(0.05, -0.02),xycoords='axes fraction',
                                    arrowprops=dict(arrowstyle='->',lw=1))
                    ax_pop.text(0.5, -0.03,graphs['pop_arrow_text'],transform=ax_pop.transAxes,
                                ha='center',va='top',fontsize=10,fontweight='bold')
                    pop_labelpad = 20
                elif not variables[0][1].get('bivariate_cmap',False): # Une seule variable
                    ax_pop.set_xticks(plot_dataset_pop.index,labels=labels) 
                ax_pop.set_xlabel(graphs['pop_xlabel'], loc='right', labelpad=pop_labelpad,
                                  size=12, weight='bold')
                ax_pop.set_ylabel(graphs['pop_ylabel'], loc='top',
                                  size=12, weight='bold')
                plt.show()
            if not rev_dataset is None:
                plot_dataset_rev = list_overlay([dataset[['global_class','geometry']],rev_dataset[[rev_variable,'geometry']]],
                                                pop_dataset=pop_dataset, pop_variable=pop_variable,proportional=[False,False],
                                                corr_admin=corr_admin) # Fusion avec les données de revenu
                plot_dataset_rev.columns = plot_dataset_rev.columns.str.removesuffix('_0')
                plot_dataset_rev.columns = plot_dataset_rev.columns.str.removesuffix('_1')
                # plot_dataset_rev = plot_dataset_rev.drop(columns='code_insee')# TODO:filtration des colonnes pour éviter les erreurs de duplication
                plot_dataset_rev = plot_dataset_rev.drop(columns=['geometry']).groupby('global_class').mean()
                # Logscaling si les valeurs changent d'ordre de grandeur
                if plot_dataset_rev[rev_variable].max()/plot_dataset_rev[rev_variable].min()>10:
                    logscale = True
                else:
                    logscale = False
                print("Revenu moyen dans chaque classe : ")
                if len(variables)==2 and variables[0][1].get('bivariate_cmap',False):
                    labels=plot_dataset_rev.index # Indexation du tableau par les deux classes
                print(tabulate(zip(labels,plot_dataset_rev[rev_variable]),
                               headers=['Classe','Revenu moyen'],
                               colalign=['left','right']))
                if not graphs is None:
                    # Création des graphiques
                    fig_rev, ax_rev = plt.subplots(figsize=(14, 8))
                    plt.grid(axis='y',zorder=0)
                    ax_rev.bar(x=plot_dataset_rev.index,height=plot_dataset_rev[rev_variable], log=logscale)
                    ax_rev.set_title(graphs['rev_title'],fontsize=16, fontweight='bold')
                    rev_labelpad = 4
                    if graphs['rev_arrow']: # Remplacement des valeurs de la classe par une flèche indiquant le sens de variation
                        ax_rev.set_xticks([])
                        ax_rev.annotate('',xy=(0.95, -0.02),xytext=(0.05, -0.02),xycoords='axes fraction',
                                        arrowprops=dict(arrowstyle='->',lw=1))
                        ax_rev.text(0.5, -0.03,graphs['rev_arrow_text'],transform=ax_rev.transAxes,
                                    ha='center',va='top',fontsize=10,fontweight='bold')
                        rev_labelpad = 20
                    elif not variables[0][1].get('bivariate_cmap',False):
                        ax_rev.set_xticks(plot_dataset_rev.index,labels=labels)       
                    ax_rev.set_xlabel(graphs['rev_xlabel'], loc='right', labelpad=rev_labelpad,
                                      size=12, weight='bold')
                    ax_rev.set_ylabel(graphs['rev_ylabel'], loc='top',
                                      size=12, weight='bold')
                    plt.show()
    return None