# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 08:46:05 2026

@author: ternilp
"""
from adjustText import adjust_text
import atlas_modules.import_donnees as imp
from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.font_manager import FontProperties
from matplotlib_map_utils.core.scale_bar import scale_bar
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib import patheffects
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import mapclassify
import numpy as np
import re
from shapely.geometry import box
from shapely.geometry import Polygon
from shapely.ops import unary_union

base = cm.get_cmap("RdYlGn")
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

grids={
'reg' : {'edgecolor':'#232323', 'linewidth':3.9},
'dep' : {'edgecolor':'#232323', 'linewidth':1.95},
'com' : {'edgecolor':'#232323', 'linewidth':0.1625},
'iris' : {'edgecolor':'#232323', 'linewidth':0}, 
'pays_v' : {'edgecolor':'#232323', 'linewidth':2.6, 'facecolor':'#dddddd'},
'reg_v' : {'edgecolor':'#232323', 'linewidth':2.6, 'facecolor':'#eeeeee'},
'dep_v' : {'edgecolor':'#232323', 'linewidth':0.325},
'pref' : {'color':'#444444', 'marker':'s', 'markersize':200, 'fontsize':40},
's_pref' : {'color':'#666666', 'marker':'o', 'markersize':100, 'fontsize':30}
}

grids_masked = ['reg','dep','pays_v','reg_v','dep_v']
grids_no_mask = set(grids.keys())-set(grids_masked)-set(('pref','s_pref'))

def masks(texts, fig, ax):
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
    vertices = []
    codes = []
    for polygon in getattr(geom, "geoms", [geom]):
        exterior = polygon.exterior.coords
        vertices.extend(exterior)
        codes.extend([Path.MOVETO] + [Path.LINETO]*(len(exterior)-2) + [Path.CLOSEPOLY])
        for interior in polygon.interiors:
            coords = interior.coords
            vertices.extend(coords)
            codes.extend([Path.MOVETO] + [Path.LINETO]*(len(coords)-2) + [Path.CLOSEPOLY])
    return Path(vertices, codes)

def draw_prefs(ax, prefs):
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

def draw_background(fig, ax, texts, geogrids):
    # Masques pour lisibilité des textes
    text_masks = masks(texts,fig,ax)
    # Fonds de couleur
    ax.patch.set_facecolor('#a6cee3')
    for grid in ['reg_v','pays_v']:
        geogrids[grid].boundary.plot(ax=ax,edgecolor=None,
                                  facecolor=grids[grid]['facecolor'],
                                  zorder=0.8)
    geogrids_masked = {}
    # Soustraction des masques aux géométries
    for grid in grids_masked:
        grid_ = geogrids[grid].copy()
        grid_["geometry"] = grid_.geometry.boundary.apply(lambda line: line.difference(text_masks))
        geogrids_masked[grid] = grid_
    # Tracé des géométries masquées
    for grid in geogrids_masked.keys():
        geogrids_masked[grid].plot(ax=ax,edgecolor=grids[grid]['edgecolor'],
                                linewidth=grids[grid]['linewidth'],zorder=2.1)
    # Tracé des géométries non masquées
    for grid in grids_no_mask:
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
    ax.text(0.47e6,6.15e6,"Espagne", fontsize=48, weight='demi', zorder=3)
    andorre=ax.text(0.556e6,6.16e6,"Andorre", fontsize=48, weight='demi', zorder=3)
    andorre.set_path_effects([patheffects.withStroke(linewidth=5, foreground='#dddddd')])
    ax.text(0.43e6,6.4e6,"Nouvelle-Aquitaine", fontsize=42, weight='demi', zorder=3)
    ax.text(0.765e6,6.44e6,"Auvergne-Rhône-Alpes", fontsize=42, weight='demi', zorder=3)
    ax.text(0.85e6,6.3e6,"Provence-Alpes-\nCôte-d'Azur", fontsize=42, weight='demi', zorder=3)

def format_bin_labels(labels):
    """
    Passe les entrées fournies au format utilisé dans l'Atlas'

    Parameters
    ----------
    labels : list(str)
        Entrées de légende, telles que créées par mapclassify.

    Returns
    -------
    formatted_labels : list(str)
        Entrées de légende au format utilisé.

    """
    formatted_labels = []
    # évalue l'ordre de grandeur pour effectuer l'arrondi
    odg = int(np.ceil(np.log(float(re.findall(r'[\d\.\-]+', labels[-1])[1]))))
    rounding = max(0,2-odg)
    for n in range(len(labels)):
        # Extrait les nombres
        numbers = re.findall(r'[\d\.\-]+', labels[n])
        if rounding==0:
            lower = int(np.floor(float(numbers[0])))
        else:
            lower = np.round(float(numbers[0]),rounding)
        if n<len(labels)-1: # pas la dernière étiquette
            if rounding==0:
                upper = int(np.floor(float(numbers[1])))
            else:
                upper = np.round(float(numbers[1]),rounding)
        else:
            if rounding==0:
                upper = int(np.ceil(float(numbers[1])))
            else:
                upper = np.round(float(numbers[1]),rounding)
        formatted_labels.append(f"{lower} - {upper}")
    return formatted_labels

def wrap_text_to_width(ax, text, required_width, fontsize=12, **text_kwargs):
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

# TODO : 2D colorbar. Semble faisable assez facilement
def plot_data(ax, datasets, variables, pop_dataset=None, titre='', source=''):
    """
    Dessine les données sur ax comme spécifié dans variables. N'affiche qu'une variable à la fois
    """
    last_cmap = None
    last_norm = None
    last_label = None
    
    for dataset in variables:
        for n_var in range(variables[dataset][0]):
            if variables[dataset][1][n_var]['type'] in imp.cmap_data:
                cmap = variables[dataset][1][n_var]['couleur']
                if cmap in set(custom_cmaps.keys()):
                    cmap = custom_cmaps[cmap]
                if variables[dataset][1][n_var]['classification'] is None:
                    classification_kwds=None
                elif variables[dataset][1][n_var]['classification']=='HeadTailBreaks':
                    classification_kwds=None
                else:
                    classification_kwds=variables[dataset][1][n_var]['bins'].copy()
                plot_obj = datasets[dataset].plot(
                    ax=ax,
                    column=variables[dataset][1][n_var]['nom'],
                    cmap=cmap,
                    scheme=variables[dataset][1][n_var]['classification'],
                    classification_kwds=classification_kwds,
                    missing_kwds={
                        "color": "lightgrey",
                        "edgecolor": "k",
                        "hatch": "///",
                        "label": "Pas de données",
                    },
                    zorder=2,
                    label=variables[dataset][1][n_var]['nom_legende'],
                    legend=False
                )
                if variables[dataset][1][n_var]['classification']=='UserDefined':
                    legend_labels = getattr(mapclassify,variables[dataset][1][n_var]['classification'])(
                        datasets[dataset][variables[dataset][1][n_var]['nom']],
                        **variables[dataset][1][n_var]['bins'])\
                        .get_legend_classes()
                    legend_labels = format_bin_labels(legend_labels)
                elif variables[dataset][1][n_var]['classification']=='HeadTailBreaks':
                    legend_labels = getattr(mapclassify,variables[dataset][1][n_var]['classification'])(
                        datasets[dataset][variables[dataset][1][n_var]['nom']].dropna().to_numpy())\
                        .get_legend_classes()
                    legend_labels = format_bin_labels(legend_labels)
                elif variables[dataset][1][n_var]['classification'] in ['FisherJenks'] : # En prévision d'un ajout de classificaitons possibles
                    legend_labels = getattr(mapclassify,variables[dataset][1][n_var]['classification'])(
                        datasets[dataset][variables[dataset][1][n_var]['nom']].dropna().to_numpy(),
                        **variables[dataset][1][n_var]['bins'])   \
                        .get_legend_classes()
                    legend_labels = format_bin_labels(legend_labels)
                else : # Cas sans classification
                    legend_labels = variables[dataset][1][n_var]['labels']
                if hasattr(plot_obj, 'collections') and len(plot_obj.collections) > 0:
                    last_cmap = cmap
                    last_norm = plot_obj.collections[0].norm
                    last_label = variables[dataset][1][n_var]['nom_legende']
                    last_labels = legend_labels
                    last_scheme = variables[dataset][1][n_var]['classification']
                    last_type = variables[dataset][1][n_var]['type']
                    last_dataset = dataset
            elif variables[dataset][1][n_var]['type'] == 'Localisation de points':
                #todo:ors isochrones
                pass
            elif variables[dataset][1][n_var]['type'] == 'Compte':
                pass
                #todo: stuff
            elif variables[dataset][1][n_var]['type'] == 'Tracés':
                cmap = variables[dataset][1][n_var]['couleur']
                if cmap in set(custom_cmaps.keys()):
                    cmap = custom_cmaps[cmap]
                datasets[dataset][variables[dataset][1][n_var]['nom']] =\
                    datasets[dataset][variables[dataset][1][n_var]['nom']].astype("float64")
                plot_obj = datasets[dataset].geometry.plot(
                    ax=ax,
                    cmap=cmap,
                    zorder=2,
                    label=variables[dataset][1][n_var]['nom_legende'],
                    legend=False
                )
            #todo:check
    if last_cmap is not None:
        # Écrit le titre dans le vide pour mesurer la place nécessaire
        fp_title = FontProperties(weight='bold', size=48)
        temp_fig = plt.figure()
        temp_ax = temp_fig.add_subplot(111)
        title_text = temp_ax.text(0, 0, last_label, fontproperties=fp_title, ha='left', va='center')
        temp_fig.canvas.draw()
        title_bbox = title_text.get_window_extent(renderer=temp_fig.canvas.get_renderer())
        title_width = title_bbox.width
        title_height = title_bbox.height
        
        longest_label = ''
        for legend_label in last_labels:
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
        # Dimensions de la légende
        colorbar_width = 0.02
        title_margin_top = 0.008
        title_to_cbar_margin = 0.005
        if last_scheme is None:
            cbar_height = 0.1
        else:
            cbar_height = 0.03*len(last_labels)
        cbar_to_labels_margin = 0.005
        cbar_margin_bottom = 0.008
        required_width = (0.008 
                          + max(colorbar_width+cbar_to_labels_margin+label_width_axes, title_width_axes)
                          + 0.008)
        required_height = (title_margin_top + title_height_axes
                           + title_to_cbar_margin + cbar_height
                           + cbar_margin_bottom)
        # Positionnement depuis le coin inférieur droit
        x0 = max(0.65,0.9 - required_width*0.34/(0.34+0.09))
        y0 = 0.05
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
            0.008/required_width, 1-title_margin_top/required_height,
            last_label,
            ha='left', va='top',
            fontsize=48, fontweight='bold'
        )
        # Création de la légende
        cbar_ax = legend_box.inset_axes([0.008/required_width,
                                         0.008/required_height,
                                         0.02/required_width,
                                         cbar_height/required_height])
        if not last_scheme is None: # Données classées
            # Discrétisation de la colormap, ajout d'espaces entre les entrées de légende
            if isinstance(last_cmap,ListedColormap):
                legend_cmap = last_cmap.resampled(len(last_labels))
            else:
                legend_cmap = plt.get_cmap(last_cmap, len(last_labels))
            legend_cmap = [legend_cmap(i) for i in range(legend_cmap.N)]
            legend_cmap = [[legend_cmap[i],legend_cmap[i],legend_cmap[i],(0,0,0,0),(0,0,0,0)]
                           for i in range(len(legend_cmap))]
            legend_cmap=[c for c_list in legend_cmap for c in c_list]
            legend_cmap = legend_cmap[:-2]
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
            cbar.set_ticks([(1.5+5*i)/(5*len(last_labels)-2) for i in range(len(last_labels))],
                           labels=last_labels, fontsize=42)
            cbar_ax.tick_params(size=0, pad=cbar_to_labels_margin*fig_dpi*ax_width_inches)
        else: # Données non classées
            cbar = ColorbarBase(
                cbar_ax,
                cmap=last_cmap,
                norm=last_norm,
                orientation='vertical',
            )
            # Étiquettes
            cbar.set_ticks(
                [last_norm.vmin*0.9+last_norm.vmax*0.1, last_norm.vmin*0.1+last_norm.vmax*0.9],
                labels=last_labels, fontsize=42
            )
        if last_type=='Densité':
            cbar.ax.invert_yaxis()
        cbar_ax.set_facecolor('none')
        cbar_ax.patch.set_alpha(0)
        # Titre
        ax.text(0.41e6,6.495e6,
            titre,
            fontsize=100, fontweight='bold', va='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
        )
        # Source
        ax.text(x0,y0-0.005,
            "Source :", fontweight='bold',fontsize=32,va='top',
            transform=ax.transAxes
        )
        wrapped = wrap_text_to_width(
            ax,
            source,
            required_width,
            fontsize=32
        )
        ax.text(x0, y0-0.005,
            wrapped,
            fontsize=32,va='top',
            transform = ax.transAxes
        )
        # Échelle
        scale_bar(ax, location="lower left", style="ticks",
                  bar={"projection": datasets[last_dataset].crs,
                       'height':0.5, 'tickwidth':5,
                       'minor_div':5,'minor_type':'first'},
                  labels={'fontsize':42},
                  units={'fontsize':48})
    return ax