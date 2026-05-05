# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:06:35 2026

@author: ternilp
"""
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import fiona
import operator
from matplotlib.colors import is_color_like
import numpy as np

# Types de géométries de raccordement
geom_dict = {'département':'dep','departement':'dep','dép':'dep','dep':'dep',
             'commune':'com','comm':'com','com':'com',
             'iris':'iris',
             'maille safran':'maille_safran','safran':'maille_safran',
             'maille drias':'maille_drias','drias':'maille_drias',
             'autre':None
             }

# Noms de colonnes dans les données pouvant être raccordées automatiquement aux références
geom_columns_dict = {'dep':['département','departement','dép','dep',
                            'code_département','code_departement',
                            'code_dép','code_dep','dept'],
                     'com':['commune','comm','com',
                            'code_commune','code_comm','code_com',
                            'code_insee','codgeo'],
                     'iris':['iris','code_iris','code iris'],
                     'maille_safran':['id safran','id_safran','maille_safran'], # Au moins un jeu de données confond les mailles safran et drias
                     'maille_drias':['id safran','id_safran','maille_drias'] # Raccordement automatique peu fiable
                     }
# Noms de colonnes des géométries de référence
geom_grid_dict = {'dep':'code','com':'code_insee','iris':'code_iris',
                  'maille_safran':'maille_safran','maille_drias':'maille_drias'}
# Noms de colonnes canoniques, utilisés pour les aggrégations
expected_admin_columns = {'code_iris':'iris','code_insee':'com','EPCI':'epci','DEP':'dep'}
expected_geom_columns = {'maille_safran','maille_drias','id_car'}
# Type de données affichables
data_types = ['Densité','Score','Compte','Localisation de points','Tracés']
cmap_data = ['Densité','Score']

cmap_classification={'Aucune':None,'Jenks':'FisherJenks','Tête/Queue':'HeadTailBreaks','Personnalisée':'UserDefined'}
cmap_classification_entries={'aucune':'Aucune','none':'Aucune',
                             'jenks':'Jenks',
                             'tête/queue':'Tête/Queue','tete/queue':'Tête/Queue','têtequeue':'Tête/Queue','tetequeue':'Tête/Queue',
                             'personnalisée':'Personnalisée','personnalisee':'Personnalisée','perso':'Personnalisée'}

# Liste des fichiers contenant des données
file_list = [x for x in Path(r"Données traitées").glob("**/*") if x.is_file()]
for georef in (Path(r"Données traitées\Région.gpkg"),
               Path(r"Données traitées\Départements.gpkg"),
               Path(r"Données traitées\EPCI.gpkg"),
               Path(r"Données traitées\Communes.gpkg"),
               Path(r"Données traitées\IRIS.gpkg"),
               Path(r"Données traitées\Pays limitrophes.gpkg"),
               Path(r"Données traitées\Régions limitrophes.gpkg"),
               Path(r"Données traitées\Départements limitrophes.gpkg"),
               Path(r"Données traitées\maille drias.gpkg"),
               Path(r"Données traitées\maille safran.gpkg"),
               Path(r"Données traitées\Correspondance_echelle_admin.csv")):
    file_list.remove(georef)
file_list = [file.stem for file in file_list]

deprecated_codes={'12076':'12218','120760000':'122180000'}

# Liste des formats supportés
format_list=['GeoPackage (.gpkg)','Comma Separated Values (.csv)']

# Traitements possibles à l'importation
treatments = ['MeanRatio','MedianRatio','Aucun']

# Méthodes d'opérations entre variables
methods = ["Indicateur synthétique","Intersection et union géographique"]
methods_entries = {'indicateur':"Indicateur synthétique",'indic':"Indicateur synthétique",'synth':"Indicateur synthétique",
                   'inter':"Intersection et union géographique",'intersection':"Intersection et union géographique",
                   'cumul':"Intersection et union géographique",'union':"Intersection et union géographique"}
inter_methods = ['Union','Intersection']
inter_methods_entries = {'inter':'intersection','1':'intersection','0':'union'}
# Méthodes de sélection du seuil pour intersection géographique
level_method = ["Valeur","Quantile"]
level_dir_list = ["Au-dessus du seuil","En-dessous du seuil"]
level_dir_entries = {'1':"Au-dessus du seuil",'0':"En-dessous du seuil"}
# Échelles d'affichage
admin_scales = {'département':['epci','com','iris'],'epci':['com','iris'],'commune':['iris'],'iris':None}
admin_scales_entries = {'dep':'département',
                        'com':'commune'}
admin_scales_names = {'département':'DEP','epci':'EPCI','commune':'code_insee','iris':'code_iris'}
admin_scales_size = {'dep':4,'epci':3,'com':2,'iris':1}

def safe_ask(query,datatype):
    """
    Demande à l'utilisateur des données d'un type spécifié.
    Réitère la demande jusqu'à ce que les types correspondent

    Parameters
    ----------
    query : str
        Texte affiché lors de la demande de données.
    datatype : type, list(type) ou fonction
        Type de données à demander ou fonction vérifiant l'interprétabilité

    Returns
    -------
    var : datatype
        Donnée demandée à l'utilisateur.

    """
    var = input(query)
    if isinstance(datatype,type):
        while not isinstance(var, datatype):
            try:
                if datatype is bool:
                    try:
                        var = int(var)
                    except:
                        if var=='False':
                            var=False
                var = datatype(var)
            except:
                print("Type invalide.")
                var = input(query)
        return var
    elif isinstance(datatype,list):
        valid_cast = False
        while not valid_cast:
            for datatype_ in datatype:
                try:
                    var = datatype_(var)
                    valid_cast = True
                    break
                except:
                    pass
            if not valid_cast:
                print("Type invalide.")
                var = input(query)
        return var
    else: # datatype est une fonction de vérification
        valid_cast = datatype(var)
        while not valid_cast:
            var = input(query)
            valid_cast = datatype(var)
        return var
                
def select_list(choices, query=None, fail_query=None, catch_dict=None):
    """
    Demande à l'utilisateur de choisir un élément dans une liste de str

    Parameters
    ----------
    choices : list(str)
        Listes des valeurs possibles.
    query : str, optional
        Demande à transmettre à l'utilisateur. Par défaut, None.
    fail_query : str, optional
        Demande réitérée en cas de valeur invalide. Par défaut, None.
    catch_list : dict, optional
        Tentatives de rattrapage des erreurs d'input
    
    Returns
    -------
    str
        Choix retenu.

    """
    if query is None:
        query = f"Sélectionnez un élément parmi :\n\
        \r\t- {'\n\r\t- '.join(choices)}\n"
    element = input(query)
    if not catch_dict is None:
        try:
            element = catch_dict[element.lower()]
        except:
            pass
    found = False
    while not found:
        for n in range(len(choices)):
            if element.lower() == choices[n].lower():
                return choices[n]
        if fail_query is None :
            fail_query = f"'{element}' n'est pas un choix valide.\n"+query
        element = input(fail_query)
        if not catch_dict is None:
            try:
                element = catch_dict[element]
            except:
                pass

def import_progress(filepath, treatment=None, compact=True):
    """
    Importe des données géographiques ou non en affichant l'état d'avancement.
    Formats supportés :
        GeoPackage (.gpkg)
        Comma Separated Values (.csv)
    
    Parameters
    ----------
    filepath : str
        Chemin vers le fichier cible.
    compact : bool
        True pour un affichage restreint, False pour un affichage détaillé
    
    Returns
    -------
    data : GeoPandas.GeoDataFrame
        Données du fichier cible.
    
    """
    if isinstance(filepath,str):
        filepath = Path(filepath)
    if filepath.suffix=='.gpkg':
        with fiona.open(filepath) as src:
            # Initialisation des variables de comptage
            if not compact:
                print(f"{filepath}")
            total = len(src)
            features = []
            last_percent = -1
            # Efface la ligne avant de commencer
            sys.stdout.write("\x1b[1K\r")
            # Importation des données par ligne et comptage
            for i, feat in enumerate(src, 1):
                features.append(feat)
                percent = int((i / total) * 100)
                if percent != last_percent:
                    last_percent = percent
                    if compact:
                        sys.stdout.write(f"\r{filepath} : Importation... {percent}% ({i}/{total})")
                    else:
                        sys.stdout.write(f"\rImportation... {percent}% ({i}/{total})")
                    sys.stdout.flush()
        data = gpd.GeoDataFrame.from_features(features, crs=src.crs)
        data.attrs['name'] = filepath.stem
        if not compact:
            print("\nImportation terminée !", flush=True)
        return data
    elif filepath.suffix=='.csv':
        if not compact:
            print(f"{filepath}")
        # Initialisation des variables de comptage, en blocs d'environ 1% du nombre d'IRIS
        chunk_size = 50
        total = sum(1 for _ in open(filepath)) - 1
        chunks = []
        last_percent = -1
        read_rows = 0
        # Efface la ligne avant de commencer
        sys.stdout.write("\x1b[1K\r")
        # Importation des données par bloc et comptage
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, dtype='string'):
            chunks.append(chunk)
            read_rows += len(chunk)
            percent = int((read_rows / total) * 100)
            if percent != last_percent:
                last_percent = percent
                if compact:
                    sys.stdout.write(f"\r{filepath} : Importation... {percent}% ({read_rows}/{total})")
                else:
                    sys.stdout.write(f"\rImportation... {percent}% ({read_rows}/{total})")
                sys.stdout.flush()
        data = pd.concat(chunks, ignore_index=True)
        data.attrs['name'] = filepath.stem
        data.loc[:,data.columns.str.endswith('_data')]=data.loc[:,data.columns.str.endswith('_data')].apply(pd.to_numeric).convert_dtypes()
        if treatment == 'MeanRatio':
            for column in data.columns[data.columns.str.endswith('_data')]:
                data[column] = data[column]/data[column].mean()
        elif treatment == 'MedianRatio':
            for column in data.columns[data.columns.str.endswith('_data')]:
                data[column] = data[column]/data[column].median()
        data.columns = data.columns.str.removesuffix('_data')
        if not compact:
            print("\nImportation terminée !", flush=True)
        return data
    else:
        raise TypeError("Format de données non supporté. Formats valides :\n\
                    \r\t- {'\n\r\t- '.join(format_list)}\n")

def import_fast(filepath, treatment=None, compact=True):
    """
    Importe des données géographiques ou non.
    Formats supportés :
        GeoPackage (.gpkg)
        Comma Separated Values (.csv)
    
    Parameters
    ----------
    filepath : str
        Chemin vers le fichier cible.
    compact : bool
        True : Sans affichage
        False : Affichage restreint
    
    Returns
    -------
    data : GeoPandas.GeoDataFrame
        Données du fichier cible.
    
    """
    if isinstance(filepath,str):
        filepath = Path(filepath)
        if not compact:
            print(f"{filepath}")
    if filepath.suffix=='.gpkg':
        data = gpd.read_file(filepath)
        if treatment == 'MeanRatio':
            for column in data.columns.drop('geometry'):
                data[column] = data[column]/data[column].mean()
        elif treatment == 'MedianRatio':
            for column in data.columns.drop('geometry'):
                data[column] = data[column]/data[column].median()
        data.attrs['name'] = filepath.stem
        return data
    elif filepath.suffix=='.csv':
        data = pd.read_csv(filepath, dtype='string')
        for col in data.columns[data.columns.str.endswith('_data')]:
            data[col] = pd.to_numeric(data[col])
        if treatment == 'MeanRatio':
            for column in data.columns[data.columns.str.endswith('_data')]:
                data[column] = data[column]/data[column].mean()
        elif treatment == 'MedianRatio':
            for column in data.columns[data.columns.str.endswith('_data')]:
                data[column] = data[column]/data[column].median()
        data.columns = data.columns.str.removesuffix('_data')
        data.attrs['name'] = filepath.stem
        return data
    else:
        raise TypeError(f"Format de données non supporté. Formats valides :\n\
                    \r\t- {'\n\r\t- '.join(format_list)}\n")

def ask_filepath():
    """
    Demande à l'utilisateur un nom de fichier.
    Le vérifie et le reconstruit si nécessaire avec check_filepath
    En cas d'échec, demande un nouveau nom de fichier.
    
    Returns
    -------
    filepath : pathlib.Path
        Chemin vers le fichier demandé à l'utilisateur.
    
    """
    filepath = Path(input("\rNom du fichier contenant les données :\n"))
    while not check_filepath(filepath)[0]==3:
        if check_filepath(filepath)[0]==-1:
            filepath = Path(input(f"Erreur : Le nom {filepath} correspond à plusieurs fichiers dans l'environnement de travail.\n\
                                  \rVeuillez préciser le format du fichier.\n\
                                  \rNom du fichier contenant les données (extension incluse):"))
        elif check_filepath(filepath)[0]<2:
            filepath = Path(input(f"Erreur : {filepath} ne correspond à aucun fichier dans l'environnement de travail.\n\
                                  \rListe des fichiers de données :\n\
                                  \r\t- {'\n\r\t- '.join(file_list)}\n\
                                  \rNom du fichier contenant les données :\n"))
        else:
            filepath = Path(input(f"Erreur : Format de données non supporté. Formats valides :\n\r\t- \
                                 {'\n\r\t- '.join(format_list)}\n\
                                 \rNom du fichier contenant les données :\n"))
    return filepath

def check_filepath(filepath):
    """
    Vérifie la validité du chemin fourni.
    Tente de le reconstruire si le chemin est invalide en l'état.
    
    Parameters
    ----------
    filepath : pathlib.Path
        Chemin vers le fichier à importer.
    
    Returns
    -------
    valid : int
        0 ou 1 si le fichier n'existe pas
        2 si le fichier existe, mais que son format n'est pas supporté
        -1 si plusieurs fichiers correspondent au nom fourni
        3 si le fichier existe et est dans un format supporté
    
    filepath : pathlib.Path
        Chemin vers le fichier à importer, reconstruit si nécessaire.
    
    """
    # Vérification de l'existence du fichier
    if len(filepath.parts)==0: # Chemin vide
        return 0,filepath
    valid=2
    if not filepath.exists(): # Non : Chemin depuis le dossier de travail, avec extension
        if ("Données traitées"/filepath).exists():  # Nom du fichier, avec extension
            filepath = "Données traitées"/filepath
        elif (matches:=[p for p in Path("Données traitées").iterdir()    # Nom du fichier, sans extension
                       if p.is_file() and p.stem.lower() == str(filepath).lower()])!=[]:
            if len(matches)>1:
                return -1, filepath
            else: filepath=matches[0]
        elif Path("Données traitées\\"+filepath.parts[-1]).exists():    # Chemin depuis un dossier parent
                filepath = Path("Données traitées\\"+filepath.parts[-1])
        elif Path("Données traitées\\"+filepath.parts[-1].strip('"')).exists(): # Chemin depuis un dossier parent, avec ""
            filepath = Path("Données traitées\\"+filepath.parts[-1].strip('"'))
        else:
            valid=0
    # Vérification de la validité du format
    stypes = ['.gpkg','.csv']
    stypes = [(stype,len(stype)) for stype in stypes]
    for (stype,i) in stypes:
        if filepath.parts[-1][-i:]==stype:
            valid = valid+1
    return valid,filepath

def ask_geom(data):
    """
    Demande à l'utilisateur un nom de géométrie à laquelle raccrocher les données.
    En cas d'échec, répète la demande.
    
    Returns
    -------
    geom : str
        Nom de la géométrie à laquelle raccrocher les données.
    
    """
    geom = input("Géométrie de référence parmi :\n\t- Département/dep\n\
                \r\t- Commune/com\n\t- IRIS\n\t- Maille Safran\n\t- Maille Drias\n\t- Autre\n\
                \rSélectionnez 'Autre' pour des données dont la localisation est incluse.\n").lower()
    while not geom in geom_dict.keys():
        geom = input(f"\nErreur :'{geom}' n'est pas une échelle valide.\n\
                   \rSélectionnez une échelle parmi :\n\t- Département/dep\n\
                   \r\t- Commune/com\n\t- IRIS\n\t- Maille Safran\n\t- Maille Drias\n\t- Autre\n\
                   \r Sélectionnez 'Autre' pour des données dont la localisation est incluse.\n").lower()
    geom = geom_dict[geom]
    if geom !=None:
        for column in data.columns:
            if data[column].dtype=='geometry':
                del_geom = input(f"Attention : Le jeu de données fourni contient déjà une géométrie ({column}).\n\
                                 \rSupprimer la géométrie actuelle et procéder au raccordement ? [y]/n\n")
                while del_geom not in ['y','n','']:
                    del_geom = input(f"Attention : Le jeu de données fourni contient déjà une géométrie ({column}).\n\
                                     \rSupprimer la géométrie actuelle et procéder au raccordement ? [y]/n\n")
                if del_geom=='n':
                    del_geom = input("Utiliser la géométrie interne au jeu de données fourni ? [y]/n\n\
                                     \rSélectionnez 'n' pour recommencer l'importation des données.\n")
                    while del_geom not in ['y','n','']:
                        del_geom = input("Utiliser la géométrie interne au jeu de données fourni ? [y]/n\n\
                                         \rSélectionnez 'n' pour recommencer l'importation des données.\n")
                    if del_geom in ['y','']:
                        geom = None
                    else:
                        raise ValueError("Retour au début de l'importation")
                else:
                    print("Suppression de la géométrie des données.")
                    data = data.drop(column, axis=1)
    return geom

def search_geom(geom_grid, data, geom_data=None):
    """
    Obtient le nom exact d'une colonne à partir d'approximations de l'utilisateur
    
    Parameters
    ----------
    geom_grid : str
        Nom de la géométrie de raccordement.
        None si les données contiennents leur géométrie
    data : Pandas.DataFrame ou GeoPandas.GeoDataFrame
        Données.
    geom_data : str, optional
        Nom de la colonne de data proposée pour être la clé de raccordement.
        Nom de la colonne proposée pour être la géométrie si geom_grid est None
        Par défaut, None.
    
    Returns
    -------
    geom_data : str
        Nom de la colonne de data permettant la jointure sur la géométrie de référence.
        Si la géométrie est incluse dans les données, nom de la colonne contenant la géométrie
    
    """
    if geom_grid is None:    # Géométrie interne aux données 
        if geom_data is None:   # Premier essai, recherche de la géométrie
            geoms = data.dtypes[data.dtypes=='geometry']
            if len(geoms)==1:
                return geoms.index[0]
            elif len(geoms)>1:   # Si plusieurs géométries sont trouvées, demande à l'utilisateur laquelle choisir
                geom_data = input(f"\nLes données contiennent plusieurs géométries.\n\
                                  \rSélectionnez la géométrie voulue parmi :\n\
                                  \r\t- {'\n\r\t- '.join(geoms)}\n")
                return search_geom(geom_grid, data, geom_data)
            else:
                print("Les données ne contiennent pas de géométrie.\n\
                      \rRetour à la sélection de la géométrie de référence.")
                return ''
        else:   # Deuxième essai, confirmation de la géométrie fournie par l'utilisateur
            geoms = data.dtypes[data.dtypes=='geometry']
            columns = []
            for column in geoms.index:
                if column.lower()==geom_data.lower():
                    columns.append(column)
            if len(columns)==1:
                return columns[0]
            else :
                raise ValueError(f"{data.attrs['name']} contient plusieurs colonnes portant ce nom et contenant une géométrie.\n\
                                 \rIl est impossible de l'utiliser en l'état")
    elif geom_data is None:    # Géométrie 'type' (dans geom_columns_dict)
        geoms = geom_columns_dict[geom_grid]
        columns = data.columns
        for column in columns:
            if column.lower() in geoms:
                print("Géométrie de raccordement trouvée.")
                return column
        geom_data = input(f"\nErreur : géométrie de raccordement introuvable dans les données.\n\
                     \rSélectionnez la clé de raccordement parmi :\n\
                     \r\t- {'\n\r\t- '.join(columns)}\n\
                     \rAppuyez sur Entrée pour revenir au choix de la géométrie de référence.\n")
        return search_geom(geom_grid, data, geom_data)
    elif geom_data=='':
        return geom_data
    else:   # Géométrie non prévue dans geom_columns_dict, nom de la colonne fourni
        columns=data.columns
        for column in columns:
            if geom_data.lower()==column.lower():
                return column
        geom_data = input(f"\nErreur : la clé fournie est absente des données.\n\
                     \rSélectionnez la clé de raccordement parmi :\n\
                     \r\t- {'\n\r\t- '.join(columns)}\n")
        return search_geom(geom_grid, data, geom_data)

def list_overlay(df_list,proportional=False):
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
        en fonction du ratio d'aire lors des découpages géométriques.
            - Si bool : appliqué à tous les datasets
            - Si list(bool) : un par dataset, indique si ses colonnes doivent être réparties
              proportionnellement lors des découpages ultérieurs
    
    Returns
    -------
    result
        Overlay des GeoDataFrame passés en argument.
    
    """
    # Application de proportional à tous les datasets
    if isinstance(proportional, bool):
        proportional = [proportional] * len(df_list)
    # Suppression des géométries en double, ajout de suffixes aux colonnes
    # pour tracer les données et éviter les erreurs dans gpd.overlay
    source_dfs = []
    admin_size = 'dep'
    admin_mixed = False
    for idx, gdf in enumerate(df_list):
        unique_gdf = gdf.drop_duplicates(subset='geometry')
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
    for i in range(1, len(source_dfs)):
        # Stockage des aires originales pour les sources qui nécessitent
        # une répartition proportionnelle
        for source_idx in range(i):
            if proportional[source_idx]:
                source_cols = [col for col in result.columns if col.endswith(f'_{source_idx}')
                               and not col.startswith('__')]
                if source_cols:
                    result[f'__area_original_{source_idx}'] = result.geometry.area
        # Stockage des aires pour la nouvelle source si nécessaire
        if proportional[i]:
            source_dfs[i][f'__area_original_{i}'] = source_dfs[i].geometry.area
        # Overlay
        result = gpd.overlay(result, source_dfs[i],keep_geom_type=True)
        # Application de la répartition proportionnelle pour toutes les sources
        for source_idx in range(i + 1):
            area_col = f'__area_original_{source_idx}'
            if proportional[source_idx] and area_col in result.columns:
                mask = result[area_col] > 0
                if mask.any():
                    result.loc[mask, '__ratio'] = result.loc[mask, 'geometry'].area / result.loc[mask, area_col]
                    source_cols = [col for col in result.columns if col.endswith(f'_{source_idx}')]
                    for col in source_cols:
                        if pd.api.types.is_numeric_dtype(result[col]):
                            result[col] = result[col].astype(float)
                            result.loc[mask, col] = result.loc[mask, col] * result.loc[mask, '__ratio']
                    result = result.drop(columns=['__ratio'])
                result = result.drop(columns=[area_col])
        # Suppression des artéfacts (très petits polygones)
        if len(result) > 0:
            result = result[result.geometry.area > 1]
            result = result.reset_index(drop=True)
    # Nettoyage des colonnes temporaires
    result = result.drop([col for col in result.columns if col.startswith('__')], axis=1)
    result.attrs = {'scale':admin_size,'admin_mixed':admin_mixed}
    return result

def operate(datasets, names, operations, true_geom=False, pop_dataset=None, pop_variable='C21_PMEN'):
    """
    Calcule le résultat des opérations fournies sur les données fournies, dans l'ordre de priorité usuel.
    Ne fonctionne qu'avec les 4 opérations de base
    
    Parameters
    ----------
    datasets : list(pd.DataFrame)
        Jeux de données contenant les séries sur lesquelles appliquer les opérations.
    names : list(str)
        Noms des colonnes sur lesquelles appliquer les opérations
    operations : list(str)
        Opérations à appliquer, parmi les 4 opérations de base
    true_geom : bool, optional
        Si True, opération sur les localisations des données.
        Si False, opération sur les indices.
        Pas défaut, False.
    pop_dataset : pd.geoDataFrame, optional
        Jeu de données contenant la population, pour analyse statistique.
        Par défaut, None.
    pop_variable : str, optional
        Nom de la colonne de pop_dataset contenant la variable de population à étudier.
        Par défaut 'C21_PMEN' (population générale).
    
    Returns
    -------
    pd.Series
        Série contenant le résultat des opérations fournies.
    pd.Series : optional
        Séries contenant la population, si pop=True

    """
    if true_geom:
        # Construit la géométrie correspondant aux intersections de chaque géométrie,
        # sans répétition pour éviter les artéfacts
        if not pop_dataset is None: # Ajoute la population aux variables calculées
            datasets.append(pop_dataset)
        print("Préparation de la jointure des données : pour chaque variable, entrer :\n\
              \r\t- 0 si c'est une grandeur intensive (à ne pas répartir, i.e. âge)\n\
              \r\t- 1 si c'est une grandeur extensive (à répartir, i.e. population)")
        proportional=[]
        for name in names:
            proportional.append(bool(int(input(f"{name} : "))))
        if not pop_dataset is None:
            proportional.append(True)
        overlay_result = list_overlay(datasets, proportional)
        # Construit l'expression arithmétique à partir des colonnes suffixées
        expr_terms = []
        for i, (name, op) in enumerate(zip(names, operations + [''])):
            col_name = f"{name}_{i}"
            if col_name in overlay_result.columns:
                expr_terms.append(f"overlay_result['{col_name}']")
            if op:
                expr_terms.append(op)
        # Évalue l'expression
        expression = ' '.join(expr_terms)
        overlay_result['result'] = eval(expression)
        # Colonnes d'indicateur géographique pour aggrégation éventuelle, premières d'un dataset par construction
        cols_to_keep = [overlay_result.columns[0]]
        for col in list(overlay_result.columns)[:-2]:
            if col[-1]!=cols_to_keep[-1][-1]:
                cols_to_keep.append(col)
        if not pop_dataset is None:
            overlay_result['population'] = overlay_result[f'{pop_variable}_{len(datasets)-1}']
            overlay_result = overlay_result[cols_to_keep+['result','population','geometry']]
            overlay_result = overlay_result.rename(
                dict(zip(cols_to_keep,[col[:-2] for col in cols_to_keep])),axis=1)
            # Suppression des doublons
            overlay_result = overlay_result.loc[:,~overlay_result.columns.duplicated()]
            return overlay_result
        else:
            overlay_result = overlay_result[cols_to_keep+['result','geometry']]
            overlay_result = overlay_result.rename(
                dict(zip(cols_to_keep,[col[:-2] for col in cols_to_keep])),axis=1)
            overlay_result = overlay_result.loc[:,~overlay_result.columns.duplicated()]
            return overlay_result
    else:
        series = [datasets[n][names[n]] for n in range(len(datasets))]
        prio_dict={'*':1,'/':1,'+':2,'-':2}
        op_dict={'*':operator.mul,
                 '/':operator.truediv,
                 '+':operator.add,
                 '-':operator.sub}
        ops_prio=[]
        for op in operations:
            ops_prio.append(prio_dict[op])
        while len(operations)>0:
            current_prio=max(ops_prio)
            for n in range(len(operations)):
                if ops_prio[n]==current_prio:
                    op = op_dict[operations[n]]
                    result = op(series[n], series[n+1])
                    series[n] = result
                    series.pop(n+1)
                    operations.pop(n)
                    ops_prio.pop(n)
                    break # Une seule opération par parcours d'operations
        return series[0]

def intersect(datasets, names, levels, level_qs, level_dirs, methods=None,
              true_geom=False, pop_dataset=None, pop_variable='POP21_POP'):
    if methods is None:
        methods = ['Intersection' for n in range(len(datasets)-1)]
    if true_geom:
        # Construit la géométrie correspondant aux intersections de chaque géométrie,
        # sans répétition pour éviter les artéfacts
        if not pop_dataset is None: # Ajoute la population aux variables calculées
            datasets.append(pop_dataset)
        print("Préparation de la jointure des données : pour chaque variable, entrer :\n\
              \r\t- 0 si c'est une grandeur intensive (à ne pas répartir, i.e. âge)\n\
              \r\t- 1 si c'est une grandeur extensive (à répartir, i.e. population)\n\
              \rNote : les seuils en valeur sur des grandeurs extensives peuvent perdre leur sens")
        proportional=[]
        for name in names:
            proportional.append(bool(int(input(f"{name} : "))))
        if not pop_dataset is None:
            proportional.append(True)
        overlay_result = list_overlay(datasets, proportional)
        # Pour chaque variable, remplace les valeurs par la proportion de seuils franchis
        # Le sens de franchissement du seuil est spécifié dans level_dirs
        for i, (name,level,level_q,level_dir) in enumerate(zip(names,levels,level_qs,level_dirs)):
            if level_q == "Valeur":
                true_level = np.array(level)[None,:]
            else:
                true_level = np.array([overlay_result[f"{name}_{i}"].quantile(lev) for lev in level])[None,:]
            if level_dir == "Au-dessus du seuil":
                counts = (overlay_result[f"{name}_{i}"].to_numpy()[:,None] >= true_level).sum(axis=1)
            else:
                counts = (overlay_result[f"{name}_{i}"].to_numpy()[:,None] <= true_level).sum(axis=1)
            overlay_result[f"{name}_{i}"] = counts/counts.max()
        # Calcul des unions (max) et intersections (min)
            # Les unions sont prioritaires sur les intersections
        if methods==['Intersection' for n in range(len(datasets)-1)]:
            overlay_result['result'] = overlay_result[[f"{name}_{i}" for i,name in enumerate(names)]].min(axis=1)
        elif methods==['Union' for n in range(len(datasets)-1)]:
            overlay_result['result'] = overlay_result[[f"{name}_{i}" for i,name in enumerate(names)]].max(axis=1)
        else:
            temp_result=pd.DataFrame()
            n=0
            while n<len(methods):
                if methods[n]=='Union':
                    temp_result[f"{n}"] = pd.concat([overlay_result[f"{names[n]}_{n}"],overlay_result[f"{names[n+1]}_{n+1}"]],axis=1).max(axis=1)
                    n+=1
                else:
                    temp_result[f"{n}"] = overlay_result[f"{names[n]}_{n}"]
                n+=1
                overlay_result['result'] = temp_result.min(axis=1)
        # Sélection des colonnes utiles
        # Colonnes d'indicateur géographique pour aggrégation éventuelle, premières d'un dataset par construction
        cols_to_keep = [overlay_result.columns[0]]
        for col in list(overlay_result.columns)[:-2]:
            if col[-1]!=cols_to_keep[-1][-1]:
                cols_to_keep.append(col)
        if not pop_dataset is None:
            overlay_result['population'] = overlay_result[f'{pop_variable}_{len(datasets)-1}']
            overlay_result = overlay_result[cols_to_keep+['result','population','geometry']]
            overlay_result = overlay_result.rename(
                dict(zip(cols_to_keep,[col[:-2] for col in cols_to_keep])),axis=1)
            # Suppression des doublons
            overlay_result = overlay_result.loc[:,~overlay_result.columns.duplicated()]
            return overlay_result
        else:
            overlay_result = overlay_result[cols_to_keep+['result','geometry']]
            overlay_result = overlay_result.rename(
                dict(zip(cols_to_keep,[col[:-2] for col in cols_to_keep])),axis=1)
            overlay_result = overlay_result.loc[:,~overlay_result.columns.duplicated()]
        return overlay_result
    else:
        series = [datasets[n][names[n]] for n in range(len(datasets))]
        for i, (level,level_q,level_dir) in enumerate(zip(levels,level_qs,level_dirs)):
            if level_q == "Valeur":
                true_level = np.array(level)[None,:]
            else:
                true_level = np.array([series[i].quantile(lev) for lev in level])[None,:]
            if level_dir == "Au-dessus du seuil":
                counts = (series[i].to_numpy()[:,None] >= true_level).sum(axis=1)
            else:
                counts = (series[i].to_numpy()[:,None] <= true_level).sum(axis=1)
            series[i] = pd.Series(counts/counts.max())
        if methods==['Intersection' for n in range(len(datasets)-1)]:
            intersection = pd.DataFrame(pd.concat(series,axis=1).min(axis=1))
        elif methods==['Union' for n in range(len(datasets)-1)]:
            intersection = pd.DataFrame(pd.concat(series,axis=1).max(axis=1))
        else:
            temp_result=pd.DataFrame()
            n=0
            while n<(len(methods)):
                if methods[n]=='Union':
                    temp_result[f"{n}"] = pd.concat([series[n],series[n+1]],axis=1).max(axis=1)
                    n+=1
                else:
                    temp_result[f"{n}"] = series[n]
                n+=1
            intersection = temp_result.min(axis=1)
        return intersection

def build_variables(datasets,pop_dataset):
    relations = input("Calcul de variables supplémentaires à partir des données existantes ? y/[n] ")
    while relations not in ['y','n','']:
        relations = input("Calcul de variables supplémentaires à partir des données existantes ? y/[n] ")
    if relations =='y':
        n_rel = safe_ask("Nombre de variables à calculer :\n", int)
        rel=0
        for rel in range(n_rel):
            print(f"\nVariable composite n°{rel+1}\n{'-'*(22+(rel+1)//10)}")
            new_var_name = input("Nom de la variable à créer :\n")
            n_var = safe_ask("Nombre de variables source :\n",int)
            method = select_list(methods,
                                 query=f"Méthode de création de la variable :\n\
                                     \r\t- {'\n\r\t- '.join(methods)}\n",
                                 catch_dict=methods_entries)
            true_geom_op = False
            rel_sets = []
            var_names = []
            if method == "Indicateur synthétique":
                operations = []
                for var in range(n_var):
                    rel_sets.append(
                        datasets[select_list(
                            list(datasets.keys()),
                            f"Nom du jeu de données source n°{var+1} parmi :\n\
                            \r\t- {'\n\r\t- '.join(list(datasets.keys()))}\n")])
                    if rel_sets[0].shape[0]!=rel_sets[-1].shape[0]:
                        true_geom_op=True
                    var_names.append(
                        select_list(
                            rel_sets[-1].columns,
                            f"Nom de la série de données n°{var+1} parmi :\n\
                            \r\t- {'\n\r\t- '.join(rel_sets[-1].columns)}\n"))
                    if var < n_var-1:
                        operations.append(
                            select_list(
                                ['+','-','/','*'],
                                f"Opération entre ce jeu et le suivant parmi {', '.join(['+','-','/','*'])}:\n"))
                if not true_geom_op:
                    new_var = operate(rel_sets, var_names, operations)
                    rel_sets[0][new_var_name] = new_var
                else:
                    pop_variable = select_list(pop_dataset.columns[4:],
                                               query=f"Nom de la variable de population à décrire parmi :\n\
                                               \r\t- {'\n\r\t- '.join(pop_dataset.columns[4:])}\n")
                    new_var = operate(rel_sets, var_names, operations,
                                      true_geom=true_geom_op,
                                      pop_dataset=pop_dataset,pop_variable=pop_variable)
                    attrs = new_var.attrs
                    new_var = new_var.rename({'result':new_var_name}, axis=1)
                    new_var = gpd.GeoDataFrame(new_var)
                    attrs.update({'name':new_var_name})
                    new_var.attrs = attrs
                    datasets[new_var_name] = new_var
            elif method == "Intersection et union géographique":
                inter_only = input("Approche par intersection uniquement ? [y]/n ")
                if inter_only =='n':
                    print("Note : L'union sera prioritaire sur l'intersection\n")
                levels=[]
                level_dirs=[]
                level_qs=[]
                inter=[]
                for var in range(n_var):
                    rel_sets.append(
                        datasets[select_list(
                            list(datasets.keys()),
                            f"Nom du jeu de données source n°{var+1} parmi :\n\
                            \r\t- {'\n\r\t- '.join(list(datasets.keys()))}\n")])
                    if rel_sets[0].shape[0]!=rel_sets[-1].shape[0]:
                        true_geom_op=True
                    var_names.append(
                        select_list(
                            rel_sets[-1].columns,
                            f"Nom de la série de données n°{var+1} parmi :\n\
                            \r\t- {'\n\r\t- '.join(rel_sets[-1].columns)}\n"))
                    level_qs.append(select_list(level_method,
                                          query=f"Méthode de sélection du seuil :\n\
                                          \r\t- {'\n\r\t- '.join(level_method)}\n"))
                    level_n = safe_ask("Nombre de seuils : ", int)
                    level=[]
                    for n in range(level_n):
                        level.append(safe_ask(f"Valeur du seuil {n+1} : ", float))
                    level_dirs.append(select_list(level_dir_list,
                                                  query="Valeurs représentant la contrainte :\n\
                                                      \r\t- Au-dessus des seuils (1)\n\
                                                      \r\t- En-dessous des seuils (0)\n",
                                                  catch_dict=level_dir_entries))
                    levels.append(level)
                    if var!=n_var-1:
                        if inter_only!='n':
                            inter.append('Intersection')
                        else:
                            inter.append(select_list(inter_methods,
                                                     query="Mode de liaison avec le jeu suivant :\n\
                                                         \r\t- Intersection (1)\n\
                                                         \r\t- Union (0)\n",
                                                     catch_dict=inter_methods_entries))
                if not true_geom_op:
                    new_var = intersect(rel_sets, var_names, levels, level_qs, level_dirs)
                    rel_sets[0][new_var_name] = new_var
                else:
                    pop_variable = select_list(pop_dataset.columns[4:],
                                               query=f"Nom de la variable de population à décrire parmi :\n\
                                               \r\t- {'\n\r\t- '.join(pop_dataset.columns[4:])}\n")
                    new_var = intersect(rel_sets, var_names, levels, level_qs, level_dirs,
                                        true_geom=true_geom_op,
                                        pop_dataset=pop_dataset, pop_variable=pop_variable)
                    attrs = new_var.attrs
                    new_var = new_var.rename({'result':new_var_name}, axis=1)
                    new_var = gpd.GeoDataFrame(new_var)
                    attrs.update({'name':new_var_name})
                    new_var.attrs = attrs
                    datasets[new_var_name] = new_var
        datasets = build_variables(datasets, pop_dataset)
    return datasets

def weighted_mean(series,pop_dataset=None,pop_variable=None):
    """
    Moyenne pondérée par la population

    Parameters
    ----------
    series : pd.Series
        Données à moyenner.
    pop_dataset : pd.DataFrame, optional
        Répartition de la population. Par défaut, None.
    pop_variable : str, optional
        Nom de la variable contenant la population. Par défaut, None.

    Returns
    -------
    float
        Moyenne pondérée par la population.

    """
    weights = pop_dataset.loc[series.index, pop_variable]
    return (series * weights).sum() / weights.sum()

admin_scale_aggfuncs = {'Somme':'sum','Moyenne pondérée':weighted_mean,'Suppression':None}
admin_scale_aggfuncs_entries = {'sum':'Somme',
                                'mean':'Moyenne pondérée','wmean':'Moyenne pondérée','moy':'Moyenne pondérée',
                                'del':'Suppression'}
def aggregate(dfs,min_scale,corr_admin,pop_dataset,geoms):
    """
    Agrège les données fournies à un échelon administratif de taille fournie.
    Si les données sont déjà à un niveau agrégé, les retourne telles quelles

    Parameters
    ----------
    dfs : dict
        Données d'entrée.
    min_scale : str
        Échelle d'agrégation.
    corr_table : list(pd.DataFrame,dict(gpd.GeoDataFrame))
        Corespondance entre les échelles administratives et géométries administratives
    pop_dataset : pd.DataFrame
        Répartition de la population, pour calcul de l'agrégation.
    geoms : list(tuple)
        Liste des colonnes contenant les géométries des données d'entrée.

    Returns
    -------
    dict
        Données agrégées.

    """
    if min_scale=='iris':
        return dfs
    corr_table = corr_admin[0]
    agg_df_list = []
    pop_dataset = pop_dataset.drop(columns='code_insee').join(corr_table.set_index('code_iris'), on='code_iris')
    pop_dataset = pop_dataset[['code_iris','code_insee','EPCI','DEP','P21_POP']]
    n=0
    for df_name in dfs:
        if dfs[df_name].attrs['scale'] in admin_scales[min_scale]:
            attrs = dfs[df_name].attrs
            print(df_name)
            print('-'*len(df_name))
            df=dfs[df_name]
            # Agrégation de la population à l'échelle des données pour traitement
            pop_dataset_ = pop_dataset.set_index('code_iris').groupby(admin_scales_names[df.attrs['scale']]).sum()
            # Demande des fonctions pour l'agrégation des données
            aggfuncs={}
            for col in df.columns:
                if col!=geom_grid_dict[attrs['scale']]\
                and not col in expected_admin_columns\
                and not col in expected_geom_columns\
                and col!='geometry' :
                    aggfunc = select_list(list(admin_scale_aggfuncs),
                                          query=f"{col} :\nMéthode d'agrégation des données :\n\
                                              \r\t- {'\n\r\t- '.join(list(admin_scale_aggfuncs))}\n",
                                          catch_dict=admin_scale_aggfuncs_entries)
                    if aggfunc=='Moyenne pondérée':
                        aggfuncs.update({col: lambda s: weighted_mean(s, pop_dataset=pop_dataset_, pop_variable='P21_POP')})
                    elif aggfunc=='Suppression':
                        df = df.drop(col, axis=1)
                    else:
                        aggfuncs.update({col:admin_scale_aggfuncs[aggfunc]})
                elif col!=geom_grid_dict[attrs['scale']]\
                and col in expected_admin_columns\
                and col!='geometry':
                    df = df.drop(col, axis=1)
            # Agrégation
            if not df.attrs.get('admin_mixed',False):
                # Géométrie selon un découpage administratif
                # Suppression de la géométrie pour travail uniquement en indice
                df = df.drop('geometry',axis=1)
                new_df = df.set_index(geom_grid_dict[attrs['scale']]).groupby(by=corr_table.set_index(admin_scales_names[df.attrs['scale']])[admin_scales_names[min_scale]])
                new_df = new_df.agg(func=aggfuncs)
                # Mise à jour de la géométrie
                new_df = new_df.join(corr_admin[1][min_scale].set_index(admin_scales_names[min_scale])['geometry'])
                # Passage au format GeoDataFrame
                new_df = gpd.GeoDataFrame(new_df,geometry='geometry')
                # Mise à jour des métadonnées
                new_df.attrs = attrs
                new_df.attrs.update({'scale':min_scale})
                agg_df_list.append(new_df)
            else:
                # Géométrie selon un découpage mixte
                # Récupération des indicateurs géographiques pertinents
                admincolumns = []
                geocolumns = []
                min_admin_scale='dep'
                for col in df.columns:
                    if col in expected_admin_columns:
                        # Indicateur administratif le plus fin possible
                        admincolumns.append(col)
                        if admin_scales_size[expected_admin_columns[col]]<=admin_scales_size[min_admin_scale]:
                            min_admin_col = col
                            min_admin_scale = expected_admin_columns[col]
                    elif col in expected_geom_columns:
                        # Indicateurs non administratifs
                        geocolumns.append(col)
                # Suppression des indicateurs géographiques non pertinents
                admincolumns.remove(min_admin_col)
                new_df = df.drop(columns=admincolumns)
                # Appariement à la table de correspondance
                new_df = new_df.set_index(min_admin_col).join(corr_table[[min_admin_col,admin_scales_names[min_scale]]].set_index(min_admin_col))
                geocolumns.append(admin_scales_names[min_scale])
                # Agrégation
                new_df = new_df.dissolve(by=geocolumns, aggfunc=aggfuncs, as_index=False)
                new_df.attrs = attrs
                new_df.attrs.update({'scale':min_scale})
                agg_df_list.append(new_df)
        else:
            agg_df_list.append(dfs[df_name])
        n+=1
    agg_dfs = dict(zip(dfs.keys(),agg_df_list))
    return agg_dfs

#TODO : gestion d'erreur sur les palettes de couleur
def ask_carto(datasets,pop_dataset):
    """
    Demande à l'utilisateur les variables à afficher, ainsi que les caractéristiques de l'affichage.
    
    Parameters
    ----------
    datasets : dict(pd.DataFrame)
        Dictionnaire contenant les données à afficher.
    pop_dataset : gpd.GeoDataFrame
        Donées sur la répartition de population, pour traitement
    
    Returns
    -------
    variables : dict(list(int,list(dict(str,dict)))
        Pour chaque jeu de données,
            Nombre de variables à afficher,
            Liste des variables à afficher,
                Nom de la variable,
                Type de la variable,
                Colorisation de l'affichage.
    
    """
    variables = dict(zip(datasets.keys(),[[0,[]] for n in range(len(datasets))])) # Création du contenant des données de sortie
    for dataset in variables:
        print(f"\n{dataset}\n{'-'*len(dataset)}\n")
        variables[dataset][0] = safe_ask("Nombre de variables à afficher :\n",int)
        for n_var in range(variables[dataset][0]):
            print(f"\nVariable n°{n_var+1}\n{'-'*(12+(n_var+1)//10)}")
            var={}
            var['nom'] = select_list(datasets[dataset].columns,
                f"Nom de la série de données parmi :\n\
                \r\t- {'\n\r\t- '.join(datasets[dataset].columns)}\n")
            var['nom_legende'] = input("Nom de variable à inscrire sur la carte :\n").replace('\\n','\n')
            var['type'] = select_list(data_types,
                f"Type des données parmi :\n\
                \r\t- {'\n\r\t- '.join(data_types)}\n")
            if var['type'] in cmap_data:
                var['classification'] = select_list(
                    list(cmap_classification.keys()),
                    f"Type de classification parmi :\n\
                    \r\t- {'\n\r\t- '.join(cmap_classification.keys())}\n",
                    catch_dict=cmap_classification_entries)
                var['classification'] = cmap_classification[var['classification']]
                if var['classification'] == 'UserDefined':
                    n_bins = safe_ask("Nombre de classes : ", int)
                    var['bins'] = [safe_ask("Borne inférieure de la première classe : ", float)]
                    for i_bin in range(n_bins-1):
                        var['bins'].append(safe_ask(f"Borne supérieure de la classe n°{i_bin+1} : ", float))
                    var['bins'].append(int(np.ceil(datasets[dataset][var['nom']].max())))
                    var['bins'] = dict(bins=var['bins'][1:],lowest=var['bins'][0])
                elif var['classification'] in ['FisherJenks']:
                    var['bins'] = dict(k=safe_ask("Nombre de classes : ", int))
                elif var['classification'] == None:
                    var['labels'] = []
                    var['labels'].append(input('Légende : étiquette pour les valeurs faibles :\n'))
                    var['labels'].append(input('Légende : étiquette pour les valeurs élevées :\n'))
                var['couleur'] = input(
                    "Palette de couleurs :\n\
                    \rPalettes standard :\n\
                    \r\t- YlOrRd\n\r\t- coolwarm ou RdBu\n\
                    \r\t- RdYlGn_corr ou GnYlRd_corr\n")
            else :
                var['couleur'] = safe_ask(
                    "Couleur :\n", is_color_like)
            variables[dataset][1].append(var)
    return variables
