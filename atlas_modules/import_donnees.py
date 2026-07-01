# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:06:35 2026
Last authenticated version : Wed Jul 01 11:15 2026
@author: ternilp
"""
import sys
import atlas_modules.carto as carto
from pathlib import Path
import pandas as pd
import geopandas as gpd
import fiona
import operator
from matplotlib.colors import is_color_like
import numpy as np

# Types de géométries de raccordement
geom_dict = {'département':'dep','departement':'dep','dép':'dep','dep':'dep',
             'epci':'epci',
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
geom_grid_dict = {'dep':'dep','epci':'epci','com':'code_insee','iris':'code_iris',
                  'maille_safran':'maille_safran','maille_drias':'maille_drias',
                  None:'geometry'}
# Noms de colonnes canoniques, utilisés pour les agrégations
expected_admin_columns = {'code_iris':'iris','code_insee':'com','EPCI':'epci','dep':'dep'}
expected_geom_columns = {'maille_safran','maille_drias','id_car'}
# Type de données affichables
cmap_data = carto.cmap_data
data_types = cmap_data+['Compte','Localisation de points','Tracés']
# Types de classifications des données
cmap_classification={'Aucune':None,'Jenks':'FisherJenks','Tête/Queue':'HeadTailBreaks','Personnalisée':'UserDefined'}
cmap_classification_entries={'aucune':'Aucune','none':'Aucune',
                             'jenks':'Jenks',
                             'tête/queue':'Tête/Queue','tete/queue':'Tête/Queue','têtequeue':'Tête/Queue','tetequeue':'Tête/Queue',
                             'personnalisée':'Personnalisée','personnalisee':'Personnalisée','perso':'Personnalisée'}
# Liste des fichiers contenant des données : tous les fichier à l'exception des grilles administratives et fonds de carte
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
# Nom du fichier uniquement
file_list = [file.stem for file in file_list]

# Codes insee ayant changé récemment, présents sous deux formes
# Ici, uniquement Conques-en-Rouergue
deprecated_codes={'12076':'12218','120760000':'122180000'}

# Liste des formats supportés
format_list=['GeoPackage (.gpkg)','Comma Separated Values (.csv)']

# Traitements possibles à l'importation
treatments = ['MeanRatio','MedianRatio','Logscaling','Aucun']

# Méthodes d'opérations entre variables
methods = ["Indicateur synthétique","Intersection et union géographique"]
methods_entries = {'indicateur':"Indicateur synthétique",'indic':"Indicateur synthétique",'synth':"Indicateur synthétique",
                   'inter':"Intersection et union géographique",'intersection':"Intersection et union géographique",
                   'cumul':"Intersection et union géographique",'union':"Intersection et union géographique"}
inter_methods = ['Union','Intersection','Mixte']
inter_methods_entries = {'0':'union','inter':'intersection','1':'intersection','2':'mixte','mixed':'mixte'}
# Méthodes de sélection du seuil pour intersection géographique
level_method = ["Valeur","Quantile"]
level_method_entries = {'v':'Valeur','val':'Valeur','q':'Quantile','quant':'Quantile'}
level_dir_list = ["Au-dessus du seuil","En-dessous du seuil"]
level_dir_entries = {'1':"Au-dessus du seuil",'0':"En-dessous du seuil"}


def safe_ask(query,datatype):
    """
    Demande à l'utilisateur des données d'un type spécifié.
    Réitère la demande jusqu'à ce que les types correspondent

    Parameters
    ----------
    query : str
        Texte affiché lors de la demande de données.
    datatype : type, list(type) ou fonction
        Type de données à demander ou fonction vérifiant l'interprétabilité.
        Dans le cas ou datatype est une liste, le type retenu sera le premier pour lequel une correspondance existe
        La liste doit donc être triée par ordre de préférence

    Returns
    -------
    var : datatype
        Donnée demandée à l'utilisateur.

    """
    var = input(query)
    if isinstance(datatype,type):
        while not isinstance(var, datatype):
            # Tente d'interpréter les données au type demandé
            # Demande un nouvel input en cas d'échec
            try:
                # Traitement particulier des booléens,
                # dont le casting renvoie True pour tout str en entrée
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
        # Cherche si l'input correspond à l'un des types spécifiés
        # S'arrête à la première correspondance
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
                element = catch_dict[element.lower()]
            except:
                pass

def import_progress(filepath, treatment=None, compact=True):
    """
    Importe des données en affichant l'état d'avancement.
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
            # Importation des données par ligne, comptage
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
    Importe des données.
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
        # Importation
        data = gpd.read_file(filepath)
        # Traitement des colonnes de données, si spécifié
        if treatment == 'MeanRatio':
            for column in data.columns.drop('geometry'):
                if pd.api.types.is_numeric_dtype(data.dtypes[column]):
                    data[column] = data[column]/data[column].mean()
        elif treatment == 'MedianRatio':
            for column in data.columns.drop('geometry'):
                if pd.api.types.is_numeric_dtype(data.dtypes[column]):
                    data[column] = data[column]/data[column].median()
        data.attrs['name'] = filepath.stem
        return data
    elif filepath.suffix=='.csv':
        # Importation
        data = pd.read_csv(filepath, dtype='string')
        # Conversion des colonnes de données à un type numérique
        for col in data.columns[data.columns.str.endswith('_data')]:
            data[col] = pd.to_numeric(data[col])
        # Traitement des colonnes de données, si spécifié
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
    if not filepath.exists(): # Chemin hors format "Chemin depuis le dossier de travail, avec extension"
        if ("Données traitées"/filepath).exists():  # Nom du fichier, avec extension
            filepath = "Données traitées"/filepath
        elif (matches:=[p for p in Path("Données traitées").iterdir()    # Nom du fichier, sans extension
                       if p.is_file() and p.stem.lower() == str(filepath).lower()])!=[]:
            if len(matches)>1:
                return -1, filepath
            else: filepath=matches[0]
        elif Path("Données traitées\\"+filepath.parts[-1]).exists():    # Chemin depuis un dossier parent du dossier de travail
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
    # Détection de la géométrie des données, d'après le nom du fichier
    name = data.attrs.get('name',None)
    if isinstance(name,str):
        if name.endswith('_IRIS'):
            geom = 'iris'
        elif name.endswith('_commune'):
            geom = 'com'
        elif name.endswith('_EPCI'):
            geom = 'epci'
        elif name.endswith('_dept'):
            geom = 'dep'
        elif name.endswith('_maille_safran'):
            geom = 'maille_safran'
        elif name.endswith('_maille_drias'):
            geom = 'maille_drias'
        # Géométrie non spécifiée
        else:
            geom = input("Géométrie de référence parmi :\n\t- Département/dep\n\
                        \r\t- Commune/com\n\t- IRIS\n\t- Maille Safran\n\t- Maille Drias\n\t- Autre\n\
                        \rSélectionnez 'Autre' pour des données dont la localisation est incluse.\n").lower()
            while not geom in geom_dict.keys():
                geom = input(f"\nErreur :'{geom}' n'est pas une échelle valide.\n\
                           \rSélectionnez une échelle parmi :\n\t- Département/dep\n\
                           \r\t- Commune/com\n\t- IRIS\n\t- Maille Safran\n\t- Maille Drias\n\t- Autre\n\
                           \r Sélectionnez 'Autre' pour des données dont la localisation est incluse.\n").lower()
            geom = geom_dict[geom]
    else:
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
                return geoms.index[0] # Une seule colonne trouvée : c'est la bonne
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
                return columns[0] # Une seule colonne trouvée : c'est la bonne
            else :
                raise ValueError(f"{data.attrs['name']} contient plusieurs colonnes portant ce nom et contenant une géométrie.\n\
                                 \rIl est impossible de l'utiliser en l'état")
    elif geom_data is None:    # Géométrie 'type' (dans geom_columns_dict). Cas d'usage classique (voir main.py)
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

def operate(datasets, names, operations, true_geom=False,
            pop_dataset=None, pop_variable='P21_PMEN', corr_admin=None):
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
        Par défaut, False.
    pop_dataset : pd.geoDataFrame, optional
        Jeu de données contenant la population, pour désagrégation.
        Par défaut, None.
    pop_variable : str, optional
        Nom de la colonne de pop_dataset contenant la variable de population à utiliser.
        Par défaut 'C21_PMEN' (population générale).
    corr_admin : list(pd.DataFrame, dict(gpd.GeoDataFrame))
        Correspondances entre les échelons administratifs (corr_admin[0])
        et géométries administratives supérieures à l'IRIS (corr_admin[1])
        Par défaut, None.
    
    Returns
    -------
    gpd.GeoDataFrame        
        Résultat des opérations fournies, avec les indices administratifs correspondants.

    """
    if true_geom: # Toutes les données ne sont pas sur une maille administrative
        # Construit la géométrie correspondant aux intersections de chaque géométrie,
        # sans répétition des géométries pour éviter les artéfacts
        print("Préparation de la jointure des données : pour chaque variable, entrer :\n\
              \r\t- 0 si c'est une grandeur intensive (à ne pas répartir, i.e. âge)\n\
              \r\t- 1 si c'est une grandeur extensive (à répartir, i.e. population)")
        proportional=[]
        for name in names:
            proportional.append(bool(int(input(f"{name} : "))))
        if not pop_dataset is None: # Ajout de la population aux variables à calculer
            datasets.append(pop_dataset)
            proportional.append(True)
        overlay_result = carto.list_overlay(datasets, proportional, pop_dataset, pop_variable, corr_admin)
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
        # Colonnes d'indicateur géographique pour agrégation éventuelle, premières d'un dataset par construction
        cols_to_keep = [overlay_result.columns[0]]
        for col in list(overlay_result.columns)[:-2]: # les deux dernières colonnes sont result et geometry
            if col[-1]!=cols_to_keep[-1][-1]: # Le dernier caractère d'un nom de colonne est le numéro du dataset
                cols_to_keep.append(col) # On obtient donc la première colonne de chaque dataset
        if not pop_dataset is None:
            overlay_result['population'] = overlay_result[f'{pop_variable}_{len(datasets)-1}'] # population de pop_dataset
            overlay_result = overlay_result[cols_to_keep+['result','population','geometry']]
            overlay_result = overlay_result.rename( # Suppression du suffixe indiquant le numéro de dataset
                dict(zip(cols_to_keep,[col[:-2] for col in cols_to_keep])),axis=1) # Ne fonctionne proprement qu'avec moins de 10 datasets
            # Suppression des doublons
            overlay_result = overlay_result.loc[:,~overlay_result.columns.duplicated()]
            return overlay_result
        else:
            overlay_result = overlay_result[cols_to_keep+['result','geometry']]
            overlay_result = overlay_result.rename(
                dict(zip(cols_to_keep,[col[:-2] for col in cols_to_keep])),axis=1)
            overlay_result = overlay_result.loc[:,~overlay_result.columns.duplicated()]
            return overlay_result
    else: # Toutes les données sont sur une maille administrative
        # Extraction de la maille la plus fine
        min_scale = datasets[0].attrs['scale']
        join_need = False
        for dataset in datasets:
            if carto.admin_scales_size[dataset.attrs['scale']]<carto.admin_scales_size[min_scale]: # Comparaison des tailles de maille
                min_scale = dataset.attrs['scale']
                join_need = True
            elif carto.admin_scales_size[dataset.attrs['scale']]>carto.admin_scales_size[min_scale]:
                join_need = True
        if join_need: # Plusieurs mailles différentes
            print("Préparation de la jointure des données : pour chaque variable, entrer :\n\
                  \r\t- 0 si c'est une grandeur intensive (à ne pas répartir, e.g. âge)\n\
                  \r\t- 1 si c'est une grandeur extensive (à répartir, e.g. population)\n\
                  \rNote : les seuils en valeur sur des grandeurs extensives peuvent perdre leur sens")
            for i,dataset in enumerate(datasets):
                if dataset.attrs['scale']!=min_scale: # = nécessité d'adaptation de l'échelle
                    proportional = bool(int(input(f"{names[i]} : ")))
                    attrs = dataset.attrs # Sauvegarde des attributs du dataset
                    dataset = corr_admin[0].join(dataset, # Jointure avec la grille correspondant à min_scale
                        on=carto.admin_scales_names[carto.admin_scales_entries[attrs['scale']]])\
                        .set_index(carto.admin_scales_names[carto.admin_scales_entries[min_scale]])
                    if proportional:
                        pop_dataset_adapted = corr_admin[0].join(pop_dataset[['code_iris',pop_variable]].set_index('code_iris'),
                                                                 on='code_iris')\
                            .set_index(carto.admin_scales_names[carto.admin_scales_entries[min_scale]])\
                            [[carto.admin_scales_names[carto.admin_scales_entries[attrs['scale']]],pop_variable]] # Ajout des codes administratifs de la maille de dataset à pop_dataset
                        pop_dataset_agg = pop_dataset_adapted.groupby(
                            carto.admin_scales_names[carto.admin_scales_entries[attrs['scale']]])\
                                                                 .sum() # Agrégation de pop_dataset sur la maille de dataset
                        pop_dataset_adapted['pop_agg'] = pop_dataset_adapted[carto.admin_scales_names[carto.admin_scales_entries[attrs['scale']]]]\
                            .map(pop_dataset_agg.iloc[:,0])
                        dataset[names[i]] = dataset[names[i]]*pop_dataset_adapted[pop_variable]/pop_dataset_adapted['pop_agg'] # Répartition proportionnelle à la population
                    attrs.update({'scale':min_scale}) # Mise à jour de l'échelle des attributs
                    dataset.attrs=attrs # Réattribution des attributs sauvegardés
                    datasets[i] = dataset
        # Réalisation de l'opération demandée sur les jeux mis à l'échelle
        # Note : une simplification pourrait être apportée en utilisant eval() comme dans le cas true_geom=True
        series = [datasets[n][names[n]] for n in range(len(datasets))]
        prio_dict={'*':1,'/':1,'+':2,'-':2}
        op_dict={'*':operator.mul,
                 '/':operator.truediv,
                 '+':operator.add,
                 '-':operator.sub}
        ops_prio=[]
        for op in operations:
            ops_prio.append(prio_dict[op])
        # Opérations avec ordres de priorité usuels
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
        result = gpd.GeoDataFrame(series[0], columns=['result'],index=datasets[0].index,
                                  geometry=datasets[0].geometry)
        result.attrs={'scale':datasets[0].attrs['scale'],'admin_mixed':False}
        return result

def intersect(datasets, names, levels, level_qs, level_dirs, methods=None,
              true_geom=False, pop_dataset=None, pop_variable='POP21_POP',
              corr_admin=None):
    """
    Calcule le résultat des opérations fournies sur les données fournies.
    L'union et la méthode mixte sont prioritaires sur l'intersection.
    
    Parameters
    ----------
    datasets : list(pd.DataFrame)
        Jeux de données contenant les séries sur lesquelles appliquer les opérations.
    names : list(str)
        Noms des colonnes sur lesquelles appliquer les opérations
    levels : list(float)
        Liste des seuils dont le dépassement est vérifié.
    level_qs : list(str)
        Liste des types de seuils.
        Valeur : Seuil en valeur qu'on compare aux données
        Quantile : Seuil de quantile, qu'on utilise pour classer les données
    level_dirs : list(str)
        Liste des sens des seuils.
        Au-dessus du seuil : On compte les valeurs au-dessus des seuils.
        En-dessous du seuil : On compte les valeurs en-dessous des seuils.
    methods : list(str), optional
        Liste des méthodes de calcul pour l'intersection.
        Intersection : compte du cumul des seuils dépassés, sélection du minimum de dépassement.
        Union : compte de la présence d'au moins un dépassement, sélection du maximum de dépassement.
        Mixte : compte de la proportion de dépassements, sélection de la moyenne des dépassements.
        Par défaut, None.
    true_geom : bool, optional
        Si True, opération sur les localisations des données.
        Si False, opération sur les indices.
        Par défaut, False.
    pop_dataset : pd.geoDataFrame, optional
        Jeu de données contenant la population, pour désagrégation.
        Par défaut, None.
    pop_variable : str, optional
        Nom de la colonne de pop_dataset contenant la variable de population à utiliser.
        Par défaut 'C21_PMEN' (population générale).
    corr_admin : list(pd.DataFrame, dict(gpd.GeoDataFrame))
        Correspondances entre les échelons administratifs (corr_admin[0])
        et géométries administratives supérieures à l'IRIS (corr_admin[1])
        Par défaut, None.

    Returns
    -------
    gpd.GeoDataFrame        
        Résultat des opérations fournies, avec les indices administratifs correspondants.

    """
    if methods is None:
        methods = ['Intersection' for n in range(len(datasets)-1)]
    if true_geom:
        # Construit la géométrie correspondant aux intersections de chaque géométrie,
        # sans répétition de géométrie pour éviter les artéfacts
        print("Préparation de la jointure des données : pour chaque variable, entrer :\n\
              \r\t- 0 si c'est une grandeur intensive (à ne pas répartir, e.g. âge)\n\
              \r\t- 1 si c'est une grandeur extensive (à répartir, e.g. population)\n\
              \rNote : les seuils en valeur sur des grandeurs extensives peuvent perdre leur sens")
        proportional=[]
        for name in names: # Demande si les variables doivent être réparties
            proportional.append(bool(int(input(f"{name} : "))))
        if not pop_dataset is None: # Ajout de la population aux variables à calculer
            datasets.append(pop_dataset)
            proportional.append(True)
        overlay_result = carto.list_overlay(datasets, proportional, pop_dataset, pop_variable, corr_admin)
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
        # Calcul des unions et intersections
            # Les unions sont prioritaires sur les intersections
            # La méthode mixte compte la proportion totale de seuils franchis
        if all(m=='Intersection' for m in methods): # Cumul de seuils franchis : l'intersection est le min
            overlay_result['result'] = overlay_result[[f"{name}_{i}" for i,name in enumerate(names)]].min(axis=1)
        elif all(m=='Union' for m in methods): # Exposition à au moins un dépassement : l'union est le max
            overlay_result['result'] = overlay_result[[f"{name}_{i}" for i,name in enumerate(names)]].max(axis=1)
        elif all(m=='Mixte' for m in methods): # Prise en compte mixte avec une moyenne
            overlay_result['result'] = overlay_result[[f"{name}_{i}" for i,name in enumerate(names)]].mean(axis=1)
        else:
            temp_result=pd.DataFrame()
            n=0
            while n<len(methods):
                if methods[n]=='Union':
                    temp_result[f"{n}"] = pd.concat([overlay_result[f"{names[n]}_{n}"],overlay_result[f"{names[n+1]}_{n+1}"]],axis=1).max(axis=1)
                    n+=1
                elif methods[n]=='Mixte':
                    temp_result[f"{n}"] = pd.concat([overlay_result[f"{names[n]}_{n}"],overlay_result[f"{names[n+1]}_{n+1}"]],axis=1).mean(axis=1)
                    n+=1
                else:
                    temp_result[f"{n}"] = overlay_result[f"{names[n]}_{n}"]
                n+=1
                overlay_result['result'] = temp_result.min(axis=1)
        # Sélection des colonnes utiles
        # Colonnes d'indicateur géographique pour agrégation éventuelle, premières d'un dataset par construction
        cols_to_keep = [overlay_result.columns[0]]
        for col in list(overlay_result.columns)[:-2]: # les deux dernières colonnes sont result et geometry
            if col[-1]!=cols_to_keep[-1][-1]: # Le dernier caractère d'un nom de colonne est le numéro du dataset
                cols_to_keep.append(col) # On obtient donc la première colonne de chaque dataset
        if not pop_dataset is None:
            overlay_result['population'] = overlay_result[f'{pop_variable}_{len(datasets)-1}'] # population de pop_dataset
            overlay_result = overlay_result[cols_to_keep+['result','population','geometry']]
            overlay_result = overlay_result.rename(
                dict(zip(cols_to_keep,[col[:-2] for col in cols_to_keep])),axis=1)
            # Suppression des doublons
            overlay_result = overlay_result.loc[:,~overlay_result.columns.duplicated()]
            return overlay_result
        else:
            overlay_result = overlay_result[cols_to_keep+['result','geometry']]
            overlay_result = overlay_result.rename( # Suppression du suffixe indiquant le numéro de dataset
                dict(zip(cols_to_keep,[col[:-2] for col in cols_to_keep])),axis=1) # Ne fonctionne proprement qu'avec moins de 10 datasets
            overlay_result = overlay_result.loc[:,~overlay_result.columns.duplicated()]
        return overlay_result
    else: # Toutes les données sont sur une maille administrative
        # Extraction de la maille la plus fine
        min_scale = datasets[0].attrs['scale']
        join_need=False
        for dataset in datasets:
            if carto.admin_scales_size[dataset.attrs['scale']]<carto.admin_scales_size[min_scale]:
                min_scale = datasets.attrs['scale']
                join_need = True
            elif carto.admin_scales_size[dataset.attrs['scale']]>carto.admin_scales_size[min_scale]:
                join_need = True
        if join_need: # Plusieurs mailles différentes
            print("Préparation de la jointure des données : pour chaque variable, entrer :\n\
                  \r\t- 0 si c'est une grandeur intensive (à ne pas répartir, e.g. âge)\n\
                  \r\t- 1 si c'est une grandeur extensive (à répartir, e.g. population)\n\
                  \rNote : les seuils en valeur sur des grandeurs extensives peuvent perdre leur sens")
            for i,dataset in enumerate(datasets):
                if dataset.attrs['scale']!=min_scale: # = nécessité d'adaptation de l'échelle
                    proportional = bool(int(input(f"{names[i]} : ")))
                    attrs = dataset.attrs # Sauvegarde des attributs du dataset
                    dataset = corr_admin[0].join(dataset, # Jointure avec la grille correspondant à min_scale
                        on=carto.admin_scales_names[carto.admin_scales_entries[attrs['scale']]])\
                        .set_index(carto.admin_scales_names[carto.admin_scales_entries[min_scale]])
                    if proportional:
                        pop_dataset_adapted = corr_admin[0].join(pop_dataset[['code_iris',pop_variable]].set_index('code_iris'),
                                                                 on='code_iris')\
                            .set_index(carto.admin_scales_names[carto.admin_scales_entries[min_scale]])\
                            [[carto.admin_scales_names[carto.admin_scales_entries[attrs['scale']]],pop_variable]] # Ajout des codes administratifs de la maille de dataset à pop_dataset
                        pop_dataset_agg = pop_dataset_adapted.groupby(
                            carto.admin_scales_names[carto.admin_scales_entries[attrs['scale']]])\
                                                                 .sum() # Agrégation de pop_dataset sur la maille de dataset
                        pop_dataset_adapted['pop_agg'] = pop_dataset_adapted[carto.admin_scales_names[carto.admin_scales_entries[attrs['scale']]]]\
                            .map(pop_dataset_agg.iloc[:,0])
                        dataset[names[i]] = dataset[names[i]]*pop_dataset_adapted[pop_variable]/pop_dataset_adapted['pop_agg'] # Répartition proportionnelle à la population
                    datasets[i] = dataset
        series = [datasets[n][names[n]] for n in range(len(datasets))]
        # Pour chaque variable, remplace les valeurs par la proportion de seuils franchis
        # Le sens de franchissement du seuil est spécifié dans level_dirs
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
        # Calcul des unions et intersections
            # Les unions sont prioritaires sur les intersections
            # La méthode mixte compte la proportion totale de seuils franchis
        if all(m=='Intersection' for m in methods): # Cumul de seuils franchis : l'intersection est le min
            intersection = pd.DataFrame(pd.concat(series,axis=1).min(axis=1))
        elif all(m=='Union' for m in methods): # Exposition à au moins un dépassement : l'union est le max
            intersection = pd.DataFrame(pd.concat(series,axis=1).max(axis=1))
        elif all(m=='Mixte' for m in methods): # Prise en compte mixte avec une moyenne
            intersection = pd.DataFrame(pd.concat(series,axis=1).mean(axis=1))
        else:
            temp_result=pd.DataFrame()
            n=0
            while n<(len(methods)):
                if methods[n]=='Union':
                    temp_result[f"{n}"] = pd.concat([series[n],series[n+1]],axis=1).max(axis=1)
                    n+=1
                elif methods[n]=='Mixte':
                    temp_result[f"{n}"] = pd.concat([series[n],series[n+1]],axis=1).mean(axis=1)
                    n+=1
                else:
                    temp_result[f"{n}"] = series[n]
                n+=1
            intersection = temp_result.min(axis=1)
        intersection.index = datasets[0].index
        result = gpd.GeoDataFrame(data=intersection[intersection.columns[0]].values, columns=['result'],
                                index=datasets[0].index, geometry=datasets[0].geometry)
        result.attrs={'scale':datasets[0].attrs['scale'],'admin_mixed':False}#todo:scale
        return result

def build_variables(datasets,pop_dataset,pop_variable,corr_admin):
    """
    Demande à l'utilisateur les méthodes de construction des variables à calculer, si nécessaire

    Parameters
    ----------
    datasets : dict(gpd.GeoDataFrame)
        Jeux de données disponibles comme base des variables à calculer.
    pop_dataset : gpd.GeoDataFrame
        Jeu de données contenant la répartition de la population.
    pop_variable : str
        Nom de la colonne de pop_dataset contenant la population à utiliser.
    corr_admin : list(pd.DataFrame, dict(gpd.GeoDataFrame))
        Correspondances entre les échelons administratifs (corr_admin[0])
        et géométries administratives supérieures à l'IRIS (corr_admin[1])

    Returns
    -------
    datasets : dict(gpd.GeoDataFrame)
        Jeux de données contenant les données d'entrée non modifiées et les données calculées.

    """
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
            # Construction des opérations à réaliser
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
                    if rel_sets[-1].attrs['scale'] is None or rel_sets[-1].attrs.get('admin_mixed',False)==True:
                        true_geom_op=True # Données pas uniquement sur unde maille administrative
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
                    new_var = operate(rel_sets, var_names, operations, pop_dataset=pop_dataset,
                                      pop_variable=pop_variable, corr_admin=corr_admin)
                else:
                    print("Interpolation des données :")
                    print('-'*len("Interpolation des données :"))
                    # Population utilisée pour l'interpolation en operate
                    pop_variable = select_list(pop_dataset.columns[4:],
                                               query=f"Nom de la variable de population à décrire parmi :\n\
                                               \r\t- {'\n\r\t- '.join(pop_dataset.columns[4:])}\n")
                    new_var = operate(rel_sets, var_names, operations,
                                      true_geom=true_geom_op,
                                      pop_dataset=pop_dataset,pop_variable=pop_variable,
                                      corr_admin=corr_admin)
                # Mise en forme des attributs du résultat
                attrs = new_var.attrs
                new_var = new_var.rename({'result':new_var_name}, axis=1)
                new_var = gpd.GeoDataFrame(new_var)
                attrs.update({'name':new_var_name})
                new_var.attrs = attrs
                datasets[new_var_name] = new_var
            elif method == "Intersection et union géographique":
                inter_only = input("Approche par intersection uniquement ? y/[n] ")
                if inter_only !='y':
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
                                          \r\t- {'\n\r\t- '.join(level_method)}\n",
                                          catch_dict=level_method_entries))
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
                        if inter_only=='y':
                            inter.append('Intersection')
                        else:
                            inter.append(select_list(inter_methods,
                                                     query="Mode de liaison avec le jeu suivant :\n\
                                                         \r\t- Union (0)\n\
                                                         \r\t- Intersection (1)\n\
                                                         \r\t- Mixte (2)\n",
                                                     catch_dict=inter_methods_entries))
                if not true_geom_op:
                    new_var = intersect(rel_sets, var_names, levels, level_qs, level_dirs,
                                        methods=inter,corr_admin=corr_admin)
                else:
                    # Population utilisée pour l'interpolation en intersect
                    pop_variable = select_list(pop_dataset.columns[1:],
                                               query=f"Nom de la variable de population à décrire parmi :\n\
                                               \r\t- {'\n\r\t- '.join(pop_dataset.columns[4:])}\n")
                    new_var = intersect(rel_sets, var_names, levels, level_qs, level_dirs,
                                        methods=inter, true_geom=true_geom_op,
                                        pop_dataset=pop_dataset, pop_variable=pop_variable,
                                        corr_admin=corr_admin)
                # Mise en forme des attributs du résultat
                attrs = new_var.attrs
                new_var = new_var.rename({'result':new_var_name}, axis=1)
                new_var = gpd.GeoDataFrame(new_var)
                attrs.update({'name':new_var_name,'admin_mixed':true_geom_op})
                new_var.attrs = attrs
                datasets[new_var_name] = new_var
        datasets = build_variables(datasets, pop_dataset, pop_variable, corr_admin)
    return datasets

def weighted_mean(series,pop_dataset=None,pop_variable=None):
    """
    Moyenne pondérée par la population

    Parameters
    ----------
    series : pd.Series
        Données à moyenner.
    pop_dataset : pd.DataFrame ou pd.Series, optional
        Répartition de la population. Par défaut, None.
    pop_variable : str, optional
        Nom de la variable contenant la population. Par défaut, None.

    Returns
    -------
    float
        Moyenne pondérée par la population.

    """
    series = series.dropna()
    if isinstance(pop_dataset, pd.Series):
        weights = pop_dataset.loc[series.index]
    elif isinstance(pop_dataset, pd.DataFrame):
        weights = pop_dataset.loc[series.index, pop_variable]
    return (series * weights).sum() / weights.sum()

admin_scale_aggfuncs = {'Somme':'sum','Moyenne pondérée':weighted_mean,}
admin_scale_aggfuncs_entries = {'sum':'Somme',
                                'mean':'Moyenne pondérée','wmean':'Moyenne pondérée','moy':'Moyenne pondérée'}
def aggregate(dfs, min_scale, corr_admin, pop_dataset, pop_variable,
              geoms, var_names,full_agg=False, grids=None):
    """
    Agrège les données fournies à un échelon administratif de taille fournie.
    Si les données sont déjà à un niveau agrégé, les retourne telles quelles

    Parameters
    ----------
    dfs : dict
        Données d'entrée.
    min_scale : str
        Échelle d'agrégation.
    corr_admin : list(pd.DataFrame, dict(gpd.GeoDataFrame))
        Correspondances entre les échelons administratifs (corr_admin[0])
        et géométries administratives supérieures à l'IRIS (corr_admin[1])
    pop_dataset : pd.DataFrame
        Répartition de la population, pour calcul de l'agrégation.
    pop_variable : str
        Nom de la colonne de pop_dataset contenant la population à utiliser.
    geoms : list(tuple)
        Liste des colonnes contenant les géométries des données d'entrée.
    var_names : list(str)
        Liste des variables d'intérêt, à conserver
    full_agg : bool
        Si False, les données non basées sur un échelon administratif ne seront pas agrégées.
        Si True, elles seront agrégées
    grids : dict(gpd.GeoDataFrame)
        Contours des échelons administratifs, pour agrégation des données à géométrie propre.
        
    Returns
    -------
    dict
        Données agrégées.

    """
    if min_scale=='iris': # Échelle la plus fine : pas d'agrégation
        return dfs
    corr_table = corr_admin[0]
    agg_df_list = []
    # Ajout de tous les identifiants administratifs au jeu de population
    pop_dataset_func = pop_dataset.drop(columns='code_insee').join(corr_table.set_index('code_iris'), on='code_iris')
    pop_dataset_func = pop_dataset_func[['code_iris','code_insee','EPCI','dep',pop_variable]]
    n=0
    for df_name in dfs:
        attrs = dfs[df_name].attrs # Sauvegarde des attributs du jeu de données en cours d'agrégation
        print(df_name)
        print('-'*len(df_name))
        df = dfs[df_name]
        if dfs[df_name].attrs['scale'] in carto.admin_scales[min_scale] and not full_agg:
            # Agrégation de la population à l'échelle des données pour traitement
            pop_dataset_ = pop_dataset_func.set_index('code_iris').groupby(carto.admin_scales_names[carto.admin_scales_entries[df.attrs['scale']]]).sum()
            # Demande des fonctions pour l'agrégation des données
            aggfuncs={}
            for col in df.columns:
                if col in var_names :
                    aggfunc = select_list(list(admin_scale_aggfuncs),
                                          query=f"{col} :\nMéthode d'agrégation des données :\n\
                                              \r\t- {'\n\r\t- '.join(list(admin_scale_aggfuncs))}\n",
                                          catch_dict=admin_scale_aggfuncs_entries)
                    if aggfunc=='Moyenne pondérée':
                        aggfuncs.update({col: lambda s: weighted_mean(s, pop_dataset=pop_dataset_, pop_variable=pop_variable)})
                    elif aggfunc=='Suppression':
                        df = df.drop(col, axis=1)
                    else:
                        aggfuncs.update({col:admin_scale_aggfuncs[aggfunc]})
                elif col!=geom_grid_dict[attrs['scale']]\
                and col in expected_admin_columns\
                and col!='geometry':
                    # Suppression des colonnes d'identifiants administratifs non utilisées
                    df = df.drop(col, axis=1)
            # Agrégation
            if not df.attrs.get('admin_mixed',False): # Géométrie selon un découpage administratif uniquement
                # Suppression de la géométrie pour travail uniquement en indice
                df = df.drop('geometry',axis=1)
                new_df = df.groupby(by=corr_table
                                    .drop_duplicates(subset=carto.admin_scales_names[carto.admin_scales_entries[df.attrs['scale']]])
                                    .set_index(carto.admin_scales_names[carto.admin_scales_entries[df.attrs['scale']]])\
                                    [carto.admin_scales_names[min_scale]]
                                    )
                new_df = new_df.agg(func=aggfuncs)
                # Mise à jour de la géométrie
                new_df = new_df.join(corr_admin[1][min_scale].set_index(carto.admin_scales_names[min_scale])['geometry'])
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
                        if carto.admin_scales_size[expected_admin_columns[col]]<=carto.admin_scales_size[min_admin_scale]:
                            min_admin_col = col
                            min_admin_scale = expected_admin_columns[col]
                    elif col in expected_geom_columns:
                        # Indicateurs non administratifs
                        geocolumns.append(col)
                # Suppression des indicateurs géographiques non pertinents
                admincolumns.remove(min_admin_col)
                new_df = df.drop(columns=admincolumns)
                # Appariement à la table de correspondance
                new_df = new_df.set_index(min_admin_col).join(corr_table[[min_admin_col,carto.admin_scales_names[min_scale]]].set_index(min_admin_col))
                geocolumns.append(carto.admin_scales_names[min_scale])
                # Agrégation
                new_df = new_df.dissolve(by=geocolumns, aggfunc=aggfuncs, as_index=False)
                new_df.attrs = attrs
                new_df.attrs.update({'scale':min_scale})
                agg_df_list.append(new_df)
        elif full_agg and not dfs[df_name].attrs['scale'] is None:
            df.attrs = attrs
            # Appariement des géométries avec les indices administratifs
            new_df = carto.list_overlay([grids[carto.admin_scales_grids[min_scale]],df], proportional=True,
                                  pop_dataset=pop_dataset, pop_variable=pop_variable,
                                  corr_admin=corr_admin)
            # Estimation des populations pour répartition des données
            pop_dataset_ = carto.infer_pop_by_geom(new_df.geometry, pop_dataset, pop_variable)
            new_df.columns = new_df.columns.str.removesuffix('_0')
            new_df.columns = new_df.columns.str.removesuffix('_1')
            new_df = new_df.loc[:,~new_df.columns.duplicated()].copy()
            # Demande des fonctions pour l'agrégation des données
            aggfuncs={}
            for col in new_df.columns:
                if col in var_names :
                    aggfunc = select_list(list(admin_scale_aggfuncs),
                                          query=f"{col} :\nMéthode d'agrégation des données :\n\
                                              \r\t- {'\n\r\t- '.join(list(admin_scale_aggfuncs))}\n",
                                          catch_dict=admin_scale_aggfuncs_entries)
                    if aggfunc=='Moyenne pondérée':
                        aggfuncs.update({col: lambda s: weighted_mean(s, pop_dataset=pop_dataset_, pop_variable=pop_variable)})
                    elif aggfunc=='Suppression':
                        new_df = new_df.drop(col, axis=1)
                    else:
                        aggfuncs.update({col:admin_scale_aggfuncs[aggfunc]})
                elif col!=carto.admin_scales_names[min_scale]\
                and col in expected_admin_columns\
                and col!='geometry':
                    # Suppression des colonnes d'identifiants administratifs non utilisées
                    new_df = new_df.drop(col, axis=1)
            # Agrégation
            new_df = new_df.dissolve(by=carto.admin_scales_names[min_scale], aggfunc=aggfuncs)
            new_df = new_df.drop(columns='geometry')
            new_df = new_df.join(grids[carto.admin_scales_grids[min_scale]].set_index(carto.admin_scales_names[min_scale])[['geometry']],
                                 on=carto.admin_scales_names[min_scale])
            new_df = gpd.GeoDataFrame(new_df, geometry='geometry')
            # Mise à jour des métadonnées du jeu agrégé
            new_df.attrs = attrs
            new_df.attrs.update({'scale':min_scale})
            agg_df_list.append(new_df)
        elif full_agg and dfs[df_name].attrs['scale'] is None:
            # Données non basées sur un échelon administratif
            # Estimation des populations pour répartition des données
            pop_dataset_ = carto.infer_pop_by_geom(df.geometry, pop_dataset, pop_variable)
            # Demande des fonctions pour l'agrégation des données
            aggfuncs={}
            for col in df.columns:
                if col in var_names :
                    aggfunc = select_list(list(admin_scale_aggfuncs),
                                          query=f"{col} :\nMéthode d'agrégation des données :\n\
                                              \r\t- {'\n\r\t- '.join(list(admin_scale_aggfuncs))}\n",
                                          catch_dict=admin_scale_aggfuncs_entries)
                    if aggfunc=='Moyenne pondérée':
                        aggfuncs.update({col: lambda s: weighted_mean(s, pop_dataset=pop_dataset_, pop_variable=pop_variable)})
                    elif aggfunc=='Suppression':
                        df = df.drop(col, axis=1)
                    else:
                        aggfuncs.update({col:admin_scale_aggfuncs[aggfunc]})
                elif col!=geom_grid_dict[attrs['scale']]\
                and col in expected_admin_columns\
                and col!='geometry':
                    # Suppression des colonnes d'identifiants administratifs non utilisées
                    df = df.drop(col, axis=1)
            # Découpage en vue de l'agrégation
            new_df = carto.list_overlay([df,grids[carto.admin_scales_grids[min_scale]]], proportional=True,
                                        pop_dataset=pop_dataset, pop_variable=pop_variable,
                                        corr_admin=corr_admin)
            new_df.columns = new_df.columns.str.removesuffix('_0')
            new_df.columns = new_df.columns.str.removesuffix('_1')
            # Agrégation
            new_df = new_df.dissolve(by=carto.admin_scales_names[min_scale], aggfunc=aggfuncs)
            new_df = new_df.drop(columns='geometry')
            new_df = new_df.join(grids[carto.admin_scales_grids[min_scale]].set_index(carto.admin_scales_names[min_scale])[['geometry']],
                                 on=carto.admin_scales_names[min_scale])
            new_df = gpd.GeoDataFrame(new_df, geometry='geometry')
            # Mise à jour des métadonnées du jeu agrégé
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
    variables : list(list(str,dict(str,dict)))
        Pour chaque variable à afficher ,
            Nom du jeu de donnée,
            Nom de la série de données,
                Nom de la variable,
                Type de la variable,
                Colorisation de l'affichage.
    
    """
    n_var = safe_ask("Nombre de variables à afficher :\n",int)
    variables = [[None,None] for n in range(n_var)] # Création du contenant des données de sortie
    if n_var==2:
        bivariate_cmap = safe_ask("Légende bivariée ? ", bool)
    for n in range(n_var):
        if n>0:
            print(f"Variable n°{n+1}\n")
            print('-'*(12+(n+1)//10))
        dataset = select_list(list(datasets.keys()),
                                      f"Nom du jeu de données source parmi :\n\
                                      \r\t- {'\n\r\t- '.join(list(datasets.keys()))}\n")
        variable = select_list(datasets[dataset].columns,
                               f"Nom de la série de données parmi :\n\
                               \r\t- {'\n\r\t- '.join(datasets[dataset].columns)}\n")
        variables[n][0] = [dataset,variable]
        var={}

        var['nom_legende'] = input("Nom de variable à inscrire sur la carte :\n").replace('\\n','\n')
        if n_var==2:
            var['bivariate_cmap'] = bivariate_cmap
            var['type'] = select_list(cmap_data,
                f"Type des données parmi :\n\
                \r\t- {'\n\r\t- '.join(cmap_data)}\n")
        else:
            var['type'] = select_list(data_types,
                f"Type des données parmi :\n\
                \r\t- {'\n\r\t- '.join(data_types)}\n")
        if var['type'] in cmap_data or bivariate_cmap:
            if n_var==2 and bivariate_cmap and n==1:
                if variables[0][1]['classification'] is None:
                    var['classification'] = None
                else:
                    var['classification'] = select_list(
                        [key for key in cmap_classification if key!='Aucune'],
                        f"Type de classification parmi :\n\
                        \r\t- {'\n\r\t- '.join([key for key in cmap_classification if key!='Aucune'])}\n",
                        catch_dict=cmap_classification_entries)
                    var['classification'] = cmap_classification[var['classification']]
            else:
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
                var['bins'] = dict(bins=var['bins'][1:],lowest=var['bins'][0])
            elif var['classification'] in ['FisherJenks']:
                var['bins'] = dict(k=safe_ask("Nombre de classes : ", int))
            elif var['classification'] == None:
                var['treatment'] = select_list(treatments,
                                               f"Traitement à appliquer aux données parmi :\n\
                                               \r\t- {'\n\r\t- '.join(treatments)}\n")
                var['labels'] = []
                var['labels'].append(input('Légende : étiquette pour les valeurs faibles :\n'))
                var['labels'].append(input('Légende : étiquette pour les valeurs élevées :\n'))
            if n_var==2 and bivariate_cmap and var['classification'] != None:
                var['default_labels'] = not safe_ask("Remplacer les étiquettes par défaut par une flèche directionnelle ? ", bool)
                if not var['default_labels']:
                    var['label'] = input("Légende : étiquette pour le sens d'augmentation de la variable :\n")
            if n_var==2 and bivariate_cmap and n==0:
                pass
            elif n_var==2 and bivariate_cmap and n==1:
                var['couleur'] = select_list(list(carto.bivariate_cmaps),
                                             f"Palette de couleur bivariée parmi :\n\
                                             \r\t- {'\n\r\t- '.join(carto.bivariate_cmaps)}\n")
            else:
                var['couleur'] = input(
                    "Palette de couleurs :\n\
                    \rPalettes standard :\n\
                    \r\t- YlOrRd\n\r\t- coolwarm ou RdBu\n\
                    \r\t- RdYlGn_corr ou GnYlRd_corr\n")
        else :
            var['couleur'] = safe_ask(
                "Couleur :\n", is_color_like)
        variables[n][1]=var
    return variables
