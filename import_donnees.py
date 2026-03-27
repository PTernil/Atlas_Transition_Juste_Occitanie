# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:06:35 2026

@author: ternilp
"""
import sys
from pathlib import Path
import geopandas as gpd
import fiona

# Types de géométries de raccordement
geom_dict = {'département':'dep','departement':'dep','dép':'dep','dep':'dep',
             'commune':'com','comm':'com','com':'com',
             'iris':'iris',
             'autre':None
             }

# Noms de colonnes dans les données pouvant être raccordées automatiquement aux références
geom_columns_dict = {'dep':['département','departement','dép','dep',
                            'code_département','code_departement',
                            'code_dép','code_dep'],
                     'com':['commune','comm','com',
                            'code_commune','code_comm','code_com',
                            'code_insee'],
                     'iris':['iris','code_iris'],
                     }

# Liste des fichiers contenant des données
file_list = [x for x in Path(r"Données traitées").glob("**/*") if x.is_file()]
for georef in (Path(r"Données traitées\Région.gpkg"),
               Path(r"Données traitées\Départements.gpkg"),
               Path(r"Données traitées\Communes.gpkg"),
               Path(r"Données traitées\IRIS.gpkg"),
               Path(r"Données traitées\Pays limitrophes.gpkg"),
               Path(r"Données traitées\Régions limitrophes.gpkg"),
               Path(r"Données traitées\Départements limitrophes.gpkg")):
    file_list.remove(georef)
file_list = [file.stem for file in file_list]

# Liste des formats supportés
format_list=['GeoPackage (.gpkg)','Comma Separated Values (.csv)']


def import_progress(filepath, compact=True):
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
    if type(filepath)==str:
        filepath = Path(filepath)
    if filepath.parts[-1][-5:]==".gpkg":
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
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(f"\rImportation... {percent}% ({i}/{total})")
                        sys.stdout.flush()
        if not compact:
            print("\nImportation terminée !", flush=True)
        data = gpd.GeoDataFrame.from_features(features, crs=src.crs)
        data.name=filepath
        return data
    elif filepath.parts[-1][-4:]==".csv":
        return None
    raise TypeError("Format de données non supporté. Formats valides :\n\
                    \r\tGeoPackage (.gpkg)\n\
                    \r\tComma Separated Values (.csv)")

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
                                  \r-\t{'\n\r-\t'.join(file_list)}\n\
                                  \rNom du fichier contenant les données :\n"))
        else:
            filepath = Path(input(f"Erreur : Format de données non supporté. Formats valides :\n\r-\t\
                                 {'\n\r-\t'.join(format_list)}\n\
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

def ask_geom():
    """
    Demande à l'utilisateur un nom de géométrie à laquelle raccrocher les données.
    En cas d'échec, répète la demande.

    Returns
    -------
    geom : str
        Nom de la géométrie à laquelle raccrocher les données.

    """
    geom = input("Géométrie de référence parmi :\n\t- Département/dep\n\
                \r\t- Commune/com\n\t- IRIS\n\t- Autre\n\
                \rSélectionnez 'Autre' pour des données dont la localisation est incluse.\n").lower()
    while not geom in geom_dict.keys():
        geom = input(f"\nErreur :'{geom}' n'est pas une échelle valide.\n\
                   \rSélectionnez une échelle parmi :\n\t- Département/dep\n\
                   \r\t- Commune/com\n\t- IRIS\n\t- Autre\n\
                   \r Sélectionnez 'Autre' pour des données dont la localisation est incluse.\n").lower()
    geom = geom_dict[geom]
    return geom

def search_geom(geom_grid, data, geom_data=None):
    """
    

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
                                  \r-\t{'\n\r-\t'.join(geoms)}\n")
                geom_data = search_geom(geom_grid, data, geom_data)
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
                raise ValueError(f"{data.name} contient plusieurs colonnes portant ce nom et contenant une géométrie.\n\
                                 \rIl est impossible de l'utiliser en l'état")
    elif geom_data is None:    # Géométrie 'type'
        geoms = geom_columns_dict[geom_grid]
        columns = data.columns
        for column in columns:
            if column.lower() in geoms:
                print("Géométrie de raccordement trouvée.")
                return column
        geom_data = input(f"\nErreur : géométrie de raccordement introuvable dans les données.\n\
                     \rSélectionnez la clé de raccordement parmi :\n\
                     \r-\t{'\n\r-\t'.join(columns)}\n\
                     \rAppuyez sur Entrée pour revenir au choix de la géométrie de référence.\n")
        geom_data = search_geom(geom_grid, data, geom_data)
    elif geom_data=='':
        return geom_data
    else:   # Géométrie non prévue, nom de la colonne fourni
        columns=data.columns
        for column in columns:
            if geom_data.lower()==column.lower():
                return column
        geom_data = input(f"\nErreur : la clé fournie est absente des données.\n\
                     \rSélectionnez la clé de raccordement parmi :\n\
                     \r-\t{'\n\r-\t'.join(columns)}\n")
        geom_data = search_geom(geom_grid, data, geom_data)

