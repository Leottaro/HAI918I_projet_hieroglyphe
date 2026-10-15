# Projet hiéroglyphes

Projet du cours HAI918I consacré à la reconnaissance de caractères hiéroglyphiques à partir d'images.

## Installation

Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Utilisation

Le point d'entrée prévu est `src/main.py` :

```bash
.venv/bin/python src/main.py
```

## Dépendances

Les dépendances Python sont listées dans `requirements.txt`. Le fichier est actuellement minimal car le code source est en cours de construction. Après l'ajout d'une bibliothèque, installez-la dans le venv puis mettez à jour la liste, par exemple :

```bash
python -m pip install "paquet"
python -m pip freeze > requirements.txt
```