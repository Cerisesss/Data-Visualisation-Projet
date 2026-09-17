# Data Visualisation - Projet fil rouge

## Installation 
1. Cloner le projet :

```bash
git clone https://github.com/Cerisesss/Data-Visualisation-Projet.git
```

2. Créer un environnement virtuel avec le terminal (première méthode) :

```bash
python -m venv .venv
source .venv/Scripts/activate
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

4. Sélectionner l'environnement virtuel :

Allez dans le fichier `main.py` et cliquez sur `Select Kernel > Python Environments... > .venv`.

5. Créer un environement virtuel pour le projet avec VSCode (deuxième méthode) :

Allez dans le fichier `main.py` et cliquez sur `Select Kernel > Python Environments... > Create Python Environment > Quick Create venv`.


## Dataset 
- Récupérer le dataset depuis ce lien : [https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres](Dataset)
- Téléchargez le fichier `Valeurs foncières 2025`.
- Dézippez le fichier et placez-le à la racine du projet.
- Vérifiez que le nom du fichier est bien `ValeursFoncieres-2025.txt` sinon renommez-le pour que le projet puisse le reconnaître.

## Lancer le projet
- Allez dans le fichier `main.py` et cliquez sur `Run all` ou utilisez la commande `python main.py`.