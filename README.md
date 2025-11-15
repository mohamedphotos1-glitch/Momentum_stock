# Momentum Stock Scanner

Scanner hebdomadaire de momentum actions US fonctionnant en local. Il repose sur un univers de tickers fourni par fichier CSV, télécharge les prix via **yfinance**, calcule les métriques clés (rendements, drawdown, proximité des plus hauts, moyennes mobiles) puis applique des profils de filtres paramétrables (B, C).

## Installation rapide

```bash
python -m venv .venv
source .venv/bin/activate   # sous Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install pandas yfinance numpy
```

Placez votre fichier d'univers (par défaut `univers_us.csv`) dans le dossier `data/`. Le fichier doit contenir une colonne de symboles (`Symbol`, `NASDAQ Symbol` ou `ACT Symbol`).

## Lancer un scan

Dans tous les cas, exécutez les commandes depuis la racine du projet.

Scanner le profil B (momentum "classique") :

```bash
python -m src.scanner_cli --profile B
```

Scanner le profil C (momentum plus strict) :

```bash
python -m src.scanner_cli --profile C
```

Vous pouvez limiter le nombre de tickers traités pour vos tests avec `--limit 50` par exemple.

## Résultats & logs

* Les fichiers CSV produits sont enregistrés dans `momentum_results/` (ex : `momentum_weekly_profile_B.csv`). Ils contiennent toutes les métriques calculées, les colonnes OK/KO et le score global.
* Les logs texte sont disponibles dans `logs/momentum_scanner.log` et les principaux messages sont également affichés dans le terminal.

## Structure du projet

```
momentum_system/
├─ data/                      # univers d'actions US (CSV)
├─ momentum_results/          # résultats des scans
├─ logs/                      # fichiers de log
└─ src/
   ├─ config_profiles.py      # définition des profils B, C, …
   ├─ universe_loader.py      # chargement & nettoyage de l'univers
   ├─ price_loader.py         # téléchargement yfinance
   ├─ momentum_metrics.py     # calcul des indicateurs + scoring
   ├─ scanner_cli.py          # CLI du scanner
   └─ utils_logging.py        # configuration centralisée du logging
```

Chaque module est autonome et peut évoluer indépendamment (ajout de nouveaux profils, watchlists, signaux d'entrée/sortie, etc.).
