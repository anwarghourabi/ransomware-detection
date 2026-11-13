# Détection précoce de ransomware (RanSMAP)

Projet du module **Big Data & Deep Learning**, binôme **Anwar / Ghada**.

Objectif : détecter un ransomware à partir de traces comportementales (accès disque et mémoire,
dataset RanSMAP) en n'observant qu'une **fraction** de l'exécution (10 / 25 / 50 / 75 / 100 %),
mesurer le compromis **précocité / faux positifs**, et tester la généralisation à une famille inconnue (zero-day).

| Question | Contenu |
|---|---|
| RQ1 | Fiabilité de la détection avec 10 %, 25 %, 50 %, 75 % de la séquence |
| RQ2 | Précision, rappel, F1, taux de faux positifs selon la fraction observée |
| RQ3 | Détection d'une famille absente de l'entraînement (leave-one-family-out) |

---

## 1. État d'avancement

| # | Tâche | Resp. | État | Notebook |
|---|---|---|---|---|
| 1 | Dépôt, environnement Spark / Python | Anwar | ✅ | – |
| 3 | Exploration de RanSMAP | Anwar | ✅ | `01_exploration_ransmap.py` |
| 4 | Ingestion PySpark → Parquet étiqueté | Anwar | ✅ | `02_ingestion_parquet.py` |
| 5 | Features par fenêtre de 1 s | Anwar | ✅ | `03_features_fenetres.py` |
| 6 | Séquences, préfixes, découpages figés | Anwar | ✅ | `04_sequences_prefixes.py` |
| 7 | Passage à l'échelle Spark | Anwar | ✅ | `05_passage_echelle.py` |
| 8 | Références Random Forest / GBT (Spark MLlib) | Anwar | ✅ | `06_references_classiques.py` |
| 9–12 | CNN 1D, LSTM, GRU, LSTM + attention, RQ1, RQ2 | **Ghada** | à faire | `07_…` |
| 13 | RQ3 leave-one-family-out, tests matériels | Anwar | à faire | – |

**Pour commencer les tâches 9 à 12, Ghada n'a besoin que de la section 4 (voie rapide).**

---

## 2. Structure du dépôt

```
configs/config.yaml        chemins, paramètres Spark, graine (seed = 42)
data/raw/                  archive.zip RanSMAP (non versionné)
data/processed/            Parquet, features, séquences (non versionné)
notebooks/                 notebooks VS Code au format .py (cellules « # %% »)
src/ingestion/             lecture du zip, étiquetage, conversion Parquet   (tâches 3-4)
src/features/              features par fenêtre temporelle                  (tâche 5)
src/sequences/             séquences, normalisation, préfixes, découpages   (tâche 6)
src/scaling/               mesure du passage à l'échelle Spark              (tâche 7)
src/models/                modèles (baselines.py = RF / GBT)                (tâche 8, puis 9-12)
src/utils/                 configuration, SparkSession
reports/                   tableaux .csv (splits.csv, résultats) et figures
tests/                     tests pytest
```

---

## 3. Installation (Windows, une seule fois)

### 3.1 Prérequis
- **Python 3.11** (pas 3.12+ ni 3.14 : PySpark 3.5 n'est pas compatible avec toutes les versions récentes).
- **Java JDK 17** (Adoptium Temurin 17). Vérifier : `java -version`.
- **Git**, **VS Code** avec les extensions *Python* et *Jupyter*.
- Un dossier **sans espace ni caractère spécial** dans le chemin, par ex. `D:\projets\` (Spark échoue avec des chemins contenant des espaces ou `&`).

### 3.2 Hadoop pour Windows (obligatoire pour que Spark écrive du Parquet)
1. Créer `D:\hadoop\bin\` (ou `C:\hadoop\bin\`).
2. Y copier `winutils.exe` et `hadoop.dll` pour Hadoop 3.3.x (dépôt GitHub *cdarlint/winutils*, dossier `hadoop-3.3.5/bin`).
3. Variables d'environnement Windows (Paramètres → « Modifier les variables d'environnement système ») :
   - `HADOOP_HOME` = `D:\hadoop`
   - ajouter `D:\hadoop\bin` au `Path`
4. **Fermer et rouvrir VS Code** pour que les variables soient prises en compte.

### 3.3 Cloner et installer
```powershell
cd D:\projets
git clone https://github.com/anwarghourabi/ransomware-detection.git
cd ransomware-detection
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4 Vérifier
```powershell
python -m pytest -q
```
Tous les tests doivent passer (dont `test_spark_smoke` : Spark démarre et écrit / relit du Parquet).

Dans VS Code : ouvrir un notebook `.py`, cliquer **Run Cell**, et choisir le noyau **`.venv (Python 3.11)`**
en haut à droite (pas un autre Python installé sur la machine).

---

## 4. Voie rapide pour les modèles (Ghada) : récupérer les séquences

Refaire tout le pipeline prend plusieurs heures et ~200 Go de disque. Pour entraîner les modèles,
il suffit du résultat de la tâche 6.

1. Anwar partage le dossier **`data/processed/sequences/`** (≈ 60 Mo, via Drive) contenant :
   - `sequences.npz` : `X`, `lengths`, `labels`
   - `runs.csv` : une ligne par exécution (programme, famille, configuration, **split**)
   - `preprocessing.json` : liste des 24 features, normalisation, graine
2. Le copier dans le clone, au même endroit : `ransomware-detection\data\processed\sequences\`.
3. `reports/splits.csv` est déjà dans Git (mêmes découpages que `runs.csv`).

### 4.1 Charger les données
```python
from pathlib import Path
from src.sequences import build as B

SEQ = Path("data/processed/sequences")
X, lengths, labels, runs = B.load(SEQ)
# X       : (1968, 322, 24) float32, déjà normalisé (log + centrage-réduction appris sur le train)
# lengths : (1968,) nombre de fenêtres réelles de chaque exécution (129 à 322) ; au-delà = padding à 0
# labels  : (1968,) 1 = ransomware, 0 = bénin
# runs    : DataFrame, colonnes rel, category, cpu, ram, family, sample, label, n_windows, split

train = (runs.split == "train").to_numpy()
val   = (runs.split == "val").to_numpy()
test  = (runs.split == "test").to_numpy()
```

### 4.2 Préfixes (RQ1 / RQ2)
```python
for frac in B.FRACTIONS:                         # [0.10, 0.25, 0.50, 0.75, 1.00]
    Xp, lp = B.prefix(X, lengths, frac)          # fenêtres après le préfixe mises à 0
    # entraîner / évaluer avec Xp et les longueurs lp
```
Les séquences sont **remplies de 0 à droite** : utiliser les longueurs (`lengths` ou `lp`) pour masquer
le padding (`pack_padded_sequence` en PyTorch, couche `Masking` en Keras), ou tronquer à `lp.max()`.

### 4.3 Jeux de données
| split | Contenu | Exécutions | bénin / rans. | Usage |
|---|---|---|---|---|
| `train` | original, 70 % | 1 006 | 504 / 502 | entraînement |
| `val` | original, 10 % | 144 | 72 / 72 | réglage, arrêt précoce |
| `test` | original, 20 % | 288 | 144 / 144 | **RQ1, RQ2** |
| `test_hw` | extra : matériel jamais vu | 360 | 180 / 180 | généralisation matérielle |
| `test_mix` | ransomware + programme bénin simultanés | 100 | 0 / 100 | rappel seulement |
| `test_variants` | 7 variantes de Conti | 70 | 0 / 70 | rappel seulement |

Règles :
- **Ne jamais** régler un modèle sur `test`, `test_hw`, `test_mix`, `test_variants` (uniquement `val`).
- Les découpages sont **figés** (graine 42) : ne pas les refaire, utiliser la colonne `split`.
- RQ3 : `B.lofo_folds(runs)` donne, pour chaque famille, les indices `(train, test)` (Conti + variantes regroupés).

### 4.4 Résultats à battre (Random Forest, tâche 8)
| Fraction | F1 test | FPR test | F1 test_hw | Rappel test_mix | Rappel test_variants |
|---|---|---|---|---|---|
| 10 % | 0,969 | 0,028 | 0,900 | 0,43 | 0,59 |
| 25 % | 0,986 | 0,021 | 0,958 | 0,62 | 0,31 |
| 50 % | 0,993 | 0,014 | 0,994 | 0,51 | 0,46 |
| 75 % | 0,993 | 0,014 | 0,989 | 0,64 | 0,61 |
| 100 % | 0,993 | 0,014 | 0,989 | 0,68 | 0,63 |

Sur `test` les références sont déjà presque parfaites : l'apport des modèles séquentiels se mesurera surtout
sur **`test_mix`, `test_variants`, `test_hw` à 10-25 %** et **RQ3**. Détail : `reports/baseline_results.csv`.

Utiliser `src.models.baselines.metrics(y, p)` pour calculer les mêmes métriques
(accuracy, précision, rappel, F1, FPR, AUC ROC, AUC PR) et comparer à l'identique.

---

## 5. Voie complète : refaire tout le pipeline (optionnel)

1. Télécharger RanSMAP (Kaggle / GitHub *manabu-hirano/RanSMAP*) et placer le zip dans `data/raw/archive.zip` (28,7 Go).
2. Prévoir **~60 Go libres** (Parquet 38 Go + un morceau de CSV à la fois).
3. Lancer les notebooks dans l'ordre, cellule par cellule, PC branché :

| Notebook | Durée | Sortie |
|---|---|---|
| `01_exploration_ransmap.py` | ~15 min | `reports/inventaire_ransmap.csv`, figures |
| `02_ingestion_parquet.py` | **~3 h** | `data/processed/parquet/` (37,8 Go) |
| `03_features_fenetres.py` | ~10 min | `data/processed/features/w=1.0/`, `trace_offsets.csv` |
| `04_sequences_prefixes.py` | ~3 min | `data/processed/sequences/`, `reports/splits.csv` |
| `05_passage_echelle.py` | ~30 min | `reports/scaling_*.csv`, figure |
| `06_references_classiques.py` | ~15 min | `reports/baseline_results.csv`, figures |

L'ingestion (02) peut être interrompue : relancer les cellules 1, 2, 3 puis 8, elle reprend où elle s'est arrêtée.

---

## 6. Méthode de travail Git

```powershell
git pull                                   # toujours avant de commencer
git checkout -b feature/lstm               # une branche par tâche
# ... travail ...
git add src notebooks tests reports
git commit -m "feat: LSTM sur préfixes"
git push -u origin feature/lstm            # puis Pull Request sur GitHub, relue par l'autre
```
- Jamais de données, de modèles entraînés (`*.pt`, `*.h5`) ni de `.venv` dans Git (déjà dans `.gitignore`).
- Les tableaux de résultats (`reports/*.csv`) **doivent** être versionnés : vérifier que `.gitignore`
  contient la ligne `!reports/*.csv` (sinon `*.csv` les ignore).
- Code réutilisable dans `src/`, notebooks pour lancer et afficher.

---

## 7. Problèmes fréquents

| Message | Solution |
|---|---|
| `Python was not found` / `Python worker failed to connect back` | Spark ne trouve pas le Python du venv : vérifier que `src/utils/spark_session.py` définit `PYSPARK_PYTHON` (c'est déjà le cas) et lancer depuis le noyau `.venv`. |
| `HADOOP_HOME and hadoop.home.dir are unset` | Section 3.2, puis redémarrer VS Code. |
| `ModuleNotFoundError: No module named 'src'` | Ouvrir VS Code à la **racine** du dépôt ; la cellule 1 de chaque notebook ajoute la racine au chemin. |
| `ModuleNotFoundError` (matplotlib, pandas…) dans un notebook | Mauvais noyau : choisir `.venv (Python 3.11)`. |
| `Undefined variable 'display'` (souligné rouge) | Simple avertissement de Pylint ; ou ajouter `from IPython.display import display`. |
| `Py4JJavaError` sur `toPandas()` | Pour de petites tables, lire avec `pd.read_parquet(...)` (nécessite `pyarrow`). |
| `No space left on device` | Libérer de la place sur C: (cache pip, `%TEMP%`) ou mettre `TEMP`/`TMP` sur D:. |
| Spark échoue sur un chemin | Le chemin du projet ne doit contenir ni espace ni `&`. |

---

## 8. Données et chiffres clés

- RanSMAP : 1 970 exécutions, 172 Go de CSV, 12 programmes (6 ransomwares : Conti, Darkside, LockBit, REvil, Ryuk, WannaCry ; 6 bénins : AESCrypt, Firefox, Idle, Office, SDelete, Zip), 15 configurations matérielles.
- Étiquette déduite du nom du dossier (les CSV n'en contiennent pas).
- Parquet : 37,8 Go (×4,5 plus compact), requêtes ~20× plus rapides.
- Fenêtre de 1 s ; 24 features par fenêtre ; 1 968 exécutions retenues (2 WannaCry incomplètes écartées).
- Rapport de suivi des tâches 4 à 7 : `Rapport_suivi_taches4-7.pdf`.
