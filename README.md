# Détection précoce de ransomware (RanSMAP)

Projet du module Big Data & Deep Learning, binôme Anwar / Ghada.
Objectif : détecter un ransomware à partir de séquences comportementales
(accès mémoire et stockage, dataset RanSMAP) en n'observant qu'une fraction
de la séquence, et mesurer le compromis précocité / faux positifs.

## Structure

```
data/raw/          traces RanSMAP brutes (non versionnées)
data/processed/    Parquet, features, séquences (non versionnés)
notebooks/         exploration uniquement (préfixe NN_initiales_sujet.ipynb)
src/ingestion/     lecture des traces, conversion Parquet
src/features/      fenêtres temporelles, features, préfixes
src/models/        modèles classiques et profonds
src/evaluation/    métriques, RQ1 / RQ2 / RQ3
src/utils/         configuration, SparkSession
configs/           config.yaml (chemins, paramètres Spark, graine)
reports/figures/   figures du rapport
tests/             tests (pytest)
```

## Installation

Prérequis : Python 3.10 ou 3.11, Java JDK 17 (Spark 3.5 ne supporte pas Java 21).

```bash
git clone <url-du-depot>
cd ransomware-detection
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Vérifier l'installation

```bash
python -m pytest tests/ -v
```

Le test `test_spark_smoke` doit passer : Spark démarre et écrit/relit du Parquet.

## Conventions

- Branche `main` protégée ; une branche par tâche : `feature/<sujet>` (ex. `feature/ingestion`).
- Tout passe par une Pull Request relue par l'autre membre du binôme.
- Messages de commit : `type: description` avec type parmi `feat`, `fix`, `data`, `exp`, `docs`, `refactor`.
- Code formaté avec `black` et vérifié avec `ruff` avant chaque commit.
- Les notebooks servent à explorer ; le code réutilisé va dans `src/`.
- Aucune donnée ni modèle entraîné dans Git ; aucun chemin en dur (utiliser `configs/config.yaml`).
- Graine aléatoire unique : `experiment.seed` dans la configuration.
