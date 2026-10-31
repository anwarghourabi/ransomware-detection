# Synthèse de l'exploration de RanSMAP (tâche 3)

Binôme Anwar / Ghada · Notebook exécuté sur Kaggle (copie du dataset RanSMAP 2024).

## 1. Volume et structure

| Catégorie | Exécutions | Volume | Contenu |
|---|---|---|---|
| original | 1 440 | 120,1 Go | Chaque programme seul, 12 configurations matérielles |
| extra | 360 | 30,7 Go | Mêmes programmes, 3 configurations supplémentaires |
| mix | 100 | 11,4 Go | Ransomware lancé en même temps qu'un programme bénin |
| variants | 70 | 10,0 Go | 7 variantes de Conti |
| **Total** | **1 970** | **172,1 Go** | |

- original = 12 programmes × 12 configurations × 10 exécutions ; extra = 12 × 3 × 10.
- Ce volume (172 Go une fois décompressé) justifie l'usage de Spark.

## 2. Programmes et étiquettes

- **Ransomwares (6)** : Conti, Darkside, LockBit, REvil, Ryuk, WannaCry.
- **Programmes bénins (6)** : AESCrypt, Firefox, Idle, Office, SDelete, Zip.
- mix et variants sont étiquetés ransomware.
- original : 720 ransomware / 720 bénin (classes équilibrées) ; total : 1 070 / 900.

## 3. Configurations matérielles (15)

- original : celeron-gen6 (DDR4 1600 / 1866 / 2133, 8 ou 16 Go) et i3-gen12 (DDR4 2133 / 2666 / 3200, 8 ou 16 Go).
- extra : i3-gen12 DDR5-4800 16 Go, i5-gen12 et i7-gen12 DDR4-2133 16 Go.
- mix et variants : i3-gen12 DDR4-2133 16 Go.

## 4. Format des fichiers

Six CSV par exécution, sans ligne d'en-tête : `sec, ns, adresse (LBA ou GPA), taille, entropie, type`.

| Fichier | Taille moyenne | Remarque |
|---|---|---|
| ata_read | 17,4 Mo | lectures disque, entropie non mesurée (-1) |
| ata_write | 36,4 Mo (max 340 Mo) | écritures disque avec entropie : fichier le plus riche |
| mem_read | 15,9 Mo | lectures mémoire |
| mem_write | 17,7 Mo | écritures mémoire avec entropie |
| mem_readwrite | ≈ 0 Mo | quasi vide, inexploitable |
| mem_exec | ≈ 0 Mo | quasi vide, inexploitable |

## 5. Premières observations (sous-ensemble : original / i3-gen12 / DDR4-2133-16g, 3 exécutions par programme)

- **Durée** : environ 200 s par exécution (Ryuk plus variable, 180 à 300 s). Les 20 premières secondes sont quasi vides (démarrage du programme).
- **Entropie moyenne des écritures disque** : Zip 0,97 et AESCrypt 0,94 (bénins) dépassent Conti 0,67, Ryuk 0,58, Darkside 0,56. Une règle « entropie élevée = ransomware » produirait des faux positifs.
- **Nombre d'écritures disque** : ransomwares 80 000 à 600 000 ; Firefox, Office, Idle 7 000 à 75 000 ; mais SDelete (bénin) 3,1 millions.
- **Dynamique temporelle** (Conti vs AESCrypt) : AESCrypt passe brutalement à une entropie ≈ 1 et reste stable ; Conti a une entropie irrégulière qui augmente progressivement.

Ces chiffres portent sur 36 exécutions d'une seule configuration : ce sont des observations exploratoires, pas des résultats.

## 6. Conséquences pour la suite

1. Les indicateurs agrégés (moyenne d'entropie, volume) ne suffisent pas : la différence est dans l'évolution temporelle, ce qui motive les modèles séquentiels (CNN 1D, LSTM, GRU).
2. RQ1 : un préfixe de 10 % (≈ 20 s) contient très peu de signal ; on s'attend à une détection difficile à 10 % et nettement meilleure dès 25 %. Les préfixes seront définis en pourcentage de chaque exécution, car les durées varient.
3. Features à privilégier : `ata_write` (débit, entropie), puis `ata_read`, `mem_read`, `mem_write`.
4. Découpage train/test par exécution et par famille (jamais par fenêtre) ; extra servira au test sur matériel inédit, mix au test « ransomware masqué », variants au test sur variantes.
5. En local, le dataset complet ne tient pas décompressé : travailler par sous-ensembles ou convertir en Parquet au fil de l'eau.
