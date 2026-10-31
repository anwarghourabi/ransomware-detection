"""Accès à RanSMAP directement dans archive.zip, sans tout décompresser.

Le dataset complet fait 172 Go une fois décompressé : on fait l'inventaire à partir
de la liste des fichiers de l'archive, puis on n'extrait que le sous-ensemble utile.
"""
import shutil
import zipfile
from pathlib import Path

import pandas as pd

CATEGORIES = {"original", "extra", "mix", "variants"}
TRACES = ["ata_read", "ata_write", "mem_read", "mem_write", "mem_readwrite", "mem_exec"]
USEFUL = ["ata_read", "ata_write", "mem_read", "mem_write"]  # mem_readwrite et mem_exec sont vides


def _parse(name: str):
    """'RanSMAP/dataset/original/i3-gen12/ddr4-2133-16g/Conti/Conti-2024.../ata_write.csv'
    -> ('original', 'i3-gen12', 'ddr4-2133-16g', 'Conti', 'Conti-2024...', 'ata_write')"""
    parts = name.replace("\\", "/").split("/")
    if not parts[-1].endswith(".csv"):
        return None
    for i, p in enumerate(parts):
        if p in CATEGORIES and len(parts) - i == 6:
            return (*parts[i:i + 5], parts[-1][:-4])
    return None


def inventory_from_zip(zip_path) -> pd.DataFrame:
    """Une ligne par exécution, avec la taille décompressée de chaque fichier de traces."""
    rows = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parsed = _parse(info.filename)
            if parsed is None:
                continue
            category, cpu, ram, sample, run_name, trace = parsed
            key = (category, cpu, ram, sample, run_name)
            row = rows.setdefault(key, {"category": category, "cpu": cpu, "ram": ram,
                                        "sample": sample, "run_name": run_name})
            row[f"{trace}_mb"] = round(info.file_size / 1e6, 2)
    inv = pd.DataFrame(list(rows.values()))
    for t in TRACES:
        if f"{t}_mb" not in inv:
            inv[f"{t}_mb"] = 0.0
    inv[[f"{t}_mb" for t in TRACES]] = inv[[f"{t}_mb" for t in TRACES]].fillna(0.0)
    inv["total_mb"] = inv[[f"{t}_mb" for t in TRACES]].sum(axis=1).round(2)
    inv["rel"] = inv["category"] + "/" + inv["cpu"] + "/" + inv["ram"] + "/" + inv["sample"] + "/" + inv["run_name"]
    return inv.sort_values("rel").reset_index(drop=True)


def extract_subset(zip_path, dest, runs: pd.DataFrame, traces=USEFUL) -> int:
    """Extrait dans dest/<category>/<cpu>/<ram>/<sample>/<run>/<trace>.csv les fichiers
    des exécutions données. Les fichiers déjà extraits sont ignorés. Renvoie le nombre extrait."""
    wanted = {(r, t) for r in runs["rel"] for t in traces}
    dest = Path(dest)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parsed = _parse(info.filename)
            if parsed is None:
                continue
            rel, trace = "/".join(parsed[:5]), parsed[5]
            if (rel, trace) not in wanted:
                continue
            target = dest / rel / f"{trace}.csv"
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
            n += 1
            print(f"[{n}] {rel}/{trace}.csv ({info.file_size / 1e6:.0f} Mo)")
    return n
