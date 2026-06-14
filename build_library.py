"""
build_library.py — buduje bibliotekę wzorcowych smoke'ów z wielu dem.

Idea: zbieramy pozycje LĄDOWANIA wszystkich smoke'ów z wielu meczów, grupujemy
je per mapa + strona algorytmem DBSCAN. Gęste skupiska = standardowe lineupy
(wiele osób rzuca w to samo miejsce). Rzuty odstające DBSCAN oznacza jako szum
i je pomijamy. Żadnych ręcznych etykiet — biblioteka wyłania się z danych.

To jest narzędzie OFFLINE, uruchamiane raz na jakiś czas — nie część aplikacji.

Wymagania (jednorazowo):
    pip install awpy scikit-learn numpy

Użycie:
    python build_library.py sciezka/do/folderu/z/demami
    (domyślnie folder "demos")

Wynik:
    smoke_library.json — słownik "mapa|strona" -> lista spotów {x, y, radius, count}
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from awpy import Demo
from sklearn.cluster import DBSCAN

# --- Parametry klasteryzacji (do kalibracji na realnych danych) ---
EPS = 130          # promień sąsiedztwa w jednostkach mapy: jak blisko muszą być
                   # dwa lądowania, by uznać je za "ten sam spot"
MIN_SAMPLES = 3    # ile rzutów w to samo miejsce, by uznać spot za standardowy


def collect_smokes(folder: str):
    """Parsuje wszystkie dema z folderu i zbiera lądowania smoke'ów per (mapa, strona)."""
    by_key = defaultdict(list)   # (mapa, strona) -> [(x, y, miejsce_rzutu), ...]
    dems = sorted(Path(folder).glob("*.dem"))
    if not dems:
        print(f"Brak plików .dem w folderze '{folder}'")
        return by_key

    print(f"Znaleziono {len(dems)} dem do przetworzenia\n")
    for i, p in enumerate(dems, 1):
        try:
            dem = Demo(str(p))
            dem.parse()
            map_name = (dem.header or {}).get("map_name", "?")
            sm = dem.smokes
            if sm is None or len(sm) == 0:
                print(f"[{i}/{len(dems)}] {p.name}: brak smoke'ów, pomijam")
                continue
            n = 0
            for row in sm.to_dicts():
                x, y = row.get("X"), row.get("Y")
                if x is None or y is None:
                    continue
                side = row.get("thrower_side") or "?"
                by_key[(map_name, side)].append((x, y, row.get("thrower_place")))
                n += 1
            print(f"[{i}/{len(dems)}] {p.name}: {map_name}, zebrano {n} smoke'ów")
        except Exception as e:
            print(f"[{i}/{len(dems)}] {p.name}: BŁĄD — {e}")
    print()
    return by_key


def cluster(by_key: dict):
    """Grupuje lądowania w spoty algorytmem DBSCAN; zwraca bibliotekę."""
    library = {}
    for (map_name, side), pts in sorted(by_key.items()):
        if len(pts) < MIN_SAMPLES:
            continue
        coords = np.array([[x, y] for x, y, _ in pts])
        labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(coords)

        spots = []
        for lab in set(labels):
            if lab == -1:
                continue  # -1 = szum = rzuty odstające (nie spot)
            mask = labels == lab
            cl = coords[mask]
            cx, cy = cl.mean(axis=0)
            radius = float(np.sqrt(((cl - [cx, cy]) ** 2).sum(axis=1)).max())
            # najczęstsze miejsce rzutu w tym skupisku (skąd zwykle się to rzuca)
            places = [pts[j][2] for j in range(len(pts)) if mask[j] and pts[j][2]]
            common_from = max(set(places), key=places.count) if places else None
            spots.append({
                "x": round(float(cx), 1),
                "y": round(float(cy), 1),
                "radius": round(radius, 1),
                "count": int(mask.sum()),
                "common_from": common_from,
            })

        spots.sort(key=lambda s: s["count"], reverse=True)
        noise = int((labels == -1).sum())
        library[f"{map_name}|{side}"] = spots
        print(f"{map_name} {side}: {len(spots)} standardowych spotów "
              f"z {len(pts)} rzutów ({noise} odstających)")
    return library


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "demos"
    by_key = collect_smokes(folder)
    library = cluster(by_key)
    out = Path("smoke_library.json")
    out.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano {out} — {sum(len(v) for v in library.values())} spotów łącznie")
