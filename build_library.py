"""
build_library.py — buduje bibliotekę wzorcowych smoke'ów i mołotowów z wielu dem.

Dwa etapy, ROZDZIELONE celowo:
  1) extract  — parsuje dema i zapisuje surowe lądowania do cache'u (WOLNE)
  2) cluster  — czyta cache i grupuje lądowania w spoty przez DBSCAN (SZYBKIE)

extract działa PRZYROSTOWO: pamięta, które dema już przerobił, i przy kolejnym
uruchomieniu parsuje tylko NOWE pliki. Cały folder od nowa zrobisz przez --rebuild.

Wymagania (jednorazowo):
    pip install awpy scikit-learn numpy

Użycie:
    python build_library.py extract demos             # parsuj tylko nowe dema -> cache
    python build_library.py extract demos --rebuild   # sparsuj wszystko od nowa
    python build_library.py cluster                   # cache -> biblioteka
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from awpy import Demo
from sklearn.cluster import DBSCAN

# --- Parametry klasteryzacji (strój i odpalaj ponownie samo 'cluster') ---
EPS = 130          # jak blisko muszą być dwa lądowania, by były "tym samym spotem"
MIN_SAMPLES = 3    # ile rzutów w miejsce, by uznać je za standardowy spot

CACHE_FILE = Path("landings_cache.json")
LIBRARY_FILE = Path("utility_library.json")

UTILITY_TABLES = {"smoke": "smokes", "molotov": "infernos"}


def _load_cache():
    """Zwraca (lista_przetworzonych_dem | None, dane). None = stary płaski format."""
    if not CACHE_FILE.exists():
        return [], defaultdict(list)
    raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw and "_processed" in raw:
        return list(raw["_processed"]), defaultdict(list, raw["data"])
    return None, defaultdict(list, raw)   # stary format -> wymaga migracji


def _save_cache(processed, data):
    CACHE_FILE.write_text(
        json.dumps({"_processed": sorted(processed), "data": data}, ensure_ascii=False),
        encoding="utf-8",
    )


def extract(folder: str, rebuild: bool = False):
    """Parsuje nowe dema i dokłada ich lądowania do cache'u."""
    dems = sorted(Path(folder).glob("*.dem"))
    if not dems:
        print(f"Brak plików .dem w folderze '{folder}'")
        return

    if rebuild:
        processed, by_key = [], defaultdict(list)
        print("Tryb --rebuild: parsuję wszystkie dema od nowa.\n")
    else:
        processed, by_key = _load_cache()
        if processed is None:
            # stary płaski cache -> zakładamy, że powstał z obecnego folderu
            processed = [p.name for p in dems]
            print(f"Wykryto stary cache — uznaję obecne {len(processed)} dem za już przetworzone.")

    done = set(processed)
    todo = [p for p in dems if p.name not in done]

    if not todo:
        print(f"Wszystkie {len(dems)} dem już sparsowane — nic nowego.")
    else:
        print(f"{len(dems)} dem w folderze, {len(todo)} nowych do sparsowania\n")
        for i, p in enumerate(todo, 1):
            try:
                dem = Demo(str(p))
                dem.parse()
                map_name = (dem.header or {}).get("map_name", "?")
                total = 0
                for utype, attr in UTILITY_TABLES.items():
                    df = getattr(dem, attr, None)
                    if df is None or len(df) == 0:
                        continue
                    for row in df.to_dicts():
                        x, y = row.get("X"), row.get("Y")
                        if x is None or y is None:
                            continue
                        side = row.get("thrower_side") or "?"
                        by_key[f"{map_name}|{side}|{utype}"].append(
                            [round(x, 1), round(y, 1), row.get("thrower_place")]
                        )
                        total += 1
                processed.append(p.name)
                print(f"[{i}/{len(todo)}] {p.name}: {map_name}, {total} rzutów (smoke+molo)")
            except Exception as e:
                print(f"[{i}/{len(todo)}] {p.name}: BŁĄD — {e}")

    _save_cache(processed, dict(by_key))
    total_pts = sum(len(v) for v in by_key.values())
    print(f"\nCache: {len(processed)} dem, {total_pts} rzutów łącznie")
    print("Teraz uruchom:  python build_library.py cluster")


def cluster_from_cache():
    """Czyta cache i grupuje lądowania w spoty algorytmem DBSCAN."""
    if not CACHE_FILE.exists():
        print(f"Brak {CACHE_FILE}. Najpierw: python build_library.py extract demos")
        return

    _, cache = _load_cache()
    library = {}
    print(f"Klasteryzacja (EPS={EPS}, MIN_SAMPLES={MIN_SAMPLES})\n")
    for key, pts in sorted(cache.items()):
        if len(pts) < MIN_SAMPLES:
            continue
        coords = np.array([[p[0], p[1]] for p in pts])
        labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(coords)

        spots = []
        for lab in set(labels):
            if lab == -1:
                continue
            mask = labels == lab
            cl = coords[mask]
            cx, cy = cl.mean(axis=0)
            radius = float(np.sqrt(((cl - [cx, cy]) ** 2).sum(axis=1)).max())
            places = [pts[j][2] for j in range(len(pts)) if mask[j] and pts[j][2]]
            common_from = max(set(places), key=places.count) if places else None
            spots.append({
                "x": round(float(cx), 1),
                "y": round(float(cy), 1),
                "radius": round(radius, 1),
                "count": int(mask.sum()),
                "common_from": common_from,
            })

        if spots:
            spots.sort(key=lambda s: s["count"], reverse=True)
            library[key] = spots
            noise = int((labels == -1).sum())
            print(f"{key}: {len(spots)} spotów z {len(pts)} rzutów ({noise} odstających)")

    LIBRARY_FILE.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")
    total_spots = sum(len(v) for v in library.values())
    print(f"\nZapisano {LIBRARY_FILE} — {total_spots} spotów łącznie")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "extract":
        args = [a for a in sys.argv[2:] if not a.startswith("--")]
        folder = args[0] if args else "demos"
        extract(folder, rebuild="--rebuild" in sys.argv)
    elif cmd == "cluster":
        cluster_from_cache()
    else:
        print("Użycie:")
        print("  python build_library.py extract demos            # parsuj tylko nowe dema")
        print("  python build_library.py extract demos --rebuild  # od nowa wszystko")
        print("  python build_library.py cluster                  # zbuduj bibliotekę")
