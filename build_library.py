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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from awpy import Demo
from sklearn.cluster import DBSCAN

# --- Parametry klasteryzacji (strój i odpalaj ponownie samo 'cluster') ---
EPS = 130          # jak blisko muszą być dwa lądowania, by były "tym samym spotem"
MIN_SAMPLES = 3    # ile rzutów w miejsce, by uznać je za standardowy spot

CACHE_FILE = Path("landings_cache.json")
LIBRARY_FILE = Path("utility_library.json")
PLAYS_FILE = Path("plays_library.json")

UTILITY_TABLES = {"smoke": "smokes", "molotov": "infernos"}


def _load_cache():
    """Zwraca (przetworzone | None, dane_utility, zagrania). None = stary format."""
    if not CACHE_FILE.exists():
        return [], defaultdict(list), []
    raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw and "_processed" in raw:
        return list(raw["_processed"]), defaultdict(list, raw["data"]), list(raw.get("plays", []))
    return None, defaultdict(list, raw), []   # stary format -> migracja


def _save_cache(processed, data, plays):
    CACHE_FILE.write_text(
        json.dumps({"_processed": sorted(processed), "data": data, "plays": plays},
                   ensure_ascii=False),
        encoding="utf-8",
    )


def _parse_one_demo(path_str: str) -> dict:
    """Parsuje JEDNO demo i zwraca lądowania utility + zagrania flash+kill.
    Funkcja na poziomie modułu — może działać w osobnym procesie (pula)."""
    from pathlib import Path
    name = Path(path_str).name
    try:
        dem = Demo(path_str)
        dem.parse()
        map_name = (dem.header or {}).get("map_name", "?")

        def safe_table(attr):
            try:
                return getattr(dem, attr, None)
            except Exception:
                return None

        # najwcześniejszy tick każdego granatu: (entity_id, round) -> tick
        gren_first = {}
        gdf = safe_table("grenades")
        if gdf is not None and len(gdf) > 0:
            gren_rows = gdf.to_dicts()
            for r in gren_rows:
                if r.get("X") is None:
                    continue
                key = (r.get("entity_id"), r.get("round_num"))
                t = r["tick"]
                if key not in gren_first or t < gren_first[key]:
                    gren_first[key] = t
        else:
            gren_rows = []

        # Flashe: w dem.grenades (brak własnej tabeli). Jeden rzut = (entity_id, round)
        # z CFlashbangProjectile; wybuch = pozycja w ostatnim ticku toru.
        flash_by_throw = defaultdict(list)
        for r in gren_rows:
            if r.get("grenade_type") == "CFlashbangProjectile" and r.get("X") is not None:
                flash_by_throw[(r.get("entity_id"), r.get("round_num"))].append(r)

        # pozycja rzutu z dem.ticks — teraz z dołączoną STRONĄ (CT/T) i calloutem.
        # Pary (steamid, tick_rzutu) zbieramy ze smoke/molo (z tabel) ORAZ z flashy.
        tdf = safe_table("ticks")
        throw_lut = {}
        if tdf is not None and len(tdf) > 0:
            pairs = set()
            for attr in UTILITY_TABLES.values():
                df = safe_table(attr)
                if df is None:
                    continue
                for row in df.to_dicts():
                    ft = gren_first.get((row.get("entity_id"), row.get("round_num")))
                    if ft is not None:
                        pairs.add((row.get("thrower_steamid"), ft))
            for (eid, rnd), rows in flash_by_throw.items():
                ft = gren_first.get((eid, rnd))
                steam = rows[0].get("thrower_steamid")
                if ft is not None:
                    pairs.add((steam, ft))
            if pairs:
                sub = tdf.filter(
                    pl.col("steamid").is_in(list({a for a, _ in pairs})) &
                    pl.col("tick").is_in(list({b for _, b in pairs}))
                ).to_dicts()
                for r in sub:
                    throw_lut[(r["steamid"], r["tick"])] = (
                        r.get("X"), r.get("Y"), r.get("place"), r.get("side")
                    )

        items = {}
        total = 0
        # smoke + molotov: lądowanie z tabeli, strona z thrower_side
        for utype, attr in UTILITY_TABLES.items():
            df = safe_table(attr)
            if df is None or len(df) == 0:
                continue
            for row in df.to_dicts():
                x, y = row.get("X"), row.get("Y")
                if x is None or y is None:
                    continue
                side = row.get("thrower_side") or "?"
                tx = ty = None
                tplace = row.get("thrower_place")
                ft = gren_first.get((row.get("entity_id"), row.get("round_num")))
                pos = throw_lut.get((row.get("thrower_steamid"), ft)) if ft else None
                if pos and pos[0] is not None:
                    tx, ty, tplace = round(pos[0], 1), round(pos[1], 1), pos[2] or tplace
                items.setdefault(f"{map_name}|{side}|{utype}", []).append(
                    [round(x, 1), round(y, 1), tx, ty, tplace]
                )
                total += 1

        # flash: wybuch = ostatni tick toru; strona + pozycja rzutu z dem.ticks
        for (eid, rnd), rows in flash_by_throw.items():
            rows.sort(key=lambda r: r["tick"])
            det = rows[-1]
            steam = rows[0].get("thrower_steamid")
            ft = gren_first.get((eid, rnd))
            pos = throw_lut.get((steam, ft)) if ft else None
            tx = ty = tplace = side = None
            if pos and pos[0] is not None:
                tx, ty, tplace, side = round(pos[0], 1), round(pos[1], 1), pos[2], pos[3]
            side = side or "?"
            items.setdefault(f"{map_name}|{side}|flash", []).append(
                [round(det["X"], 1), round(det["Y"], 1), tx, ty, tplace]
            )
            total += 1

        plays = []
        kdf = safe_table("kills")
        if kdf is not None and len(kdf) > 0:
            for kr in kdf.to_dicts():
                if not kr.get("assistedflash"):
                    continue
                plays.append({
                    "map":          map_name,
                    "victim_place": kr.get("victim_place"),
                    "victim_side":  kr.get("victim_side"),
                    "flash_from":   kr.get("assister_place"),
                    "killer_from":  kr.get("attacker_place"),
                })

        return {"file": name, "map": map_name, "items": items,
                "plays": plays, "total": total, "error": None}
    except Exception as e:
        return {"file": name, "map": None, "items": {},
                "plays": [], "total": 0, "error": str(e)}


def extract(folder: str, rebuild: bool = False, workers: int = 3):
    """Parsuje nowe dema RÓWNOLEGLE i dokłada wyniki do cache'u.
    workers = ile dem parsować naraz (domyślnie 3 — łagodnie dla pracy w tle)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    dems = sorted(Path(folder).glob("*.dem"))
    if not dems:
        print(f"Brak plików .dem w folderze '{folder}'")
        return

    if rebuild:
        processed, by_key, plays = [], defaultdict(list), []
        print("Tryb --rebuild: parsuję wszystkie dema od nowa.")
    else:
        processed, by_key, plays = _load_cache()
        if processed is None:
            processed = [p.name for p in dems]
            plays = []
            print(f"Wykryto stary cache — uznaję obecne {len(processed)} dem za już przetworzone.")

    done = set(processed)
    todo = [p for p in dems if p.name not in done]

    if not todo:
        print(f"Wszystkie {len(dems)} dem już sparsowane — nic nowego.")
    else:
        workers = max(1, workers)
        print(f"{len(dems)} dem w folderze, {len(todo)} nowych — parsuję {workers} naraz\n")
        finished = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_parse_one_demo, str(p)): p for p in todo}
            for fut in as_completed(futures):
                res = fut.result()
                finished += 1
                if res["error"]:
                    print(f"[{finished}/{len(todo)}] {res['file']}: BŁĄD — {res['error']}")
                    continue
                for key, lst in res["items"].items():
                    by_key[key].extend(lst)
                plays.extend(res["plays"])
                processed.append(res["file"])
                print(f"[{finished}/{len(todo)}] {res['file']}: {res['map']}, "
                      f"{res['total']} rzutów (smoke+molo+flash)")
                # zapis przyrostowy co 5 dem — przerwanie nie traci całej roboty
                if finished % 5 == 0:
                    _save_cache(processed, dict(by_key), plays)

    _save_cache(processed, dict(by_key), plays)
    total_pts = sum(len(v) for v in by_key.values())
    print(f"\nCache: {len(processed)} dem, {total_pts} rzutów utility, {len(plays)} zagrań flash+kill")
    print("Teraz uruchom:  python build_library.py cluster   (smoke'i)")
    print("           oraz: python build_library.py plays     (zagrania)")


def cluster_from_cache():
    """Czyta cache i grupuje lądowania w spoty algorytmem DBSCAN."""
    if not CACHE_FILE.exists():
        print(f"Brak {CACHE_FILE}. Najpierw: python build_library.py extract demos")
        return

    _, cache, _ = _load_cache()
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
            dists = np.sqrt(((cl - [cx, cy]) ** 2).sum(axis=1))
            radius = float(dists.max())          # zasięg (do najdalszego rzutu)
            spread = float(np.median(dists))      # typowy rozrzut (odporny na outliery)
            members = [pts[j] for j in range(len(pts)) if mask[j]]

            # callout rzutu (z poprawionej pozycji rzutu)
            places = [m[4] for m in members if len(m) > 4 and m[4]]
            common_from = max(set(places), key=places.count) if places else None

            # wzorcowa pozycja rzutu = średnia pozycji rzutów w tym skupisku
            throws = [(m[2], m[3]) for m in members
                      if len(m) > 3 and m[2] is not None and m[3] is not None]
            throw_x = throw_y = None
            if throws:
                throw_x = round(sum(t[0] for t in throws) / len(throws), 1)
                throw_y = round(sum(t[1] for t in throws) / len(throws), 1)

            spots.append({
                "x": round(float(cx), 1),
                "y": round(float(cy), 1),
                "radius": round(radius, 1),
                "spread": round(spread, 1),
                "count": int(mask.sum()),
                "common_from": common_from,
                "throw_x": throw_x,
                "throw_y": throw_y,
            })

        if spots:
            # Krytyczny smoke = pro trafiają go bardzo ciasno (mały spread = każda
            # luka boli) i nie jest to marginalny, rzadki rzut.
            #  spread = typowy rozrzut (mediana odległości od centroidu, odporny na outliery)
            total = sum(s["count"] for s in spots)
            for s in spots:
                s["freq"] = round(s["count"] / total, 3) if total else 0
                s["critical"] = bool(s["spread"] <= 50 and s["count"] >= 8)

            spots.sort(key=lambda s: (s["critical"], s["count"]), reverse=True)
            library[key] = spots
            noise = int((labels == -1).sum())
            ncrit = sum(1 for s in spots if s["critical"])
            spreads = sorted(s["spread"] for s in spots)
            print(f"{key}: {len(spots)} spotów ({ncrit} kryt.) z {len(pts)} rzutów "
                  f"| spread min/med/max = {spreads[0]:.0f}/{spreads[len(spreads)//2]:.0f}/{spreads[-1]:.0f}")

    LIBRARY_FILE.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")
    total_spots = sum(len(v) for v in library.values())
    print(f"\nZapisano {LIBRARY_FILE} — {total_spots} spotów łącznie")


def build_plays():
    """Agreguje zagrania flash+kill z cache'u w bibliotekę per miejsce.

    Klucz: mapa|miejsce_smierci|strona_ofiary. Wynik: jak często pro zdobywają
    tam killa z flasha i skąd ten flash najczęściej leci.
    """
    if not CACHE_FILE.exists():
        print(f"Brak {CACHE_FILE}. Najpierw: python build_library.py extract demos")
        return
    _, _, plays = _load_cache()
    if not plays:
        print("Brak zagrań flash+kill w cache. Zrób extract --rebuild nową wersją skryptu.")
        return

    MIN_PLAYS = 4   # min. liczba flash-killi w miejscu, by uznać za wzorzec
    agg = {}
    for pl_ in plays:
        place = pl_.get("victim_place")
        if not place:
            continue
        key = f"{pl_.get('map')}|{place}|{pl_.get('victim_side')}"
        a = agg.setdefault(key, {"flash_kills": 0,
                                 "flash_from": Counter(), "killer_from": Counter()})
        a["flash_kills"] += 1
        if pl_.get("flash_from"):
            a["flash_from"][pl_["flash_from"]] += 1
        if pl_.get("killer_from"):
            a["killer_from"][pl_["killer_from"]] += 1

    library = {}
    for key, a in agg.items():
        if a["flash_kills"] < MIN_PLAYS:
            continue
        library[key] = {
            "flash_kills": a["flash_kills"],
            "flash_from":  a["flash_from"].most_common(3),
            "killer_from": a["killer_from"].most_common(3),
        }

    PLAYS_FILE.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano {PLAYS_FILE} — {len(library)} miejsc-zagrań (min {MIN_PLAYS} flash-killi)\n")
    for key, v in sorted(library.items(), key=lambda x: -x[1]["flash_kills"])[:15]:
        ff = v["flash_from"][0] if v["flash_from"] else ("?", 0)
        print(f"  {key}: {v['flash_kills']} flash-killi | flash zwykle z {ff[0]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "extract":
        args = [a for a in sys.argv[2:] if not a.startswith("--")]
        folder = args[0] if args else "demos"
        workers = 3
        for a in sys.argv[2:]:
            if a.startswith("--workers"):
                # forma --workers=N albo --workers N
                if "=" in a:
                    workers = int(a.split("=")[1])
                else:
                    idx = sys.argv.index(a)
                    if idx + 1 < len(sys.argv):
                        workers = int(sys.argv[idx + 1])
        extract(folder, rebuild="--rebuild" in sys.argv, workers=workers)
    elif cmd == "cluster":
        cluster_from_cache()
    elif cmd == "plays":
        build_plays()
    else:
        print("Użycie:")
        print("  python build_library.py extract demos                 # parsuj nowe dema (3 naraz)")
        print("  python build_library.py extract demos --workers 6     # szybciej (gdy laptop wolny)")
        print("  python build_library.py extract demos --rebuild       # od nowa wszystko")
        print("  python build_library.py cluster                       # biblioteka smoke'ów")
        print("  python build_library.py plays                         # biblioteka zagrań flash+kill")
