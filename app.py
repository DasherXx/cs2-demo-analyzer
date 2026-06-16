"""
CS2 Demo API — prototyp (krok 2)

Nowy endpoint: POST /api/demos/{demo_id}/parse
Parsuje wgrane demo przez awpy i zapisuje wynik jako raport JSON.
Parsowanie trwa kilka-kilkanaście sekund — na razie synchronicznie,
w kolejnym kroku przeniesie się do kolejki zadań w tle.

Uruchomienie (bez zmian):
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import datetime
import json
import uuid
from pathlib import Path

import polars as pl
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

BASE = Path(__file__).parent
STORAGE = BASE / "storage"
STORAGE.mkdir(exist_ok=True)

app = FastAPI(title="CS2 Demo API (prototyp)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_type(head: bytes) -> str:
    if head[:4] == b"\x28\xb5\x2f\xfd":
        return "skompresowane (zstd) — pewnie .dem.zst z FACEIT"
    if head[:7] == b"PBDEMS2":
        return "demo CS2"
    if head[:7] == b"HL2DEMO":
        return "demo CS:GO"
    return "nieznany format"


def load_meta(demo_id: str) -> dict:
    meta_file = STORAGE / f"{demo_id}.json"
    if not meta_file.exists():
        raise HTTPException(404, "Nie ma takiego dema")
    return json.loads(meta_file.read_text(encoding="utf-8"))


def save_meta(demo_id: str, meta: dict) -> None:
    (STORAGE / f"{demo_id}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Parsowanie (awpy) — wywoływane w tle przez BackgroundTasks
# ---------------------------------------------------------------------------

def compute_findings(players: list, rounds: list, kills: list, grenades: list,
                     smokes: list, infernos: list, tick_rate,
                     map_name: str = "?", library: dict = None) -> dict:
    library = library or {}
    """
    Silnik wniosków — dla każdego gracza liczy kilka reguł analizy.
    Działa na już sparsowanych danych (kills, grenades, rounds).
    """
    rate = tick_rate or 64           # CS2 zwykle 64 tick
    trade_window = int(rate * 5)     # 5 sekund na pomszczenie śmierci

    # Pierwsze starcie (opening duel) każdej rundy = kill o najmniejszym ticku
    first_kill_by_round = {}
    for k in kills:
        r = k["round"]
        if r not in first_kill_by_round or k["tick"] < first_kill_by_round[r]["tick"]:
            first_kill_by_round[r] = k

    # Rundy, w których gracz rzucił choć jeden granat
    nades_by_player_round = {}
    for g in grenades:
        nades_by_player_round.setdefault(g["thrower"], set()).add(g["round"])

    findings = {}
    for player in players:
        deaths = [k for k in kills if k["victim"] == player]
        rounds_died = sorted(set(k["round"] for k in deaths))

        # REGUŁA 1 — zginąłeś nie rzucając żadnego granatu
        threw_in = nades_by_player_round.get(player, set())
        deaths_no_utility = [r for r in rounds_died if r not in threw_in]

        # REGUŁA 2 — śmierć bez trade'a (kolega nie pomścił w oknie czasowym)
        untraded = []
        traded = 0
        for d in deaths:
            killer = d["attacker"]
            my_side = d["victim_side"]
            t = d["tick"]
            was_traded = any(
                k["round"] == d["round"]
                and k["victim"] == killer            # ktoś zabił mojego zabójcę
                and k["attacker_side"] == my_side     # i był z mojej drużyny
                and k["attacker"] != player           # (nie ja, bo nie żyję)
                and t < k["tick"] <= t + trade_window
                for k in kills
            )
            if was_traded:
                traded += 1
            else:
                untraded.append({"round": d["round"], "killer": killer, "weapon": d["weapon"]})

        total_deaths = len(deaths)
        trade_rate = round(100 * traded / total_deaths) if total_deaths else 0

        # REGUŁA 3 — pierwsze starcia rundy (entry duels)
        entry_kills = sum(1 for fk in first_kill_by_round.values() if fk["attacker"] == player)
        entry_deaths = sum(1 for fk in first_kill_by_round.values() if fk["victim"] == player)

        # REGUŁA 4 — miejsca (callouty): gdzie giniesz vs gdzie zabijasz
        place_stats = {}
        for k in deaths:
            p = k.get("victim_place") or "?"
            place_stats.setdefault(p, {"deaths": 0, "kills": 0})["deaths"] += 1
        for k in kills:
            if k["attacker"] == player:
                p = k.get("attacker_place") or "?"
                place_stats.setdefault(p, {"deaths": 0, "kills": 0})["kills"] += 1
        # Najgorsze spoty = najwięcej śmierci, sortowane po deaths
        worst_spots = sorted(
            ({"place": p, **v, "net": v["kills"] - v["deaths"]} for p, v in place_stats.items()),
            key=lambda x: (x["deaths"], -x["net"]), reverse=True
        )[:5]

        # KONTEKST per śmierć — łączymy sygnały w tagi (over-peek itp.)
        untraded_set = {(u["round"]) for u in untraded}
        death_contexts = []
        for d in deaths:
            tags = []
            place = d.get("victim_place") or "?"
            was_entry = any(
                fk["victim"] == player and fk["round"] == d["round"]
                for fk in [first_kill_by_round.get(d["round"])] if fk
            )
            is_untraded = any(
                u["round"] == d["round"] and u["killer"] == d["attacker"] for u in untraded
            )
            threw_nade = d["round"] in threw_in
            if was_entry:
                tags.append("pierwsza śmierć rundy")
            if is_untraded:
                tags.append("bez trade'a")
            if not threw_nade:
                tags.append("bez utility")
            if d.get("thrusmoke"):
                tags.append("przez dym")
            death_contexts.append({
                "round": d["round"], "place": place,
                "killer": d["attacker"], "weapon": d["weapon"], "tags": tags,
            })

        # REGUŁA 5 — celność smoke'ów (dystans rzutu)
        # Bardzo krótki rzut = smoke prawdopodobnie odbił się o przeszkodę.
        # Próg orientacyjny, do kalibracji na większej liczbie rzutów.
        SHORT_THROW = 250
        my_smokes = [s for s in smokes if s["thrower"] == player and s["distance"] is not None]
        suspicious_smokes = [s for s in my_smokes if s["distance"] < SHORT_THROW]
        avg_throw = round(sum(s["distance"] for s in my_smokes) / len(my_smokes)) if my_smokes else 0

        # REGUŁA 6 — celność wg biblioteki wzorcowej (z klasteryzacji dem pro)
        # Dla każdego smoke'a szukamy najbliższego kanonicznego spotu.
        #  - w promieniu spotu          -> trafiony lineup
        #  - blisko, ale poza promieniem -> "celowałeś tu, ale wyszło niecelnie"
        #  - daleko od wszystkich        -> improwizacja (nie oceniamy)
        MISS_MARGIN = 250     # ile poza strefą akceptacji liczymy jeszcze jako "near miss"
        SMOKE_TOLERANCE = 130  # realny zasięg dymu — w tym promieniu od wzorca smoke spełnia rolę
        smokes_ontarget = 0
        smokes_offtarget = []
        for sm in [s for s in smokes if s["thrower"] == player and s["land_x"] is not None]:
            spots = library.get(f"{map_name}|{sm['side']}|smoke", [])
            if not spots:
                continue
            best = min(spots, key=lambda sp: (sp["x"] - sm["land_x"]) ** 2 + (sp["y"] - sm["land_y"]) ** 2)
            dist = ((best["x"] - sm["land_x"]) ** 2 + (best["y"] - sm["land_y"]) ** 2) ** 0.5
            accept = best["radius"] + SMOKE_TOLERANCE   # strefa, w której smoke nadal spełnia rolę
            if dist <= accept:
                smokes_ontarget += 1
            elif dist <= accept + MISS_MARGIN:
                smokes_offtarget.append({
                    "round":       sm["round"],
                    "from":        sm["from"],
                    "target_from": best["common_from"],
                    "off_by":      round(dist - accept),
                    "from_x":      sm["from_x"],
                    "from_y":      sm["from_y"],
                    "from_z":      sm["from_z"],
                    "land_x":      sm["land_x"],
                    "land_y":      sm["land_y"],
                    "land_z":      sm["land_z"],
                    "target_x":    best["x"],
                    "target_y":    best["y"],
                    "traj":        sm.get("traj", []),
                })
            # else: za daleko od jakiegokolwiek spotu = improwizacja, pomijamy

        findings[player] = {
            "deaths_total":     total_deaths,
            "deaths_no_utility": deaths_no_utility,
            "untraded_deaths":  untraded,
            "traded_deaths":    traded,
            "trade_rate":       trade_rate,
            "entry_kills":      entry_kills,
            "entry_deaths":     entry_deaths,
            "worst_spots":      worst_spots,
            "death_contexts":   death_contexts,
            "smokes_total":     len(my_smokes),
            "smokes_avg_dist":  avg_throw,
            "smokes_suspicious": suspicious_smokes,
            "smokes_list":      my_smokes,
            "smokes_ontarget":  smokes_ontarget,
            "smokes_offtarget": smokes_offtarget,
            "has_library":      bool(library.get(f"{map_name}|ct|smoke") or library.get(f"{map_name}|t|smoke")),
        }
    return findings


def _do_parse(demo_id: str, dem_path: Path) -> None:
    """
    Parsuje plik .dem i zapisuje raport JSON obok metadanych.
    Uruchamiane jako zadanie w tle (BackgroundTasks) — nie blokuje odpowiedzi API.
    """
    meta = load_meta(demo_id)
    meta["parse_status"] = "running"
    meta["parse_started_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_meta(demo_id, meta)

    try:
        from awpy import Demo  # import tutaj — awpy jest opcjonalne

        dem = Demo(str(dem_path))
        dem.parse()

        # ----------------------------------------------------------------
        # Budujemy czytelny raport — bez surowych dataframe'ów Polars
        # ----------------------------------------------------------------
        header = dem.header or {}

        # Rundy — liczymy wynik zliczając wygrane rundy per strona
        rounds_df = dem.rounds
        rounds = []
        t_score = 0
        ct_score = 0
        if rounds_df is not None and len(rounds_df) > 0:
            for row in rounds_df.to_dicts():
                winner = row.get("winner", "")
                if winner == "t":
                    t_score += 1
                elif winner == "ct":
                    ct_score += 1
                rounds.append({
                    "round":    row.get("round_num"),
                    "winner":   winner.upper() if winner else "?",
                    "reason":   row.get("reason", ""),
                    "t_score":  t_score,
                    "ct_score": ct_score,
                    "bomb_site": row.get("bomb_site", ""),
                })

        # Zabójstwa — dodajemy strony (potrzebne do wykrywania trade'ów)
        kills_df = dem.kills
        kills = []
        if kills_df is not None and len(kills_df) > 0:
            for row in kills_df.to_dicts():
                kills.append({
                    "round":         row.get("round_num"),
                    "tick":          row.get("tick"),
                    "attacker":      row.get("attacker_name"),
                    "attacker_side": row.get("attacker_side"),
                    "attacker_place": row.get("attacker_place"),
                    "victim":        row.get("victim_name"),
                    "victim_side":   row.get("victim_side"),
                    "victim_place":  row.get("victim_place"),
                    "weapon":        row.get("weapon"),
                    "headshot":      row.get("headshot"),
                    "assistedby":    row.get("assister_name"),
                    "thrusmoke":     row.get("thrusmoke", False),
                    "blind":         row.get("attackerblind", False),
                })

        # Granaty — każdy rzut istnieje jako kilka obiektów (projektil + efekt),
        # więc liczymy TYLKO fazę "Projectile" = jeden obiekt na fizyczny rzut.
        # Decoy ma tylko jeden typ, więc dodajemy go osobno.
        THROW_TYPES = {
            "CSmokeGrenadeProjectile": "smoke",
            "CFlashbangProjectile":    "flash",
            "CHEGrenadeProjectile":    "HE",
            "CMolotovProjectile":      "molotov",
            "CDecoyGrenade":           "decoy",
        }
        grenades_df = dem.grenades
        grenades = []
        seen_entities = set()
        if grenades_df is not None and len(grenades_df) > 0:
            for row in grenades_df.to_dicts():
                raw_type = row.get("grenade_type")
                if raw_type not in THROW_TYPES:
                    continue  # pomijamy fazę efektu (dym, ogień) — liczymy tylko rzut
                key = (row.get("round_num"), row.get("entity_id"))
                if key in seen_entities:
                    continue  # ten sam projektil w kolejnym ticku — pomijamy
                seen_entities.add(key)
                grenades.append({
                    "round":   row.get("round_num"),
                    "thrower": row.get("thrower"),
                    "type":    THROW_TYPES[raw_type],   # czytelna nazwa
                    "tick":    row.get("tick"),
                })

        # --- Drużyny i połowy (wykrywane z danych, bez założeń o formacie) ---
        kills_sorted = sorted(kills, key=lambda x: (x["round"] or 0, x["tick"] or 0))

        # 1) strona startowa gracza = strona z najwcześniejszej rundy, w której się pojawia
        first_side = {}
        for k in kills_sorted:
            for who, side in [(k["attacker"], k["attacker_side"]), (k["victim"], k["victim_side"])]:
                if who and side and who not in first_side:
                    first_side[who] = side

        # 2) wykryj rundę zmiany stron = pierwsza runda, gdy ktoś gra inną stroną niż startową
        swap_round = None
        for k in kills_sorted:
            for who, side in [(k["attacker"], k["attacker_side"]), (k["victim"], k["victim_side"])]:
                if who in first_side and side and side != first_side[who]:
                    if swap_round is None or (k["round"] or 0) < swap_round:
                        swap_round = k["round"]
        if not swap_round:
            swap_round = 9999  # brak zmiany stron (krótki mecz) — wszystko jako 1. połowa

        def half_of(rnd):
            return 1 if (rnd or 0) < swap_round else 2

        # 3) statystyki per gracz, z rozbiciem na połowy
        def _new_stat():
            return {"kills": 0, "deaths": 0, "hs": 0, "assists": 0, "grenades": 0,
                    "h1": {"k": 0, "a": 0, "d": 0}, "h2": {"k": 0, "a": 0, "d": 0}}

        player_stats: dict[str, dict] = {}
        for k in kills:
            h = "h1" if half_of(k["round"]) == 1 else "h2"
            a, v, ast = k["attacker"], k["victim"], k["assistedby"]
            if a:
                s = player_stats.setdefault(a, _new_stat())
                s["kills"] += 1
                s[h]["k"] += 1
                if k["headshot"]:
                    s["hs"] += 1
            if v:
                s = player_stats.setdefault(v, _new_stat())
                s["deaths"] += 1
                s[h]["d"] += 1
            if ast:
                s = player_stats.setdefault(ast, _new_stat())
                s["assists"] += 1
                s[h]["a"] += 1

        for g in grenades:
            if g.get("thrower"):
                player_stats.setdefault(g["thrower"], _new_stat())["grenades"] += 1

        for name in player_stats:
            player_stats[name]["start_side"] = first_side.get(name, "?")

        # 4) wyniki per połowa i per drużyna (A = zaczynała CT, B = zaczynała T)
        h1_ct = h1_t = h2_ct = h2_t = 0
        for r in rounds:
            rn, w = r.get("round"), r.get("winner")
            if not rn or w not in ("T", "CT"):
                continue
            if half_of(rn) == 1:
                h1_ct += (w == "CT"); h1_t += (w == "T")
            else:
                h2_ct += (w == "CT"); h2_t += (w == "T")

        teams = {
            "ct_start_score": h1_ct + h2_t,   # A: CT w 1. poł, T w 2. poł
            "t_start_score":  h1_t + h2_ct,   # B: T w 1. poł, CT w 2. poł
            "swap_round":     swap_round,
            "h1": {"a": h1_ct, "b": h1_t},
            "h2": {"a": h2_t,  "b": h2_ct},
        }

        # Smoke'i i mołotowy. UWAGA: thrower_X/Y w tabeli smokes to pozycja gracza
        # w momencie LĄDOWANIA dymu (start_tick), nie rzutu. Prawdziwą pozycję rzutu
        # bierzemy z dem.ticks w pierwszym ticku granatu, a tor lotu z dem.grenades.

        # 1) Indeks ticków granatów: (entity_id, round) -> posortowane [(tick,x,y,z), ...]
        gren_df = getattr(dem, "grenades", None)
        gren_by_ent: dict = {}
        if gren_df is not None and len(gren_df) > 0:
            for r in gren_df.to_dicts():
                if r.get("X") is None or r.get("Y") is None:
                    continue
                key = (r.get("entity_id"), r.get("round_num"))
                gren_by_ent.setdefault(key, []).append((r["tick"], r["X"], r["Y"], r.get("Z") or 0))
            for k in gren_by_ent:
                gren_by_ent[k].sort()

        ticks_df = getattr(dem, "ticks", None)

        # Bezpieczny dostęp do tabel utility — brak mołotowów w demie sprawia,
        # że awpy rzuca przy dem.infernos; nie może to ubić parsowania smoke'ów.
        def safe_table(attr):
            try:
                return getattr(dem, attr, None)
            except Exception:
                return None

        # 2) Pozycja rzutu = gracz w dem.ticks przy pierwszym ticku granatu.
        #    Zbieramy potrzebne pary (steamid, tick) i filtrujemy ticki raz.
        def first_tick_of(row):
            tk = gren_by_ent.get((row.get("entity_id"), row.get("round_num")))
            return tk[0][0] if tk else None

        throw_lut: dict = {}
        if ticks_df is not None and len(ticks_df) > 0:
            pairs = set()
            for df in (safe_table("smokes"), safe_table("infernos")):
                if df is None:
                    continue
                for row in df.to_dicts():
                    ft = first_tick_of(row)
                    if ft is not None:
                        pairs.add((row.get("thrower_steamid"), ft))
            if pairs:
                steamids = list({p[0] for p in pairs})
                tlist = list({p[1] for p in pairs})
                sub = ticks_df.filter(
                    pl.col("steamid").is_in(steamids) & pl.col("tick").is_in(tlist)
                ).to_dicts()
                for r in sub:
                    throw_lut[(r["steamid"], r["tick"])] = (
                        r.get("X"), r.get("Y"), r.get("Z"), r.get("place")
                    )

        def parse_utility(df):
            items = []
            if df is None or len(df) == 0:
                return items
            for row in df.to_dicts():
                eid, rnd = row.get("entity_id"), row.get("round_num")
                start_tick = row.get("start_tick")
                steam = row.get("thrower_steamid")
                ticks = gren_by_ent.get((eid, rnd), [])

                # prawdziwa pozycja + callout rzutu
                fx = fy = fz = None
                fplace = row.get("thrower_place")
                if ticks:
                    ftick = ticks[0][0]
                    pos = throw_lut.get((steam, ftick))
                    if pos and pos[0] is not None:
                        fx, fy, fz, fplace = pos[0], pos[1], pos[2], pos[3] or fplace
                    else:
                        fx, fy, fz = ticks[0][1], ticks[0][2], ticks[0][3]

                # tor lotu (ticki do momentu lądowania), próbkowany do ~40 punktów
                flight = [(t, x, y) for (t, x, y, z) in ticks
                          if start_tick is None or t < start_tick]
                traj = []
                if len(flight) >= 2:
                    step = max(1, len(flight) // 40)
                    traj = [[round(x, 1), round(y, 1)] for (t, x, y) in flight[::step]]
                    lx_, ly_ = flight[-1][1], flight[-1][2]
                    if traj[-1] != [round(lx_, 1), round(ly_, 1)]:
                        traj.append([round(lx_, 1), round(ly_, 1)])

                lx, ly = row.get("X"), row.get("Y")
                dist = None
                if None not in (fx, fy, lx, ly):
                    dist = round(((fx - lx) ** 2 + (fy - ly) ** 2) ** 0.5)

                items.append({
                    "round":    rnd,
                    "thrower":  row.get("thrower_name"),
                    "side":     row.get("thrower_side"),
                    "from":     fplace,
                    "from_x":   round(fx, 1) if fx is not None else None,
                    "from_y":   round(fy, 1) if fy is not None else None,
                    "from_z":   round(fz, 1) if fz is not None else 0,
                    "land_x":   round(lx, 1) if lx is not None else None,
                    "land_y":   round(ly, 1) if ly is not None else None,
                    "land_z":   round(row.get("Z"), 1) if row.get("Z") is not None else 0,
                    "traj":     traj,
                    "distance": dist,
                })
            return items

        smokes = parse_utility(safe_table("smokes"))
        infernos = parse_utility(safe_table("infernos"))

        # Wczytujemy bibliotekę wzorcową (zbudowaną przez build_library.py).
        # Wczytujemy świeżo przy każdej analizie, więc podmiana biblioteki działa od razu.
        lib_path = BASE / "utility_library.json"
        library = {}
        if lib_path.exists():
            try:
                library = json.loads(lib_path.read_text(encoding="utf-8"))
            except Exception:
                library = {}

        map_name = header.get("map_name", "nieznana")

        # Silnik wniosków — analiza per gracz
        players = list(player_stats.keys())
        player_findings = compute_findings(
            players, rounds, kills, grenades, smokes, infernos,
            header.get("tick_rate"), map_name, library
        )

        report = {
            "demo_id":        demo_id,
            "map":            header.get("map_name", "nieznana"),
            "server":         header.get("server_name", ""),
            "tick_rate":      header.get("tick_rate"),
            "score":          {"t": t_score, "ct": ct_score},
            "teams":          teams,
            "rounds_total":   len(rounds),
            "rounds":         rounds,
            "kills_total":    len(kills),
            "kills":          kills,
            "grenades_total": len(grenades),
            "grenades":       grenades,
            "smokes":         smokes,
            "infernos":       infernos,
            "player_stats":   player_stats,
            "player_findings": player_findings,
            "parsed_at":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        report_path = STORAGE / f"{demo_id}_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        meta["parse_status"] = "done"
        meta["report_file"] = report_path.name
        meta["map"] = report["map"]
        meta["rounds_total"] = report["rounds_total"]
        meta["kills_total"] = report["kills_total"]

    except Exception as exc:
        meta["parse_status"] = "error"
        meta["parse_error"] = str(exc)

    meta["parse_finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_meta(demo_id, meta)


# ---------------------------------------------------------------------------
# Endpointy
# ---------------------------------------------------------------------------

@app.post("/api/demos")
async def upload_demo(file: UploadFile = File(...)):
    """Przyjmuje plik dema, zapisuje go i zwraca metadane."""
    demo_id = uuid.uuid4().hex
    suffix = "".join(Path(file.filename or "").suffixes) or ".dem"
    dest = STORAGE / f"{demo_id}{suffix}"

    size = 0
    head = b""
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            if not head:
                head = chunk[:8]
            size += len(chunk)
            out.write(chunk)

    meta = {
        "id":           demo_id,
        "filename":     file.filename,
        "stored_as":    dest.name,
        "size_bytes":   size,
        "type":         detect_type(head),
        "parse_status": "pending",
        "uploaded_at":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    save_meta(demo_id, meta)
    return meta


@app.post("/api/demos/{demo_id}/parse")
async def parse_demo(demo_id: str, background_tasks: BackgroundTasks):
    """
    Uruchamia parsowanie dema w tle.
    Odpowiedź wraca natychmiast — status sprawdzasz przez GET /api/demos/{demo_id}
    """
    meta = load_meta(demo_id)
    if meta.get("parse_status") == "running":
        return {"message": "Parsowanie już trwa", "status": "running"}

    dem_path = STORAGE / meta["stored_as"]
    if not dem_path.exists():
        raise HTTPException(404, "Plik dema nie istnieje w magazynie")

    background_tasks.add_task(_do_parse, demo_id, dem_path)
    return {"message": "Parsowanie uruchomione w tle", "demo_id": demo_id, "status": "running"}


@app.get("/api/demos/{demo_id}/report")
def get_report(demo_id: str):
    """Zwraca pełny raport z parsowania (gdy parse_status == done)."""
    meta = load_meta(demo_id)
    if meta.get("parse_status") != "done":
        status = meta.get("parse_status", "pending")
        raise HTTPException(
            425 if status == "running" else 404,
            f"Raport niedostępny — status: {status}"
        )
    report_file = STORAGE / meta["report_file"]
    return json.loads(report_file.read_text(encoding="utf-8"))


@app.get("/api/demos")
def list_demos():
    demos = [
        json.loads(f.read_text(encoding="utf-8"))
        for f in STORAGE.glob("*.json")
        if "_report" not in f.name
    ]
    demos.sort(key=lambda d: d["uploaded_at"], reverse=True)
    return demos


@app.get("/api/demos/{demo_id}")
def get_demo(demo_id: str):
    return load_meta(demo_id)


@app.get("/api/demos/{demo_id}/download")
def download_demo(demo_id: str):
    meta = load_meta(demo_id)
    path = STORAGE / meta["stored_as"]
    if not path.exists():
        raise HTTPException(404, "Plik zniknął z magazynu")
    return FileResponse(path, filename=meta["filename"] or meta["stored_as"])


@app.get("/api/demos/{demo_id}/debug")
def debug_demo(demo_id: str):
    """Tymczasowy endpoint diagnostyczny — pokazuje surową strukturę granatów."""
    meta = load_meta(demo_id)
    dem_path = STORAGE / meta["stored_as"]
    from awpy import Demo

    dem = Demo(str(dem_path))
    dem.parse()
    g = dem.grenades
    rows = g.to_dicts()
    types = sorted(set(r.get("grenade_type") for r in rows))
    round1 = [
        {
            "thrower":   r.get("thrower"),
            "type":      r.get("grenade_type"),
            "tick":      r.get("tick"),
            "entity_id": r.get("entity_id"),
        }
        for r in rows if r.get("round_num") == 1
    ][:15]
    return {
        "total_rows":    len(rows),
        "grenade_types": types,
        "round1_sample": round1,
    }


@app.get("/api/debug-nades")
def debug_nades_auto():
    """Diagnostyka BEZ ID — bierze najnowsze demo z magazynu i pokazuje smokes/infernos/grenades."""
    dems = sorted(STORAGE.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dems:
        raise HTTPException(404, "Brak plików .dem w magazynie — wgraj demo najpierw")
    dem_path = dems[0]
    from awpy import Demo

    dem = Demo(str(dem_path))
    dem.parse()
    out = {"plik": dem_path.name}
    for attr in ["smokes", "infernos", "grenades"]:
        df = getattr(dem, attr, None)
        try:
            if df is not None and len(df) > 0:
                out[attr] = {
                    "columns": list(df.columns),
                    "count": len(df),
                    "sample": df.to_dicts()[:3],
                }
            else:
                out[attr] = "brak / pusty"
        except Exception as e:
            out[attr] = f"blad: {e}"
    return out


@app.get("/api/demos/{demo_id}/smoke-map")
def smoke_map(demo_id: str, player: str, round: int):
    """Renderuje radar mapy dla niecelnego smoke'a danego gracza w danej rundzie."""
    meta = load_meta(demo_id)
    report_file = STORAGE / (meta.get("report_file") or f"{demo_id}_report.json")
    if not report_file.exists():
        raise HTTPException(404, "Brak raportu — najpierw przeanalizuj demo")
    report = json.loads(report_file.read_text(encoding="utf-8"))

    findings = (report.get("player_findings") or {}).get(player, {})
    offtarget = findings.get("smokes_offtarget", [])
    smoke = next((o for o in offtarget if o.get("round") == round), None)
    if not smoke:
        raise HTTPException(404, "Nie znaleziono takiego niecelnego smoke'a")

    out_png = STORAGE / f"{demo_id}_smoke_{player}_{round}.png"
    if not out_png.exists():
        import matplotlib
        matplotlib.use("Agg")
        from awpy.plot import plot

        z = smoke.get("land_z", 0) or 0
        traj = smoke.get("traj") or []

        # Kolejność punktów: rzut, tor lotu (próbka), lądowanie, cel wzorcowy
        points = [(smoke["from_x"], smoke["from_y"], smoke.get("from_z", 0) or 0)]
        settings = [{"color": "#5b9dd9", "marker": "o", "size": 11}]   # rzut

        for tx, ty in traj:
            points.append((tx, ty, z))
            settings.append({"color": "#f0c040", "marker": ".", "size": 4})  # tor

        points.append((smoke["land_x"], smoke["land_y"], z))           # lądowanie
        settings.append({"color": "#e05c5c", "marker": "o", "size": 11})
        points.append((smoke["target_x"], smoke["target_y"], z))       # cel wzorcowy
        settings.append({"color": "#3ecf8e", "marker": "o", "size": 11})

        try:
            fig, ax = plot(report["map"], points=points, point_settings=settings)
        except Exception:
            fig, ax = plot(report["map"], points=points)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

    return FileResponse(out_png, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "index.html").read_text(encoding="utf-8")
