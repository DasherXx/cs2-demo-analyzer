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

def compute_findings(players: list, rounds: list, kills: list, grenades: list, tick_rate) -> dict:
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

        # Statystyki per gracz — kills, deaths, HS, granaty
        player_stats: dict[str, dict] = {}
        if kills_df is not None and len(kills_df) > 0:
            for row in kills_df.to_dicts():
                for role, name_key in [("kill", "attacker_name"), ("death", "victim_name")]:
                    name = row.get(name_key)
                    if not name:
                        continue
                    s = player_stats.setdefault(name, {"kills": 0, "deaths": 0, "hs": 0, "grenades": 0})
                    if role == "kill":
                        s["kills"] += 1
                        if row.get("headshot"):
                            s["hs"] += 1
                    else:
                        s["deaths"] += 1

        # Granaty per gracz — liczymy z odfiltrowanej listy RZUTÓW (nie ticków)
        for g in grenades:
            name = g.get("thrower")
            if name:
                s = player_stats.setdefault(name, {"kills": 0, "deaths": 0, "hs": 0, "grenades": 0})
                s["grenades"] += 1

        # Silnik wniosków — analiza per gracz
        players = list(player_stats.keys())
        player_findings = compute_findings(
            players, rounds, kills, grenades, header.get("tick_rate")
        )

        report = {
            "demo_id":        demo_id,
            "map":            header.get("map_name", "nieznana"),
            "server":         header.get("server_name", ""),
            "tick_rate":      header.get("tick_rate"),
            "score":          {"t": t_score, "ct": ct_score},
            "rounds_total":   len(rounds),
            "rounds":         rounds,
            "kills_total":    len(kills),
            "kills":          kills,
            "grenades_total": len(grenades),
            "grenades":       grenades,
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


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "index.html").read_text(encoding="utf-8")
