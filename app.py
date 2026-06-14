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

        # Rundy
        rounds_df = dem.rounds
        rounds = []
        if rounds_df is not None and len(rounds_df) > 0:
            for row in rounds_df.to_dicts():
                rounds.append({
                    "round":       row.get("round_num"),
                    "winner":      row.get("winner_side"),
                    "reason":      row.get("reason"),
                    "t_score":     row.get("t_score"),
                    "ct_score":    row.get("ct_score"),
                })

        # Zabójstwa
        kills_df = dem.kills
        kills = []
        if kills_df is not None and len(kills_df) > 0:
            for row in kills_df.to_dicts():
                kills.append({
                    "round":       row.get("round_num"),
                    "tick":        row.get("tick"),
                    "attacker":    row.get("attacker_name"),
                    "victim":      row.get("victim_name"),
                    "weapon":      row.get("weapon"),
                    "headshot":    row.get("headshot"),
                    "assistedby":  row.get("assister_name"),
                })

        # Granaty — smoki, flashe, mołotowy, HE
        grenades_df = dem.grenades
        grenades = []
        if grenades_df is not None and len(grenades_df) > 0:
            for row in grenades_df.to_dicts():
                grenades.append({
                    "round":      row.get("round_num"),
                    "thrower":    row.get("thrower_name"),
                    "type":       row.get("grenade_type"),
                    "x":          round(row.get("grenade_x", 0) or 0, 1),
                    "y":          round(row.get("grenade_y", 0) or 0, 1),
                })

        # Skrócone statystyki per gracz
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

        if grenades_df is not None and len(grenades_df) > 0:
            for row in grenades_df.to_dicts():
                name = row.get("thrower_name")
                if name and name in player_stats:
                    player_stats[name]["grenades"] += 1

        report = {
            "demo_id":      demo_id,
            "map":          header.get("map_name", "nieznana"),
            "server":       header.get("server_name", ""),
            "tick_rate":    header.get("tick_rate"),
            "rounds_total": len(rounds),
            "rounds":       rounds,
            "kills_total":  len(kills),
            "kills":        kills,
            "grenades_total": len(grenades),
            "grenades":     grenades,
            "player_stats": player_stats,
            "parsed_at":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
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


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "index.html").read_text(encoding="utf-8")
