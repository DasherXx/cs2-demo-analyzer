# CS2 Demo Analyzer 🎯

Aplikacja webowa do analizy meczów CS2 — wgraj demo, dostań coaching oparty na danych z dem pro.

## Co potrafi

### Scoreboard w stylu CS2
- Podział na drużyny (CT / T) z wynikiem per drużyna
- Tabele 1. i 2. połowy z K / A / D każdego gracza
- Kliknięcie w gracza → pełna analiza jego meczu

### Analiza per gracz
- **Śmierci bez utility** — ile razy zginąłeś nie rzucając żadnego granatu w rundzie
- **Trade rate** — jak często Twoje śmierci są pomśczone przez kolegów w ciągu 5 sekund
- **Pierwsze starcia rundy** — wygrany / przegrany entry duel
- **Najgorsze miejsca** — callout po calloucie, gdzie giniesz częściej niż zabijasz
- **Kontekst śmierci** — tag każdej śmierci (bez trade'a, bez utility, entry...)

### Detektor niecelnych smoke'ów 💨
- Porównuje każdy rzut gracza z biblioteką wzorcowych spotów (klasteryzacja z dem pro)
- **Krytyczne spoty** — smoke'i, które pro kładą zawsze w to samo miejsce i gdzie każda luka boli (sądzony surowiej)
- Kliknięcie „mapa" → radar mapy z torem lotu granatu (prawdziwa trajektoria z `dem.grenades`, z odbiciami), miejscem lądowania i wzorcowym celem
- Pozycja rzutu pobierana z `dem.ticks` (prawdziwa pozycja gracza w momencie rzutu, nie lądowania)

## Biblioteka wzorcowych spotów

Skrypt `build_library.py` parsuje dema pro i przez DBSCAN klasteryzuje miejsca lądowania smoke'ów / mołotowów w standardowe spoty. Działa przyrostowo (parsuje tylko nowe dema).

```bash
# zbuduj / rozbuduj bibliotekę
python build_library.py extract demos    # parsuj nowe dema z folderu demos/
python build_library.py cluster          # klasteryzuj -> utility_library.json

# pełne przeliczenie od nowa
python build_library.py extract demos --rebuild
python build_library.py cluster
```

Aktualna biblioteka: **57 dem pro** (Vitality, NaVi, G2, FaZe, Spirit, Furia...), **296 spotów** na 6 mapach (Dust2, Mirage, Inferno, Nuke, Anubis, Overpass).

## Instalacja i uruchomienie

```bash
pip install -r requirements.txt
pip install scikit-learn numpy matplotlib  # do biblioteki i map

# (jednorazowo) pobierz radary map
awpy get maps

uvicorn app:app --host 0.0.0.0 --port 8000
```

Aplikacja dostępna na `http://localhost:8000`.  
Z telefonu w tej samej sieci: `http://<IP-peceta>:8000`.

## Stack

- **Backend:** FastAPI + awpy (parsowanie dem CS2, Rust core)
- **Frontend:** single-page HTML/JS (dark theme, responsywny)
- **Analiza:** DBSCAN (scikit-learn), matplotlib + awpy.plot (radary)
- **Dane:** dema pro z HLTV

## Struktura projektu

```
app.py               — backend FastAPI (parsowanie, silnik wniosków, endpointy)
build_library.py     — offline: buduje bibliotekę spotów z dem pro
index.html           — frontend (scoreboard, analiza gracza, mapy)
utility_library.json — biblioteka wzorcowych spotów (296 spotów, 6 map)
requirements.txt     — zależności backendu
demos/               — (gitignore) dema pro do budowania biblioteki
storage/             — (gitignore) wgrane dema użytkowników + raporty JSON
```

## Roadmap

- [ ] Czwarty punkt na mapie: „stąd rzucają pro" (dane już w bibliotece: `throw_x/y`)
- [ ] Warstwa LLM — naturalny język zamiast suchych liczb (Anthropic API)
- [ ] Obsługa FACEIT link (auto-pobieranie dem)
- [ ] Deploy (Render / Hetzner VPS)
- [ ] Więcej map i dem w bibliotece
