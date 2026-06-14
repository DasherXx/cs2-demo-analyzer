# CS2 Demo — magazyn (krok 1)

Najmniejsze API + strona, która udowadnia jedną rzecz: **demo wgrane na PC ląduje
w magazynie i widać je z telefonu**. Bez parsowania i analizy — to kolejny krok.

## Co tu jest
- `app.py` — backend (FastAPI): wgrywanie, lista, pobieranie dema
- `index.html` — strona działająca i na PC (wrzucanie), i na telefonie (podgląd)
- `requirements.txt` — zależności
- `storage/` — tworzy się sam; tu lądują pliki dem + ich metadane

## Uruchomienie
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` jest kluczowe — dzięki temu serwer jest widoczny w sieci lokalnej,
a nie tylko na samym pececie.

## Jak przetestować „PC → telefon"
1. Na **PC** otwórz `http://localhost:8000` i wrzuć plik `.dem`.
2. Sprawdź adres IP peceta w sieci lokalnej:
   - Windows: `ipconfig` → pole „IPv4 Address" (np. `192.168.1.20`)
   - macOS/Linux: `ip addr` lub `ifconfig`
3. Na **telefonie** (to samo wifi) otwórz `http://192.168.1.20:8000` — zobaczysz to samo demo.

Podgląd API (Swagger): `http://localhost:8000/docs`

## Skąd wziąć demo do testu
Najprościej — pobierz dowolne demo z FACEIT albo z historii meczów CS2.
Strona rozpoznaje typ po nagłówku pliku: `PBDEMS2` = demo CS2, `HL2DEMO` = CS:GO,
a `.dem.zst` z FACEIT pokaże się jako „skompresowane (zstd)".

## Następny krok
Gdy demo niezawodnie się wgrywa: parsowanie (awpy) jednego dema do JSON-a ze
statystykami — czyli pierwszy realny „wniosek" z meczu.
