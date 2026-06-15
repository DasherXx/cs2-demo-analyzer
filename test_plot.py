"""
test_plot.py — próba: czy awpy.plot narysuje radar mapy z punktami?

Bierze pierwszy smoke z dema Dust2 i nanosi na radar dwa punkty:
  - skąd rzucił gracz (thrower_X/Y)
  - gdzie smoke wylądował (X/Y)

Jeśli powstanie plik test_smoke.png z mapą i dwoma punktami — narzędzie działa
i możemy wpiąć je w aplikację.

Wymagania:
    pip install matplotlib   (awpy zwykle ciągnie je samo, ale na wszelki wypadek)

Użycie:
    python test_plot.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # tryb bez okienka — tylko zapis do pliku

from awpy import Demo
from awpy.plot import plot

# Znajdź jakieś demo Dust2 (w demos/ albo storage/)
candidates = list(Path("demos").glob("*dust2*.dem")) + list(Path("storage").glob("*.dem"))
if not candidates:
    raise SystemExit("Nie znalazłem żadnego dema. Wrzuć .dem do folderu demos/ lub storage/")

dem_path = candidates[0]
print(f"Parsuję: {dem_path.name}")
dem = Demo(str(dem_path))
dem.parse()

map_name = (dem.header or {}).get("map_name", "de_dust2")
smokes = dem.smokes.to_dicts()
if not smokes:
    raise SystemExit("To demo nie ma smoke'ów")

sm = smokes[0]
print(f"Mapa: {map_name}, smoke rzucony przez {sm.get('thrower_name')} "
      f"z {sm.get('thrower_place')}")

points = [
    (sm["thrower_X"], sm["thrower_Y"], sm["thrower_Z"]),  # skąd rzucił
    (sm["X"], sm["Y"], sm["Z"]),                          # gdzie wylądował
]

fig, ax = plot(map_name, points=points)
fig.savefig("test_smoke.png", dpi=120, bbox_inches="tight")
print("Zapisano test_smoke.png — otwórz i zobacz czy mapa + 2 punkty są widoczne")
