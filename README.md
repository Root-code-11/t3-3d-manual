# Tatra T3 6101 · interaktivní 3D model

> **Neoficiální fanouškovský projekt.** Model, kód i texty vznikly s pomocí AI (Claude) z typového výkresu, barevných pohledů a fotografií vozů 6102 a 6149. Kde výkres nic neříká, jde o odhad; karty dílů takové údaje označují.
>
> **Unofficial fan project.** The model, code and texts were made with AI (Claude) from the type drawing, coloured elevations and photos of cars 6102 and 6149. Where the drawing says nothing, values are estimates, and the part cards mark them.

![Pohled 3/4](docs/preview.png)

Interaktivní 3D model první pražské tramvaje Tatra T3, prototypu 6101 z roku 1960, postavený v rozměrech druhého prototypu 6102. Otáčení, přiblížení, rozložení na díly, otevírání dveří, pohled do interiéru a karta ke každému dílu s popisem a technickými údaji. Česky a anglicky.

## Interiér / Interior

24 jednotlivých sedadel podle půdorysu typového výkresu, tvary a barvy podle fotografií vozu 6149 (salon, plošiny) a 6102 (stanoviště řidiče).
24 single seats laid out from the plan of the type drawing; shapes and colours from photos of car 6149 (saloon, platforms) and 6102 (cab).

| | |
|---|---|
| ![Salon](docs/interior_saloon.png) | ![Přední plošina](docs/interior_front.png) |
| ![Zadní plošina](docs/interior_rear.png) | ![Stanoviště](docs/interior_cab.png) |

![Interiér v prohlížeči / interior in the viewer](docs/interior_web.png)

## Spuštění / Running it

- **Online / live demo:** [root-code-11.github.io/t3-3d-manual](https://root-code-11.github.io/t3-3d-manual/)
- **Lokálně / locally:** stránka načítá model ze souboru, proto ji otevřete přes lokální server, ne dvojklikem:

```
python3 -m http.server 8000
```

a pak / then open http://localhost:8000

## Co je kde / What is where

| Cesta | Obsah |
|---|---|
| `index.html` | stránka s 3D prohlížečem, kartami dílů a přepínačem CZ/EN |
| `t3_6101.txt` | model pro stránku (GLB s 2K texturami, zakódovaný base64) |
| `model/t3_6101.glb` | plný model se 4K texturami pro Blender, three.js a další nástroje |
| `model/t3_6101_web.glb` | lehčí model pro web (stejný obsah jako `t3_6101.txt`) |
| `blender/build_t3.py` | celý model postavený kódem v Blenderu |
| `blender/t3_textures.py` | malování PBR textur (nátěr, spáry, lehké opotřebení) |
| `blender/fonts/` | písma DejaVu pro čísla vozu a orientaci |
| `docs/` | rendery: 3/4 pohled, oba boky, čelo; kontroly proti výkresu |

## Úpravy modelu / Rebuilding the model

Testováno s Blenderem 5.2 jako Python modulem (`pip install bpy`):

```
python3 blender/build_t3.py -- out --glb --web       # model .glb (4K) a lehčí verze pro web
python3 blender/build_t3.py -- out --render          # render 1920 × 1080 na trati
python3 blender/build_t3.py -- out --retex --glb      # po změně tvaru čel znovu namalovat textury (jinak se použijí uložené)
base64 -w0 out/t3_6101_web.glb > t3_6101.txt         # model pro stránku
```

Kontrola čela proti barevnému výkresu / front-end check against the coloured drawing (the drawing itself is not in the repository):

```
python3 blender/build_t3.py -- out --idpass                       # ploché pohledy zepředu a z boku, 1 mm na pixel
python3 blender/compare_front.py <celo_vykres.jpg> out/id_front.png  # tabulka odchylek v mm
python3 blender/compare_side.py <bok_vykres.jpg> out/id_right.png    # pravý bok / right side
python3 blender/compare_left.py <levy_bok.jpg> out/id_left.png       # levý bok / left side
python3 blender/compare_rear.py <zad_vykres.jpg> out/id_rear.png      # záď / rear
python3 blender/compare_plan.py <pudorys.jpg> out/in_plan.png         # sedadla v půdorysu / seats in plan
```

Výsledky / results: [docs/front-check](docs/front-check/index.html), [docs/side-check](docs/side-check/index.html), [docs/left-check](docs/left-check/index.html), [docs/rear-check](docs/rear-check/index.html), [docs/corner-check](docs/corner-check/index.html), [docs/left-rear-check](docs/left-rear-check/index.html)

Souřadnice: x dopředu, y nahoru, z doprava, metry, počátek na temeni kolejnice pod středem vozu. Každý díl je prázdný objekt s id dílu a směrem rozložení; křídla dveří mají počátek v pantu, dvojkolí v nápravě.

Coordinates: x forward, y up, z to the right, metres, origin on the rail top under the vehicle centre. Every part is an empty carrying its id and explode direction; door leaves have their origin on the hinge, wheelsets on the axle.

## Prameny / Sources

- Wikipedia: Tatra T3 (Maruna 1974; Linert, Fojtík 2005)
- prazsketramvaje.cz, tram-bus.cz, abczech.cz
- Wikimedia Commons: Trams in Prague by registration number
- Typový výkres T3 a barevný výkres (oba boky, čelo, záď, půdorys) / T3 type drawing and coloured drawing (both sides, front, rear, plan)
- Fotografie vozů 6102 a 6149 / photos of cars 6102 and 6149

Inspirace / inspired by: [BMW R 12 · interaktivní 3D manuál](https://github.com/lukasersil/bmw-r12-3d-manual) (Lukáš Eršil)
