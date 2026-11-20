"""Plan check of the interior: red seats in a top-down render of the model (in_plan.png, ortho 14.6 m
across 1920 px, car centre in the middle, front to the right, door side at the bottom) against the red
seats of the plan of the coloured type drawing. Reports rear edge, front edge and centre across of each
seat footprint in mm.  usage: compare_plan.py drawing.jpg in_plan.png [--json out.json]"""
import sys, json
import numpy as np
from PIL import Image
from scipy import ndimage

def blobs(mask, to_m, min_px):
    lab, n = ndimage.label(mask)
    out = []
    for sl in ndimage.find_objects(lab):
        if (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start) < min_px:
            continue
        xa, za = to_m(sl[1].start, sl[0].start)
        xb, zb = to_m(sl[1].stop, sl[0].stop)
        out.append(dict(x0=xa, x1=xb, z0=za, z1=zb))
    return out

def main():
    a = np.asarray(Image.open(sys.argv[1]).convert('RGB')).astype(int)
    ppm = 2348 / 14; x0 = 94 + 7 * ppm; cr = 461.5
    d = blobs((a[..., 0] > 180) & (a[..., 1] < 90) & (a[..., 2] < 80), lambda c, r: ((c - x0) / ppm, (r - cr) / ppm), 2000)
    m_img = np.asarray(Image.open(sys.argv[2]).convert('RGB')).astype(int)
    H, W = m_img.shape[:2]
    mp = W / 14.6
    R, G, B = m_img[..., 0], m_img[..., 1], m_img[..., 2]
    mm_ = ndimage.binary_closing((R > 45) & (R > 2.0 * G) & (R > 2.0 * B), structure=np.ones((3, 9)))   # backrest top + cushion as one
    m = blobs(mm_, lambda c, r: ((c - W / 2) / mp, (r - H / 2) / mp), 900)
    rows = []
    for k, s in enumerate(sorted(d, key=lambda b: ((b['z0'] + b['z1']) > 0, b['x0']))):
        zc = (s['z0'] + s['z1']) / 2
        side = 'door side' if zc > 0 else 'left side'
        near = min(m, key=lambda b: abs((b['x0'] + b['x1']) / 2 - (s['x0'] + s['x1']) / 2) + abs((b['z0'] + b['z1']) / 2 - zc), default=None)
        nm = f'{side} seat at x {(s["x0"] + s["x1"]) / 2:+.2f}'
        for lab, dv, mv in (('rear edge', s['x0'], near and near['x0']), ('front edge', s['x1'], near and near['x1']),
                            ('centre across', zc, near and (near['z0'] + near['z1']) / 2)):
            dev = None if mv is None else round((mv - dv) * 1000)
            rows.append([f'{nm} {lab}', round(dv * 1000), None if mv is None else round(mv * 1000), dev])
    ok = sum(1 for r in rows if r[3] is not None and abs(r[3]) <= 15)
    for r in rows:
        print(f'{r[0]:52s} {r[1]:7d} {r[2] if r[2] is not None else "-":>7} {r[3] if r[3] is not None else "-":>5}  {"ok" if r[3] is not None and abs(r[3]) <= 15 else "!!"}')
    print(f'{ok}/{len(rows)} within 15 mm; seats found: drawing {len(d)}, model {len(m)}')
    if '--json' in sys.argv:
        json.dump({'rows': rows}, open(sys.argv[sys.argv.index('--json') + 1], 'w'))

main()
