"""Numeric comparison of the model's front view with the coloured front elevation.

Both images are reduced to the same colour classes (red, cream, glass/white, dark, grey, amber,
background). Every feature is then measured the same way in both, in a fixed window given in
metres, and reported in millimetres:  zv = across, positive to the viewer's right; y = height
above the top of rail.

Drawing:  scale and origin from its own dimension lines (2,500 mm body width, 1,435 mm gauge,
          3,028 mm rail-to-roof); the body centreline is the midpoint of the body sides.
Model:    flat-colour orthographic render from build_t3.py --idpass (1,000 px per metre).

usage: python3 compare_front.py <drawing.jpg> <id_front.png> [label] [--json out.json]
"""
import json, sys
import numpy as np
from PIL import Image, ImageFilter

TOL = 15  # mm

# ---------------------------------------------------------------- classification
D_PAL = {'red': (240, 58, 30), 'cream': (248, 245, 184), 'bg': (241, 232, 202), 'light': (214, 231, 240),
         'dark': (40, 38, 40), 'grey': (125, 122, 124), 'amber': (214, 150, 42), 'olive': (196, 186, 128),
         'pink': (236, 160, 150)}
M_PAL = {'red': (198, 34, 30), 'cream': (228, 216, 178), 'cream2': (204, 190, 156), 'light': (195, 225, 243), 'dark': (35, 33, 32),
         'grey': (140, 140, 140), 'amber': (240, 150, 20), 'white': (246, 243, 236), 'chrome': (250, 250, 250)}

def classify(rgb, pal):
    names = list(pal)
    ref = np.array([pal[k] for k in names], np.float32)
    out = np.empty(rgb.shape[:2], dtype=np.int16)
    for r in range(0, rgb.shape[0], 256):                # in row chunks to keep memory small
        blk = rgb[r:r + 256].astype(np.float32)
        d = ((blk[:, :, None, :] - ref[None, None]) ** 2).sum(-1)
        out[r:r + 256] = d.argmin(-1)
    return np.array(names)[out]

class View:
    def __init__(s, lab, ppm, cx, row_of_y0, alpha=None):
        s.lab, s.ppm, s.cx, s.r0, s.alpha = lab, ppm, cx, row_of_y0, alpha
    def col(s, zv): return int(round(s.cx + zv * s.ppm))
    def row(s, y): return int(round(s.r0 - y * s.ppm))
    def zv(s, c): return (c - s.cx) / s.ppm
    def y(s, r): return (s.r0 - r) / s.ppm
    def win(s, z0, z1, y0, y1):
        return s.row(y1), s.row(y0) + 1, s.col(z0), s.col(z1) + 1
    def mask(s, labels):
        m = np.isin(s.lab, list(labels))
        return m
    # extents of a class inside a window: rows/cols whose class fraction exceeds frac
    def extent(s, labels, z0, z1, y0, y1, frac=0.5):
        r0, r1, c0, c1 = s.win(z0, z1, y0, y1)
        m = s.mask(labels)[r0:r1, c0:c1]
        rows = np.where(m.mean(1) > frac)[0] if m.size else []
        cols = np.where(m.mean(0) > frac)[0] if m.size else []
        if len(rows) == 0 or len(cols) == 0:
            return None
        return dict(left=s.zv(c0 + cols[0]), right=s.zv(c0 + cols[-1] + 1),
                    top=s.y(r0 + rows[0]), bottom=s.y(r0 + rows[-1] + 1))
    # first/last run of a class along one column (y edges) or one row (z edges)
    def col_edges(s, labels, zv, y0, y1, half=3):
        c = s.col(zv)
        r0, r1 = s.row(y1), s.row(y0) + 1
        m = s.mask(labels)[r0:r1, c - half:c + half + 1].mean(1) > 0.5
        idx = np.where(m)[0]
        return None if len(idx) == 0 else (s.y(r0 + idx[0]), s.y(r0 + idx[-1] + 1))
    def row_edges(s, labels, y, z0, z1, half=3):
        r = s.row(y)
        c0, c1 = s.col(z0), s.col(z1) + 1
        m = s.mask(labels)[r - half:r + half + 1, c0:c1].mean(0) > 0.5
        idx = np.where(m)[0]
        return None if len(idx) == 0 else (s.zv(c0 + idx[0]), s.zv(c0 + idx[-1] + 1))
    def blob(s, labels, z0, z1, y0, y1):
        r0, r1, c0, c1 = s.win(z0, z1, y0, y1)
        m = s.mask(labels)[r0:r1, c0:c1]
        if m.sum() < 10:
            return None
        rr, cc = np.nonzero(m)
        return dict(z=s.zv(c0 + cc.mean()), y=s.y(r0 + rr.mean()), d=2 * np.sqrt(m.sum() / np.pi) / s.ppm,
                    w=(cc.max() - cc.min() + 1) / s.ppm, h=(rr.max() - rr.min() + 1) / s.ppm)
    def silhouette(s, y, half=2, gap=8):
        """Body extent along one row: walk out from the centre until `gap` background columns."""
        r = s.row(y)
        if s.alpha is not None:
            body = s.alpha[r - half:r + half + 1].mean(0) > 0.5
        else:
            body = np.isin(s.lab[r - half:r + half + 1], ['bg']).mean(0) < 0.5
        c = int(round(s.cx))
        ends = []
        for step in (-1, 1):
            k, run, last = c, 0, c
            while 0 < k < len(body) - 1 and run < gap:
                k += step
                if body[k]: run, last = 0, k
                else: run += 1
            ends.append(last)
        return (s.zv(ends[0]), s.zv(ends[1] + 1))

def load_drawing(path):
    im = Image.open(path).convert('RGB').filter(ImageFilter.GaussianBlur(2.0))
    lab = classify(np.asarray(im), D_PAL)
    raw = np.asarray(Image.open(path).convert('L')).astype(float)
    ppm = 819.0 / 2.5                    # 2,500 mm between the dimension lines at columns 303.5 / 1122.5
    rail = 1878                          # top of the ground line = top of rail (3,028 mm line ends there)
    # body centreline: midpoint of the body sides at 1.0 m and 2.2 m
    v = View(lab, ppm, 712.5, rail)
    mids = []
    for y in (1.0, 1.2, 1.5):
        l, r = v.silhouette(y)
        mids.append(v.cx + (l + r) / 2 * ppm)
    v.cx = float(np.mean(mids))
    v.lum = raw
    return v

def load_model(path):
    im = Image.open(path).convert('RGBA')
    a = np.asarray(im)
    lab = classify(a[:, :, :3], M_PAL)
    lab = np.where(a[:, :, 3] < 128, 'bg', lab)
    v = View(lab, 1000.0, a.shape[1] / 2, a.shape[0] - 0, a[:, :, 3] / 255.0)   # row 3400 = y 0
    v.lum = np.where(a[:, :, 3] > 128, a[:, :, :3].mean(2), 255.0)
    return v

LIGHT = ('light', 'white')
BODY_RED = ('red',)
CREAMS = ('cream', 'cream2', 'olive')

def features(v, model):
    f = {}
    def put(k, val):
        f[k] = None if val is None else round(val * 1000)
    glass = ('light',) if not model else ('light',)
    # windscreen opening (glass between the corner pillars), mid-height and mid-width scans
    e = v.row_edges(glass, 2.10, -1.0, 1.0)
    if e:
        put('windscreen left edge', e[0]); put('windscreen right edge', e[1]); put('windscreen width', e[1] - e[0])
    for zz in (-0.45, 0.45):
        e = v.col_edges(glass, zz, 1.55, 2.60)          # (stops under the route sign's glass)
        if e:
            put(f'windscreen top at zv {zz:+.2f}', e[0]); put(f'windscreen bottom at zv {zz:+.2f}', e[1])
    # corner shape: glass width 20, 50 and 100 mm below the top edge
    e = v.col_edges(glass, -0.45, 1.55, 2.60)
    if e:
        for dy in (0.02, 0.05, 0.10):
            w = v.row_edges(glass, e[0] - dy, -1.0, 1.0, half=1)
            if w: put(f'windscreen width {int(dy*1000)} mm below top', w[1] - w[0])
    e = v.col_edges(glass, -0.45, 1.55, 2.60)
    if e:
        for dy in (0.02, 0.05):
            w = v.row_edges(glass, e[1] + dy, -1.0, 1.0, half=1)
            if w: put(f'windscreen width {int(dy*1000)} mm above bottom', w[1] - w[0])
    # destination sign: the non-cream panel in the dome
    r0, r1, c0, c1 = v.win(-0.75, 0.75, 2.56, 2.86)
    m = np.isin(v.lab[r0:r1, c0:c1], ['grey', 'dark']) if not model else ~np.isin(v.lab[r0:r1, c0:c1], list(CREAMS) + ['bg'])
    rows = np.where(m.mean(1) > 0.4)[0]; cols = np.where(m.mean(0) > 0.4)[0]
    if len(rows) and len(cols):
        put('sign left', v.zv(c0 + cols[0])); put('sign right', v.zv(c0 + cols[-1] + 1))
        put('sign top', v.y(r0 + rows[0])); put('sign bottom', v.y(r0 + rows[-1] + 1))
        put('sign width', (cols[-1] + 1 - cols[0]) / v.ppm); put('sign height', (rows[-1] + 1 - rows[0]) / v.ppm)
    # red panel: top edge in the middle and near the corners, bottom edge
    for zz in (-0.30, 0.30, -0.70, 0.70):
        e = v.col_edges(BODY_RED, zz, 0.6, 1.75)
        if e:
            put(f'red top at zv {zz:+.2f}', e[0]); put(f'red bottom at zv {zz:+.2f}', e[1])
    # headlights
    for nm, z0, z1 in (('L', -0.75, -0.25), ('R', 0.25, 0.75)):
        r0, r1, c0, c1 = v.win(z0, z1, 0.93, 1.22)
        m = ~np.isin(v.lab[r0:r1, c0:c1], ['red', 'pink'])
        rr, cc = np.nonzero(m)
        if len(rr) > 50:
            put(f'headlight {nm} outer width', (cc.max() - cc.min() + 1) / v.ppm)
        b = v.blob(('light', 'white'), z0, z1, 0.93, 1.22)
        if b:
            put(f'headlight {nm} centre zv', b['z']); put(f'headlight {nm} centre y', b['y']); put(f'headlight {nm} lens diameter', b['d'])
    # position lamps (upright ovals)
    for nm, z0, z1 in (('L', -0.95, -0.78), ('R', 0.78, 0.95)):
        b = v.blob(LIGHT, z0, z1, 1.20, 1.55)
        if b:
            put(f'position lamp {nm} centre zv', b['z']); put(f'position lamp {nm} centre y', b['y']); put(f'position lamp {nm} height', b['h'])
    # amber indicators
    for nm, z0, z1 in (('L', -1.45, -1.05), ('R', 1.05, 1.45)):
        b = v.blob(('amber',), z0, z1, 1.75, 2.70)
        if b:
            put(f'indicator {nm} centre zv', b['z']); put(f'indicator {nm} centre y', b['y'])
    # fleet number (yellow/gold digits between the headlights)
    r0, r1, c0, c1 = v.win(-0.33, 0.33, 0.95, 1.20)
    m = np.isin(v.lab[r0:r1, c0:c1], ['amber', 'gold'])
    rr, cc = np.nonzero(m)
    if len(rr) > 30:
        put('number digit height', (rr.max() - rr.min() + 1) / v.ppm); put('number centre y', v.y(r0 + (rr.max() + rr.min()) / 2))
        put('number width', (cc.max() - cc.min() + 1) / v.ppm)
    # bright beading at the top of the cream band, bumper band (pink in the drawing, chrome/white in the model)
    bright = ('pink',) if not model else ('white', 'chrome')
    e = v.row_edges(bright, 0.80, -1.3, 1.3)
    if e: put('bumper left end', e[0]); put('bumper right end', e[1])
    # plate outline under the windscreen: centres of its dark outline strokes (raw luminance)
    def stroke_centres(prof, lo):
        dark = np.where(prof < 120)[0]
        if len(dark) == 0: return []
        groups, cur = [], [dark[0]]
        for k in dark[1:]:
            if k - cur[-1] <= 2: cur.append(k)
            else: groups.append(cur); cur = [k]
        groups.append(cur)
        return [lo + (g[0] + g[-1]) / 2 for g in groups]
    lum = v.lum
    r0, r1, c0, c1 = v.win(-0.08, 0.08, 1.38, 1.575)
    rows = stroke_centres(lum[r0:r1, c0:c1].mean(1), r0)
    if len(rows) >= 2:
        put('plate outline top', v.y(rows[0] + 0.5)); put('plate outline bottom', v.y(rows[-1] + 0.5))
    r0, r1, c0, c1 = v.win(-0.30, 0.30, 1.46, 1.51)
    cols = stroke_centres(lum[r0:r1, c0:c1].mean(0), c0)
    if len(cols) >= 2:
        put('plate outline left', v.zv(cols[0] + 0.5)); put('plate outline right', v.zv(cols[-1] + 0.5))
    for zz in (-0.40, 0.40):
        r0, r1, c0, c1 = v.win(zz - 0.01, zz + 0.01, 1.57, 1.648)
        rows = stroke_centres(lum[r0:r1, c0:c1].mean(1), r0) if not model else []
        if model:
            m = np.isin(v.lab[r0:r1, c0:c1], ['white', 'chrome']).mean(1) > 0.5
            idx = np.where(m)[0]
            rows = [r0 + idx[0], r0 + idx[-1] + 1] if len(idx) else []
        if len(rows) >= 2:
            put(f'beading top (stroke centre) at zv {zz:+.2f}', v.y(rows[0] + 0.5)); put(f'beading bottom (stroke centre) at zv {zz:+.2f}', v.y(rows[-1] + 0.5))
        r0, r1, c0, c1 = v.win(zz - 0.01, zz + 0.01, 0.70, 0.90)
        if model:
            m = np.isin(v.lab[r0:r1, c0:c1], ['white', 'chrome']).mean(1) > 0.5
            idx = np.where(m)[0]
            rows = [r0 + idx[0], r0 + idx[-1] + 1] if len(idx) else []
        else:
            rows = stroke_centres(lum[r0:r1, c0:c1].mean(1), r0)
        if len(rows) >= 2:
            put(f'bumper top (stroke centre) at zv {zz:+.2f}', v.y(rows[0] + 0.5)); put(f'bumper bottom (stroke centre) at zv {zz:+.2f}', v.y(rows[-1] + 0.5))
    # strips beside the face: door leaves on the viewer's left, cab window on the right
    for nm, z0, z1 in (('L', -1.15, -0.80), ('R', 0.80, 1.15)):
        e = v.row_edges(CREAMS, 2.10, z0, z1, half=4)
        if e:
            put(f'corner pillar {nm} inner', e[0] if nm == 'R' else e[1]); put(f'corner pillar {nm} outer', e[1] if nm == 'R' else e[0])
    e = v.row_edges(('light',), 2.10, 0.97, 1.30)
    if e: put('cab window glass inner', e[0]); put('cab window glass outer', e[1])
    # skirt lobes and the gap between them
    e = v.row_edges(CREAMS, 0.55, -0.70, 0.0)
    g = None
    lobes = v.mask(('cream', 'cream2'))
    r = v.row(0.45)
    for side, zs in (('L', np.arange(-0.2, -0.75, -0.002)), ('R', np.arange(0.2, 0.75, 0.002))):
        for zz in zs:
            c = v.col(zz)
            if lobes[r - 2:r + 3, c].mean() > 0.5:
                put(f'lobe {side} inner edge', zz); break
    for zz in (-0.70, 0.70):
        e = v.col_edges(('cream', 'cream2'), zz, 0.2, 0.74)
        if e: put(f'lobe bottom at zv {zz:+.2f}', e[1])
    # guard plate under the front (dark/grey band below the lobes)
    for zz in (-0.65, 0.65):
        e = v.col_edges(('dark', 'grey') + (() if model else ('olive',)), zz, 0.0, 0.36)
        if e: put(f'guard plate top at zv {zz:+.2f}', e[0]); put(f'guard plate bottom at zv {zz:+.2f}', e[1])
    e = v.row_edges(('dark', 'grey') + (() if model else ('olive',)), 0.16, -1.2, 1.2)
    if e: put('guard plate left', e[0]); put('guard plate right', e[1])
    # silhouette widths
    for y in (1.0, 2.0, 2.75, 2.90, 2.98):
        l, rr = v.silhouette(y)
        put(f'body left at y {y:.2f}', l); put(f'body right at y {y:.2f}', rr)
    return f

def main():
    d = load_drawing(sys.argv[1])
    m = load_model(sys.argv[2])
    fd, fm = features(d, False), features(m, True)
    rows = []
    for k in fd:
        a, b = fd.get(k), fm.get(k)
        dev = None if a is None or b is None else b - a
        rows.append((k, a, b, dev))
    for k in fm:
        if k not in fd:
            rows.append((k, None, fm[k], None))
    out = {'centreline_px': d.cx, 'rows': rows}
    if '--json' in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
    print(f"drawing centreline at column {d.cx:.1f}")
    print(f"{'feature':44s} {'drawing':>8s} {'model':>8s} {'dev':>6s}")
    for k, a, b, dev in rows:
        flag = '' if dev is None else ('  ok' if abs(dev) <= TOL else '  !!')
        print(f"{k:44s} {str(a):>8s} {str(b):>8s} {str(dev):>6s}{flag}")

if __name__ == '__main__':
    main()
