"""Numeric comparison of the model's rear end with the coloured rear elevation.

Same method as compare_front.py: both images reduced to colour classes, every feature measured the same
way inside a window given in metres, deviations in millimetres.  zv = across the car, + to the viewer's
right (seen from behind that is the door side, the tram's right); y = height above the top of rail.

Drawing: scale and origin from its own dimension lines (2,500 mm between columns 203 / 1028, i.e. 330 px
         per metre, centre column 616; rail top = top of the ground line, row 1899.5).
Model:   flat-colour orthographic render id_rear.png from build_t3.py --idpass (1,000 px per metre,
         centre column = zv 0, bottom row = y 0).

usage: python3 compare_rear.py <rear_drawing.jpg> <id_rear.png> [--json out.json]
"""
import json, sys
import numpy as np
from PIL import Image, ImageFilter
from compare_front import classify, View, D_PAL, M_PAL, TOL

M_PALR = dict(M_PAL, grey2=(90, 90, 92), roofgrey=(60, 60, 60), olive=(152, 148, 122), oliveD=(118, 116, 96),
              panto=(196, 80, 56))

def load_drawing(path):
    im = Image.open(path).convert('RGB')
    v = View(classify(np.asarray(im.filter(ImageFilter.GaussianBlur(2.0))), D_PAL), 330.0, 611.6, 1899.5)   # centre: midpoint of the body sides
    v.lum = np.asarray(im.convert('L')).astype(float)
    return v

def load_model(path):
    a = np.asarray(Image.open(path).convert('RGBA'))
    lab = classify(a[:, :, :3], M_PALR)
    lab = np.where(a[:, :, 3] < 128, 'bg', lab)
    v = View(lab, 1000.0, a.shape[1] / 2, a.shape[0], a[:, :, 3] / 255.0)
    v.lum = np.where(a[:, :, 3] > 128, a[:, :, :3].mean(2), 255.0)
    return v

def runs(v, labels, y, z0, z1, min_w=0.02, half=2):
    r = v.row(y)
    c0, c1 = v.col(z0), v.col(z1)
    m = v.mask(labels)[r - half:r + half + 1, c0:c1].mean(0) > 0.5
    out, st = [], None
    for k, on in enumerate(list(m) + [False]):
        if on and st is None: st = k
        elif not on and st is not None:
            a, b = v.zv(c0 + st), v.zv(c0 + k)
            if b - a >= min_w: out.append((a, b))
            st = None
    return out

def groups(idx, conv):
    if len(idx) == 0: return []
    g, cur = [], [idx[0]]
    for k in idx[1:]:
        if k - cur[-1] <= 2: cur.append(k)
        else: g.append(cur); cur = [k]
    g.append(cur)
    return [conv((q[0] + q[-1]) / 2) for q in g]

def strokes_v(v, z, y0, y1, half=2, thr=80):
    c = v.col(z); r0, r1 = v.row(y1), v.row(y0)
    prof = v.lum[r0:r1, c - half:c + half + 1].mean(1)
    return groups(np.where(prof < thr)[0], lambda k: v.y(r0 + k + 0.5))

def strokes_h(v, y, z0, z1, half=2, thr=80):
    r = v.row(y); c0, c1 = v.col(z0), v.col(z1)
    prof = v.lum[r - half:r + half + 1, c0:c1].mean(0)
    return groups(np.where(prof < thr)[0], lambda k: v.zv(c0 + k + 0.5))

def features(v, model):
    f = {}
    def put(k, val):
        f[k] = None if val is None else round(val * 1000)
    G = ('light',)
    DK = ('dark', 'roofgrey', 'grey2') if model else ('dark',)
    # ---- rear window: glass runs across at two heights; the bars are the gaps between them
    for y in (2.10, 1.85):
        rs = [r for r in runs(v, G, y, -1.06, 0.93, min_w=0.15)]
        if rs:
            put(f'rear window left edge (y {y:.2f})', rs[0][0]); put(f'rear window right edge (y {y:.2f})', rs[-1][1])
        if len(rs) == 3:
            put(f'left bar centre (y {y:.2f})', (rs[0][1] + rs[1][0]) / 2); put(f'right bar centre (y {y:.2f})', (rs[1][1] + rs[2][0]) / 2)
            put(f'left pane width (y {y:.2f})', rs[0][1] - rs[0][0]); put(f'right pane width (y {y:.2f})', rs[2][1] - rs[2][0])
        put(f'rear window panes (y {y:.2f})', len(rs) / 1000)
    for z in (-0.85, 0.0, 0.75):
        e = v.col_edges(G, z, 1.55, 2.62, half=2)
        if e: put(f'rear window top at zv {z:+.2f}', e[0]); put(f'rear window bottom at zv {z:+.2f}', e[1])
    # ---- left side window seen edge-on, and its one horizontal bar
    rs = runs(v, G, 2.0, -1.26, -1.03, min_w=0.04)
    if rs:
        put('left side window inner edge', rs[-1][1]); put('left side window outer edge', rs[0][0])
        zc = rs[-1][1] - 0.025                      # (clear of the vertical bar in the upper pane)
        c = v.col(zc); r0 = v.row(2.62); r1 = v.row(1.6)
        m = v.mask(G)[r0:r1, c - 2:c + 3].mean(1) > 0.5
        idx = np.where(m)[0]
        if len(idx):
            put('left side window top', v.y(r0 + idx[0])); put('left side window bottom', v.y(r0 + idx[-1] + 1))
            gaps = [k for k in range(idx[0], idx[-1]) if not m[k]]
            bars = groups(np.array(gaps), lambda k: v.y(r0 + k + 0.5)) if gaps else []
            put('left side window bars', len(bars) / 1000)
            if bars: put('left side window bar height', bars[0])
    # ---- door strip on the right, seen edge-on
    for y in (2.0, 1.2):                              # dark columns, bridging the light leaf edges (<= 25 mm)
        r = v.row(y); c0, c1 = v.col(0.87), v.col(1.27)
        cols = np.where(v.mask(DK)[r - 3:r + 4, c0:c1].mean(0) > 0.5)[0]
        if len(cols):
            grp, cur = [], [cols[0]]
            for k in cols[1:]:
                if k - cur[-1] <= 0.025 * v.ppm: cur.append(k)
                else: grp.append(cur); cur = [k]
            grp.append(cur)
            g = max(grp, key=lambda q: q[-1] - q[0])
            put(f'door strip inner edge (y {y:.1f})', v.zv(c0 + g[0])); put(f'door strip outer edge (y {y:.1f})', v.zv(c0 + g[-1] + 1))
    c = v.col(1.04); r0 = v.row(2.75); r1 = v.row(0.15)
    m = v.mask(DK)[r0:r1, c - 3:c + 4].mean(1) > 0.5
    k0 = v.row(1.2) - r0
    gap = int(0.015 * v.ppm)
    if m[k0]:                                          # the dark run through 1.2 m (bridging gaps <= 30 mm)
        a = k0
        while a > 0 and m[max(0, a - gap):a].any():
            a -= 1
            while a > 0 and not m[a]: a -= 1
        b = k0
        while b < len(m) - 1 and m[b + 1:b + 1 + gap].any():
            b += 1
            while b < len(m) - 1 and not m[b]: b += 1
        put('door strip top', v.y(r0 + a)); put('door strip bottom', v.y(r0 + b + 1))
        if not model:                                 # (the gutter strip just above it joins the run: take its outline)
            st = [t for t in strokes_v(v, 1.04, 2.40, 2.70, half=1, thr=80)]
            if len(st) >= 2 and st[0] > 2.57: put('door strip top', st[1])
    # ---- side gutters at window top
    for nm, z in (('left', -1.05), ('right', 0.90)):
        if model:
            e = v.col_edges(DK, z, 2.55, 2.70, half=2)
            put(f'{nm} gutter height', (e[0] + e[1]) / 2 if e else None)
        else:
            s = strokes_v(v, z, 2.55, 2.70, thr=90)
            put(f'{nm} gutter height', np.mean(s[:2]) if s else None)
    # ---- turn indicators on the side edges
    for nm, w in (('left', (-1.40, -1.15, 2.0, 2.5)), ('right', (1.15, 1.40, 2.0, 2.5))):
        b = v.blob(('amber',), *w)
        if b: put(f'{nm} indicator zv', b['z']); put(f'{nm} indicator y', b['y']); put(f'{nm} indicator height', b['h'])
    # ---- red band, beading and the bumper strip
    for z in (-0.75, 0.70):
        if model:
            e = v.col_edges(('red',), z, 0.6, 1.9, half=3)
            if e: put(f'red top at zv {z:+.1f}', e[0]); put(f'red bottom at zv {z:+.1f}', e[1])
            r0 = v.row(0.95)
            m = v.mask(('white', 'chrome'))[r0:v.row(0.65), v.col(z) - 2:v.col(z) + 3].mean(1) > 0.5
            idx = np.where(m)[0]
            if len(idx): put(f'strip top at zv {z:+.1f}', v.y(r0 + idx[0])); put(f'strip bottom at zv {z:+.1f}', v.y(r0 + idx[-1] + 1))
        else:
            s = strokes_v(v, z, 1.55, 1.72, thr=80)
            if s: put(f'red top at zv {z:+.1f}', s[-1])
            s = strokes_v(v, z, 0.70, 0.90, thr=70)
            if len(s) >= 2:
                put(f'red bottom at zv {z:+.1f}', s[0]); put(f'strip top at zv {z:+.1f}', s[0]); put(f'strip bottom at zv {z:+.1f}', s[-1])
    # where the thick strip section starts at the left corner (its top steps up)
    def strip_top(z):
        if model:
            r0 = v.row(0.95)
            m = v.mask(('white', 'chrome'))[r0:v.row(0.65), v.col(z)].astype(bool)
            idx = np.where(m)[0]
            return v.y(r0 + idx[0]) if len(idx) else None
        s = strokes_v(v, z, 0.65, 0.95, half=0, thr=80)
        return s[0] if s else None
    zs = np.arange(-1.24, -0.6, 0.004)
    t = [strip_top(z) for z in zs]
    ref = t[-1]
    hit = next((zs[i] for i in range(len(t) - 8) if all(x is not None and ref is not None and abs(x - ref) < 0.01 for x in t[i:i + 8])), None)
    put('thick strip left end', hit)
    # ---- plate frame (outline) under the window
    s = strokes_v(v, -0.08, 1.38, 1.575, half=1, thr=80 if not model else 60)
    if len(s) >= 2: put('plate frame top', s[0]); put('plate frame bottom', s[-1])
    s = strokes_h(v, 1.51, -0.30, 0.30, half=1, thr=80 if not model else 60)
    if len(s) >= 2: put('plate frame left', s[0]); put('plate frame right', s[-1])
    # ---- fleet number
    r0, r1, c0, c1 = v.win(-0.30, 0.30, 1.00, 1.20)
    m = np.isin(v.lab[r0:r1, c0:c1], ['amber', 'cream2', 'cream'] if model else ['amber', 'cream', 'olive', 'light'])
    rr, cc = np.nonzero(m)
    if len(rr) > 30:
        put('number height', (rr.max() - rr.min() + 1) / v.ppm); put('number width', (cc.max() - cc.min() + 1) / v.ppm)
        put('number centre y', v.y(r0 + (rr.min() + rr.max()) / 2)); put('number centre zv', v.zv(c0 + (cc.min() + cc.max()) / 2))
    # ---- tail-light units: outline extents, centre, and the colour at the middle and both ends
    for nm, (z0, z1) in (('left', (-0.72, -0.20)), ('right', (0.20, 0.72))):    # (to 0.72: the 6102 units sit further out)
        r0, r1, c0, c1 = v.win(z0, z1, 0.885, 1.007)
        m = np.isin(v.lab[r0:r1, c0:c1], ['amber', 'dark'] + (['white', 'chrome'] if model else [])) | (v.lum[r0:r1, c0:c1] < 70)   # (model rim is bright metal now)
        rr, cc = np.nonzero(m)
        if len(cc) > 20:
            a, b = v.zv(c0 + cc.min()), v.zv(c0 + cc.max() + 1)
            yt, yb = v.y(r0 + rr.min()), v.y(r0 + rr.max() + 1)
            put(f'{nm} tail unit width', b - a); put(f'{nm} tail unit height', yt - yb)
            put(f'{nm} tail unit centre zv', (a + b) / 2); put(f'{nm} tail unit centre y', (yt + yb) / 2)
            yc = (yt + yb) / 2
            order = []
            for zz in (a + 0.22 * (b - a), (a + b) / 2, a + 0.78 * (b - a)):
                lab = v.lab[v.row(yc) - 2:v.row(yc) + 3, v.col(zz) - 2:v.col(zz) + 3].ravel()
                vals, cnt = np.unique(lab, return_counts=True)
                order.append(vals[np.argmax(cnt)])
            f[f'{nm} tail unit colours'] = '-'.join(order)
    # ---- skirt lobes, the recess between them and the coupler
    rs = runs(v, ('cream', 'cream2', 'white', 'olive', 'oliveD') if model else ('cream',), 0.55, -1.3, 1.0, min_w=0.05)
    lft = [r for r in rs if -0.8 < r[1] < -0.2]; rgt = [r for r in rs if 0.2 < r[0] < 0.8]
    if lft and rgt:
        put('recess left edge', lft[-1][1]); put('recess right edge', rgt[0][0]); put('recess width', rgt[0][0] - lft[-1][1])
    for z in (-0.8, 0.75):
        c = v.col(z); r0 = v.row(0.75); r1 = v.row(0.15)
        m = v.mask(('cream', 'cream2', 'white', 'olive', 'oliveD') if model else ('cream',))[r0:r1, c - 2:c + 3].mean(1) > 0.5
        k = 0
        while k < len(m) and not m[k]: k += 1
        while k < len(m) and m[k]: k += 1
        put(f'lobe bottom at zv {z:+.2f}', v.y(r0 + k))
    r0, r1, c0, c1 = v.win(-0.12, 0.12, 0.45, 0.72)
    m = (v.lum[r0:r1, c0:c1] < 70) if not model else ~np.isin(v.lab[r0:r1, c0:c1], ['bg', 'cream', 'cream2'])
    rr = np.where(m.mean(1) > 0.3)[0]
    if len(rr): put('coupler head centre y', v.y(r0 + (rr[0] + rr[-1]) / 2))
    # ---- roof box, pantograph, raised panel on the dome
    if model:
        rs = runs(v, ('grey', 'grey2', 'chrome', 'white'), 3.30, -0.42, 0.42, min_w=0.1)
        if rs: put('roof box left', rs[0][0]); put('roof box right', rs[-1][1])
        e = v.col_edges(('grey', 'grey2', 'chrome', 'white'), 0.0, 3.1, 3.6, half=2)
        if e: put('roof box top', e[0])
    else:
        s = strokes_h(v, 3.30, -0.42, 0.42, half=1, thr=80)
        if len(s) >= 2: put('roof box left', s[0]); put('roof box right', s[-1])
        s = strokes_v(v, 0.1, 3.1, 3.6, half=1, thr=80)
        if s: put('roof box top', s[0])
    P = ('red', 'panto', 'amber', 'dark', 'grey', 'grey2') if model else ('dark', 'grey')
    rs = runs(v, P, 4.3, -1.0, 1.0, min_w=0.01, half=1) if model else [(s - 0.01, s + 0.01) for s in strokes_h(v, 4.3, -1.0, 1.0, thr=90)]
    side = [r for r in rs if abs((r[0] + r[1]) / 2) > 0.3]
    if len(side) >= 2:
        put('pantograph left upright', (side[0][0] + side[0][1]) / 2); put('pantograph right upright', (side[-1][0] + side[-1][1]) / 2)
    inner = [r for r in rs if abs((r[0] + r[1]) / 2) < 0.3]
    put('pantograph bracing crossings at 4.30 m', len([r for r in inner if abs((r[0] + r[1]) / 2) > 0.05]) / 1000)
    r0, r1, c0, c1 = v.win(-1.0, 1.0, 4.70, 5.18)
    m = v.mask(P)[r0:r1, c0:c1] if model else (v.lum[r0:r1, c0:c1] < 90)
    cols = np.where(m.sum(0) > 2)[0]
    if len(cols): put('pan bow span', (cols[-1] - cols[0] + 1) / v.ppm)
    s = strokes_h(v, 2.86, -0.9, 0.9, half=1, thr=110 if not model else 120)
    if len(s) >= 2 and not model: put('dome panel width', s[-1] - s[0])
    if model:
        rs_ = runs(v, ('white', 'chrome', 'cream'), 2.86, -0.9, 0.9, min_w=0.0, half=0)
        m = v.mask(('dark', 'roofgrey', 'grey2', 'white'))[v.row(2.86) - 1:v.row(2.86) + 2, v.col(-0.9):v.col(0.9)].mean(0) > 0.3
        idx = np.where(m)[0]
        put('dome panel width', (idx[-1] - idx[0] + 1) / v.ppm if len(idx) > 1 else None)
    # ---- body outline
    for y in (2.98, 2.90, 2.75, 2.0, 1.2, 0.5):
        r = v.row(y)
        body = (v.alpha[r - 1:r + 2].mean(0) > 0.5) if model else (np.isin(v.lab[r - 1:r + 2], ['bg']).mean(0) < 0.5)
        c = int(round(v.cx)); ends = []
        for st in (-1, 1):
            k, run, last = c, 0, c
            while 0 < k < len(body) - 1 and run < 8:
                k += st
                if body[k]: run, last = 0, k
                else: run += 1
            ends.append(last)
        put(f'body left at y {y:.2f}', v.zv(ends[0])); put(f'body right at y {y:.2f}', v.zv(ends[1] + 1))
    return f

def main():
    d, m = load_drawing(sys.argv[1]), load_model(sys.argv[2])
    fd, fm = features(d, False), features(m, True)
    rows = []
    for k in list(fd) + [k for k in fm if k not in fd]:
        a, b = fd.get(k), fm.get(k)
        dev = b - a if isinstance(a, int) and isinstance(b, int) else (None if a is None or b is None else (0 if a == b else 999))
        rows.append((k, a, b, dev))
    if '--json' in sys.argv:
        json.dump({'rows': rows}, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
    print(f"{'feature':46s} {'drawing':>16s} {'model':>16s} {'dev':>6s}")
    for k, a, b, dev in rows:
        flag = '' if dev is None else ('  ok' if abs(dev) <= TOL else '  !!')
        print(f"{k:46s} {str(a):>16s} {str(b):>16s} {str(dev):>6s}{flag}")

if __name__ == '__main__':
    main()
