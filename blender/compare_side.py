"""Numeric comparison of the model's right side (door side) with the coloured side elevation.

Same method as compare_front.py: both images reduced to colour classes, every feature measured the same
way inside a window given in metres, deviations in millimetres.  x = along the car, + towards the front
(image right); y = height above the top of rail.

Drawing: scale and origin from its own dimension lines (bogie centres 6,400 mm apart at columns 744 /
         1814, body ends at 2448 / 110, rail top = top of the ground line at row 949).
Model:   flat-colour orthographic render id_right.png from build_t3.py --idpass (500 px per metre,
         centre column = x 0, row 1300 = y 2.5 m).

usage: python3 compare_side.py <side_drawing.jpg> <id_right.png> [--json out.json]
"""
import json, sys
import numpy as np
from PIL import Image, ImageFilter
from compare_front import classify, View, D_PAL, M_PAL, TOL

M_PAL2 = dict(M_PAL, grey2=(90, 90, 92), roofgrey=(60, 60, 60))

def load_drawing(path):
    im = Image.open(path).convert('RGB').filter(ImageFilter.GaussianBlur(2.0))
    v = View(classify(np.asarray(im), D_PAL), 1070 / 6.4, 1279.0, 949)
    v.lum = np.asarray(Image.open(path).convert('L')).astype(float)
    return v

def load_model(path):
    a = np.asarray(Image.open(path).convert('RGBA'))
    lab = classify(a[:, :, :3], M_PAL2)
    lab = np.where(a[:, :, 3] < 128, 'bg', lab)
    v = View(lab, 500.0, a.shape[1] / 2, 1300 + 2.5 * 500, a[:, :, 3] / 255.0)
    v.lum = np.where(a[:, :, 3] > 128, a[:, :, :3].mean(2), 255.0)
    return v

GLASS = ('light',)
CREAM = ('cream', 'cream2', 'olive')
DARK = ('dark', 'roofgrey')

def runs_on_row(v, labels, y, x0, x1, min_w=0.04, half=2):
    r = v.row(y)
    c0, c1 = v.col(x0), v.col(x1)
    m = v.mask(labels)[r - half:r + half + 1, c0:c1].mean(0) > 0.5
    out, start = [], None
    for k, on in enumerate(list(m) + [False]):
        if on and start is None:
            start = k
        elif not on and start is not None:
            a, b = v.zv(c0 + start), v.zv(c0 + k)
            if b - a >= min_w:
                out.append((a, b))
            start = None
    return out

def stroke_rows(v, x, y0, y1, half=3, thr=90):
    """Centres of dark outline strokes crossing a vertical line at x (raw luminance)."""
    c = v.col(x)
    r0, r1 = v.row(y1), v.row(y0)
    prof = v.lum[r0:r1, c - half:c + half + 1].mean(1)
    dark = np.where(prof < thr)[0]
    if len(dark) == 0:
        return []
    groups, cur = [], [dark[0]]
    for k in dark[1:]:
        if k - cur[-1] <= 2: cur.append(k)
        else: groups.append(cur); cur = [k]
    groups.append(cur)
    return [v.y(r0 + (g[0] + g[-1]) / 2 + 0.5) for g in groups]

def features(v, model):
    f = {}
    def put(k, val):
        f[k] = None if val is None else round(val * 1000)
    # ---- passenger windows and door panes along y = 2.0 (glass runs)
    wins = [r for r in runs_on_row(v, GLASS, 2.05, -6.2, 6.1) if r[1] - r[0] > 0.6]
    for i, (a, b) in enumerate(wins):
        k = f'window {i + 1}'
        put(f'{k} rear edge', a); put(f'{k} front edge', b)
        xc = (a + b) / 2
        e = v.col_edges(GLASS, xc - 0.25, 1.55, 2.70, half=3)
        if e:
            put(f'{k} top', e[0]); put(f'{k} bottom', e[1])
            for dy in (0.02,):
                w = v.row_edges(GLASS, e[0] - dy, a - 0.1, b + 0.1, half=1)
                if w: put(f'{k} width {int(dy*1000)} mm below top', w[1] - w[0])
    # ---- door panes: upper at 2.05, lower at 1.20; report each door's outermost panes and pane heights
    for nm, (x0, x1) in (('rear door', (-6.1, -4.5)), ('middle door', (-1.55, 0.05)), ('front door', (4.45, 6.1))):
        up = [r for r in runs_on_row(v, GLASS, 2.05, x0, x1, min_w=0.06) if r[1] - r[0] < 0.4]
        lo = [r for r in runs_on_row(v, GLASS, 1.20, x0, x1, min_w=0.06) if r[1] - r[0] < 0.4]
        if up:
            put(f'{nm} first pane rear edge', up[0][0]); put(f'{nm} last pane front edge', up[-1][1])
            put(f'{nm} upper panes', len(up) / 1000)
            xc = (up[0][0] + up[0][1]) / 2
            e = v.col_edges(GLASS, xc, 1.6, 2.65, half=1)
            if e: put(f'{nm} upper pane top', e[0]); put(f'{nm} upper pane bottom', e[1])
        if lo:
            xc = (lo[0][0] + lo[0][1]) / 2
            e = v.col_edges(GLASS, xc, 0.55, 1.65, half=1)
            if e: put(f'{nm} lower pane top', e[0]); put(f'{nm} lower pane bottom', e[1])
    # ---- red band and the bright strip under it
    for x in (-6.5, -3.8, -2.2, 1.6, 3.9, 6.3):
        e = v.col_edges(('red',), x, 0.6, 1.9, half=3)
        if e and not model:
            above = [r for r in stroke_rows(v, x, e[0] - 0.01, e[0] + 0.03)]
            below = [r for r in stroke_rows(v, x, e[1] - 0.03, e[1] + 0.01)]
            e = (above[-1] if above else e[0], below[0] if below else e[1])
        if e: put(f'red top at x {x:+.1f}', e[0]); put(f'red bottom at x {x:+.1f}', e[1])
        rows = stroke_rows(v, x, 0.70, 0.95) if not model else None
        if model:
            m = v.mask(('white', 'chrome'))[v.row(0.95):v.row(0.70), v.col(x) - 2:v.col(x) + 3].mean(1) > 0.5
            idx = np.where(m)[0]
            rows = [v.y(v.row(0.95) + idx[0]), v.y(v.row(0.95) + idx[-1] + 1)] if len(idx) else []
        if rows and len(rows) >= 2:
            put(f'strip top at x {x:+.1f}', rows[0]); put(f'strip bottom at x {x:+.1f}', rows[-1])
    # ---- gutter over the windows: dark stroke between 2.53 and 2.66 m
    for x in (-3.8, -2.2, 1.6, 3.9):
        if model:
            m = v.mask(DARK)[v.row(2.68):v.row(2.53), v.col(x) - 2:v.col(x) + 3].mean(1) > 0.5
            idx = np.where(m)[0]
            put(f'gutter at x {x:+.1f}', v.y(v.row(2.68) + (idx[0] + idx[-1]) / 2) if len(idx) else None)
        else:
            rows = stroke_rows(v, x, 2.55, 2.66)
            put(f'gutter at x {x:+.1f}', rows[0] if rows else None)
    # ---- body bottom edge and roof top
    for x in (-6.5, -1.0, 0.0, 1.0, 6.4):
        c = v.col(x)
        col = v.mask(('cream', 'cream2', 'red'))[:, c - 2:c + 3].mean(1) > 0.5
        r_lo = v.row(0.20); r_hi = v.row(0.75)
        idx = np.where(col[r_hi:r_lo])[0]
        put(f'body bottom at x {x:+.1f}', v.y(r_hi + idx[-1] + 1) if len(idx) else None)
    for x in (-3.5, -2.4):
        c = v.col(x)
        col = (v.lab[:, c - 2:c + 3] != 'bg').mean(1) > 0.5
        r_top = v.row(3.25); r_bot = v.row(2.7)
        idx = np.where(col[r_top:r_bot])[0]
        put(f'roof top at x {x:+.1f}', v.y(r_top + idx[0]) if len(idx) else None)
    # ---- body ends at several heights (outermost body pixel, couplers excluded by height)
    BODY = ('cream', 'cream2', 'red', 'light', 'dark', 'pink', 'amber', 'white', 'chrome', 'grey2', 'roofgrey', 'grey')
    for y in (2.9, 2.6, 2.3, 2.0, 1.7, 1.45, 1.2, 0.95, 0.80, 0.65, 0.40):
        r = v.row(y)
        m = np.isin(v.lab[r - 1:r + 2], BODY if model else ('cream', 'red', 'light', 'dark', 'pink', 'amber')).mean(0) > 0.5
        for nm, rng in (('rear', range(v.col(-7.35), v.col(-5.5))), ('front', range(v.col(7.35), v.col(5.5), -1))):
            if nm == 'rear' and y in (1.70, 0.65):    # drawing: window bar / coupler at these heights, no body edge
                continue
            hit = next((k for k in rng if m[k]), None)
            put(f'{nm} end at y {y:.2f}', v.zv(hit) if hit is not None else None)
    # ---- couplers (wheel and bogie positions are given by the dimensions and are not measured)
    for nm, x0, x1 in (('rear coupler', -7.64, -7.08), ('front coupler', 7.08, 7.64)):   # the drawing ends at -7.65 m
        r0, r1, c0, c1 = v.win(x0, x1, 0.40, 0.75)
        m = (v.lab[r0:r1, c0:c1] != 'bg') if model else ~np.isin(v.lab[r0:r1, c0:c1], ['bg', 'cream', 'pink'])
        cols = np.where(m.sum(0) > 3)[0]; rows = np.where(m.sum(1) > 3)[0]
        if len(cols):
            put(f'{nm} tip x', v.zv(c0 + (cols[0] if x0 < 0 else cols[-1] + 1)))
            put(f'{nm} centre y', v.y(r0 + (rows[0] + rows[-1]) / 2))
    # ---- roof equipment
    r0, r1, c0, c1 = v.win(0.7, 2.45, 3.06, 3.50)
    m = np.isin(v.lab[r0:r1, c0:c1], ['grey', 'grey2', 'chrome'])
    cols = np.where(m.mean(0) > 0.15)[0]; rows = np.where(m.mean(1) > 0.15)[0]
    if len(cols):
        put('roof cover rear end', v.zv(c0 + cols[0])); put('roof cover front end', v.zv(c0 + cols[-1] + 1)); put('roof cover top', v.y(r0 + rows[0]))
    rs = runs_on_row(v, DARK, 2.93, 1.2, 4.4, min_w=0.05)
    merged = []
    for a, b in rs:
        if merged and a - merged[-1][1] < 0.025: merged[-1] = (merged[-1][0], b)
        else: merged.append((a, b))
    for i, (a, b) in enumerate([r for r in merged if r[1] - r[0] > 0.2]):
        put(f'roof edge box {i + 1} rear', a); put(f'roof edge box {i + 1} front', b)
    r0, r1, c0, c1 = v.win(-0.4, 0.7, 2.58, 3.20)
    m = v.mask(DARK)[r0:r1, c0:c1]
    cols = np.where(m.mean(0) > 0.3)[0]; rows = np.where(m.mean(1) > 0.3)[0]
    if len(cols):
        put('box over middle door rear', v.zv(c0 + cols[0])); put('box over middle door front', v.zv(c0 + cols[-1] + 1))
        put('box over middle door top', v.y(r0 + rows[0])); put('box over middle door bottom', v.y(r0 + rows[-1] + 1))
    for nm, x0, x1 in (('rear', -6.4, -4.2), ('middle', -1.9, -0.10), ('front', 3.8, 5.45)):
        rs = runs_on_row(v, ('cream', 'cream2', 'light', 'white', 'grey', 'grey2', 'chrome', 'dark', 'roofgrey') if model else ('cream', 'olive', 'light', 'grey', 'dark'), 3.068, x0, x1, min_w=0.3, half=1)
        if rs:
            put(f'roof hatch {nm} rear', rs[0][0])
            if nm != 'front':                         # behind the front hatch the drawing's background swirl joins the run
                put(f'roof hatch {nm} front', rs[-1][1])
    # ---- end windows
    for nm, x0, x1 in (('rear corner window', -7.0, -5.95), ('cab side window', 5.95, 7.0)):
        rs = runs_on_row(v, GLASS, 2.05, x0, x1, min_w=0.08)
        if rs:
            put(f'{nm} rear edge', rs[0][0]); put(f'{nm} front edge', rs[-1][1])
            xc = (rs[0][0] + rs[-1][1]) / 2 if nm.startswith('rear') else rs[0][0] + 0.12
            e = v.col_edges(GLASS, xc, 1.55, 2.70, half=2)
            if e: put(f'{nm} top', e[0]); put(f'{nm} bottom', e[1])
    return f

def main():
    d, m = load_drawing(sys.argv[1]), load_model(sys.argv[2])
    fd, fm = features(d, False), features(m, True)
    rows = []
    for k in list(fd) + [k for k in fm if k not in fd]:
        a, b = fd.get(k), fm.get(k)
        rows.append((k, a, b, None if a is None or b is None else b - a))
    if '--json' in sys.argv:
        json.dump({'rows': rows}, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
    print(f"{'feature':44s} {'drawing':>8s} {'model':>8s} {'dev':>6s}")
    for k, a, b, dev in rows:
        flag = '' if dev is None else ('  ok' if abs(dev) <= TOL else '  !!')
        print(f"{k:44s} {str(a):>8s} {str(b):>8s} {str(dev):>6s}{flag}")

if __name__ == '__main__':
    main()
