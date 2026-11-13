"""Numeric comparison of the model's left side (no doors) with the coloured left elevation.

Same method as compare_front.py / compare_side.py: both images reduced to colour classes, every feature
measured the same way inside a window given in metres, deviations in millimetres.  Both images are
mirrored first, so that here too x = along the car, + towards the front (image right); y = height above
the top of rail.

Drawing: scale and origin from its own dimension lines (body ends at columns 108.5 / 2447, bogie
         centres at 742.5 / 1813.5, i.e. 167.19 px per metre; rail top = top of the ground line, row 888).
Model:   flat-colour orthographic render id_left.png from build_t3.py --idpass (500 px per metre,
         centre column = x 0, row 1300 = y 2.5 m).

usage: python3 compare_left.py <left_drawing.jpg> <id_left.png> [--json out.json]
"""
import json, sys
import numpy as np
from PIL import Image, ImageFilter
from compare_front import classify, View, D_PAL, M_PAL, TOL
from compare_side import runs_on_row

M_PAL3 = dict(M_PAL, grey2=(90, 90, 92), roofgrey=(60, 60, 60), olive=(152, 148, 122), oliveD=(118, 116, 96))

def load_drawing(path):
    im = Image.open(path).convert('RGB')
    # the scan is turned by ~0.08 degrees (ground line 891 -> 894 px, wire 50 -> 54 px across the sheet):
    # level it about the rail point under the body centre
    im = im.rotate(0.08, resample=Image.BICUBIC, center=(1278, 888), fillcolor=(241, 232, 202))
    lab = classify(np.asarray(im.filter(ImageFilter.GaussianBlur(2.0))), D_PAL)[:, ::-1]
    W = im.size[0]
    v = View(lab, 167.19, W - 1278.25, 888)
    v.lum = np.asarray(im.convert('L')).astype(float)[:, ::-1]
    return v

def load_model(path):
    a = np.asarray(Image.open(path).convert('RGBA'))[:, ::-1]
    lab = classify(a[:, :, :3], M_PAL3)
    lab = np.where(a[:, :, 3] < 128, 'bg', lab)
    v = View(lab, 500.0, a.shape[1] / 2, 1300 + 2.5 * 500, a[:, :, 3] / 255.0)
    v.lum = np.where(a[:, :, 3] > 128, a[:, :, :3].mean(2), 255.0)
    return v

GLASS = ('light',)

def strokes_v(v, x, y0, y1, half=2, thr=90):
    """Centres of dark outline strokes crossing a vertical line at x (raw luminance)."""
    c = v.col(x)
    r0, r1 = v.row(y1), v.row(y0)
    prof = v.lum[r0:r1, c - half:c + half + 1].mean(1)
    return _groups(np.where(prof < thr)[0], lambda k: v.y(r0 + k + 0.5))

def strokes_h(v, y, x0, x1, half=2, thr=90):
    r = v.row(y)
    c0, c1 = v.col(x0), v.col(x1)
    prof = v.lum[r - half:r + half + 1, c0:c1].mean(0)
    return _groups(np.where(prof < thr)[0], lambda k: v.zv(c0 + k + 0.5))

def _groups(idx, conv):
    if len(idx) == 0:
        return []
    groups, cur = [], [idx[0]]
    for k in idx[1:]:
        if k - cur[-1] <= 2: cur.append(k)
        else: groups.append(cur); cur = [k]
    groups.append(cur)
    return [conv((g[0] + g[-1]) / 2) for g in groups]

def first_run(v, labels, x, y_from, y_to, half=2):
    """Edges (y) of the first run of a class met when walking from y_from towards y_to along x."""
    c = v.col(x)
    step = 1 if y_from > y_to else -1
    r, r_end = v.row(y_from), v.row(y_to)
    m = v.mask(labels)[:, c - half:c + half + 1].mean(1) > 0.5
    k = r
    while k != r_end and not m[k]: k += step
    if k == r_end: return None
    a = k
    while k != r_end and m[k]: k += step
    return tuple(sorted((v.y(a + (0 if step == 1 else 1)), v.y(k + (0 if step == 1 else 1)))))[::-1]

def features(v, model):
    f = {}
    def put(k, val):
        f[k] = None if val is None else round(val * 1000)
    D = (lambda *a: a[1]) if model else (lambda *a: a[0])     # pick drawing / model class set
    # ---- passenger windows: glass runs along y = 2.05 and y = 1.85
    wins = [r for r in runs_on_row(v, GLASS, 1.95, -6.0, 5.95) if r[1] - r[0] > 0.9]
    for i, (a, b) in enumerate(reversed(wins)):
        k = f'window {i + 1}'                                  # numbered from the front
        put(f'{k} front edge', b); put(f'{k} rear edge', a)
        e = v.col_edges(GLASS, (a + b) / 2 + 0.25, 1.55, 2.70, half=3)
        if e: put(f'{k} top', e[0]); put(f'{k} bottom', e[1])
        # the sash bar under the sliding panes and the vertical bar between them
        xs = a + 0.20
        segs = []
        c = v.col(xs); r0, r1 = v.row(2.60), v.row(1.60)
        m = v.mask(GLASS)[r0:r1, c - 2:c + 3].mean(1) > 0.5
        idx = np.where(~m[1:] & m[:-1])[0]
        gaps = [v.y(r0 + k_ + 1) for k_ in idx if 1.9 < v.y(r0 + k_ + 1) < 2.45]
        if gaps: put(f'{k} sash bar', gaps[0])
        rs = [r for r in runs_on_row(v, GLASS, 2.40, a - 0.05, b + 0.05, min_w=0.1, half=1)]
        if len(rs) >= 2:                              # the bar between the two sliding panes, from the rear glass edge
            put(f'{k} vent bar centre from rear edge', (rs[0][1] + rs[1][0]) / 2 - a)
    # ---- end windows
    rs = runs_on_row(v, GLASS, 2.05, 5.9, 7.0, min_w=0.1)
    if rs:
        put('cab window rear edge at y 2.05', rs[0][0])
    for y in (1.85, 2.30):
        rs = runs_on_row(v, GLASS, y, 5.9, 7.0, min_w=0.05)
        if rs: put(f'cab window rear edge at y {y:.2f}', rs[0][0])
    for x in (6.10, 6.30, 6.50):
        e = v.col_edges(GLASS, x, 1.55, 2.56, half=2)
        if e: put(f'cab window top at x {x:.2f}', e[0]); put(f'cab window bottom at x {x:.2f}', e[1])
    rs = runs_on_row(v, GLASS, 2.05, -7.0, -5.9, min_w=0.05)
    if rs:
        put('rear corner window front edge', rs[-1][1]); put('rear corner window rear edge', rs[-1][0] if len(rs) == 1 else rs[-1][0])
    for x in (-6.10, -6.45):
        e = v.col_edges(GLASS, x, 1.55, 2.70, half=2)
        if e: put(f'rear corner window top at x {x:.2f}', e[0]); put(f'rear corner window bottom at x {x:.2f}', e[1])
    # ---- beading over the red band and the aluminium strip under it (outline-stroke centres in the drawing,
    #      the bright metal's extent in the model)
    for x in (-6.6, -4.4, -2.9, 1.5, 3.0, 5.2, 6.5):
        for nm, y0, y1 in (('beading', 1.55, 1.67), ('strip', 0.68, 0.92)):
            if model:
                r0 = v.row(y1)
                m = v.mask(('white', 'chrome'))[r0:v.row(y0), v.col(x) - 2:v.col(x) + 3].mean(1) > 0.5
                idx = np.where(m)[0]
                rows = [v.y(r0 + idx[0]), v.y(r0 + idx[-1] + 1)] if len(idx) else []
            else:
                rows = strokes_v(v, x, y0, y1, thr=70)
            if len(rows) >= 2:
                put(f'{nm} top at x {x:+.1f}', rows[0]); put(f'{nm} bottom at x {x:+.1f}', rows[-1])
    # where the thick strip ends meet the thin strip: scan along y = 0.855 (only the thick part reaches it)
    def strip_top(x):
        if model:
            r0 = v.row(0.95)
            m = v.mask(('white', 'chrome'))[r0:v.row(0.70), v.col(x)].astype(bool)
            idx = np.where(m)[0]
            return v.y(r0 + idx[0]) if len(idx) else None
        st = strokes_v(v, x, 0.70, 0.95, half=0, thr=70)
        return st[0] if st else None
    for nm, xs_ in (('thick strip front section, rear end', np.arange(5.5, 6.6, 0.004)),
                    ('thick strip rear section, front end', np.arange(-5.5, -6.6, -0.004))):
        tops = [(x, strip_top(x)) for x in xs_]
        ok = [t is not None and t > 0.848 for x, t in tops]
        hit = next((tops[i][0] for i in range(len(ok) - 10) if all(ok[i:i + 10])), None)
        put(nm, hit)
    # ---- gutter over the windows
    for x in (-5.0, -2.2, 1.6, 4.4):
        if model:
            m = v.mask(('dark', 'roofgrey'))[v.row(2.66):v.row(2.53), v.col(x) - 2:v.col(x) + 3].mean(1) > 0.5
            idx = np.where(m)[0]
            put(f'gutter at x {x:+.1f}', v.y(v.row(2.66) + (idx[0] + idx[-1]) / 2 + 0.5) if len(idx) else None)
        else:
            rows = strokes_v(v, x, 2.55, 2.63)
            put(f'gutter at x {x:+.1f}', rows[0] if rows else None)
    s = strokes_h(v, 2.55, 5.5, 6.25, half=1) if not model else None
    if model:
        rs = runs_on_row(v, ('dark', 'roofgrey'), 2.555, 5.5, 6.3, min_w=0.008, half=1)
        s = [(a + b) / 2 for a, b in rs]
    if s: put('gutter front hook x', s[-1])
    if model:
        rs = runs_on_row(v, ('dark', 'roofgrey'), 2.555, -6.3, -5.5, min_w=0.008, half=1)
        s = [(a + b) / 2 for a, b in rs]
    else:
        s = strokes_h(v, 2.55, -6.25, -5.6, half=1)
    if s: put('gutter rear hook x', s[0])
    # ---- roof colour: lower edge of the grey/olive roof on the side band, and where it rises at the ends
    ROOFC = ('olive', 'oliveD', 'roofgrey', 'grey', 'grey2') if model else ('olive',)
    CR = ('cream', 'cream2') if model else ('cream',)
    for x in (-6.0, -5.0, -4.2, -2.0, 0.6, 2.4, 4.6, 5.6):
        c = v.col(x)
        m = v.mask(ROOFC)[:, c - 2:c + 3].mean(1) > 0.5
        r = v.row(2.95)
        while not m[r] and r < v.row(2.80): r += 1          # (start inside the roof colour)
        while m[r] and r < v.row(2.50): r += 1
        put(f'roof colour lower edge at x {x:+.1f}', v.y(r))
    # rise of the boundary: along y = 2.72 find the cream runs next to the middle
    rs = runs_on_row(v, CR, 2.72, -6.5, 6.5, min_w=0.05, half=1)
    mids = [r for r in rs if r[0] > 1.0]; rear = [r for r in rs if r[1] < -1.0]
    if mids: put('cream side band front part, rear end (y 2.72)', mids[0][0])
    if rear: put('cream side band rear part, front end (y 2.72)', rear[-1][1])
    rs = runs_on_row(v, ROOFC + (('dark',) if model else ()), 2.90, -7.0, 7.0, min_w=0.1, half=1)
    if rs: put('roof colour rear end (y 2.90)', rs[0][0]); put('roof colour front end (y 2.90)', rs[-1][1])
    # ---- roof equipment
    if model:
        rs = runs_on_row(v, ('dark', 'roofgrey'), 2.93, 1.3, 4.6, min_w=0.05, half=1)
        edges = [rs[0][0]] + [e for r in rs for e in r][1:-1] + [rs[-1][1]] if rs else []
        gaps = [(rs[i][1] + rs[i + 1][0]) / 2 for i in range(len(rs) - 1)]
    else:
        st = strokes_h(v, 2.93, 1.3, 4.6, half=1, thr=70)
        edges = st
        gaps = [np.mean([t for t in st if lo < t < hi]) for lo, hi in ((2.3, 2.7), (3.35, 3.8)) if any(lo < t < hi for t in st)]
    if edges:
        put('roof edge boxes rear end', edges[0]); put('roof edge boxes front end', edges[-1])
    for i, g in enumerate(sorted(gaps)[:2]):
        put(f'roof edge boxes joint {i + 1}', g)
    e = v.col_edges(('dark', 'grey') if not model else ('dark', 'roofgrey'), 3.85, 2.70, 3.15, half=2)
    if e: put('roof edge box top at x 3.85', e[0]); put('roof edge box bottom at x 3.85', e[1])
    r0, r1, c0, c1 = v.win(0.7, 2.45, 3.07, 3.50)
    m = np.isin(v.lab[r0:r1, c0:c1], ['grey', 'grey2', 'chrome'])
    cols = np.where(m.mean(0) > 0.15)[0]; rows = np.where(m.mean(1) > 0.15)[0]
    if len(cols):
        put('roof cover rear end', v.zv(c0 + cols[0])); put('roof cover front end', v.zv(c0 + cols[-1] + 1)); put('roof cover top', v.y(r0 + rows[0]))
    # platform plate under the pantograph: a thin band at ~3.06 m
    if model:
        rs = runs_on_row(v, ('grey', 'grey2', 'chrome', 'dark'), 3.065, 2.3, 4.1, min_w=0.3, half=0)
        if rs: put('pantograph platform rear end', rs[0][0]); put('pantograph platform front end', rs[-1][1])
    else:                                         # pale plate on an olive roof: its outline strokes
        st = strokes_h(v, 3.055, 2.3, 3.9, half=0, thr=80)
        a = [t for t in st if 2.43 < t < 2.55]; b = [t for t in st if 3.55 < t < 3.63]
        if a: put('pantograph platform rear end', a[0])
        if b: put('pantograph platform front end', b[0])
    # pantograph knees: outermost non-background pixels between 3.6 and 4.4 m
    r0, r1, c0, c1 = v.win(1.0, 5.4, 3.75, 4.30)
    m = (v.lab[r0:r1, c0:c1] == 'dark') if not model else (v.lab[r0:r1, c0:c1] != 'bg')
    cols = np.where(m.sum(0) > 0)[0]
    if len(cols):
        a, b = v.zv(c0 + cols[0]), v.zv(c0 + cols[-1] + 1)
        put('pantograph rear knee x', a); put('pantograph front knee x', b); put('pantograph centre x', (a + b) / 2)
        cr = v.col(a + 0.01) - c0
        rows_ = np.where(m[:, max(0, cr - 2):cr + 3].any(1))[0]
        if len(rows_): put('pantograph rear knee y', v.y(r0 + (rows_[0] + rows_[-1]) / 2))
        cf = v.col(b - 0.01) - c0
        rows_ = np.where(m[:, max(0, cf - 2):cf + 3].any(1))[0]
        if len(rows_): put('pantograph front knee y', v.y(r0 + (rows_[0] + rows_[-1]) / 2))
    # anything inside the diamond on its centre line, between the knees' height and the pan head
    xc = f.get('pantograph centre x')
    if xc is not None:
        r0, r1, c0, c1 = v.win(xc / 1000 - 0.15, xc / 1000 + 0.15, 3.55, 4.55)
        m = (v.lab[r0:r1, c0:c1] != 'bg') if model else (v.lum[r0:r1, c0:c1] < 90)
        f['pantograph: pixels inside the diamond on the centre line'] = int(m.sum() if model else 0)
    for nm, x0, x1 in (('front', 4.0, 5.45), ('middle', -1.6, 0.0), ('rear', -6.0, -4.4)):
        rs = runs_on_row(v, ('cream', 'cream2', 'white', 'grey', 'light') if model else ('cream', 'light', 'grey'), 3.075, x0, x1, min_w=0.4, half=0)
        if rs: put(f'roof hatch {nm} rear', rs[0][0]); put(f'roof hatch {nm} front', rs[-1][1])
    rs = runs_on_row(v, ('dark', 'roofgrey', 'grey') if model else ('cream', 'light', 'grey', 'dark'), 3.075, -0.25, 0.5, min_w=0.2, half=0)
    if rs: put('roof vent cap rear', rs[0][0]); put('roof vent cap front', rs[-1][1])
    # ---- grille in the red band
    G = ('dark', 'grey', 'grey2', 'white', 'chrome', 'cream2') if model else ('dark', 'grey', 'light', 'cream')
    r0, r1, c0, c1 = v.win(-2.0, 0.6, 0.86, 1.45)
    m = np.isin(v.lab[r0:r1, c0:c1], G) | ((v.lab[r0:r1, c0:c1] == 'red') & (v.lum[r0:r1, c0:c1] < 90 if not model else False))
    cols = np.where(m.mean(0) > 0.3)[0]
    if len(cols):
        put('grille rear edge', v.zv(c0 + cols[0])); put('grille front edge', v.zv(c0 + cols[-1] + 1))
        xm = (v.zv(c0 + cols[0]) + v.zv(c0 + cols[-1] + 1)) / 2
        c = v.col(xm) - c0
        rows = np.where(m[:, :].mean(1) > 0.3)[0]
        if len(rows): put('grille top', v.y(r0 + rows[0]))
        # does red show between the grille and the strip? lowest grille row at the middle of a bay
        rr = np.where(np.isin(v.lab[r0:v.row(0.80), c0 + c - 30:c0 + c + 30], G).mean(1) > 0.3)[0]
        if len(rr): put('grille bottom', v.y(r0 + rr[-1] + 1))
    # ---- number, coat of arms, lamps
    r0, r1, c0, c1 = v.win(2.0, 4.6, 1.05, 1.40)
    m = (v.lab[r0:r1, c0:c1] == ('amber' if not model else 'amber')) | (v.lab[r0:r1, c0:c1] == 'cream2') if model else \
        (v.lab[r0:r1, c0:c1] == 'amber') | ((v.lab[r0:r1, c0:c1] != 'red') & (v.lab[r0:r1, c0:c1] != 'pink'))
    rr, cc = np.nonzero(m)
    if len(cc) > 20:
        put('fleet number centre x', v.zv(c0 + (cc.min() + cc.max()) / 2)); put('fleet number height', (rr.max() - rr.min() + 1) / v.ppm)
        put('fleet number centre y', v.y(r0 + (rr.min() + rr.max()) / 2))
    r0, r1, c0, c1 = v.win(-3.6, -2.4, 0.95, 1.55)
    m = ~np.isin(v.lab[r0:r1, c0:c1], ['red', 'pink'])
    rr, cc = np.nonzero(m)
    if len(cc) > 20:
        put('coat of arms centre x', v.zv(c0 + (cc.min() + cc.max()) / 2)); put('coat of arms centre y', v.y(r0 + (rr.min() + rr.max()) / 2))
        put('coat of arms diameter', (cc.max() - cc.min() + 1) / v.ppm)
    for nm, win in (('indicator between windows 2 and 3', (2.85, 3.30, 2.05, 2.45)),
                    ('rear amber light', (-7.05, -6.75, 0.86, 1.05))):
        b = v.blob(('amber',), *win)
        if b: put(f'{nm} x', b['z']); put(f'{nm} y', b['y'])
    r0, r1, c0, c1 = v.win(5.9, 6.45, 1.30, 1.48)
    m = (v.lum[r0:r1, c0:c1] > 170) if not model else np.isin(v.lab[r0:r1, c0:c1], ['white', 'chrome', 'light'])
    rr, cc = np.nonzero(m)
    if len(cc) > 10:
        put('side marker x', v.zv(c0 + cc.mean())); put('side marker y', v.y(r0 + rr.mean()))
    # ---- skirt bottom, panel seams and the grey boxes under the skirt
    for x in (-6.4, -5.6, -0.4, 1.0, 5.4):
        e = first_run(v, ('cream', 'cream2', 'white', 'olive', 'oliveD') if model else ('cream',), x, 0.70, 0.15)
        if e: put(f'skirt bottom at x {x:+.1f}', e[1])
    sx = strokes_h(v, 0.55, -6.3, 6.3, half=2, thr=110 if not model else 70)
    for t in (6.04, 4.63, 1.33, 0.07, -1.55, -4.85, -6.02):         # seams from the front (drawing list)
        near = [s for s in sx if abs(s - t) < 0.12]
        put(f'skirt seam near x {t:+.2f}', near[0] if near else None)
    if model:
        rs = runs_on_row(v, ('grey',), 0.29, -6.6, 5.0, min_w=0.3, half=1)
    else:
        rs = runs_on_row(v, ('grey', 'dark'), 0.30, -6.6, 5.0, min_w=0.3, half=1)
    for nm, lo, hi in (('rear', -6.2, -4.5), ('middle', -1.7, 0.1), ('front', 0.1, 1.5)):
        cand = [r for r in rs if lo < (r[0] + r[1]) / 2 < hi]
        if cand: put(f'grey box under skirt, {nm}, rear end', cand[0][0]); put(f'grey box under skirt, {nm}, front end', cand[0][1])
    # ---- body ends and couplers
    BODY = ('cream', 'cream2', 'red', 'light', 'dark', 'pink', 'amber', 'white', 'chrome', 'grey2', 'roofgrey', 'grey', 'olive', 'oliveD')
    for y in (2.9, 2.6, 2.3, 2.0, 1.45, 1.2, 0.95, 0.80, 0.40):
        r = v.row(y)
        m = np.isin(v.lab[r - 1:r + 2], BODY if model else ('cream', 'red', 'light', 'dark', 'pink', 'amber', 'olive')).mean(0) > 0.5
        for nm, rng in (('rear', range(v.col(-7.35), v.col(-5.5))), ('front', range(v.col(7.35), v.col(5.5), -1))):
            hit = next((k for k in rng if m[k]), None)
            put(f'{nm} end at y {y:.2f}', v.zv(hit + (0 if nm == 'rear' else 1)) if hit is not None else None)
    for nm, x0, x1 in (('rear coupler', -7.64, -7.08), ('front coupler', 7.08, 7.64)):
        r0, r1, c0, c1 = v.win(x0, x1, 0.40, 0.75)
        m = (v.lab[r0:r1, c0:c1] != 'bg') if model else ~np.isin(v.lab[r0:r1, c0:c1], ['bg', 'cream', 'pink'])
        cols = np.where(m.sum(0) > 3)[0]; rows = np.where(m.sum(1) > 3)[0]
        if len(cols):
            put(f'{nm} tip x', v.zv(c0 + (cols[0] if x0 < 0 else cols[-1] + 1)))
            put(f'{nm} centre y', v.y(r0 + (rows[0] + rows[-1]) / 2))
    # ---- the cream stripe under the beading at the front end
    for x in (6.2, 6.6):
        e = first_run(v, ('cream', 'cream2') if model else ('cream',), x, 1.59, 1.30)
        put(f'front cream stripe bottom at x {x:.1f}', e[1] if e else None)
    rs = runs_on_row(v, ('cream', 'cream2') if model else ('cream',), 1.575, 5.5, 7.0, min_w=0.05, half=1)
    if rs: put('front cream stripe rear end (y 1.575)', rs[0][0])
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
    print(f"{'feature':52s} {'drawing':>8s} {'model':>8s} {'dev':>6s}")
    for k, a, b, dev in rows:
        flag = '' if dev is None else ('  ok' if abs(dev) <= TOL else '  !!')
        print(f"{k:52s} {str(a):>8s} {str(b):>8s} {str(dev):>6s}{flag}")

if __name__ == '__main__':
    main()
