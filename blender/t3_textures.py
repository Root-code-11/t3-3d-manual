# Procedural PBR textures for the Tatra T3 model, painted with numpy straight into the UV layout
# that build_t3.py gives the body, so livery bands, seams and wear land exactly on the geometry.
#
# Body atlas (one 4K set shared by the body sides, roof and both end masks):
#   v 0.70-1.00  right side, u = x along the car, v = arc length round the section (bottom -> roof crown)
#   v 0.40-0.70  left side, same mapping
#   v 0.00-0.40  front mask (u 0.00-0.50) and rear mask (u 0.50-1.00), mapped by arc length
# Maps: base colour (sRGB), ORM (R = ambient occlusion, G = roughness, B = metallic), normal (OpenGL).
import os
import numpy as np
from PIL import Image
from scipy import ndimage

RNG_SEED = 6101

def srgb(c):
    return np.array(c, dtype=np.float32) / 255.0

RED = srgb((198, 32, 30))          # Prague red, matched to the photos of 6102 / 6149
CREAM = srgb((234, 222, 184))
MAROON = srgb((80, 18, 14))
UNDER = srgb((34, 32, 30))
DIRT = srgb((92, 82, 66))          # road dust and brake dust, brown-grey
SOOT = srgb((70, 68, 66))          # graphite dust from the pantograph contact strips
DUST = srgb((178, 170, 152))
ROOF = srgb((92, 92, 90))          # grey roof between the cream end domes (coloured drawing)

def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def noise(shape, cells, seed, order=3):
    """Smooth value noise: a coarse random grid of (cells_y, cells_x), spline-upsampled to shape."""
    cy, cx = cells
    rng = np.random.default_rng(seed)
    g = rng.random((cy + 4, cx + 4)).astype(np.float32)
    zy, zx = (shape[0] + 0.0) / cy, (shape[1] + 0.0) / cx
    yy = np.linspace(1.5, cy + 1.5, shape[0], dtype=np.float32)
    xx = np.linspace(1.5, cx + 1.5, shape[1], dtype=np.float32)
    Y, X = np.meshgrid(yy, xx, indexing='ij')
    return ndimage.map_coordinates(g, [Y, X], order=order, mode='nearest').astype(np.float32)

def fbm(shape, cells, seed, octaves=4, gain=0.5):
    out = np.zeros(shape, np.float32)
    amp, tot = 1.0, 0.0
    cy, cx = cells
    for o in range(octaves):
        out += amp * noise(shape, (max(1, int(cy * 2 ** o)), max(1, int(cx * 2 ** o))), seed + 17 * o)
        tot += amp
        amp *= gain
    return out / tot

def white(shape, seed):
    return np.random.default_rng(seed).random(shape, dtype=np.float32)

def height_to_normal(h, dx, dy, strength=1.0):
    """h in metres; dx, dy metres per pixel along u (columns) and v (rows). Rows run top to bottom,
    v runs bottom to top, so the v gradient is the negative row gradient."""
    gy, gx = np.gradient(h)
    nx = -gx / dx * strength
    ny = gy / dy * strength
    nz = np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / l, ny / l, nz / l], axis=-1)
    return np.clip(n * 0.5 + 0.5, 0, 1)

def save_rgb(path, a, quality=92):
    Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGB').save(path, quality=quality, subsampling=0)

def gauss_line(d, w):
    return np.exp(-(d / w) ** 2)

# ------------------------------------------------------------------------------------------------ livery
def livery(y, L):
    col = np.where(y[..., None] < L['SKIRT'], CREAM, np.where(y[..., None] < L['BELT'], RED, CREAM)).astype(np.float32)
    pin = ((y > L['BELT'] - 0.004) & (y < L['BELT'] + 0.010)) | ((y > L['EAVE']) & (y < L['EAVE'] + 0.012))
    col[pin] = MAROON
    return col

def rect_dist(x, y, r):
    """Distance from points to the outside of rectangle r = (x0, x1, y0, y1); 0 inside."""
    dx = np.maximum(np.maximum(r[0] - x, x - r[1]), 0)
    dy = np.maximum(np.maximum(r[2] - y, y - r[3]), 0)
    return np.sqrt(dx * dx + dy * dy)

def wear(col, rough, ao, x, y, L, seed, dirt_amount, roof_mask, streak_rects, region_shape, bogie=True):
    """Shared wear: dirt along the skirt, rain streaks under openings, roof dust and slight fading."""
    H, W = region_shape
    n_mid = fbm((H, W), (6, 40), seed + 1, 4)
    n_hi = fbm((H, W), (40, 260), seed + 2, 3)
    n_low = fbm((H, W), (2, 8), seed + 3, 2)
    # fading: low-frequency unevenness of the paint
    col *= (1.0 + 0.05 * (n_low[..., None] - 0.5))
    # dirt rising from the rail: strongest at the skirt bottom and around the bogies
    d_low = smooth((1.12 - y) / 0.55) ** 1.4
    bog = np.exp(-((np.abs(x) - 3.2) / 1.7) ** 2) if bogie else 0.0
    speck = (white((H, W), seed + 4) > 0.985).astype(np.float32) * smooth((0.85 - y) / 0.3)
    dirt = np.clip(d_low * (0.22 + 0.30 * bog) * (0.55 + 0.9 * n_mid) * dirt_amount + 0.35 * speck, 0, 0.8)
    # rain streaks under windows: vertical, fading downwards, strongest at the corners
    st = np.zeros((H, W), np.float32)
    streak_n = noise((H, W), (3, 700), seed + 5, order=1)
    streak_n = np.clip((streak_n - 0.45) * 2.2, 0, 1)
    for (x0, x1, y0, y1) in streak_rects:
        inx = (x > x0 - 0.02) & (x < x1 + 0.02)
        below = np.clip((y0 - y) / 0.75, 0, 1)
        fall = np.where((y < y0) & inx, (1 - below) ** 1.6, 0)
        corner = 0.35 + np.exp(-((x - x0) / 0.05) ** 2) + np.exp(-((x - x1) / 0.05) ** 2)
        st = np.maximum(st, fall * corner * streak_n)
    st = np.clip(st, 0, 1) * 0.22
    # roof: light dust everywhere, graphite soot round the pantograph
    soot = roof_mask * np.exp(-((x - 3.2) / 1.3) ** 2) * (0.25 + 0.25 * n_mid)
    dust = roof_mask * (0.18 + 0.22 * n_mid)
    col = col * (1 - dust[..., None]) + DUST * dust[..., None]
    col = col * (1 - soot[..., None]) + SOOT * soot[..., None]
    col = col * (1 - st[..., None]) + DIRT * 0.8 * st[..., None]
    col = col * (1 - dirt[..., None]) + DIRT * dirt[..., None]
    rough += 0.05 * (n_hi - 0.5) + dirt * 0.35 + st * 0.4 + dust * 0.9 + soot * 0.5
    return col, np.clip(rough, 0.05, 1.0), ao

def paint_side(L, side, H, W, seed):
    v = 1.0 - (np.arange(H, dtype=np.float32) + 0.5) / H          # within the band: top row = high v
    # this function paints the band rows; v_band 0..1 bottom -> top
    u_prof = v                                                     # profile parameter (arc fraction)
    y_row = np.interp(u_prof, L['prof_u'], L['prof_y']).astype(np.float32)
    x_col = ((np.arange(W, dtype=np.float32) + 0.5) / W) * 2 * L['XS'] - L['XS']
    Y, X = np.meshgrid(y_row, x_col, indexing='ij')
    U = np.broadcast_to(u_prof[:, None], (H, W))
    col = livery(Y, L)
    under = U < L['u_bottom']
    col[under] = UNDER
    col[U > L['u_grey']] = ROOF
    roof = smooth((U - L['u_roof']) / 0.04)
    rough = np.full((H, W), 0.36, np.float32)
    ao = np.ones((H, W), np.float32)
    wins = L['win'][side]
    col, rough, ao = wear(col, rough, ao, X, Y, L, seed, 1.0, roof, wins, (H, W))
    # ambient occlusion: round the openings, under the chrome strip, the bottom crease, the eave
    for r in wins + (L['doors'] if side == 1 else []) + [L['cab_corner']]:
        d = rect_dist(X, Y, r)
        ao *= 1 - 0.28 * np.exp(-d / 0.012) * (d > 0)
    ao *= 1 - 0.25 * gauss_line(Y - (L['SKIRT'] - 0.012), 0.01)
    ao *= 1 - 0.45 * smooth((L['BODY_BOTTOM'] + 0.10 - Y) / 0.10) * ~under
    ao[under] = 0.45
    # height: orange peel, slight panel waviness below the windows, seams as soft grooves
    h = 0.000012 * fbm((H, W), (120, 600), seed + 9, 2)
    wav = 0.00035 * (fbm((H, W), (3, 30), seed + 10, 2) - 0.5) * (Y < L['BELT'])
    h += wav
    for sx in L['seams'][side]:
        h -= 0.0009 * gauss_line(X - sx, 0.0018) * (Y < L['BELT']) * (Y > L['BODY_BOTTOM'] + 0.005)
    h -= 0.0012 * (gauss_line(X - L['XS'], 0.002) + gauss_line(X + L['XS'], 0.002)) * (Y < 2.75)
    return col, rough, ao, h

def paint_mask(L, end, H, W, seed):
    E = L['nose'][end]
    v = 1.0 - (np.arange(H, dtype=np.float32) + 0.5) / H
    B = v * E['B_tot']
    y_row = np.interp(B, E['B'], E['y']).astype(np.float32)
    A_col = ((np.arange(W, dtype=np.float32) + 0.5) / W - 0.5) * 2 * L['ARCMAX']
    Y, A = np.meshgrid(y_row, A_col, indexing='ij')
    col = livery(Y, L)
    roof = smooth((Y - 2.78) / 0.12)
    rough = np.full((H, W), 0.36, np.float32)
    ao = np.ones((H, W), np.float32)
    # streak sources: the corners of the windscreen / rear window, in arc units
    if end == 'front':
        a_c = np.interp(0.78, E['s'], E['A_at_ws'])
        rects = [(-a_c - 0.05, -a_c + 0.05, L['WS'][0], L['WS'][1]), (a_c - 0.05, a_c + 0.05, L['WS'][0], L['WS'][1]),
                 (-0.6, 0.6, L['WS'][0], L['WS'][1])]
    else:
        a_c = np.interp(0.78, E['s'], E['A_at_rw'])
        rects = [(-a_c, a_c, L['RW'][2], L['RW'][3])]
    col, rough, ao = wear(col, rough, ao, A, Y, L, seed, 1.25, roof, rects, (H, W), bogie=False)
    ao *= 1 - 0.30 * gauss_line(Y - L['WS'][0], 0.015) if end == 'front' else 1
    ao *= 1 - 0.25 * gauss_line(Y - 0.75, 0.03)                      # under the bumper
    h = 0.000012 * fbm((H, W), (100, 300), seed + 9, 2)
    h += 0.00025 * (fbm((H, W), (3, 10), seed + 10, 2) - 0.5) * (Y < L['BELT'])
    return col, rough, ao, h

def body_atlas(out_dir, L, size=4096):
    base = np.zeros((size, size, 3), np.float32)
    orm = np.zeros((size, size, 3), np.float32)
    nrm = np.zeros((size, size, 3), np.float32)
    def put(rows, cols, col, rough, ao, h, dx, dy):
        r0, r1 = rows
        c0, c1 = cols
        base[r0:r1, c0:c1] = col
        orm[r0:r1, c0:c1, 0] = ao
        orm[r0:r1, c0:c1, 1] = rough
        orm[r0:r1, c0:c1, 2] = 0.0
        nrm[r0:r1, c0:c1] = height_to_normal(h, dx, dy)
    def rows_for(v0, v1):
        return int(round((1 - v1) * size)), int(round((1 - v0) * size))
    for side, (v0, v1) in ((1, (0.70, 1.00)), (-1, (0.40, 0.70))):
        r0, r1 = rows_for(v0, v1)
        col, rough, ao, h = paint_side(L, side, r1 - r0, size, RNG_SEED + (0 if side == 1 else 100))
        put((r0, r1), (0, size), col, rough, ao, h, 2 * L['XS'] / size, L['ARC_TOT'] / (r1 - r0))
        print('  painted side', side)
    r0, r1 = rows_for(0.0, 0.40)
    for end, (c0, c1) in (('front', (0, size // 2)), ('rear', (size // 2, size))):
        col, rough, ao, h = paint_mask(L, end, r1 - r0, c1 - c0, RNG_SEED + (200 if end == 'front' else 300))
        put((r0, r1), (c0, c1), col, rough, ao, h, 2 * L['ARCMAX'] / (c1 - c0), L['nose'][end]['B_tot'] / (r1 - r0))
        print('  painted mask', end)
    paths = {k: os.path.join(out_dir, f'body_{k}.jpg') for k in ('base', 'orm', 'normal')}
    save_rgb(paths['base'], base)
    save_rgb(paths['orm'], orm)
    save_rgb(paths['normal'], nrm, quality=95)
    return paths

# ------------------------------------------------------------------------------------------------ door leaves
def door_atlas(out_dir, size=2048):
    """Door leaves: u across a leaf, v = height 0.42 - 2.62 m. Cream, dirt rising from the step."""
    H = W = size
    v = 1.0 - (np.arange(H, dtype=np.float32) + 0.5) / H
    y = 0.42 + v * 2.2
    Y = np.broadcast_to(y[:, None], (H, W)).copy()
    X = np.broadcast_to(((np.arange(W, dtype=np.float32) + 0.5) / W * 0.32)[None, :], (H, W)).copy()
    col = np.broadcast_to(CREAM, (H, W, 3)).copy()
    rough = np.full((H, W), 0.36, np.float32)
    ao = np.ones((H, W), np.float32)
    n_mid = fbm((H, W), (8, 4), RNG_SEED + 50, 4)
    col *= (1.0 + 0.04 * (fbm((H, W), (2, 2), RNG_SEED + 51, 2)[..., None] - 0.5))
    dirt = np.clip(smooth((1.0 - Y) / 0.55) ** 1.4 * 0.45 * (0.5 + n_mid), 0, 0.75)
    hand = np.exp(-((Y - 1.25) / 0.25) ** 2) * np.exp(-((X - 0.03) / 0.04) ** 2) * 0.18      # hand marks at the edge
    dirt = np.maximum(dirt, hand)
    col = col * (1 - dirt[..., None]) + DIRT * dirt[..., None]
    rough += dirt * 0.3 + 0.04 * (fbm((H, W), (60, 30), RNG_SEED + 52, 2) - 0.5)
    ao *= 1 - 0.3 * np.exp(-np.minimum(X, 0.32 - X) / 0.01)
    h = 0.000012 * fbm((H, W), (300, 150), RNG_SEED + 53, 2)
    paths = {k: os.path.join(out_dir, f'door_{k}.jpg') for k in ('base', 'orm', 'normal')}
    save_rgb(paths['base'], col)
    save_rgb(paths['orm'], np.stack([ao, np.clip(rough, 0, 1), np.zeros_like(ao)], -1))
    save_rgb(paths['normal'], height_to_normal(h, 0.32 / W, 2.2 / H), quality=95)
    return paths

# ------------------------------------------------------------------------------------------------ tiling sets
def tiling_sets(out_dir, size=2048):
    """Seamless-enough tiling textures: painted cast steel (bogies), rubber, ribbed floor, leatherette."""
    H = W = size
    out = {}
    def wrapnoise(cells, seed, octaves=4):
        # tileable: blend the noise with its own shifted copies
        a = fbm((H, W), cells, seed, octaves)
        b = np.roll(a, H // 2, axis=0)
        c = np.roll(a, W // 2, axis=1)
        wy = np.abs(np.linspace(-1, 1, H))[:, None]
        wx = np.abs(np.linspace(-1, 1, W))[None, :]
        return a * (1 - wy) * (1 - wx) + b * wy * (1 - wx) + c * (1 - wy) * wx + np.roll(b, W // 2, 1) * wy * wx
    # bogie paint: black-grey, grime, rust at the edges of chips
    n1, n2, n3 = wrapnoise((6, 6), 901), wrapnoise((40, 40), 902, 3), wrapnoise((3, 3), 903, 2)
    chips = np.clip((wrapnoise((25, 25), 904, 3) - 0.68) * 6, 0, 1)
    col = np.stack([0.07 + 0.05 * n1, 0.068 + 0.045 * n1, 0.064 + 0.04 * n1], -1)
    grime = np.clip(0.3 + 0.5 * (n3 - 0.5) + 0.3 * n2, 0, 1)[..., None]
    col = col * (1 - grime * 0.6) + np.array([0.20, 0.17, 0.13]) * grime * 0.6
    rust = chips[..., None] * np.array([0.32, 0.15, 0.07])
    col = col * (1 - chips[..., None]) + rust
    rough = np.clip(0.55 + 0.25 * n2 + 0.2 * chips, 0, 1)
    metal = np.clip(0.15 + 0.2 * chips, 0, 1)
    h = 0.0004 * n2 + 0.0006 * n1 - 0.0003 * chips
    out['cast'] = (col, np.stack([np.ones_like(rough) * 0.95, rough, metal], -1), height_to_normal(h, 1 / W, 1 / H))
    # rubber
    n = wrapnoise((50, 50), 911, 3)
    col = np.stack([0.025 + 0.012 * n] * 3, -1)
    out['rubber'] = (col, np.stack([np.ones_like(n), 0.82 + 0.1 * n, np.zeros_like(n)], -1), height_to_normal(0.00005 * n, 0.5 / W, 0.5 / H))
    # ribbed rubber floor: ribs every 12 mm across a 0.6 m tile
    yy = np.linspace(0, 0.6, H, endpoint=False)[:, None] * np.ones((1, W))
    rib = 0.5 + 0.5 * np.cos(2 * np.pi * yy / 0.012)
    n = wrapnoise((30, 30), 921, 3)
    col = np.stack([0.11 + 0.02 * n - 0.02 * rib] * 3, -1) * np.array([1.0, 1.0, 1.02])
    out['floor'] = (col, np.stack([1 - 0.25 * rib, 0.7 + 0.15 * n, np.zeros_like(n)], -1), height_to_normal(0.0009 * rib, 0.6 / W, 0.6 / H))
    # leatherette grain (used as a normal map under a plain colour)
    n = wrapnoise((120, 120), 931, 3)
    out['leather'] = (np.ones((H, W, 3)) * 0.5, np.stack([np.ones_like(n), 0.45 + 0.15 * n, np.zeros_like(n)], -1),
                      height_to_normal(0.00012 * n, 0.4 / W, 0.4 / H))
    paths = {}
    for k, (b, o, nm) in out.items():
        p = {m: os.path.join(out_dir, f'{k}_{m}.jpg') for m in ('base', 'orm', 'normal')}
        save_rgb(p['base'], b)
        save_rgb(p['orm'], o)
        save_rgb(p['normal'], nm, quality=95)
        paths[k] = p
        print('  painted tile', k)
    return paths
