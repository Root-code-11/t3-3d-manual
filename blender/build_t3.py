# Tatra T3 tram, Prague prototype 6101 (1960), built to the specification of 6102, from code in Blender.
#   python3 build_t3.py -- <out_dir> [--glb] [--web] [--render] [--checks] [--blend] [--retex]
#                         [--variant 6102|6101] [--number 6101] [--samples N] [--res WxH] [--shots a,b]
#   (or: Blender -b -P build_t3.py -- <out_dir> ...)
#
# Coordinates (as in the BMW R 12 manual this project follows): x forward, y up, z to the right seen
# facing forward, metres, origin on the rail top under the vehicle centre. The glTF exporter turns
# Blender Z-up into Y-up, so a glTF viewer gets these coordinates back unchanged.
# Every part is an empty carrying { part, explode } as glTF extras; meshes hang under it. Door leaves
# have their origin on the hinge (leaf 1 is a child of leaf 0, leaf 2 a child of leaf 3); wheelsets
# have their origin on the axle.
#
# Sources: Wikipedia "Tatra T3" (citing Maruna 1974, Linert & Fojtík 2005) for main dimensions, the
# 1343 mm doors, the 25-seat layout, seat colours and interior finishes; prazsketramvaje.cz for 6102;
# reference photos of 6102 and 6149 for proportions, livery and details. ASSUMED marks values fitted
# to the photos that should be measured from a type drawing.
import bpy, bmesh, math, sys, os
import numpy as np
from mathutils import Vector, Matrix

ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = os.path.abspath(ARGS[0] if ARGS and not ARGS[0].startswith('--') else 'out')
VARIANT = ARGS[ARGS.index('--variant') + 1] if '--variant' in ARGS else '6102'   # geometry
NUMBER = ARGS[ARGS.index('--number') + 1] if '--number' in ARGS else '6101'     # fleet number painted on
HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, 'textures')
os.makedirs(OUT, exist_ok=True)
os.makedirs(TEX, exist_ok=True)
sys.path.insert(0, HERE)

# ============================================================================== dimensions
L_BODY = 14.000          # body length (Wikipedia)
L_COUPLERS = 15.104      # length over couplers (type drawing; coupler 552 mm beyond each end)
W_BODY = 2.500           # body width (Wikipedia)
H_ROOF = 3.060           # roof above rail top, without pantograph (spec)
GAUGE = 1.435
BOGIE_PIVOTS = 6.400     # bogie pivot centres (spec)
WHEELBASE = 1.900        # bogie wheelbase (spec)
WHEEL_R = 0.350          # 700 mm wheels (spec)
DOORS = {'6102': (1.343, 1.343, 1.343), '6101': (1.300, 1.060, 1.300)}[VARIANT]
# Scaled off the dimensioned type drawing (side view, ~50 px per metre) and the coloured elevations:
BODY_BOTTOM = 0.33       # lower edge of the cream skirt
SKIRT_LIFT = 0.29        # skirt cut up to ~0.62 m over the bogies
SKIRT = 0.76             # cream skirt / chrome strip / red band
BELT = 1.58              # red / cream break with a maroon pinstripe
WIN_Y0, WIN_Y1, WIN_SPLIT = 1.66, 2.54, 2.12   # side windows: sill, head, top of the fixed pane
EAVE = 2.62              # upper maroon pinstripe
SIDE_TOP = 2.70          # where the side cells hand over to the roof
FLOOR = 0.90             # passenger floor
WALL = 0.05              # body wall thickness (outer skin to inner lining)
HALF = L_BODY / 2
XP = BOGIE_PIVOTS / 2

def V(x, y, z):
    return Vector((x, -z, y))
ROT = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))   # tram -> Blender

# ============================================================================== body surface
PROF_PTS = [(0.0, BODY_BOTTOM), (0.6, BODY_BOTTOM), (1.10, BODY_BOTTOM + 0.002), (1.20, BODY_BOTTOM + 0.022),
            (1.24, BODY_BOTTOM + 0.07), (1.25, BODY_BOTTOM + 0.16), (1.25, 1.66), (1.237, 2.30), (1.222, 2.62),
            (1.195, 2.72), (1.13, 2.82), (0.98, 2.925), (0.62, 3.012), (0.0, H_ROOF)]

def catmull(pts, per=12):
    P = [Vector(p) for p in pts]
    P = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(per):
            t = k / per
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2])
    return out

_dense = catmull(PROF_PTS, per=40)
for i in range(1, len(_dense)):
    if _dense[i].y < _dense[i - 1].y:
        _dense[i].y = _dense[i - 1].y
_acc = [0.0]
for i in range(1, len(_dense)):
    _acc.append(_acc[-1] + (_dense[i] - _dense[i - 1]).length)
ARC_TOT = _acc[-1]
_U = [a / ARC_TOT for a in _acc]

def prof(u):
    u = min(max(u, 0.0), 1.0)
    lo, hi = 0, len(_U) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _U[mid] <= u: lo = mid
        else: hi = mid
    t = (u - _U[lo]) / max(_U[hi] - _U[lo], 1e-12)
    p = _dense[lo].lerp(_dense[hi], t)
    return p.x, p.y

_UY = {}
def u_of_y(y):
    k = round(y, 7)
    if k in _UY:
        return _UY[k]
    lo, hi = 0.0, 1.0
    for _ in range(46):
        mid = (lo + hi) / 2
        if prof(mid)[1] < y: lo = mid
        else: hi = mid
    _UY[k] = (lo + hi) / 2
    return _UY[k]

def smooth01(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)

from scipy.interpolate import PchipInterpolator

class End:
    """A fibreglass end mask, from the type drawing.
    Plan: the cab narrows from the body side towards a slightly bowed face with rounded corners
    (the angled cab side windows sit on the narrowing part).
    Side view: cream apron set back under the bumper, upright red panel, windscreen leaning back
    ~0.35 m, then a rounded roof dome that starts ~0.95 m behind the nose. The whole plan outline is
    scaled towards the body by k(y), so the dome and the rake never fold back past the body."""
    def __init__(s, sign, xs=6.0, ws_top=2.53, rake=0.35, apron=0.06):
        s.sign, s.xs, s.xt = sign, xs, HALF
        s.ws_top, s.rake, s.apron = ws_top, rake, apron

FRONT = End(+1)
REAR = End(-1, rake=0.30)
ENDS = (FRONT, REAR)
XS = FRONT.xs

# plan outline: r = |z| / half-width (1 at the body side, 0 on the centre line) -> share of the nose length
_PR = [0.0, 0.35, 0.55, 0.68, 0.76, 0.82, 0.91, 1.0]
_PX = [1.0, 0.99, 0.965, 0.92, 0.84, 0.70, 0.35, 0.0]
PLAN = PchipInterpolator(_PR, _PX)
_R_TAB = np.linspace(0.0, 1.0, 401)
_X_TAB = PLAN(_R_TAB)

def kfac(E, y0):
    """Share of the nose length left at height y0 (1 = full length)."""
    if y0 < 0.70:
        return 1.0 - E.apron / (E.xt - E.xs)
    if y0 < 1.66:
        return 1.0
    if y0 < E.ws_top:
        return 1.0 - E.rake * (y0 - 1.66) / (E.ws_top - 1.66)
    k0 = 1.0 - E.rake
    R = H_ROOF - E.ws_top
    t = min(1.0, (y0 - E.ws_top) / R)
    return 0.05 + (k0 - 0.05) * math.sqrt(max(0.0, 1 - t * t))

def taper(y0):
    return 1.0 - 0.05 * min(1.0, max(0.0, (0.70 - y0) / (0.70 - BODY_BOTTOM)))

def skirt_lift(x):
    d = abs(abs(x) - XP)
    return SKIRT_LIFT * (1 - smooth01((d - 1.30) / 0.18))

def nose_s(E, s, y0):
    """Point on an end mask: s -1..1 across (fraction of the half-width), y0 the height."""
    zb = prof(u_of_y(y0))[0] * taper(y0)
    r = min(abs(s), 1.0)
    x = E.xs + (E.xt - E.xs) * float(PLAN(r)) * kfac(E, y0)
    return Vector((E.sign * x, y0, max(-1.0, min(1.0, s)) * zb))

def nose(E, z, y0):
    zb = prof(u_of_y(y0))[0] * taper(y0)
    return nose_s(E, z / zb if zb > 1e-9 else 0.0, y0)

def surf(xb, u, side):
    z0, y0 = prof(u)
    for E in ENDS:
        if xb * E.sign > E.xs:
            k = kfac(E, y0)
            frac = min((xb * E.sign - E.xs) / ((E.xt - E.xs) * k), 1.0)
            r = float(np.interp(-frac, -_X_TAB, _R_TAB))                  # invert the plan outline
            x = E.sign * (E.xs + (E.xt - E.xs) * k * frac)
            return Vector((x, y0, side * z0 * r * taper(y0)))
    k = smooth01((abs(xb) - 5.6) / (XS - 5.6))
    z = z0 * (1 - (1 - taper(y0)) * k)
    y = y0
    if y0 < SKIRT:
        y = y0 + skirt_lift(xb) * (SKIRT - y0) / (SKIRT - BODY_BOTTOM)
    return Vector((xb, y, side * z))

def side_pt(side):
    return lambda x, y: surf(x, u_of_y(y), side)

def nose_pt(E):
    return lambda s, y: nose_s(E, s, y)

def normal(f, a, b, h=1e-4):
    n = (f(a + h, b) - f(a - h, b)).cross(f(a, b + h) - f(a, b - h))
    p = f(a, b)
    ref = Vector((max(-5.5, min(5.5, p.x)), 1.84, 0.0))
    if n.length < 1e-12:
        n = p - ref
    n.normalize()
    return n if n.dot(p - ref) >= 0 else -n

# ============================================================================== layout
# Door centres scaled off the type drawing: front door straight behind the cab, middle door between
# the bogies, rear door at the very rear end; widths 1343 mm (prazsketramvaje.cz, Wikipedia).
PILLAR = 0.17
MARGIN = 0.22                                       # from a door frame to the first window
Y_SIDE0 = BODY_BOTTOM + 0.003
FD0, FD1 = 5.26 - DOORS[0] / 2, 5.26 + DOORS[0] / 2
MD0, MD1 = -0.78 - DOORS[1] / 2, -0.78 + DOORS[1] / 2
RD0, RD1 = -5.30 - DOORS[2] / 2, -5.30 + DOORS[2] / 2
DOOR_SPANS = [('door_front', FD0, FD1), ('door_middle', MD0, MD1), ('door_rear', RD0, RD1)]

def win_run(x_from, x_to, n, pillar=PILLAR):
    w = (x_to - x_from - pillar * (n - 1)) / n
    return [(x_from + i * (w + pillar), x_from + i * (w + pillar) + w, WIN_Y0, WIN_Y1) for i in range(n)]

# right (door) side: 3 windows between the front and middle door, 2 between the middle and rear door
WIN_R = win_run(MD1 + MARGIN, FD0 - MARGIN, 3) + win_run(RD1 + MARGIN, MD0 - MARGIN, 2)
# left side: 8 equal windows (coloured elevation)
WIN_L = win_run(-5.85, 5.85, 8, pillar=0.15)
DOOR_OPEN = [(x0, x1, Y_SIDE0 - 0.01, 2.56) for _, x0, x1 in DOOR_SPANS]
OPEN = {1: WIN_R + DOOR_OPEN, -1: list(WIN_L)}
# end glazing in mask units (s across, height): windscreen on the face, angled cab side windows on
# the narrowing part; the rear end the same with a lower window
WS_Y0, WS_Y1 = 1.68, 2.50
FRONT_HOLES = [(-0.78, 0.78, WS_Y0, WS_Y1), (0.84, 0.975, WS_Y0, 2.46), (-0.975, -0.84, WS_Y0, 2.46)]
REAR_HOLES = [(-0.78, 0.78, 1.70, 2.46), (0.84, 0.975, 1.70, 2.42), (-0.975, -0.84, 1.70, 2.42)]
REAR_WIN = REAR_HOLES[0]

def pillar_seams(wins):
    xs = sorted(wins)
    return [(xs[i][1] + xs[i + 1][0]) / 2 for i in range(len(xs) - 1) if xs[i + 1][0] - xs[i][1] < 0.3]
SEAMS = {1: pillar_seams(WIN_R), -1: pillar_seams(WIN_L)}
GROOVE_W, GROOVE_D = 0.005, 0.0022
def grooves(side):
    g = [(s - GROOVE_W / 2, s + GROOVE_W / 2, Y_SIDE0, BELT - 0.006) for s in SEAMS[side]]
    g += [(XS - GROOVE_W, XS, Y_SIDE0, 2.75), (-XS, -XS + GROOVE_W, Y_SIDE0, 2.75)]          # mask joints
    return g

# ============================================================================== UV layout (see t3_textures.py)
V_SIDE = {1: (0.70, 0.30), -1: (0.40, 0.30)}
def uv_side(side):
    v0, vh = V_SIDE[side]
    return lambda x, y: ((x + XS) / (2 * XS), v0 + vh * u_of_y(y))
def uv_side_u(side):
    v0, vh = V_SIDE[side]
    return lambda x, u: ((x + XS) / (2 * XS), v0 + vh * u)

# arc-length tables for the masks, so the texture is not stretched round the corners
S_GRID = np.linspace(-1.0, 1.0, 161)
NOSE_TAB = {}
for E, key in ((FRONT, 'front'), (REAR, 'rear')):
    ys = np.concatenate([np.linspace(Y_SIDE0, 2.70, 120), np.linspace(2.70, H_ROOF, 80)[1:]])
    A = np.zeros((len(ys), len(S_GRID)))
    for j, y in enumerate(ys):
        pts = [nose_s(E, s, y) for s in S_GRID]
        d = [0.0] + [(pts[i + 1] - pts[i]).length for i in range(len(pts) - 1)]
        c = np.cumsum(d)
        A[j] = c - np.interp(0.0, S_GRID, c)
    cen = [nose_s(E, 0.0, y) for y in ys]
    B = np.concatenate([[0.0], np.cumsum([(cen[i + 1] - cen[i]).length for i in range(len(cen) - 1)])])
    NOSE_TAB[key] = dict(y=ys, A=A, B=B, B_tot=B[-1])
ARCMAX = float(max(np.abs(NOSE_TAB[k]['A']).max() for k in NOSE_TAB)) + 0.02
U_CEN = {'front': 0.25, 'rear': 0.75}

def uv_nose(E):
    key = 'front' if E is FRONT else 'rear'
    T = NOSE_TAB[key]
    idx = np.arange(len(T['y']))
    def f(s, y):
        j = float(np.interp(y, T['y'], idx))
        j0 = min(int(j), len(T['y']) - 2)
        fy = j - j0
        a = (1 - fy) * np.interp(s, S_GRID, T['A'][j0]) + fy * np.interp(s, S_GRID, T['A'][j0 + 1])
        b = float(np.interp(y, T['y'], T['B']))
        return (U_CEN[key] + a / (4 * ARCMAX), 0.40 * b / T['B_tot'])
    return f

# ============================================================================== textures
import t3_textures as TT
def build_textures():
    L = dict(XS=XS, SKIRT=SKIRT, BELT=BELT, EAVE=EAVE, BODY_BOTTOM=BODY_BOTTOM, ARC_TOT=ARC_TOT, ARCMAX=ARCMAX,
             prof_u=np.array(_U), prof_y=np.array([p.y for p in _dense]),
             u_bottom=u_of_y(Y_SIDE0), u_roof=u_of_y(2.74), u_grey=u_of_y(2.935),
             win={1: WIN_R, -1: WIN_L}, doors=[(x0, x1, Y_SIDE0, 2.64) for _, x0, x1 in DOOR_SPANS],
             cab_corner=(99.0, 99.1, 0.0, 0.0), seams=SEAMS, WS=(WS_Y0, WS_Y1), RW=REAR_WIN, nose={})
    for key in ('front', 'rear'):
        T = NOSE_TAB[key]
        jw = int(np.searchsorted(T['y'], WS_Y0))
        jr = int(np.searchsorted(T['y'], REAR_WIN[2]))
        L['nose'][key] = dict(y=T['y'], B=T['B'], B_tot=T['B_tot'], s=S_GRID, A_at_ws=T['A'][jw], A_at_rw=T['A'][jr])
    print('painting textures ...')
    TT.body_atlas(TEX, L, 4096)
    TT.door_atlas(TEX, 2048)
    TT.tiling_sets(TEX, 2048)

if '--retex' in ARGS or not os.path.exists(os.path.join(TEX, 'body_base.jpg')):
    build_textures()

# ============================================================================== scene and materials
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
COL = bpy.data.collections.new('T3')
scene.collection.children.link(COL)

def gltf_group():
    g = bpy.data.node_groups.get('glTF Material Output')
    if g is None:
        g = bpy.data.node_groups.new('glTF Material Output', 'ShaderNodeTree')
        g.interface.new_socket(name='Occlusion', in_out='INPUT', socket_type='NodeSocketFloat')
    return g

M = {}
def mat(name, color=(0.8, 0.8, 0.8), metal=0.0, rough=0.5, coat=0.0, emit=None, emit_strength=1.0,
        trans=0.0, ior=1.5, tex=None, normal_only=None, normal_strength=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Metallic'].default_value = metal
    p.inputs['Roughness'].default_value = rough
    if coat:
        p.inputs['Coat Weight'].default_value = coat
        p.inputs['Coat Roughness'].default_value = 0.12
    if emit:
        p.inputs['Emission Color'].default_value = (*emit, 1)
        p.inputs['Emission Strength'].default_value = emit_strength
    if trans:
        p.inputs['Transmission Weight'].default_value = trans
        p.inputs['IOR'].default_value = ior
    def img(path, non_color):
        im = bpy.data.images.load(path, check_existing=True)
        if non_color:
            im.colorspace_settings.name = 'Non-Color'
        n = nt.nodes.new('ShaderNodeTexImage')
        n.image = im
        return n
    prefix = tex or normal_only
    if prefix:
        if tex:
            b = img(os.path.join(TEX, f'{prefix}_base.jpg'), False)
            nt.links.new(b.outputs['Color'], p.inputs['Base Color'])
        o = img(os.path.join(TEX, f'{prefix}_orm.jpg'), True)
        sep = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(o.outputs['Color'], sep.inputs['Color'])
        nt.links.new(sep.outputs['Green'], p.inputs['Roughness'])
        if tex:
            nt.links.new(sep.outputs['Blue'], p.inputs['Metallic'])
            gn = nt.nodes.new('ShaderNodeGroup')
            gn.node_tree = gltf_group()
            nt.links.new(sep.outputs['Red'], gn.inputs['Occlusion'])
        nm = img(os.path.join(TEX, f'{prefix}_normal.jpg'), True)
        nmap = nt.nodes.new('ShaderNodeNormalMap')
        nmap.inputs['Strength'].default_value = normal_strength
        nt.links.new(nm.outputs['Color'], nmap.inputs['Color'])
        nt.links.new(nmap.outputs['Normal'], p.inputs['Normal'])
    M[name] = m
    return m

# painted body: livery and wear live in the 4K atlas; a thin clear coat gives the semi-gloss finish
mat('paint', tex='body', coat=0.12)
mat('door', tex='door', coat=0.12)
mat('cast', tex='cast')                                         # bogies: painted cast steel with grime
mat('rubber', tex='rubber')
mat('floor', tex='floor')
mat('mat_roof', tex='floor')                                    # roof walkway mats
mat('leather_red', (0.36, 0.035, 0.03), rough=0.45, normal_only='leather')
mat('leather_grey', (0.17, 0.18, 0.19), rough=0.45, normal_only='leather')
mat('maroon', (0.13, 0.012, 0.01), rough=0.38, coat=0.4)
mat('gold', (0.80, 0.56, 0.16), metal=0.55, rough=0.32, coat=0.4)    # fleet numbers (coloured drawing, photo of 6149)
mat('brown', (0.16, 0.06, 0.03), rough=0.6)                           # step plates under the doors
mat('amber', (0.85, 0.40, 0.03), rough=0.12, emit=(0.6, 0.25, 0.0), emit_strength=0.2)
mat('roofgrey', (0.11, 0.11, 0.11), rough=0.7)                        # roof fittings on the grey roof
mat('panto', (0.55, 0.07, 0.035), rough=0.5)
mat('grey', (0.30, 0.31, 0.32), rough=0.6)
mat('under', (0.028, 0.027, 0.026), rough=0.8)
mat('glass', (0.86, 0.95, 0.90), rough=0.015, trans=1.0, ior=1.52)
mat('tread', (0.62, 0.62, 0.60), metal=1.0, rough=0.16)        # worn, shiny running surface
mat('steel', (0.5, 0.5, 0.52), metal=1.0, rough=0.35)
mat('darkmetal', (0.035, 0.035, 0.037), metal=0.5, rough=0.5)
mat('chrome', (0.95, 0.95, 0.95), metal=1.0, rough=0.07)
mat('mirror', (0.92, 0.92, 0.92), metal=1.0, rough=0.02)
mat('lens', (0.92, 0.9, 0.84), rough=0.06, emit=(1.0, 0.93, 0.78), emit_strength=0.15)
mat('redlens', (0.45, 0.02, 0.02), rough=0.12, emit=(0.4, 0.0, 0.0), emit_strength=0.3)
mat('lamp', (0.93, 0.91, 0.84), rough=0.15)
mat('signtext', (0.92, 0.92, 0.88), rough=0.4, emit=(1, 1, 0.95), emit_strength=0.2)
mat('insulator', (0.30, 0.11, 0.05), rough=0.2, coat=0.5)
mat('copper', (0.62, 0.32, 0.18), metal=1.0, rough=0.4)
mat('motor', (0.09, 0.095, 0.09), metal=0.3, rough=0.6)
mat('umakart', (0.42, 0.33, 0.22), rough=0.42, coat=0.2)
mat('lining', (0.70, 0.68, 0.60), rough=0.5)
mat('ceiling', (0.80, 0.79, 0.74), rough=0.6)
mat('cab', (0.15, 0.19, 0.17), rough=0.5)
mat('tube', (1.0, 0.98, 0.94), rough=0.2, emit=(1, 0.97, 0.9), emit_strength=0.6)
TILED = {'cast': 0.5, 'rubber': 0.4, 'floor': 0.6, 'mat_roof': 0.6, 'leather_red': 0.4, 'leather_grey': 0.4}

FONT_NUM = bpy.data.fonts.load(os.path.join(HERE, 'fonts', 'DejaVuSerif-Bold.ttf'))
FONT_SIGN = bpy.data.fonts.load(os.path.join(HERE, 'fonts', 'DejaVuSansCondensed-Bold.ttf'))

# ============================================================================== mesh helpers
PART = None
def part(pid, explode=(0, 0, 0), **props):
    global PART
    e = bpy.data.objects.new(f'P.{pid}', None)
    e.empty_display_size = 0.1
    e['part'] = pid
    e['explode'] = list(explode)
    for k, v in props.items():
        e[k] = v
    COL.objects.link(e)
    PART = e
    return e

def box_uv(bm, scale):
    uvl = bm.loops.layers.uv.verify()
    for f in bm.faces:
        f.normal_update()
        n = f.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        for l in f.loops:
            c = l.vert.co
            uv = (c.z, c.y) if ax == 0 else (c.x, c.z) if ax == 1 else (c.x, c.y)
            l[uvl].uv = (uv[0] / scale, uv[1] / scale)

def finish(bm, name, material, smooth=True, sharp=40, parent='part'):
    """bm is in tram coordinates; box-project UVs for tiled materials, rotate to Blender, link under the part."""
    if material in TILED and not bm.loops.layers.uv:
        box_uv(bm, TILED[material])
    bmesh.ops.transform(bm, matrix=ROT, verts=bm.verts)
    if smooth:
        lim = math.radians(sharp)
        for f in bm.faces:
            f.smooth = True
        for e in bm.edges:
            if len(e.link_faces) == 2 and e.calc_face_angle(0) > lim:
                e.smooth = False
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M[material])
    o = bpy.data.objects.new(name, me)
    COL.objects.link(o)
    o.parent = PART if parent == 'part' else parent
    return o

def fixed_uv(o, uv):
    """Point every vertex of a small part at one spot of the body atlas (a plain patch of paint)."""
    layer = o.data.uv_layers.new()
    for l in layer.data:
        l.uv = uv

def rehome(o, pivot, parent=None, parent_pivot=(0, 0, 0)):
    pv = V(*pivot)
    o.data.transform(Matrix.Translation(-pv))
    if parent is not None:
        o.parent = parent
    o.location = pv - V(*parent_pivot)

def _merge_tmp(bm, tmp):
    me = bpy.data.meshes.new('_tmp')
    tmp.to_mesh(me)
    tmp.free()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)

def cyl_into(bm, a, b, r, seg=24, r2=None, bevel=0.0):
    """Add a cylinder (optionally with rounded cap edges) into an existing bmesh."""
    tmp = bmesh.new()
    bmesh.ops.create_cone(tmp, cap_ends=True, cap_tris=False, segments=seg, radius1=r,
                          radius2=r if r2 is None else r2, depth=(Vector(b) - Vector(a)).length)
    if bevel:
        edges = [e for e in tmp.edges if len(e.link_faces) == 2 and e.calc_face_angle(0) > math.radians(50)]
        bmesh.ops.bevel(tmp, geom=edges, offset=bevel, segments=2, affect='EDGES', profile=0.5)
    a, b = Vector(a), Vector(b)
    q = Vector((0, 0, 1)).rotation_difference((b - a).normalized())
    bmesh.ops.transform(tmp, matrix=Matrix.Translation((a + b) / 2) @ q.to_matrix().to_4x4(), verts=tmp.verts)
    _merge_tmp(bm, tmp)

def box_into(bm, center, size, bevel=0.0, rot=None):
    tmp = bmesh.new()
    bmesh.ops.create_cube(tmp, size=1.0)
    bmesh.ops.scale(tmp, vec=Vector(size), verts=tmp.verts)
    if bevel:
        bmesh.ops.bevel(tmp, geom=list(tmp.edges), offset=min(bevel, min(size) * 0.45), segments=2, affect='EDGES', profile=0.5)
    if rot is not None:
        bmesh.ops.transform(tmp, matrix=rot.to_4x4(), verts=tmp.verts)
    bmesh.ops.translate(tmp, vec=Vector(center), verts=tmp.verts)
    _merge_tmp(bm, tmp)

def cyl(name, a, b, r, material, seg=24, r2=None, bevel=0.0, smooth=True):
    bm = bmesh.new()
    cyl_into(bm, a, b, r, seg, r2, bevel)
    return finish(bm, name, material, smooth=smooth, sharp=50)

def box(name, center, size, material, bevel=0.006, rot=None):
    bm = bmesh.new()
    box_into(bm, center, size, bevel, rot)
    return finish(bm, name, material, smooth=bevel > 0, sharp=50)

def rot_z(deg):
    return Matrix.Rotation(math.radians(deg), 3, 'Z')

def rod(name, pts, r, material, seg=10):
    bm = bmesh.new()
    for i in range(len(pts) - 1):
        cyl_into(bm, pts[i], pts[i + 1], r, seg)
    for p in pts[1:-1]:
        bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=max(4, seg // 2), radius=r,
                                  matrix=Matrix.Translation(Vector(p)))
    return finish(bm, name, material, smooth=True, sharp=70)

def sweep(name, path, r, material, seg=10, closed=False):
    """A round tube along a path of points (springs, chains)."""
    bm = bmesh.new()
    rings = []
    n = len(path)
    for i, p in enumerate(path):
        p = Vector(p)
        t = (Vector(path[min(i + 1, n - 1)]) - Vector(path[max(i - 1, 0)])).normalized()
        a = t.orthogonal().normalized()
        b = t.cross(a)
        rings.append([bm.verts.new(p + (a * math.cos(2 * math.pi * k / seg) + b * math.sin(2 * math.pi * k / seg)) * r) for k in range(seg)])
    for i in range(n - 1 + (1 if closed else 0)):
        r0, r1 = rings[i], rings[(i + 1) % n]
        for k in range(seg):
            bm.faces.new([r0[k], r0[(k + 1) % seg], r1[(k + 1) % seg], r1[k]])
    if not closed:
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return finish(bm, name, material, smooth=True, sharp=80)

def lathe(name, profile, x, sz, material, seg=48):
    """Revolve a (radius, axial offset) profile round a wheel axis at (x, axle height), along z * sz."""
    bm = bmesh.new()
    rings = []
    for r, w in profile:
        rings.append([bm.verts.new((x + r * math.cos(2 * math.pi * k / seg), WHEEL_R + r * math.sin(2 * math.pi * k / seg), sz * w))
                      for k in range(seg)])
    for i in range(len(rings) - 1):
        for k in range(seg):
            try:
                bm.faces.new([rings[i][k], rings[i][(k + 1) % seg], rings[i + 1][(k + 1) % seg], rings[i + 1][k]])
            except ValueError:
                pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return finish(bm, name, material, smooth=True, sharp=35)

def text_mesh(name, s, size, center, bx, by, material, extrude=0.004, font=None):
    cu = bpy.data.curves.new(name, 'FONT')
    cu.body = s
    cu.size = size
    cu.extrude = extrude
    cu.align_x = 'CENTER'
    cu.align_y = 'CENTER'
    cu.resolution_u = 6
    if font:
        cu.font = font
    ob = bpy.data.objects.new(name + '_tmp', cu)
    COL.objects.link(ob)
    bpy.context.view_layer.update()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(ob)
    bm = bmesh.new()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)
    bz = bx.cross(by)
    c = Vector(center)
    m = Matrix(((bx.x, by.x, bz.x, c.x), (bx.y, by.y, bz.y, c.y), (bx.z, by.z, bz.z, c.z), (0, 0, 0, 1)))
    bmesh.ops.transform(bm, matrix=m, verts=bm.verts)
    return finish(bm, name, material, smooth=False)

# ============================================================================== surface grid builder
def out_dir(c):
    return c - Vector((max(-5.5, min(5.5, c.x)), 1.84, 0.0))

BMS = {}
def acc(m):
    if m not in BMS:
        BMS[m] = bmesh.new()
    return BMS[m]

def flush(prefix, smooth=True, sharp=55):
    global BMS
    objs = {}
    for m, bm in BMS.items():
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        bmesh.ops.dissolve_degenerate(bm, dist=1e-7, edges=bm.edges)
        objs[m] = finish(bm, f'{prefix}.{m}', m, smooth=smooth, sharp=sharp)
    BMS = {}
    return objs

def add_grid(bm, f, A, B, off=0.0, inward=False, orient=None, uvf=None):
    """Quads over the parameter grid A x B of surface f, optionally pushed along the surface normal.
    uvf(a, b) gives each vertex its texture coordinate."""
    uvl = bm.loops.layers.uv.verify() if uvf else None
    rows, uvs = [], {}
    for a in A:
        row = []
        for b in B:
            p = f(a, b)
            if off:
                p = p + normal(f, a, b) * off
            v = bm.verts.new(p)
            if uvf:
                uvs[v] = uvf(a, b)
            row.append(v)
        rows.append(row)
    for i in range(len(A) - 1):
        for j in range(len(B) - 1):
            try:
                fc = bm.faces.new([rows[i][j], rows[i + 1][j], rows[i + 1][j + 1], rows[i][j + 1]])
            except ValueError:
                continue
            fc.normal_update()
            c = fc.calc_center_median()
            want = orient(c) if orient else out_dir(c)
            if (fc.normal.dot(want) < 0) != inward:
                fc.normal_flip()
            if uvf:
                for l in fc.loops:
                    l[uvl].uv = uvs[l.vert]

def lin(a, b, n):
    return [a + (b - a) * i / n for i in range(n + 1)]

def breaks(lo, hi, cuts, step):
    pts = sorted(set([lo, hi] + [c for c in cuts if lo < c < hi]))
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, math.ceil((b - a) / step((a + b) / 2) - 1e-9))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return out

def ystep(y):
    return 0.025 if y < 0.80 else 0.03 if y > 2.58 else 0.08

def inside(opens, ca, cb):
    return any(o[0] < ca < o[1] and o[2] < cb < o[3] for o in opens)

def cells(f, A_lo, A_hi, B_lo, B_hi, opens, astep, bstep, matf, off=0.0, inward=False, extra_b=(), uvf=None, groove=()):
    """Surface cells between the openings; cells inside a groove rectangle are sunk by GROOVE_D."""
    A = breaks(A_lo, A_hi, [v for o in list(opens) + list(groove) for v in o[:2]], lambda m: astep)
    B = breaks(B_lo, B_hi, [v for o in list(opens) + list(groove) for v in o[2:4]] + list(extra_b), bstep)
    for i in range(len(A) - 1):
        for j in range(len(B) - 1):
            ca, cb = (A[i] + A[i + 1]) / 2, (B[j] + B[j + 1]) / 2
            if inside(opens, ca, cb):
                continue
            o = off - (GROOVE_D if inside(groove, ca, cb) else 0.0)
            add_grid(acc(matf(cb)), f, [A[i], A[i + 1]], [B[j], B[j + 1]], o, inward, uvf=uvf)

def reveal(f, rect, material, edges='btlr', depth=WALL, astep=0.25, outward=False, uvf=None, start=0.0):
    """The wall thickness inside an opening, from `start` behind the skin to `depth`."""
    a0, a1, b0, b1 = rect
    cen = f((a0 + a1) / 2, (b0 + b1) / 2)
    bm = acc(material)
    def g(pt):
        return lambda a, d: f(*pt(a)) - normal(f, *pt(a)) * (start + (depth - start) * d)
    def gu(pt):
        return (lambda a, d: uvf(*pt(a))) if uvf else None
    orient = (lambda c: c - cen) if outward else (lambda c: cen - c)
    for e, pt, lo, hi, st in (('b', lambda a: (a, b0), a0, a1, astep), ('t', lambda a: (a, b1), a0, a1, astep),
                              ('l', lambda b: (a0, b), b0, b1, 0.12), ('r', lambda b: (a1, b), b0, b1, 0.12)):
        if e in edges:
            add_grid(bm, g(pt), breaks(lo, hi, [], lambda m: st), [0, 1], orient=orient, uvf=gu(pt))

def pane(f, rect, off=-0.014, thick=0.006, astep=0.25, material='glass'):
    """Glass with real thickness, recessed behind the gasket."""
    a0, a1, b0, b1 = rect
    A, B = breaks(a0, a1, [], lambda m: astep), breaks(b0, b1, [], lambda m: 0.15)
    add_grid(acc(material), f, A, B, off)
    add_grid(acc(material), f, A, B, off - thick, inward=True)

# gasket section (offset from the opening edge, height over the skin): an inner lip over the glass,
# then a rounded bead lying on the body
GASKET = [(-0.026, -0.011), (-0.008, -0.006)] + [(0.004 + 0.011 * math.cos(math.pi * (1 - k / 4)), 0.0045 * math.sin(math.pi * k / 4)) for k in range(5)]
def gasket(f, rect, material='rubber', edges=None, rc=0.035, ka=1.0, step=0.08, profile=GASKET):
    """Rubber window gasket with a rounded bead, following a rounded rectangle round the opening.
    ka: metres per unit of the first parameter (1 on the sides, ~1.3 for the s parameter of the masks).
    edges: None for a closed rounded loop, or a string of straight edges ('b', 't', 'l', 'r')."""
    a0, a1, b0, b1 = rect
    bm = acc(material)
    def build(path, closed):
        rows = []
        for a, b, da, db in path:
            q = f(a, b)
            n = normal(f, a, b)
            eps = 1e-3
            d3 = f(a + da * eps / ka, b + db * eps) - q
            d3 = (d3 - n * d3.dot(n)).normalized()
            rows.append([bm.verts.new(q + d3 * o + n * h) for o, h in profile])
        m = len(rows)
        for i in range(m - 1 + (1 if closed else 0)):
            r0, r1 = rows[i], rows[(i + 1) % m]
            want = normal(f, *path[i][:2])
            for k in range(len(profile) - 1):
                fc = bm.faces.new([r0[k], r1[k], r1[k + 1], r0[k + 1]])
                fc.normal_update()
                if fc.normal.dot(want) < 0:
                    fc.normal_flip()
    if edges is None:
        A0, A1 = a0 * ka, a1 * ka
        segs = [((A0 + rc, b0), (A1 - rc, b0), (0, -1)), ((A1, b0 + rc), (A1, b1 - rc), (1, 0)),
                ((A1 - rc, b1), (A0 + rc, b1), (0, 1)), ((A0, b1 - rc), (A0, b0 + rc), (-1, 0))]
        corners = [(A1 - rc, b0 + rc, -90), (A1 - rc, b1 - rc, 0), (A0 + rc, b1 - rc, 90), (A0 + rc, b0 + rc, 180)]
        path = []
        for (p, q, d), (cx, cy, ang0) in zip(segs, corners):
            L = math.hypot(q[0] - p[0], q[1] - p[1])
            n = max(1, int(L / step))
            for k in range(n):
                t = k / n
                path.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, d[0], d[1]))
            for k in range(5):
                a = math.radians(ang0 + 90 * k / 5)
                path.append((cx + rc * math.cos(a), cy + rc * math.sin(a), math.cos(a), math.sin(a)))
        build([(a / ka, b, da, db) for a, b, da, db in path], True)
    else:
        for e in edges:
            if e in 'bt':
                b = b0 if e == 'b' else b1
                build([(a, b, 0, -1 if e == 'b' else 1) for a in breaks(a0, a1, [], lambda m: step / ka)], False)
            else:
                a = a0 if e == 'l' else a1
                build([(a, b, -1 if e == 'l' else 1, 0) for b in breaks(b0, b1, [], lambda m: step)], False)

# ============================================================================== body (vozová skříň)
BODY = part('body', explode=(0, 1.3, 0))
X_MID = breaks(-XS, XS, [], lambda m: 0.15)
for side in (1, -1):
    f = side_pt(side)
    uvf = uv_side(side)
    G = grooves(side)
    cells(f, -XS, XS, Y_SIDE0, SIDE_TOP, OPEN[side], 0.15, ystep, lambda y: 'paint', extra_b=(SKIRT, BELT), uvf=uvf, groove=G)
    for g in G:                                       # walls of the panel-joint grooves
        reveal(f, g, 'paint', edges='lr', depth=GROOVE_D, uvf=uvf, astep=0.1)
    add_grid(acc('under'), lambda x, u, s=side: surf(x, u, s), X_MID, lin(0.0, u_of_y(Y_SIDE0), 3))
    add_grid(acc('paint'), lambda x, u, s=side: surf(x, u, s), X_MID, lin(u_of_y(SIDE_TOP), 1.0, 22), uvf=uv_side_u(side))
    # raised chrome trim strip over the skirt, interrupted by the doors
    gaps = [(x0, x1) for _, x0, x1 in DOOR_SPANS] if side == 1 else []
    stops = sorted([-XS, XS] + [v for g_ in gaps for v in g_])
    for i in range(0, len(stops), 2):
        X = breaks(stops[i], stops[i + 1], [], lambda m: 0.3)
        bm = acc('chrome')
        add_grid(bm, f, X, [SKIRT - 0.008, SKIRT - 0.004], 0.0015)
        add_grid(bm, f, X, [SKIRT - 0.004, SKIRT + 0.010], 0.0035)
        add_grid(bm, f, X, [SKIRT + 0.010, SKIRT + 0.014], 0.0015)
    for rect in (WIN_R if side == 1 else WIN_L):
        reveal(f, rect, 'rubber', start=0.006)
        gasket(f, rect)
    # sash bars of the windows: horizontal under the vents, vertical between the two sliding vent panes
    for (a0, a1, b0, b1) in (WIN_R if side == 1 else WIN_L):
        X = breaks(a0, a1, [], lambda m: 0.3)
        xm = a0 + 0.47 * (a1 - a0)
        for o, inw in ((-0.019, False), (-0.034, True)):
            add_grid(acc('lining'), f, X, [WIN_SPLIT - 0.016, WIN_SPLIT + 0.016], o, inward=inw)
            add_grid(acc('lining'), f, [xm - 0.016, xm + 0.016], breaks(WIN_SPLIT + 0.016, b1, [], lambda m: 0.2), o, inward=inw)
DOOR_BEAD = [(-0.004, -0.002)] + [(0.012 + 0.016 * math.cos(math.pi * (1 - k / 6)), 0.004 * math.sin(math.pi * k / 6)) for k in range(7)]
for pid, x0, x1 in DOOR_SPANS:
    f = side_pt(1)
    reveal(f, (x0, x1, Y_SIDE0, 2.56), 'maroon', edges='tlr')
    gasket(f, (x0, x1, Y_SIDE0, 2.56), material='maroon', edges='tlr', profile=DOOR_BEAD)
flush('body')
for pid, x0, x1 in DOOR_SPANS:                        # grab handle on the body beside each doorway
    xh = x1 + 0.07 if x1 + 0.07 < XS - 0.03 else x0 - 0.07
    a = surf(xh, u_of_y(0.98), 1)
    b = surf(xh, u_of_y(1.52), 1)
    nn = Vector((0, 0, 1))
    rod(f'{pid}.handle', [a, a + nn * 0.045, b + nn * 0.045, b], 0.011, 'chrome')
    # brown step plate under the doorway (coloured elevation)
    box(f'{pid}.step_plate', ((x0 + x1) / 2, 0.315, 1.215), (x1 - x0 + 0.04, 0.035, 0.07), 'brown', bevel=0.004)
for side in (1, -1):                                  # fleet number in gold on both sides (coloured elevation)
    p = surf(2.28, u_of_y(1.32), side) + Vector((0, 0, side * 0.0012))
    text_mesh(f'side_number.{side}', NUMBER, 0.13, p, Vector((side, 0, 0)), Vector((0, 1, 0)), 'gold',
              extrude=0.0006, font=FONT_NUM)
    # amber side lamp on the window pillar nearest x = 3.0
    wins = sorted(WIN_R if side == 1 else WIN_L)
    gaps = [((wins[i][1] + wins[i + 1][0]) / 2) for i in range(len(wins) - 1) if wins[i + 1][0] - wins[i][1] < 0.3]
    xl = min(gaps, key=lambda g: abs(g - 3.0))
    p = surf(xl, u_of_y(2.24), side)
    n = Vector((0, 0, side))
    bm = bmesh.new()
    cyl_into(bm, p - n * 0.003, p + n * 0.018, 0.028, 20, bevel=0.006)
    bmesh.ops.scale(bm, vec=Vector((0.8, 1.35, 1.0)), space=Matrix.Translation(-p), verts=bm.verts)
    finish(bm, f'side_lamp.{side}', 'amber', sharp=50)
# louvred grille in the red band on the left side (coloured elevation): cooling for the equipment
# under the floor
bm_back, bm_slat = bmesh.new(), bmesh.new()
for k in range(3):
    gx0 = -1.37 + k * 0.445
    gx1 = gx0 + 0.41
    zc = -(1.25 + 0.006)
    box_into(bm_back, ((gx0 + gx1) / 2, 1.06, zc), (gx1 - gx0, 0.46, 0.008), 0.003)
    for j in range(12):
        yj = 0.86 + j * 0.036
        box_into(bm_slat, ((gx0 + gx1) / 2, yj, zc - 0.009), (gx1 - gx0 - 0.03, 0.012, 0.016), 0.003,
                 rot=Matrix.Rotation(math.radians(35), 3, 'X'))
finish(bm_back, 'grille.back', 'under', sharp=50)
fixed_uv(finish(bm_slat, 'grille.slats', 'paint', sharp=50), (0.5, 0.40 + 0.30 * 0.40))

# ============================================================================== glass (one object)
part('glass', explode=(0, 1.3, 0))
for side in (1, -1):
    f = side_pt(side)
    for (a0, a1, b0, b1) in (WIN_R if side == 1 else WIN_L):
        pane(f, (a0 - 0.012, a1 + 0.012, b0 - 0.012, WIN_SPLIT - 0.012))
        xm = a0 + 0.47 * (a1 - a0)                    # two sliding vent panes on separate tracks
        pane(f, (a0 - 0.012, xm + 0.02, WIN_SPLIT + 0.012, b1 + 0.012), off=-0.026)
        pane(f, (xm - 0.02, a1 + 0.012, WIN_SPLIT + 0.012, b1 + 0.012), off=-0.036)
fz, rz = nose_pt(FRONT), nose_pt(REAR)
for E, holes, fn in ((FRONT, FRONT_HOLES, fz), (REAR, REAR_HOLES, rz)):
    for (a0, a1, b0, b1) in holes:
        pane(fn, (a0 - 0.01, a1 + 0.01, b0 - 0.012, b1 + 0.012), astep=0.03)
flush('glass')

# ============================================================================== end masks (laminate)
def eased(E, n=24):
    return [E.sign * (E.xs + (E.xt - E.xs) * (1 - (1 - i / n) ** 2)) for i in range(n + 1)]

def mask(E, holes):
    f = nose_pt(E)
    cells(f, -1.0, 1.0, Y_SIDE0, H_ROOF, holes, 0.04,
          lambda y: 0.025 if y < 0.80 else 0.015 if y > 2.85 else 0.03 if y > 2.45 else 0.07,
          lambda y: 'paint', extra_b=(SKIRT, BELT, SIDE_TOP), uvf=uv_nose(E))
    for side in (1, -1):
        add_grid(acc('under'), lambda x, u, s=side: surf(x, u, s), eased(E), lin(0.0, u_of_y(Y_SIDE0), 3))
    S = breaks(-1.0, 1.0, [], lambda m: 0.04)
    bm = acc('chrome')
    add_grid(bm, f, S, [SKIRT - 0.008, SKIRT - 0.004], 0.0015)
    add_grid(bm, f, S, [SKIRT - 0.004, SKIRT + 0.010], 0.0035)
    add_grid(bm, f, S, [SKIRT + 0.010, SKIRT + 0.014], 0.0015)
    for h in holes:
        reveal(f, h, 'rubber', astep=0.03, start=0.006)
        gasket(f, h, ka=1.25, rc=0.07 if h[1] - h[0] > 1 else 0.035)

part('front_mask', explode=(1.0, 1.3, 0))
mask(FRONT, FRONT_HOLES)
flush('front_mask')

# route box on the roof dome over the windscreen
_dc = nose(FRONT, 0.0, 2.70)
_tilt = normal(fz, 0.0, 2.70)
_up = (Vector((0, 1, 0)) - _tilt * _tilt.y).normalized()
_bx = Vector((0, 0, -1))
_RT = Matrix(((_bx.x, _up.x, _tilt.x), (_bx.y, _up.y, _tilt.y), (_bx.z, _up.z, _tilt.z)))
def tilted_box(name, w, h, d, push, material, bevel=0.008):
    bm = bmesh.new()
    box_into(bm, (0, 0, 0), (w, h, d), bevel, rot=_RT)
    bmesh.ops.translate(bm, vec=_dc + _tilt * push, verts=bm.verts)
    return finish(bm, name, material, smooth=True, sharp=50)
tilted_box('route_box.frame', 1.10, 0.20, 0.10, 0.0, 'maroon', bevel=0.012)
tilted_box('route_box.blind', 1.04, 0.15, 0.02, 0.045, 'darkmetal', bevel=0.002)
tilted_box('route_box.glass', 1.05, 0.16, 0.004, 0.058, 'glass', bevel=0.001)
text_mesh('route_box.text', '4  MOTOL', 0.095, _dc + _tilt * 0.0555, _bx, _up, 'signtext', extrude=0.0005, font=FONT_SIGN)

for s in (1, -1):                                     # wipers: spindle, arm and rubber blade
    n0 = normal(fz, s * 0.10, 2.44)
    sp = nose_s(FRONT, s * 0.10, 2.44) + n0 * 0.02
    n1 = normal(fz, s * 0.42, 1.80)
    tip = nose_s(FRONT, s * 0.42, 1.80) + n1 * 0.022
    cyl(f'wiper.spindle.{s}', sp - n0 * 0.02, sp + n0 * 0.012, 0.014, 'darkmetal', seg=12, bevel=0.002)
    rod(f'wiper.arm.{s}', [sp, tip], 0.006, 'darkmetal', seg=8)
    d = (tip - sp).normalized()
    bm = bmesh.new()
    box_into(bm, (0, 0, 0), (0.012, 0.42, 0.010), 0.002, rot=Vector((0, 1, 0)).rotation_difference(d).to_matrix())
    bmesh.ops.translate(bm, vec=tip - d * 0.2 + n1 * 0.005, verts=bm.verts)
    finish(bm, f'wiper.blade.{s}', 'rubber', sharp=50)

def bumper(name, E, y=0.69, half=1.20, out=0.035, th=0.05, h=0.08, n=72, material='chrome', r=0.012):
    """Bumper bar with a rounded section, following the mask."""
    bm = bmesh.new()
    f = nose_pt(E)
    zb = prof(u_of_y(y))[0] * taper(y)
    sec = []
    for (cx, cy, a0) in ((th / 2 - r, h / 2 - r, 0), (-(th / 2 - r), h / 2 - r, 90), (-(th / 2 - r), -(h / 2 - r), 180), (th / 2 - r, -(h / 2 - r), 270)):
        for k in range(4):
            a = math.radians(a0 + 30 * k)
            sec.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    rings = []
    for i in range(n + 1):
        z = -half + 2 * half * i / n
        s = z / zb
        p = f(s, y)
        nn = normal(f, max(-0.999, min(0.999, s)), y)
        nn.y = 0
        nn.normalize()
        up = Vector((0, 1, 0))
        rings.append([bm.verts.new(p + nn * (out + dx) + up * dy) for dx, dy in sec])
    m = len(sec)
    for i in range(n):
        for k in range(m):
            bm.faces.new([rings[i][k], rings[i + 1][k], rings[i + 1][(k + 1) % m], rings[i][(k + 1) % m]])
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return finish(bm, name, material, sharp=60)

bumper('bumper_front', FRONT)

for s in (1, -1):                                     # marker lamps on stalks at the cab corners
    p = nose(FRONT, s * 1.20, 1.78)
    n = (normal(fz, s * 0.93, 1.78) + Vector((0, 0, s * 0.8))).normalized()
    cyl(f'marker.stalk.{s}', p, p + n * 0.07, 0.012, 'chrome', seg=12)
    cyl(f'marker.body.{s}', p + n * 0.065, p + n * 0.09, 0.05, 'chrome', seg=24, bevel=0.004)
    cyl(f'marker.lens.{s}', p + n * 0.09, p + n * 0.108, 0.044, 'lamp', seg=24, r2=0.036)

FACE_X, FACE_Y = Vector((0, 0, -1)), Vector((0, 1, 0))
p = nose(FRONT, 0.0, 1.10)
text_mesh('fleet_number', NUMBER, 0.17, p + Vector((0.0012, 0, 0)), FACE_X, FACE_Y, 'gold', extrude=0.0008, font=FONT_NUM)
if '--tatra' in ARGS:
    p = nose(FRONT, 0.0, 0.92)
    text_mesh('tatra_lettering', 'TATRA', 0.11, p + Vector((0.004, 0, 0)), FACE_X, FACE_Y, 'chrome', extrude=0.006)

# horn under the front end, door side (ASSUMED position)
cyl('horn.bell', (6.35, 0.27, 0.55), (6.62, 0.27, 0.55), 0.022, 'chrome', seg=20, r2=0.055)
cyl('horn.body', (6.22, 0.27, 0.55), (6.35, 0.27, 0.55), 0.03, 'darkmetal', seg=16, bevel=0.004)

# ============================================================================== headlights, mirror
part('headlights', explode=(1.5, 1.3, 0))
for s in (1, -1):
    z, y = s * 0.66, 1.03                            # front elevation of the coloured drawing
    p = nose(FRONT, z, y)
    n = normal(fz, z / 1.2, y)
    fixed_uv(cyl(f'headlamp.body.{s}', p - n * 0.02, p + n * 0.06, 0.12, 'paint', seg=40, bevel=0.01), (0.5, 0.82))
    cyl(f'headlamp.rim.{s}', p + n * 0.058, p + n * 0.076, 0.118, 'chrome', seg=40, bevel=0.006)
    cyl(f'headlamp.lens.{s}', p + n * 0.076, p + n * 0.094, 0.10, 'lens', seg=40, r2=0.077)

part('mirror', explode=(1.0, 1.3, 0.6))
a = surf(5.97, u_of_y(1.90), 1)
b = a + Vector((0.10, 0.05, 0.28))
rod('mirror.arm', [a, a + Vector((0.04, 0.0, 0.14)), b], 0.012, 'chrome')
box('mirror.head', b + Vector((0.0, 0.0, 0.03)), (0.04, 0.24, 0.15), 'darkmetal', bevel=0.015)
box('mirror.glass', b + Vector((-0.021, 0.0, 0.03)), (0.003, 0.21, 0.125), 'mirror', bevel=0.001)

# ============================================================================== rear mask
part('rear_mask', explode=(-1.0, 1.3, 0))
mask(REAR, REAR_HOLES)
flush('rear_mask')
bumper('bumper_rear', REAR)
for s in (1, -1):                                     # paired tail and indicator lamps (coloured rear elevation)
    for z, mt in ((s * 0.36, 'redlens'), (s * 0.48, 'amber')):
        p = nose(REAR, z, 1.00)
        n = normal(rz, z / 1.2, 1.00)
        bm = bmesh.new()
        cyl_into(bm, p - n * 0.01, p + n * 0.025, 0.04, 24, bevel=0.006)
        bmesh.ops.scale(bm, vec=Vector((1.0, 0.7, 1.6)), space=Matrix.Translation(-p), verts=bm.verts)
        finish(bm, f'rearlamp.{s}.{mt}', mt, sharp=50)
p = nose(REAR, 0.0, 1.13)
text_mesh('rear_number', NUMBER, 0.15, p - Vector((0.0012, 0, 0)), Vector((0, 0, 1)), FACE_Y, 'gold', extrude=0.0008, font=FONT_NUM)

# ============================================================================== doors (four-leaf folding)
LEAF_T = 0.035
def door(pid, x0, x1):
    e = part(pid, explode=(0, 1.3, 0.8), nomerge=1)
    f = side_pt(1)
    def leaf_pt(depth):
        def g(x, y):
            yy = max(y, 0.80)
            p = f(x, yy)
            n = normal(f, x, yy)
            if y < 0.80:
                p = Vector((p.x, y, p.z))
            return p + n * (0.009 - depth)
        return g
    gap = 0.012
    w = (x1 - x0 - 3 * gap) / 4
    leaves = []
    for i in range(4):
        a, b = x0 + i * (w + gap), x0 + i * (w + gap) + w
        uvf = lambda x, y, a=a: ((x - a) / 0.32, (y - 0.33) / 2.21)
        glass = [(a + 0.055, b - 0.055, 0.95, 1.55), (a + 0.055, b - 0.055, 1.62, 2.44)]
        front, back = leaf_pt(0.0), leaf_pt(LEAF_T)
        cells(front, a, b, 0.335, 2.54, glass, 0.4, lambda y: 0.2, lambda y: 'door', uvf=uvf)
        cells(back, a, b, 0.335, 2.54, glass, 0.4, lambda y: 0.2, lambda y: 'door', inward=True, uvf=uvf)
        for rect in glass:
            G0 = [rect[0] - 0.01, rect[1] + 0.01]
            G1 = breaks(rect[2] - 0.01, rect[3] + 0.01, [], lambda m: 0.3)
            add_grid(acc('glass'), leaf_pt(LEAF_T / 2 - 0.003), G0, G1)
            add_grid(acc('glass'), leaf_pt(LEAF_T / 2 + 0.003), G0, G1, inward=True)
            reveal(front, rect, 'door', depth=LEAF_T, uvf=uvf)
            gasket(front, rect, rc=0.02, step=0.10)
        reveal(front, (a, b, 0.335, 2.54), 'door', depth=LEAF_T, outward=True, uvf=uvf)
        hx = a if i in (0, 1) else b                # hinge knuckles on this leaf's hinge edge
        zf = f(hx, 1.5).z + 0.009
        for yk in (0.75, 1.55, 2.35):
            cyl_into(acc('steel'), (hx, yk - 0.035, zf + 0.004), (hx, yk + 0.035, zf + 0.004), 0.011, 12, bevel=0.002)
        if i in (1, 2):                              # rubber edge seal where the pairs meet
            ex = b if i == 1 else a
            box_into(acc('rubber'), (ex + (0.004 if i == 1 else -0.004), 1.44, zf - LEAF_T / 2), (0.012, 2.19, 0.03), 0.004)
        objs = flush(f'{pid}.leaf{i}')
        leaves.append((objs, a, b))
    zh = f(x0, 1.5).z + 0.009 - LEAF_T / 2
    hinge = [x0, x0 + w + gap / 2, x1 - w - gap / 2, x1]
    parents = [None, 0, 3, None]
    anchors = {}
    for i in (0, 3, 1, 2):
        objs, a, b = leaves[i]
        pv = (hinge[i], 0.335, zh)
        main = objs['door']
        if parents[i] is None:
            rehome(main, pv, e)
        else:
            rehome(main, pv, anchors[parents[i]][0], anchors[parents[i]][1])
        main.name = f'{pid}.leaf{i}'
        main['hinge'] = 'jamb' if parents[i] is None else 'fold'
        anchors[i] = (main, pv)
        for m, o in objs.items():
            if o is not main:
                rehome(o, pv, main, pv)
                o.name = f'{pid}.leaf{i}.{m}'
    xm = (x0 + x1) / 2                               # step well: two ribbed treads with steel nosings
    box(f'{pid}.tread.upper', (xm, 0.64, 1.00), (x1 - x0, 0.04, 0.26), 'floor', bevel=0.004)
    box(f'{pid}.riser.upper', (xm, 0.77, 0.875), (x1 - x0, 0.26, 0.02), 'under', bevel=0.003)
    box(f'{pid}.tread.lower', (xm, 0.37, 1.14), (x1 - x0, 0.04, 0.22), 'floor', bevel=0.004)
    box(f'{pid}.riser.lower', (xm, 0.495, 1.035), (x1 - x0, 0.25, 0.02), 'under', bevel=0.003)
    for yy, zz in ((0.662, 1.125), (0.392, 1.245)):
        box(f'{pid}.nosing.{yy}', (xm, yy, zz), (x1 - x0, 0.012, 0.03), 'steel', bevel=0.003)
    for xx in (x0 + 0.06, x1 - 0.06):
        cyl(f'{pid}.pole.{xx:.2f}', (xx, FLOOR, 0.86), (xx, 2.54, 0.86), 0.016, 'chrome', seg=14)

for pid, x0, x1 in DOOR_SPANS:
    door(pid, x0, x1)

# ============================================================================== interior
part('interior', explode=(0, 1.3, 0))
def lining_mat(y):
    return 'umakart' if y < WIN_Y0 else 'lining' if y < SIDE_TOP else 'ceiling'
for side in (1, -1):
    f = side_pt(side)
    cells(f, -XS, XS, FLOOR, SIDE_TOP, OPEN[side], 0.30, ystep, lining_mat, off=-WALL, inward=True, extra_b=(WIN_Y0,))
    add_grid(acc('ceiling'), lambda x, u, s=side: surf(x, u, s), breaks(-XS, XS, [], lambda m: 0.3), lin(u_of_y(SIDE_TOP), 1.0, 16), -WALL, inward=True)
for E, holes in ((FRONT, FRONT_HOLES), (REAR, REAR_HOLES)):
    cells(nose_pt(E), -1.0, 1.0, FLOOR, H_ROOF, holes, 0.08, lambda y: 0.03 if y > 2.70 else 0.12,
          lining_mat, off=-WALL, inward=True, extra_b=(WIN_Y0, SIDE_TOP))
INNER = 1.25 - WALL
pts = [(-XS, INNER)]
for _, x0, x1 in sorted(DOOR_SPANS, key=lambda d: d[1]):
    pts += [(x0, INNER), (x0, 0.865), (x1, 0.865), (x1, INNER)]
pts += [(XS, INNER)]
zb_f = prof(u_of_y(FLOOR))[0] * taper(FLOOR)
for k in range(1, 24):
    s = 1 - 2 * k / 24
    q = nose_s(FRONT, s * (zb_f - WALL) / zb_f, FLOOR)
    pts.append((q.x - WALL, q.z))
pts += [(XS, -INNER), (-XS, -INNER)]
for k in range(1, 24):
    s = -1 + 2 * k / 24
    q = nose_s(REAR, s * (zb_f - WALL) / zb_f, FLOOR)
    pts.append((q.x + WALL, q.z))
bm = bmesh.new()
vs = [bm.verts.new((x, FLOOR, z)) for x, z in pts]
es = [bm.edges.new((vs[i], vs[(i + 1) % len(vs)])) for i in range(len(vs))]
bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=False, edges=es)
for fc in bm.faces:
    fc.normal_update()
    if fc.normal.y < 0:
        fc.normal_flip()
finish(bm, 'floor', 'floor', smooth=False)

# 25 seats (Wikipedia): doubles on the left, singles on the right, alternating red and grey
# leatherette on chrome tube frames; the heaters sit in the seat pedestals (Wikipedia).
SEAT_Y = FLOOR + 0.42
def seat(name, x, zc, width, material):
    box(f'{name}.cushion', (x, SEAT_Y, zc), (0.44, 0.09, width), material, bevel=0.03)
    box(f'{name}.back', (x - 0.235, SEAT_Y + 0.30, zc), (0.07, 0.50, width), material, bevel=0.03, rot=rot_z(-8))
    box(f'{name}.heater', (x - 0.03, FLOOR + 0.19, zc), (0.30, 0.36, width - 0.10), 'darkmetal', bevel=0.01)
    for dz in (-width / 2 + 0.03, width / 2 - 0.03):
        rod(f'{name}.frame.{dz:.2f}', [(x + 0.2, SEAT_Y - 0.05, zc + dz), (x - 0.25, SEAT_Y - 0.05, zc + dz),
                                       (x - 0.28, SEAT_Y + 0.56, zc + dz)], 0.011, 'chrome', seg=10)
    cyl(f'{name}.grab', (x - 0.28, SEAT_Y + 0.56, zc - width / 2 + 0.03), (x - 0.28, SEAT_Y + 0.56, zc + width / 2 - 0.03), 0.011, 'chrome', seg=10)
# Seats as drawn in the plan of the type drawing: 13 singles along the left side, double seats on the
# door side (3 between the rear and middle door, 5 between the middle and front door) and a single seat
# beside the front and rear doors. (Wikipedia gives 25 seats for the Prague cars; the plan shows more.)
for i, x in enumerate((-4.39, -3.69, -2.99, -2.25, -1.55, -0.85, -0.15, 0.59, 1.29, 2.01, 2.71, 3.45, 4.19)):
    seat(f'seat.L{i}', x, -0.86, 0.44, ('leather_red', 'leather_grey')[i % 2])
for i, x in enumerate((-3.64, -2.92, -2.14, 0.33, 1.03, 1.73, 2.43, 3.13)):
    seat(f'seat.D{i}', x, 0.66, 0.86, ('leather_grey', 'leather_red')[i % 2])
for i, x in enumerate((-4.40, 4.06)):
    seat(f'seat.S{i}', x, 0.90, 0.44, 'leather_red')
for zz in (-0.42, 0.42):
    cyl(f'handrail.{zz}', (-5.9, 2.28, zz), (5.45, 2.28, zz), 0.014, 'chrome', seg=12)
    for x in (-5.6, -3.9, -2.2, -0.5, 1.2, 2.9, 4.6):
        cyl(f'handrail.drop.{zz}.{x}', (x, 2.28, zz), (x, 2.93, zz), 0.010, 'chrome', seg=8)
for x in (3.6, 1.0, -2.5):
    cyl(f'stanchion.{x}', (x, FLOOR, 0.22), (x, 2.28, 0.42), 0.016, 'chrome', seg=14)
for zz in (-0.50, 0.50):
    for x in (-5.0, -3.2, -1.4, 0.4, 2.2, 4.0):
        box(f'lampcase.{zz}.{x}', (x, 2.935, zz), (1.24, 0.03, 0.12), 'lamp', bevel=0.006)
        box(f'tube.{zz}.{x}', (x, 2.915, zz), (1.18, 0.012, 0.09), 'tube', bevel=0.004)
CZ = -0.45
box('cab.desk', (6.40, FLOOR + 0.33, CZ), (0.40, 0.66, 0.95), 'cab', bevel=0.02)
box('cab.desk_top', (6.34, FLOOR + 0.70, CZ), (0.44, 0.06, 0.99), 'cab', bevel=0.015, rot=rot_z(-12))
cyl('cab.controller', (6.32, FLOOR + 0.73, CZ + 0.28), (6.32, FLOOR + 0.80, CZ + 0.28), 0.06, 'darkmetal', seg=24, bevel=0.006)
cyl('cab.controller_handle', (6.32, FLOOR + 0.80, CZ + 0.28), (6.27, FLOOR + 0.86, CZ + 0.34), 0.012, 'chrome', seg=10)
for i, dz in enumerate((-0.30, -0.14, 0.02)):
    box(f'cab.dial.{i}', (6.46, FLOOR + 0.82, CZ + dz), (0.02, 0.10, 0.10), 'darkmetal', bevel=0.004)
box('cab.seat.cushion', (5.86, FLOOR + 0.50, CZ), (0.46, 0.09, 0.48), 'leather_grey', bevel=0.03)
box('cab.seat.back', (5.60, FLOOR + 0.82, CZ), (0.07, 0.56, 0.46), 'leather_grey', bevel=0.03)
cyl('cab.seat.post', (5.86, FLOOR, CZ), (5.86, FLOOR + 0.46, CZ), 0.04, 'darkmetal', seg=16)
for i, dz in enumerate((-0.20, 0.0, 0.20)):          # the T3 is driven with three pedals
    box(f'cab.pedal.{i}', (6.16, FLOOR + 0.05, CZ + dz), (0.22, 0.03, 0.10), 'rubber', bevel=0.006, rot=rot_z(-15))
box('cab.partition', (5.45, FLOOR + 0.55, -0.30), (0.03, 1.10, 1.78), 'lining', bevel=0.005,
    rot=Matrix.Rotation(math.radians(-8), 3, 'Y'))
flush('interior')

# ============================================================================== roof
part('roof_vents', explode=(0, 1.9, 0))
for i, (x0, x1) in enumerate(((-6.04, -4.94), (-1.34, -0.24), (4.40, 5.56))):     # low roof vents (type drawing)
    box(f'vent.{i}', ((x0 + x1) / 2, H_ROOF + 0.025, 0), (x1 - x0, 0.06, 0.62), 'roofgrey', bevel=0.02)
    bm = bmesh.new()
    for k in range(int((x1 - x0) / 0.09)):
        box_into(bm, (x0 + 0.06 + k * 0.09, H_ROOF + 0.03, 0), (0.03, 0.05, 0.64), 0.004)
    finish(bm, f'vent.{i}.slats', 'darkmetal', sharp=50)

part('roof_equipment', explode=(0, 2.1, 0))
box('roof_hatch', (0.22, H_ROOF + 0.06, 0), (0.52, 0.12, 0.58), 'roofgrey', bevel=0.025)          # centre hatch
box('roof_hatch.lid', (0.22, H_ROOF + 0.125, 0), (0.56, 0.02, 0.62), 'roofgrey', bevel=0.008)
for zz in (-0.45, 0.45):                              # walkway beside the pantograph (coloured elevation)
    box(f'walkway.{zz}', (1.78, H_ROOF + 0.006, zz), (1.58, 0.02, 0.24), 'mat_roof', bevel=0.004)
cyl('roof_mast', (5.72, H_ROOF, 0), (5.72, H_ROOF + 0.16, 0), 0.012, 'roofgrey', seg=10)            # small mast near the front
cyl('roof_mast.bar', (5.72, H_ROOF + 0.16, -0.10), (5.72, H_ROOF + 0.16, 0.10), 0.010, 'roofgrey', seg=8)

part('pantograph', explode=(0, 2.6, 0))
# Large rhombic (diamond) pantograph as drawn in both elevations: two diamond frames side by side,
# bottom corners on a common base, top corners carrying the pan, X-bracing between the frames
# (front elevation). Shown partly raised.
X0 = 3.10                                             # over the front bogie (type drawing)
for sx in (-1, 1):
    for sz in (-1, 1):
        bm = bmesh.new()
        for k in range(4):
            yk = H_ROOF + 0.02 + k * 0.025
            cyl_into(bm, (X0 + sx * 0.42, yk, sz * 0.40), (X0 + sx * 0.42, yk + 0.02, sz * 0.40), 0.048 if k % 2 == 0 else 0.036, 20, bevel=0.004)
        finish(bm, f'insulator.{sx}{sz}', 'insulator', sharp=60)
box('base.frame', (X0, H_ROOF + 0.13, 0), (0.98, 0.04, 0.90), 'darkmetal', bevel=0.01)
Y_BASE = H_ROOF + 0.17
HGT, HW = 1.50, 1.12                                  # raised height and half span of the diamond (ASSUMED)
Y_MID, Y_TOP = Y_BASE + HGT / 2, Y_BASE + HGT
ZF = 0.42                                             # the two frames (front elevation)
for sz in (-1, 1):
    zz = sz * ZF
    rod(f'diamond.{sz}', [(X0, Y_BASE, zz), (X0 - HW, Y_MID, zz), (X0, Y_TOP, zz), (X0 + HW, Y_MID, zz), (X0, Y_BASE, zz)],
        0.020, 'panto', seg=12)
    cyl(f'hinge.base.{sz}', (X0, Y_BASE, zz - 0.04), (X0, Y_BASE, zz + 0.04), 0.03, 'darkmetal', seg=14, bevel=0.003)
    for sx in (-1, 1):
        cyl(f'hinge.knee.{sz}{sx}', (X0 + sx * HW, Y_MID, zz - 0.035), (X0 + sx * HW, Y_MID, zz + 0.035), 0.026, 'darkmetal', seg=14, bevel=0.003)
for sx in (-1, 1):                                    # cross tubes at the knees
    cyl(f'knee.cross.{sx}', (X0 + sx * HW, Y_MID, -ZF), (X0 + sx * HW, Y_MID, ZF), 0.014, 'panto', seg=10)
for yy0, yy1 in ((Y_BASE + 0.15, Y_MID - 0.1), (Y_MID + 0.1, Y_TOP - 0.15)):   # X-bracing between the frames
    rod(f'brace.{yy0:.2f}.a', [(X0, yy0, -ZF), (X0, yy1, ZF)], 0.009, 'panto', seg=8)
    rod(f'brace.{yy0:.2f}.b', [(X0, yy0, ZF), (X0, yy1, -ZF)], 0.009, 'panto', seg=8)
cyl('pan.tube', (X0, Y_TOP, -ZF - 0.05), (X0, Y_TOP, ZF + 0.05), 0.016, 'panto', seg=12)
for sx in (-1, 1):                                    # pan head: two bows with downturned horns, carbon strips
    xb = X0 + sx * 0.13
    cyl(f'pan.bow.{sx}', (xb, Y_TOP + 0.05, -0.70), (xb, Y_TOP + 0.05, 0.70), 0.016, 'panto', seg=12)
    box(f'pan.strip.{sx}', (xb, Y_TOP + 0.075, 0), (0.035, 0.02, 1.30), 'copper', bevel=0.004)
    for sz in (-1, 1):
        rod(f'pan.horn.{sx}{sz}', [(xb, Y_TOP + 0.05, sz * 0.70), (xb, Y_TOP + 0.02, sz * 0.84), (xb, Y_TOP - 0.08, sz * 0.92)], 0.015, 'panto')
    rod(f'pan.strut.{sx}', [(X0, Y_TOP, -0.25), (xb, Y_TOP + 0.05, -0.30)], 0.010, 'panto', seg=8)
    rod(f'pan.strut2.{sx}', [(X0, Y_TOP, 0.25), (xb, Y_TOP + 0.05, 0.30)], 0.010, 'panto', seg=8)
for sz in (-1, 1):
    cyl(f'pan.end.{sz}', (X0 - 0.13, Y_TOP + 0.05, sz * 0.70), (X0 + 0.13, Y_TOP + 0.05, sz * 0.70), 0.012, 'panto', seg=10)
for sx in (-1, 1):                                    # raising springs from the base to the lower arms
    a = Vector((X0 + sx * 0.45, Y_BASE - 0.01, 0.0))
    b = Vector((X0 + sx * 0.30, Y_BASE + 0.25, 0.0))
    d = (b - a)
    n = 140
    t0 = d.normalized()
    u0 = t0.orthogonal().normalized()
    v0 = t0.cross(u0)
    helix = [a + d * (k / n) + (u0 * math.cos(2 * math.pi * 14 * k / n) + v0 * math.sin(2 * math.pi * 14 * k / n)) * 0.025 for k in range(n + 1)]
    sweep(f'raising_spring.{sx}', helix, 0.004, 'steel', seg=6)

# ============================================================================== bogies and wheels
AXLE_Y = WHEEL_R
WZ = GAUGE / 2
B2B = 0.68                                            # wheel back face from the track centre (ASSUMED)
# wheel profile (radius, axial distance from the back face): flange, worn conical tread, chamfer
TREAD = [(0.300, 0.0), (0.360, 0.0), (0.372, 0.008), (0.372, 0.016), (0.362, 0.026), (0.352, 0.034),
         (0.350, 0.050), (0.346, 0.090), (0.343, 0.105), (0.335, 0.112), (0.300, 0.112)]
RIM = [(0.300, 0.112), (0.268, 0.112), (0.215, 0.124), (0.130, 0.140), (0.095, 0.150), (0.095, 0.178),
       (0.075, 0.178), (0.075, -0.02), (0.110, -0.02), (0.150, -0.008), (0.230, -0.004), (0.300, 0.0)]

def wheelset(pid, x, explode):
    part(pid, explode, pivot=[x, AXLE_Y, 0.0])
    for sz in (-1, 1):
        lathe(f'{pid}.tread.{sz}', [(r, B2B + w) for r, w in TREAD], x, sz, 'tread')
        lathe(f'{pid}.rim.{sz}', [(r, B2B + w) for r, w in RIM], x, sz, 'cast')
    cyl(f'{pid}.axle', (x, AXLE_Y, -B2B - 0.20), (x, AXLE_Y, B2B + 0.20), 0.065, 'cast', seg=20, bevel=0.005)

def coil(name, base, height, R=0.075, r=0.016, turns=5.5, material='steel'):
    n = int(turns * 14)
    pts = [(base[0] + R * math.cos(2 * math.pi * turns * k / n), base[1] + height * k / n,
            base[2] + R * math.sin(2 * math.pi * turns * k / n)) for k in range(n + 1)]
    return sweep(name, pts, r, material, seg=6)

def bogie(pid, xc, explode, motors_pid, motors_explode, first_axle):
    part(pid, explode)
    for sz in (-1, 1):
        zf = sz * 0.92
        box(f'{pid}.sideframe.{sz}', (xc, 0.40, zf), (2.10, 0.16, 0.09), 'cast', bevel=0.02)
        for sx in (-1, 1):
            xa = xc + sx * WHEELBASE / 2
            box(f'{pid}.pedestal.{sz}{sx}', (xa, 0.44, zf), (0.34, 0.12, 0.10), 'cast', bevel=0.02)
            box(f'{pid}.axlebox.{sz}{sx}', (xa, AXLE_Y, zf), (0.24, 0.22, 0.15), 'cast', bevel=0.025)
            cyl(f'{pid}.axlebox_cover.{sz}{sx}', (xa, AXLE_Y, zf + sz * 0.075), (xa, AXLE_Y, zf + sz * 0.10), 0.09, 'cast', seg=24, bevel=0.006)
            bm = bmesh.new()
            for k in range(6):                       # bolts on the axle box cover
                a = 2 * math.pi * k / 6
                c = (xa + 0.065 * math.cos(a), AXLE_Y + 0.065 * math.sin(a))
                cyl_into(bm, (c[0], c[1], zf + sz * 0.10), (c[0], c[1], zf + sz * 0.112), 0.011, 6)
            finish(bm, f'{pid}.bolts.{sz}{sx}', 'steel', sharp=40)
            box(f'{pid}.chevron.{sz}{sx}', (xa, 0.485, zf), (0.20, 0.05, 0.12), 'rubber', bevel=0.012)
        for dx in (-0.16, 0.16):                      # secondary coil springs under the bolster ends
            coil(f'{pid}.spring.{sz}{dx}', (xc + dx, 0.30, zf), 0.20)
            box(f'{pid}.spring_seat.{sz}{dx}', (xc + dx, 0.29, zf), (0.20, 0.02, 0.20), 'cast', bevel=0.005)
        cyl(f'{pid}.damper.{sz}', (xc + 0.42, 0.30, zf + sz * 0.06), (xc + 0.30, 0.52, zf + sz * 0.06), 0.028, 'darkmetal', seg=14, bevel=0.004)
        box(f'{pid}.trackbrake.{sz}', (xc, 0.075, sz * WZ), (0.95, 0.07, 0.085), 'cast', bevel=0.012)
        for sx in (-1, 1):
            cyl(f'{pid}.tb_hanger.{sz}{sx}', (xc + sx * 0.35, 0.11, sz * WZ), (xc + sx * 0.35, 0.33, zf), 0.015, 'darkmetal', seg=10)
            box(f'{pid}.tb_coil.{sz}{sx}', (xc + sx * 0.22, 0.125, sz * WZ), (0.30, 0.04, 0.10), 'darkmetal', bevel=0.008)
    box(f'{pid}.bolster', (xc, 0.46, 0), (0.36, 0.10, 1.95), 'cast', bevel=0.02)
    cyl(f'{pid}.pivot', (xc, 0.51, 0), (xc, BODY_BOTTOM + 0.01, 0), 0.12, 'cast', seg=28, bevel=0.008)
    for sx in (-1, 1):
        box(f'{pid}.transom.{sx}', (xc + sx * 0.62, 0.40, 0), (0.12, 0.10, 1.80), 'cast', bevel=0.015)
    for k, sx in enumerate((1, -1)):
        wheelset(f'wheelset_{first_axle + k}', xc + sx * WHEELBASE / 2, explode)
    # two traction motors lying lengthwise, each driving one axle through a bevel gear (Wikipedia),
    # with a brake drum and band on the free shaft end
    part(motors_pid, motors_explode)
    for sx, sz in ((1, 1), (-1, -1)):
        zz = sz * 0.30
        cyl(f'{motors_pid}.motor.{sx}', (xc - sx * 0.42, 0.37, zz), (xc + sx * 0.40, 0.37, zz), 0.16, 'motor', seg=36, bevel=0.015)
        bm = bmesh.new()
        for k in range(6):
            xr = xc - sx * 0.30 + sx * k * 0.12
            cyl_into(bm, (xr - 0.006, 0.37, zz), (xr + 0.006, 0.37, zz), 0.168, 36)
        finish(bm, f'{motors_pid}.ribs.{sx}', 'motor', sharp=40)
        cyl(f'{motors_pid}.drum.{sx}', (xc - sx * 0.43, 0.37, zz), (xc - sx * 0.52, 0.37, zz), 0.12, 'cast', seg=36, bevel=0.006)
        cyl(f'{motors_pid}.band.{sx}', (xc - sx * 0.44, 0.37, zz), (xc - sx * 0.51, 0.37, zz), 0.127, 'darkmetal', seg=36)
        box(f'{motors_pid}.brake_unit.{sx}', (xc - sx * 0.475, 0.55, zz), (0.10, 0.08, 0.14), 'darkmetal', bevel=0.01)
        box(f'{motors_pid}.gearbox.{sx}', (xc + sx * 0.80, AXLE_Y + 0.01, zz), (0.36, 0.28, 0.24), 'cast', bevel=0.04)

bogie('bogie_front', XP, (0.9, 0, 0), 'motors_front', (0.9, 0.6, 0), 1)
bogie('bogie_rear', -XP, (-0.9, 0, 0), 'motors_rear', (-0.9, 0.6, 0), 3)

part('accelerator', explode=(0, 0.45, 0))         # TR37 accelerator (Wikipedia); size and place ASSUMED
box('accelerator.case', (1.05, 0.40, -0.25), (1.0, 0.24, 0.80), 'cast', bevel=0.02)
bm = bmesh.new()
for k in range(5):
    box_into(bm, (0.65 + k * 0.2, 0.40, 0.155), (0.04, 0.20, 0.01), 0.003)
finish(bm, 'accelerator.ribs', 'cast', sharp=50)

# ============================================================================== couplers (Albert) and covers
for pid, s in (('coupler_front', 1), ('coupler_rear', -1)):
    part(pid, explode=(s * 1.4, 0, 0))
    tip = s * L_COUPLERS / 2
    cyl(f'{pid}.bar', (s * 6.55, 0.50, 0), (tip - s * 0.08, 0.50, 0), 0.04, 'grey', seg=16, bevel=0.004)
    box(f'{pid}.head', (tip - s * 0.05, 0.50, 0), (0.10, 0.07, 0.14), 'grey', bevel=0.012)
    cyl(f'{pid}.pin', (tip - s * 0.05, 0.43, 0), (tip - s * 0.05, 0.57, 0), 0.014, 'steel', seg=12)
    box(f'{pid}.cover', (s * 6.86, 0.52, 0), (0.22, 0.14, 0.46), 'grey', bevel=0.02)
    box(f'{pid}.head_cap', (tip - s * 0.05, 0.50, 0), (0.13, 0.09, 0.17), 'grey', bevel=0.025)
    chain = [(tip - s * (0.1 + 0.025 * k), 0.47 - 0.09 * math.sin(math.pi * k / 12), 0.08) for k in range(13)]
    sweep(f'{pid}.chain', chain, 0.006, 'steel', seg=6)

# ============================================================================== export
def kids(e):
    out = []
    for c in e.children:
        out.append(c)
        out.extend(kids(c))
    return out

def merge_by_material():
    for e in [o for o in COL.objects if o.type == 'EMPTY']:
        if e.get('nomerge'):
            continue
        groups = {}
        for c in kids(e):
            if c.type == 'MESH':
                groups.setdefault(c.data.materials[0].name, []).append(c)
        for mname, objs in groups.items():
            if any(o.data.uv_layers for o in objs):
                for o in objs:
                    if not o.data.uv_layers:
                        o.data.uv_layers.new()
            if len(objs) > 1:
                with bpy.context.temp_override(active_object=objs[0], selected_editable_objects=objs, selected_objects=objs):
                    bpy.ops.object.join()
            objs[0].name = f"{e['part']}.{mname}"
            if e.get('pivot'):
                rehome(objs[0], tuple(e['pivot']), e)

merge_by_material()
if '--debug' in ARGS: exec(open(ARGS[ARGS.index('--debug') + 1]).read())
meshes = [o for o in COL.objects if o.type == 'MESH']
tris = 0
for o in meshes:
    o.data.calc_loop_triangles()
    tris += len(o.data.loop_triangles)
print('VARIANT', VARIANT, 'NUMBER', NUMBER, 'PARTS', len([o for o in COL.objects if o.type == 'EMPTY']),
      'OBJECTS', len(meshes), 'TRIS', tris)

def export(path, fmt='AUTO'):
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=False, export_apply=True,
                              export_extras=True, export_yup=True, export_materials='EXPORT', export_cameras=False,
                              export_lights=False, export_texcoords=True, export_normals=True,
                              export_tangents=False, export_image_format=fmt)
    print('GLB', path, os.path.getsize(path))

if '--blend' in ARGS:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f't3_{NUMBER}.blend'))
if '--glb' in ARGS:
    export(os.path.join(OUT, f't3_{NUMBER}.glb'))
if '--web' in ARGS:
    # lighter copy for the browser preview: body atlas at 2K, the rest at 512 px
    for im in bpy.data.images:
        if im.size[0] == 0:
            continue
        s = 2048 if im.name.startswith('body') else 512
        if im.size[0] > s:
            im.scale(s, s)
    export(os.path.join(OUT, f't3_{NUMBER}_web.glb'), fmt='JPEG')

# ============================================================================== render scene (not exported)
def render_materials():
    mat('rail', (0.30, 0.20, 0.14), metal=0.8, rough=0.6)
    m = mat('concrete', (0.40, 0.39, 0.37), rough=0.85)
    nt = m.node_tree
    tc = nt.nodes.new('ShaderNodeTexCoord')
    nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 40
    nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.2
    nt.links.new(nz.outputs['Fac'], nt.nodes['Principled BSDF'].inputs['Roughness'])
    nt.links.new(nz.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], nt.nodes['Principled BSDF'].inputs['Normal'])
    for name, scale, c0, c1, bump in (('ballast', 9.0, (0.09, 0.085, 0.08), (0.36, 0.34, 0.31), 0.9),
                                      ('ground', 2.5, (0.07, 0.09, 0.04), (0.20, 0.21, 0.12), 0.3)):
        m = mat(name, rough=0.95)
        nt = m.node_tree
        p = nt.nodes['Principled BSDF']
        tc = nt.nodes.new('ShaderNodeTexCoord')           # world-sized stones, not stretched to the object
        vo = nt.nodes.new('ShaderNodeTexVoronoi'); vo.inputs['Scale'].default_value = scale * 3
        nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = scale; nz.inputs['Detail'].default_value = 8
        nt.links.new(tc.outputs['Object'], vo.inputs['Vector'])
        nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
        mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'
        mx.inputs[6].default_value = (*c0, 1); mx.inputs[7].default_value = (*c1, 1)
        if name == 'ballast':                       # one grey per stone, rounded stones in the bump
            vo.inputs['Scale'].default_value = 32
            nt.links.new(vo.outputs['Color'], mx.inputs['Factor'])
            bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 1.0; bp.invert = True
            bp.inputs['Distance'].default_value = 0.02
        else:
            nt.links.new(nz.outputs['Fac'], mx.inputs['Factor'])
            bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = bump
        nt.links.new(mx.outputs[2], p.inputs['Base Color'])
        nt.links.new(vo.outputs['Distance'], bp.inputs['Height'])
        nt.links.new(bp.outputs['Normal'], p.inputs['Normal'])

def scene_setup():
    global PART
    PART = None
    # standard-gauge track: rail profile (head, web, foot) extruded, concrete sleepers, ballast bed
    head_w, foot_w = 0.072, 0.150
    sec = [(-foot_w / 2, -0.172), (foot_w / 2, -0.172), (foot_w / 2, -0.160), (0.010, -0.140), (0.010, -0.050),
           (head_w / 2, -0.040), (head_w / 2, -0.008), (head_w / 2 - 0.008, 0.0), (-head_w / 2 + 0.008, 0.0),
           (-head_w / 2, -0.008), (-head_w / 2, -0.040), (-0.010, -0.050), (-0.010, -0.140), (-foot_w / 2, -0.160)]
    for sz in (-1, 1):
        zc = sz * (WZ + 0.036)
        bm = bmesh.new()
        r0 = [bm.verts.new((-12.0, y, zc + z)) for z, y in sec]
        r1 = [bm.verts.new((12.0, y, zc + z)) for z, y in sec]
        m = len(sec)
        for k in range(m):
            bm.faces.new([r0[k], r0[(k + 1) % m], r1[(k + 1) % m], r1[k]])
        bm.faces.new(list(reversed(r0)))
        bm.faces.new(r1)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        finish(bm, f'rail.{sz}', 'rail', sharp=30, parent=None)
        box(f'railtop.{sz}', (0, -0.0005, zc), (24, 0.002, 0.056), 'tread', bevel=0)    # polished running band
    for k in range(-19, 20):
        box(f'sleeper.{k}', (k * 0.6, -0.272, 0), (0.24, 0.20, 2.6), 'concrete', bevel=0.02)
    bm = bmesh.new()
    prof_b = [(-4.2, -0.62), (-2.0, -0.20), (2.0, -0.20), (4.2, -0.62)]
    rows = [[bm.verts.new((x, y, z)) for z, y in prof_b] for x in lin(-12, 12, 60)]
    for i in range(len(rows) - 1):
        for k in range(len(prof_b) - 1):
            bm.faces.new([rows[i][k], rows[i + 1][k], rows[i + 1][k + 1], rows[i][k + 1]])
    for fc in bm.faces:
        fc.normal_update()
        if fc.normal.y < 0:
            fc.normal_flip()
    finish(bm, 'ballast', 'ballast', smooth=False, parent=None)
    box('ground', (0, -0.64, 0), (2000, 0.02, 2000), 'ground', bevel=0)

def lights_and_world(sun_elev=38, sun_az=-35):
    world = bpy.data.worlds.new('w')
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes['Background']
    try:
        sky = nt.nodes.new('ShaderNodeTexSky')
        for t in ('MULTIPLE_SCATTERING', 'NISHITA', 'HOSEK_WILKIE'):
            try:
                sky.sky_type = t
                break
            except Exception:
                pass
        if hasattr(sky, 'sun_disc'):
            sky.sun_disc = False
        if hasattr(sky, 'sun_elevation'):
            sky.sun_elevation = math.radians(sun_elev)
            sky.sun_rotation = math.radians(sun_az + 180)
        nt.links.new(sky.outputs['Color'], bg.inputs['Color'])
        bg.inputs['Strength'].default_value = float(ARGS[ARGS.index('--sky') + 1]) if '--sky' in ARGS else 0.10
        print('sky', sky.sky_type)
    except Exception as ex:
        print('sky texture unavailable', ex)
        bg.inputs['Color'].default_value = (0.62, 0.72, 0.85, 1)
        bg.inputs['Strength'].default_value = 0.9
    scene.world = world
    sun = bpy.data.lights.new('sun', 'SUN')
    sun.energy = float(ARGS[ARGS.index('--sun') + 1]) if '--sun' in ARGS else 4.0
    sun.angle = math.radians(4.0)                    # soft shadow edges
    sun.color = (1.0, 0.97, 0.92)
    so = bpy.data.objects.new('sun', sun)
    scene.collection.objects.link(so)
    so.rotation_euler = (math.radians(90 - sun_elev), 0, math.radians(sun_az))

def camera(loc, look, lens=35, res=(1920, 1080)):
    cam = bpy.data.cameras.new('cam')
    co = bpy.data.objects.new('cam', cam)
    scene.collection.objects.link(co)
    scene.camera = co
    cam.lens = lens
    cam.clip_start = 0.05
    cam.clip_end = 300
    cam.sensor_width = 36
    co.location = V(*loc)
    co.rotation_euler = (V(*look) - V(*loc)).to_track_quat('-Z', 'Y').to_euler()
    scene.render.resolution_x, scene.render.resolution_y = res
    return co

def cycles(samples):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.max_bounces = 8
    scene.cycles.transmission_bounces = 8
    scene.cycles.use_denoising = True
    want = ARGS[ARGS.index('--view') + 1] if '--view' in ARGS else 'AgX'
    for vt in (want, 'AgX', 'Filmic', 'Standard'):
        try:
            scene.view_settings.view_transform = vt
            break
        except Exception:
            pass
    look = ARGS[ARGS.index('--look') + 1].replace('_', ' ') if '--look' in ARGS else 'AgX - Medium High Contrast'
    try:
        scene.view_settings.look = look
    except Exception as ex:
        print('look unavailable', look, ex)
    scene.view_settings.exposure = float(ARGS[ARGS.index('--exposure') + 1]) if '--exposure' in ARGS else -0.4

if '--render' in ARGS or '--checks' in ARGS:
    render_materials()
    scene_setup()
    lights_and_world()
    samples = int(ARGS[ARGS.index('--samples') + 1]) if '--samples' in ARGS else 64
    res = tuple(int(v) for v in ARGS[ARGS.index('--res') + 1].split('x')) if '--res' in ARGS else (1920, 1080)
    cycles(samples)
    if '--render' in ARGS:
        camera((12.2, 1.45, 6.4), (2.6, 1.55, -0.3), lens=38, res=res)      # three-quarter front, eye level
        scene.render.filepath = os.path.join(OUT, f't3_{NUMBER}_preview.png')
        bpy.ops.render.render(write_still=True)
    if '--checks' in ARGS:
        shots = ARGS[ARGS.index('--shots') + 1].split(',') if '--shots' in ARGS else ['side', 'front', 'bogie', 'interior']
        def shot(name, loc, look, lens=35, ortho=None):
            co = camera(loc, look, lens, res)
            if ortho:
                co.data.type = 'ORTHO'
                co.data.ortho_scale = ortho
            scene.render.filepath = os.path.join(OUT, name)
            bpy.ops.render.render(write_still=True)
        if 'side' in shots: shot('check_side.png', (0, 1.6, 30), (0, 1.6, 0), ortho=16.0)
        if 'left' in shots: shot('check_left.png', (0, 1.6, -30), (0, 1.6, 0), ortho=16.0)
        if 'fortho' in shots: shot('check_fortho.png', (40, 1.6, 0), (0, 1.6, 0), ortho=4.6)
        if 'q34' in shots: shot('check_34.png', (12.2, 1.45, 6.4), (2.6, 1.55, -0.3), lens=38)
        if 'front' in shots: shot('check_front.png', (9.5, 1.7, 0.6), (6.5, 1.5, 0.0), lens=40)
        if 'bogie' in shots: shot('check_bogie.png', (4.6, 0.55, 2.6), (3.2, 0.40, 0.6), lens=28)
        if 'detail' in shots: shot('check_detail.png', (7.6, 2.0, 2.6), (5.6, 1.9, 1.1), lens=35)
        if 'interior' in shots: shot('check_interior.png', (-5.6, 1.62, 0.25), (3.0, 1.35, -0.05), lens=20)
