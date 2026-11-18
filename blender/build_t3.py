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
import bpy, bmesh, math, sys, os, re
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
L_COUPLERS = 15.200      # length over couplers (coloured drawing: coupler head 600 mm beyond each end)
W_BODY = 2.500           # body width (Wikipedia)
H_ROOF = 3.028           # roof above rail top, without pantograph (coloured drawing: 3,028 mm)
GAUGE = 1.435
BOGIE_PIVOTS = 6.400     # bogie pivot centres (spec)
WHEELBASE = 1.900        # bogie wheelbase (spec)
WHEEL_R = 0.350          # 700 mm wheels (spec)
DOORS = {'6102': (1.343, 1.343, 1.343), '6101': (1.300, 1.060, 1.300)}[VARIANT]
# Scaled off the dimensioned type drawing (side view, ~50 px per metre) and the coloured elevations:
BODY_BOTTOM = 0.352      # lower edge of the cream skirt (left elevation 0.36, front elevation 0.345)
SKIRT_LIFT = 0.29        # skirt cut up to ~0.62 m over the bogies
SKIRT = 0.793            # cream skirt / red band, hidden under the aluminium strip 0.754-0.832 m (both side elevations)
BELT = 1.608             # top of the red band, under the bright beading 1.608-1.656 m (side elevation)
WIN_Y0, WIN_Y1, WIN_SPLIT = 1.699, 2.506, 2.25 # side windows: opening (glass 1.725-2.480 visible), top of the fixed pane
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
            (1.24, BODY_BOTTOM + 0.07), (1.25, BODY_BOTTOM + 0.16), (1.25, 1.66), (1.248, 2.30), (1.244, 2.62),
            # roof shoulder scaled off the silhouette of the coloured front elevation (half-widths at
            # 2.75 / 2.81 / 2.87 / 2.93 / 2.96 / 2.99 m, outline stroke taken off)
            (1.235, 2.70), (1.219, 2.75), (1.176, 2.81), (1.110, 2.87), (0.997, 2.93), (0.904, 2.96),
            (0.781, 2.99), (0.62, 3.004), (0.0, H_ROOF)]

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
_PR = [0.0, 0.35, 0.55, 0.68, 0.76, 0.82, 0.91, 1.0]
_PX = [1.0, 0.99, 0.965, 0.92, 0.84, 0.70, 0.35, 0.0]

def roof_profile(xs, L, x_ws, y_ws, wall_slope, x_flat, drop_at, drop, r):
    """Centre-line silhouette of an end above the window top, from the side elevations: the roof runs flat
    to x_flat, then slopes down tangentially (a parabola, dropping `drop` by x=drop_at), turns through a
    rounded bend of radius r into the face, which rises from (x_ws, y_ws) leaning back wall_slope (dx/dy).
    Returns heights and kfac values (share of the nose length L beyond xs) for interpolation."""
    c = drop / (drop_at - x_flat) ** 2
    ya = lambda x: H_ROOF - c * max(0.0, x - x_flat) ** 2
    # corner: where the parabola meets the face line x = x_ws - wall_slope * (y - y_ws)
    lo, hi = x_flat, x_ws + 0.3
    for _ in range(60):
        m = (lo + hi) / 2
        if m < x_ws - wall_slope * (ya(m) - y_ws): lo = m
        else: hi = m
    C = Vector((lo, ya(lo)))
    d1 = Vector((1.0, -2 * c * (lo - x_flat))).normalized()          # along the roof, outwards/down
    d2 = Vector((wall_slope, -1.0)).normalized()                      # down the face
    th = math.acos(max(-1.0, min(1.0, d1.dot(d2))))
    tl = r * math.tan(th / 2)
    A, B = C - d1 * tl, C + d2 * tl
    pts = [(x, ya(x)) for x in np.linspace(x_flat, A.x, 40)]
    for t in np.linspace(0, 1, 24)[1:]:                                # rounded bend (quadratic Bezier)
        q = A * (1 - t) ** 2 + C * 2 * t * (1 - t) + B * t * t
        pts.append((q.x, q.y))
    pts.append((x_ws, y_ws))
    pts = sorted(pts, key=lambda p: p[1])
    ys = [p[1] for p in pts]; ks = [max(0.0, (p[0] - xs) / L) for p in pts]
    ys.append(H_ROOF + 1e-4); ks.append(max(0.0, (x_flat - xs) / L))
    return np.array(ys), np.array(ks)

class End:
    """A fibreglass end mask, from the type drawing.
    Plan: the cab narrows from the body side towards a slightly bowed face with rounded corners
    (the angled cab side windows sit on the narrowing part).
    Side view: cream apron set back under the bumper, upright red panel, windscreen leaning back
    ~0.35 m, then a rounded roof dome that starts ~0.95 m behind the nose. The whole plan outline is
    scaled towards the body by k(y), so the dome and the rake never fold back past the body."""
    def __init__(s, sign, xs=6.0, ws_top=2.53, rake=0.35, apron=0.06, q=1.0, plan=None, plan_win=None, xt=HALF,
                 face_rake=0.0, low=(0.70, 0.06, 0.0), dome_p=2.0, plan_r=None):
        s.sign, s.xs, s.xt = sign, xs, xt
        s.ws_top, s.rake, s.apron = ws_top, rake, apron
        s.q = q                                       # half-width where the mask starts, share of the body's
        s.face_rake = face_rake                       # face leans back by this much from 1.62 m down to 0.86 m
        s.low = low                                   # skirt under the bumper: (top, setback at top, extra per metre down)
        s.dome_p = dome_p                             # exponent of the roof dome (2 = ellipse, larger = fuller)
        pr, px = plan if plan else (_PR, _PX)
        s.PLAN = PchipInterpolator(pr, px)
        s.PLAN_W = PchipInterpolator(*plan_win) if plan_win else s.PLAN
        s.R_TAB = np.linspace(0.0, 1.0, 401)
        s.X_TAB = s.PLAN(s.R_TAB)
        s.X_TAB_W = s.PLAN_W(s.R_TAB)
        s.PLAN_R = PchipInterpolator(*plan_r) if plan_r else None    # right (door) side, all heights
        s.X_TAB_R = s.PLAN_R(s.R_TAB) if plan_r else None
    def wmix(s, y0):                                  # 0 low on the face, 1 from just under the beading upwards
        return smooth01((y0 - 1.50) / 0.10)
    def share(s, r, y0, side=-1):
        if side > 0 and s.PLAN_R is not None:
            return float(s.PLAN_R(r))
        w = s.wmix(y0)
        return (1 - w) * float(s.PLAN(r)) + w * float(s.PLAN_W(r))
    def r_of(s, frac, y0, side=-1):
        if side > 0 and s.PLAN_R is not None:
            return float(np.interp(-frac, -s.X_TAB_R, s.R_TAB))
        w = s.wmix(y0)
        return float(np.interp(-frac, -((1 - w) * s.X_TAB + w * s.X_TAB_W), s.R_TAB))

# plan outline: r = |z| / half-width (1 at the body side, 0 on the centre line) -> share of the nose length
_PR = [0.0, 0.35, 0.55, 0.68, 0.76, 0.82, 0.91, 1.0]
_PX = [1.0, 0.99, 0.965, 0.92, 0.84, 0.70, 0.35, 0.0]
# Front end, traced from the plan of the coloured drawing (167.7 px/m) along the outer edge of the body
# wall (the brown band outside it at the nose is the bumper, which the floor-level section cuts): an
# almost flat face, then a long rounded corner back to x = 6.0 m.
_PR_F = [0.0, 0.096, 0.287, 0.478, 0.576, 0.669, 0.724, 0.784, 0.821, 0.855, 0.889, 0.924, 0.967, 0.989, 1.0]
_PX_F = [1.0, 1.0, 0.992, 0.973, 0.950, 0.901, 0.842, 0.737, 0.632, 0.526, 0.421, 0.316, 0.211, 0.105, 0.0]
# The body narrows in plan from x = 4.5 m to the start of the front mask at 6.0 m, so the front door and
# the first left window sit on angled walls. Half-width at 6.0 m: 1.05 m in the plan, 0.95-1.07 m in
# the front elevation (strips beside the face); the model takes 1.03 m.
TAPER_X0 = 4.50
# At window height the cab narrows quickly behind the windscreen, so that its glass starts 0.18 m ahead
# of the mask joint in the side elevation while keeping the front elevation's windscreen width.
_PX_FW = [1.0, 1.0, 0.992, 0.973, 0.950, 0.90, 0.84, 0.73, 0.60, 0.42, 0.22, 0.10, 0.04, 0.012, 0.0]
_PR_FW = [0.0, 0.096, 0.287, 0.478, 0.576, 0.669, 0.724, 0.784, 0.821, 0.855, 0.887, 0.94, 0.975, 0.992, 1.0]
# side elevation: red face at 6.965 m leaning back to ~6.89 m at the bumper; skirt lobes curving back to
# ~6.77 m at 0.40 m; rear face upright at 6.965 m with a fuller roof dome and little rake
FRONT = End(+1, rake=0.30, q=0.824, plan=(_PR_F, _PX_F), plan_win=(_PR_FW, _PX_FW), xt=7.000, face_rake=0.071,
            low=(0.735, 0.075, 0.27), dome_p=4.0)          # both side elevations: upright face up to ~2.95 m, then a tight corner
# rear end, traced from the plan along the outer edge of the body wall (relative to its width at 6.0 m)
_PR_R = [0.0, 0.287, 0.382, 0.473, 0.707, 0.775, 0.816, 0.849, 0.886, 0.918, 0.95, 0.978, 1.0]
_PX_R = [1.0, 0.991, 0.979, 0.975, 0.870, 0.761, 0.652, 0.543, 0.435, 0.326, 0.217, 0.109, 0.0]
# (left elevation: body ends at +/-7.00 m, as the 14,000 mm dimension; the rear mask joint sits at 5.97 m so
# that the left rear corner window reaches forward to -6.0 m)
# Plan of the coloured drawing: the body tapers in at the rear exactly as at the front, from 4.5 m off the
# centre to 1.074 m half-width at the mask joint (x -5.98 m, q = 0.859), so the rear door and the last left
# window sit on angled walls (that is the door edge and the edge-on window the rear elevation shows).
# The rear mask then follows the traced plan below the belt; from window height up it is a rounder
# "capsule" (photo of the real car), so that the rear window can wrap round both corners: the bars at
# zv +/-0.61 m (rear elevation) come out at x -6.78 m in both side elevations, and the glass reaches the
# joint on the left (-6.0 m) and to just behind the rear door on the right (-6.21 m).
_PR_RW = [0.0, 0.20, 0.40, 0.568, 0.70, 0.80, 0.90, 0.96, 1.0]
_PX_RW = [1.0, 0.985, 0.94, 0.833, 0.70, 0.52, 0.30, 0.14, 0.0]
# right elevation: the red band leans back ~70 mm towards its bottom, the skirt below the strip starts
# ~135 mm in and slopes in gently (0.18 m per m) - no step between band and skirt
# photo of 6102: the red band follows the same rounded corner as the window above it, flush from roof to
# skirt, so the whole rear mask takes one plan, halfway between the traced belt-level plan and the rounder
# window-height one (the flat middle keeps room for the tail lights); the roof curves down to the window in a large
# radius (rounder dome than the side elevations' tight corner)
_PR_RM = list(np.linspace(0.0, 1.0, 21))
_PX_RM = [0.5 * (float(PchipInterpolator(_PR_R, _PX_R)(r)) + float(PchipInterpolator(_PR_RW, _PX_RW)(r))) for r in _PR_RM]
REAR = End(-1, xs=5.98, q=0.859, rake=0.10, xt=6.99, low=(0.762, 0.135, 0.18), face_rake=0.06, dome_p=2.8,
           plan=(_PR_RM, _PX_RM), plan_win=(_PR_RM, _PX_RM))
# Roof ends from the side elevations (coloured drawing, both sides): at the front the roof is flat to
# ~1.0 m from the end, then slopes gently down (~130 mm) and turns in a bend of ~0.2 m radius into the
# face, which keeps leaning back above the windscreen as the screen does below (a peak at ~2.8 m); at the
# rear it is flat to ~0.9 m from the end, drops ~120 mm and turns in a ~0.28 m radius into a nearly
# upright rear wall (top ~2.85 m).
def _set_top(E, wall_slope, x_flat, drop_at, drop, r):
    L = E.xt - E.xs
    x_ws = E.xs + L * (1.0 - E.rake)
    E.top = roof_profile(E.xs, L, x_ws, E.ws_top, wall_slope, x_flat, drop_at, drop, r)
_set_top(FRONT, 0.20, 6.00, 6.64, 0.14, 0.20)
_set_top(REAR, 0.115, 6.10, 6.85, 0.12, 0.28)
ENDS = (FRONT, REAR)
XS = FRONT.xs
XR = REAR.xs                                          # rear end of the side panels


def plan_w(x):
    """Plan width factor of the body sides: 1, then narrowing straight towards both masks."""
    if x < -TAPER_X0:
        t = min(1.0, (-x - TAPER_X0) / (REAR.xs - TAPER_X0))
        return 1.0 - (1.0 - REAR.q) * t
    if x <= TAPER_X0:
        return 1.0
    t = min(1.0, (x - TAPER_X0) / (FRONT.xs - TAPER_X0))
    return 1.0 - (1.0 - FRONT.q) * t

def kfac(E, y0):
    """Share of the nose length left at height y0 (1 = full length)."""
    L = E.xt - E.xs
    top, s0, per = E.low
    if y0 < top:
        return 1.0 - (s0 + per * (top - y0)) / L
    if y0 < 1.45:
        return 1.0 - E.face_rake * min(1.0, (1.45 - y0) / (1.45 - 0.86)) / L
    if y0 < 1.66:
        return 1.0
    if y0 < E.ws_top:
        return 1.0 - E.rake * (y0 - 1.66) / (E.ws_top - 1.66)
    if getattr(E, 'top', None) is not None:          # end profile traced from the side elevations
        return max(0.012, float(np.interp(y0, E.top[0], E.top[1])))
    k0 = 1.0 - E.rake
    R = H_ROOF - E.ws_top
    t = min(1.0, (y0 - E.ws_top) / R)
    p = E.dome_p
    return 0.05 + (k0 - 0.05) * max(0.0, 1 - t ** p) ** (1 / p)

def taper(y0):
    return 1.0 - 0.05 * min(1.0, max(0.0, (0.70 - y0) / (0.70 - BODY_BOTTOM)))

def skirt_lift(x):
    d = abs(abs(x) - XP)
    return SKIRT_LIFT * (1 - smooth01((d - 1.30) / 0.18))

def nose_s(E, s, y0):
    """Point on an end mask: s -1..1 across (fraction of the half-width), y0 the height."""
    zb = prof(u_of_y(y0))[0] * taper(y0) * E.q
    r = min(abs(s), 1.0)
    x = E.xs + (E.xt - E.xs) * E.share(r, y0, 1 if s > 0 else -1) * kfac(E, y0)
    return Vector((E.sign * x, y0, max(-1.0, min(1.0, s)) * zb))

def nose(E, z, y0):
    zb = prof(u_of_y(y0))[0] * taper(y0) * E.q
    return nose_s(E, z / zb if zb > 1e-9 else 0.0, y0)

def surf(xb, u, side):
    z0, y0 = prof(u)
    for E in ENDS:
        if xb * E.sign > E.xs:
            k = kfac(E, y0)
            frac = min((xb * E.sign - E.xs) / ((E.xt - E.xs) * k), 1.0)
            r = E.r_of(frac, y0, side)                                     # invert the plan outline
            x = E.sign * (E.xs + (E.xt - E.xs) * k * frac)
            return Vector((x, y0, side * z0 * r * taper(y0) * E.q))
    k = smooth01((abs(xb) - 5.6) / (XS - 5.6))
    z = z0 * (1 - (1 - taper(y0)) * k) * plan_w(xb)
    y = y0
    if y0 < 0.748:                                     # the cut-out over the bogies stays under the strip
        y = y0 + skirt_lift(xb) * (0.748 - y0) / (0.748 - BODY_BOTTOM)
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
FD0, FD1 = 5.3145 - DOORS[0] / 2, 5.3145 + DOORS[0] / 2
MD0, MD1 = -0.7445 - DOORS[1] / 2, -0.7445 + DOORS[1] / 2
RD0, RD1 = -5.3145 - DOORS[2] / 2, -5.3145 + DOORS[2] / 2
DOOR_SPANS = [('door_front', FD0, FD1), ('door_middle', MD0, MD1), ('door_rear', RD0, RD1)]

def win_run(x_from, x_to, n, pillar=PILLAR):
    w = (x_to - x_from - pillar * (n - 1)) / n
    return [(x_from + i * (w + pillar), x_from + i * (w + pillar) + w, WIN_Y0, WIN_Y1) for i in range(n)]

# right (door) side: 2 windows between the rear and middle door, 3 between the middle and front door;
# openings scaled off the coloured side elevation (visible glass + 26 mm gasket lip all round)
WIN_R = [(x0, x1, WIN_Y0, WIN_Y1) for x0, x1 in ((-4.410, -3.096), (-2.933, -1.619), (0.153, 1.467), (1.631, 2.951), (3.120, 4.428))]
# left side: 8 equal windows (coloured elevation)
WIN_L = win_run(-5.832, 5.832, 8, pillar=0.161)
DOOR_OPEN = [(x0, x1, Y_SIDE0 - 0.01, 2.56) for _, x0, x1 in DOOR_SPANS]
OPEN = {1: WIN_R + DOOR_OPEN, -1: list(WIN_L)}
# end glazing in mask units (s across, height): windscreen on the face, angled cab side windows on
# the narrowing part; the rear end the same with a lower window
# Front: one large windscreen (coloured front elevation): visible glass 1,777 mm wide, 1,682-2,491 mm up;
# the opening is cut ~26 mm larger all round because the gasket lip covers that much of the glass.
WS_Y0, WS_Y1 = 1.664, 2.519
S_WS = 0.906 / (1.249 * 0.824)
WS_RC = (0.105, (0.362, 0.286))                      # corner radii of the opening: bottom, top (across, up)
FRONT_HOLES = [(-0.988, S_WS, WS_Y0, WS_Y1)]
WS_RC_L = (0.105, (0.26, 0.43))                       # corner radii on the left (s < 0) end of the opening
# gap between the two skirt lobes under the bumper, where the coupler sits (front elevation)
LOBE_GAP = (-0.53, 0.53, None, 0.735)
LOBE_GAP_R = (-0.5225, 0.5225, None, 0.746)           # the same recess at the rear (rear elevation: 1,045 mm between the lobes)
# rear: a centre window and two large corner windows wrapping round to the sides, divided by narrow bars
# (side elevation: corner window glass from -6.78 to -6.21 m, the bar just behind it)
# rear elevation: one rear window (glass zv -0.978..+0.880 m, 1.702-2.502 m) divided by two thin bars, the
# narrower outer pane on the door side; on the left the corner window seen edge-on beside it; on the right
# the right outer pane wraps on round the corner to just behind the rear door
REAR_HOLES = [(-0.9995, 0.93, 1.676, 2.528)]           # one rear window wrapping round both corners
RW_RC_R = ((0.02, 0.05), (0.02, 0.07))                # corners of the wrapped glass at its front end, on the side
RW_RC = (0.06, 0.10)
RW_RC_L = ((0.012, 0.03), (0.015, 0.06))              # front corners of the left corner window, on the joint (left elevation)
RW_BARS = (-0.608, 0.616)                             # bar centres across the rear window, zv (rear elevation)
RW_SASH = 2.245                                       # the one horizontal bar of the left corner window (rear elevation)
# (the black strip at the right of the rear elevation is the rear door seen edge-on past the tapering
# corner; it is not a part of its own)
REAR_WIN = REAR_HOLES[0]

def pillar_seams(wins):
    xs = sorted(wins)
    return [(xs[i][1] + xs[i + 1][0]) / 2 for i in range(len(xs) - 1) if xs[i + 1][0] - xs[i][1] < 0.3]
SEAMS = {1: pillar_seams(WIN_R), -1: pillar_seams(WIN_L)}
# joints between the cream skirt panels on the left (left elevation), drawn as dark lines in the texture
SKIRT_SEAMS_L = [4.631, 1.344, 0.076, -1.536, -4.828]
GROOVE_W, GROOVE_D = 0.005, 0.0022
def grooves(side):
    g = [(s - GROOVE_W / 2, s + GROOVE_W / 2, Y_SIDE0, BELT - 0.006) for s in SEAMS[side]]
    g += [(XS - GROOVE_W, XS, Y_SIDE0, 2.75), (-XR, -XR + GROOVE_W, Y_SIDE0, 2.75)]          # mask joints
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
    X = np.array([[abs(nose_s(E, sv, y).x) for sv in S_GRID] for y in ys])
    NOSE_TAB[key] = dict(y=ys, A=A, B=B, B_tot=B[-1], X=X)
# (kept at least at its earlier value, so that a change to one end does not rescale the other end's texture)
ARCMAX = max(1.745203891990116, float(max(np.abs(NOSE_TAB[k]['A']).max() for k in NOSE_TAB)) + 0.02)
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
    L = dict(XS=XS, XR=XR, SKIN_SEAMS_L=SKIRT_SEAMS_L, SKIRT=SKIRT, BELT=BELT, EAVE=EAVE, BODY_BOTTOM=BODY_BOTTOM, ARC_TOT=ARC_TOT, ARCMAX=ARCMAX,
             prof_u=np.array(_U), prof_y=np.array([p.y for p in _dense]),
             u_bottom=u_of_y(Y_SIDE0), u_roof=u_of_y(2.74), u_grey=u_of_y(2.935),
             win={1: WIN_R, -1: WIN_L}, doors=[(x0, x1, Y_SIDE0, 2.64) for _, x0, x1 in DOOR_SPANS],
             cab_corner=(99.0, 99.1, 0.0, 0.0), seams=SEAMS, WS=(WS_Y0, WS_Y1), RW=REAR_WIN, nose={})
    for key in ('front', 'rear'):
        T = NOSE_TAB[key]
        jw = int(np.searchsorted(T['y'], WS_Y0))
        jr = int(np.searchsorted(T['y'], REAR_WIN[2]))
        L['nose'][key] = dict(y=T['y'], B=T['B'], B_tot=T['B_tot'], s=S_GRID, A_at_ws=T['A'][jw], A_at_rw=T['A'][jr],
                              A=T['A'], X=T['X'])
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
mat('alu', (0.86, 0.87, 0.88), metal=1.0, rough=0.28)                  # satin aluminium body strips (photo of 6102)
mat('stripe', (0.62, 0.035, 0.03), rough=0.35, coat=0.3)               # red line along the middle of each strip
mat('mirror', (0.92, 0.92, 0.92), metal=1.0, rough=0.02)
mat('lens', (0.92, 0.9, 0.84), rough=0.06, emit=(1.0, 0.93, 0.78), emit_strength=0.15)
mat('redlens', (0.45, 0.02, 0.02), rough=0.12, emit=(0.4, 0.0, 0.0), emit_strength=0.3)
mat('lamp', (0.93, 0.91, 0.84), rough=0.15)
mat('signtext', (0.74, 0.74, 0.71), rough=0.4, emit=(1, 1, 0.95), emit_strength=0.12)   # ~20 % dimmer behind the glass
mat('cutter', (1, 0, 1))
mat('signglass', (0.96, 0.985, 1.0), rough=0.02, trans=1.0, ior=1.5)          # clear cover over the route blind: a real
                                                                               # glass sheet with clear reflections
mat('insulator', (0.30, 0.11, 0.05), rough=0.2, coat=0.5)
mat('copper', (0.62, 0.32, 0.18), metal=1.0, rough=0.4)
mat('motor', (0.09, 0.095, 0.09), metal=0.3, rough=0.6)
mat('umakart', (0.42, 0.33, 0.22), rough=0.42, coat=0.2)
mat('lining', (0.70, 0.68, 0.60), rough=0.5)
mat('ceiling', (0.80, 0.79, 0.74), rough=0.6)
mat('cab', (0.15, 0.19, 0.17), rough=0.5)
mat('tube', (1.0, 0.98, 0.94), rough=0.2, emit=(1, 0.97, 0.9), emit_strength=0.6)
mat('redpaint', (0.565, 0.0144, 0.0129), rough=0.35, coat=0.12)        # small loose red parts (livery red)
mat('lettering', (0.80, 0.71, 0.47), rough=0.4, coat=0.3)            # cream boarding signs (coloured elevation)
mat('handle', (0.78, 0.74, 0.62), rough=0.35, coat=0.25)            # painted door handles (photos of 6102 and 7807)
def mat_img(name, path, rough=0.4, coat=0.0):
    m = mat(name, rough=rough, coat=coat)
    nt = m.node_tree
    n = nt.nodes.new('ShaderNodeTexImage')
    n.image = bpy.data.images.load(path, check_existing=True)
    nt.links.new(n.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
    return m
EMBLEM = os.path.join(TEX, 'emblem.png')
TT.emblem(EMBLEM, 1024)
mat_img('emblem', EMBLEM, rough=0.35, coat=0.3)
TILED = {'cast': 0.5, 'rubber': 0.4, 'floor': 0.6, 'mat_roof': 0.6, 'leather_red': 0.4, 'leather_grey': 0.4}

FONT_NUM = bpy.data.fonts.load(os.path.join(HERE, 'fonts', 'DejaVuSerif-Bold.ttf'))
FONT_SIGN = bpy.data.fonts.load(os.path.join(HERE, 'fonts', 'DejaVuSansCondensed-Bold.ttf'))
FONT_ROUTE = bpy.data.fonts.load(os.path.join(HERE, 'fonts', 'BarlowSemiCondensed-SemiBold.ttf'))   # route blind: DIN-style (OFL, fonts/Barlow-OFL.txt)

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
# glass edge unchanged at -0.026; black ~31 mm wide in all (door windows in the side elevation ~30 mm)
GASKET = [(-0.026, -0.011), (-0.010, -0.006)] + [(-0.0005 + 0.0055 * math.cos(math.pi * (1 - k / 4)), 0.0035 * math.sin(math.pi * k / 4)) for k in range(5)]
def gasket(f, rect, material='rubber', edges=None, rc=0.035, ka=1.0, step=0.08, profile=GASKET, rcl=None, rcr=None):
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
        rb, rt = rc if isinstance(rc, tuple) else (rc, rc)
        (bx_, by_), (tx_, ty_) = [(r, r) if not isinstance(r, tuple) else r for r in (rb, rt)]
        (blx, bly), (tlx, tly) = [(r, r) if not isinstance(r, tuple) else r for r in (rcl if rcl else (rb, rt))]
        if rcr:
            (bx_, by_), (tx_, ty_) = [(r, r) if not isinstance(r, tuple) else r for r in rcr]
        A0, A1 = a0 * ka, a1 * ka
        segs = [((A0 + blx, b0), (A1 - bx_, b0), (0, -1)), ((A1, b0 + by_), (A1, b1 - ty_), (1, 0)),
                ((A1 - tx_, b1), (A0 + tlx, b1), (0, 1)), ((A0, b1 - tly), (A0, b0 + bly), (-1, 0))]
        corners = [(A1 - bx_, b0 + by_, -90, bx_, by_), (A1 - tx_, b1 - ty_, 0, tx_, ty_),
                   (A0 + tlx, b1 - tly, 90, tlx, tly), (A0 + blx, b0 + bly, 180, blx, bly)]
        path = []
        for (p, q, d), (cx, cy, ang0, rx, ry) in zip(segs, corners):
            L = math.hypot(q[0] - p[0], q[1] - p[1])
            n = max(1, int(L / step))
            for k in range(n):
                t = k / n
                path.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, d[0], d[1]))
            nk = max(5, int(max(rx, ry) / 0.025))
            for k in range(nk):
                a = math.radians(ang0 + 90 * k / nk)
                nx, ny = math.cos(a) / rx, math.sin(a) / ry
                nl = math.hypot(nx, ny)
                path.append((cx + rx * math.cos(a), cy + ry * math.sin(a), nx / nl, ny / nl))
        build([(a / ka, b, da, db) for a, b, da, db in path], True)
    else:
        for e in edges:
            if e in 'bt':
                b = b0 if e == 'b' else b1
                build([(a, b, 0, -1 if e == 'b' else 1) for a in breaks(a0, a1, [], lambda m: step / ka)], False)
            else:
                a = a0 if e == 'l' else a1
                build([(a, b, -1 if e == 'l' else 1, 0) for b in breaks(b0, b1, [], lambda m: step)], False)

def corner_fillets(f, rect, rb, rt, ka, uvf, n=10, rl=None, rr=None, material='paint', inward=False):
    """Fill the corners of a rectangular opening so it reads as rounded (radius rb at the bottom,
    rt at the top, in metres), on the same surface and with the same texture as the mask."""
    a0, a1, b0, b1 = rect
    A0, A1 = a0 * ka, a1 * ka
    (bx_, by_), (tx_, ty_) = [(r, r) if not isinstance(r, tuple) else r for r in (rb, rt)]
    (blx, bly), (tlx, tly) = [(r, r) if not isinstance(r, tuple) else r for r in (rl if rl else (rb, rt))]
    if rr:
        (bx_, by_), (tx_, ty_) = [(r, r) if not isinstance(r, tuple) else r for r in rr]
    for (cxA, cyB, rx, ry, ang0, CA, CB) in ((A0 + blx, b0 + bly, blx, bly, 180, A0, b0), (A1 - bx_, b0 + by_, bx_, by_, 270, A1, b0),
                                             (A1 - tx_, b1 - ty_, tx_, ty_, 0, A1, b1), (A0 + tlx, b1 - tly, tlx, tly, 90, A0, b1)):
        def P(t, rho, cxA=cxA, cyB=cyB, rx=rx, ry=ry, ang0=ang0, CA=CA, CB=CB):
            a = math.radians(ang0 + 90 * t)
            A = CA + rho * (cxA + rx * math.cos(a) - CA)
            B = CB + rho * (cyB + ry * math.sin(a) - CB)
            return A / ka, B
        g = lambda t, rho: f(*P(t, rho))
        add_grid(acc(material), g, [k / n for k in range(n + 1)], [0.0, 0.5, 1.0], uvf=lambda t, rho: uvf(*P(t, rho)), inward=inward)

def bright_strip(f, X, y0, y1):
    """Aluminium strip y0-y1 on surface f (rounded edges stepped down to the skin) with a 10 mm red line
    along its middle (photo of 6102)."""
    ym = (y0 + y1) / 2
    add_grid(acc('alu'), f, X, [y0, y0 + 0.004], 0.0015)
    add_grid(acc('alu'), f, X, [y0 + 0.004, ym - 0.005], 0.0035)
    add_grid(acc('stripe'), f, X, [ym - 0.005, ym + 0.005], 0.0036)
    add_grid(acc('alu'), f, X, [ym + 0.005, y1 - 0.004], 0.0035)
    add_grid(acc('alu'), f, X, [y1 - 0.004, y1], 0.0015)

# ============================================================================== body (vozová skříň)
BODY = part('body', explode=(0, 1.3, 0))
X_MID = breaks(-XR, XS, [], lambda m: 0.15)
for side in (1, -1):
    f = side_pt(side)
    uvf = uv_side(side)
    G = grooves(side)
    cells(f, -XR, XS, Y_SIDE0, SIDE_TOP, OPEN[side], 0.15, ystep, lambda y: 'paint', extra_b=(SKIRT, BELT), uvf=uvf, groove=G)
    for g in G:                                       # walls of the panel-joint grooves
        reveal(f, g, 'paint', edges='lr', depth=GROOVE_D, uvf=uvf, astep=0.1)
    add_grid(acc('under'), lambda x, u, s=side: surf(x, u, s), X_MID, lin(0.0, u_of_y(Y_SIDE0), 3))
    add_grid(acc('paint'), lambda x, u, s=side: surf(x, u, s), X_MID, lin(u_of_y(SIDE_TOP), 1.0, 22), uvf=uv_side_u(side))
    # aluminium strip under the red band (0.754-0.832 m) and bright beading over it (1.608-1.656 m),
    # both interrupted by the doors (side elevation)
    gaps = [(x0, x1) for _, x0, x1 in DOOR_SPANS] if side == 1 else []
    stops = sorted([-XR, XS] + [v for g_ in gaps for v in g_])
    for i in range(0, len(stops), 2):
        # straight wall: long segments; where the body narrows towards the ends (|x| > 4.45 m) finely
        # split, so the strip follows the curve and does not sink under the skin next to the end doors
        X = breaks(stops[i], stops[i + 1], [-4.45, 4.45], lambda m: 0.075 if abs(m) > 4.45 else 3.0)
        for y0_, y1_ in ((0.754, 0.832), (1.608, 1.656)):
            bright_strip(f, X, y0_, y1_)
    for rect in (WIN_R if side == 1 else WIN_L):
        reveal(f, rect, 'rubber', start=0.006)
        gasket(f, rect, rc=0.086)                    # clearly rounded corners in black rubber (side elevation)
        corner_fillets(f, rect, 0.086, 0.086, 1.0, uvf)
    # sash bars of the windows: horizontal under the vents, vertical between the two sliding vent panes
    for (a0, a1, b0, b1) in (WIN_R if side == 1 else WIN_L):
        X = breaks(a0, a1, [], lambda m: 0.3)
        xm = a0 + (0.47 if side == 1 else 0.494) * (a1 - a0)     # left elevation: bar 0.625 m from the glass edge
        for o, inw in ((-0.019, False), (-0.034, True)):
            add_grid(acc('lining'), f, X, [WIN_SPLIT - 0.016, WIN_SPLIT + 0.016], o, inward=inw)
            add_grid(acc('lining'), f, [xm - 0.016, xm + 0.016], breaks(WIN_SPLIT + 0.016, b1, [], lambda m: 0.2), o, inward=inw)
DOOR_BEAD = [(-0.004, -0.002)] + [(0.012 + 0.016 * math.cos(math.pi * (1 - k / 6)), 0.004 * math.sin(math.pi * k / 6)) for k in range(7)]
for pid, x0, x1 in DOOR_SPANS:
    f = side_pt(1)
    reveal(f, (x0, x1, Y_SIDE0, 2.56), 'maroon', edges='tlr')
    gasket(f, (x0, x1, Y_SIDE0, 2.56), material='maroon', edges='tlr', profile=DOOR_BEAD)
flush('body')
# brown step plate under each doorway (coloured elevation, photo of 6149), sticking out a little from the
# body side. Towards both ends the body narrows, so under the front and rear doors the bar is turned in
# plan to run parallel to the door leaves (body side at door height), keeping the middle door's offset.
_zd = lambda x: surf(x, u_of_y(1.0), 1).z
_off = 1.215 - _zd((MD0 + MD1) / 2)
for pid, x0, x1 in DOOR_SPANS:                        # (no grab handles on the body: both elevations and the photo)
    _xa, _xb = x0 - 0.02, x1 + 0.02
    _za, _zb = _zd(_xa) + _off, _zd(_xb) + _off
    _ang = math.atan2(_zb - _za, _xb - _xa)
    box(f'{pid}.step_plate', ((_xa + _xb) / 2, 0.315, (_za + _zb) / 2), (math.hypot(_xb - _xa, _zb - _za), 0.035, 0.07), 'brown',
        bevel=0.004, rot=Matrix.Rotation(-_ang, 3, 'Y'))
# fleet number in gold on both sides, digits ~0.14 m high; behind the middle of the front door-side
# window run on the right, under the third window on the left (coloured elevations)
# (left elevation: digits ~130 mm tall and narrow, the number ~0.39 m long, centred 2.96 m)
for side, xn in ((1, 2.29), (-1, 2.96)):
    p = surf(xn, u_of_y(1.22), side) + Vector((0, 0, side * 0.0012))
    text_mesh(f'side_number.{side}', NUMBER, 0.253, p, Vector((side * 0.88, 0, 0)), Vector((0, 1, 0)), 'gold',
              extrude=0.0006, font=FONT_NUM)
    # Prague coat of arms in a round badge behind the middle of the car (coloured elevations)
    xe = -3.03 if side == 1 else -2.96
    c = surf(xe, u_of_y(1.22), side) + Vector((0, 0, side * 0.0014))
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new()
    n_e, R_e = 48, 0.13                              # ~260 mm across (both elevations)
    ring = [bm.verts.new(c + Vector((side * R_e * math.cos(2 * math.pi * k / n_e), R_e * math.sin(2 * math.pi * k / n_e), 0))) for k in range(n_e)]
    mid = bm.verts.new(c)
    for k in range(n_e):
        fc = bm.faces.new([mid, ring[k], ring[(k + 1) % n_e]])
        for l in fc.loops:
            d = (l.vert.co - c) / R_e
            l[uvl].uv = (0.5 + 0.5 * side * d.x, 0.5 + 0.5 * d.y)
    finish(bm, f'emblem.{side}', 'emblem', smooth=False)
# boarding signs on the door side (coloured elevation): VÝSTUP (exit) behind the rear and middle
# doors, VSTUP (entry) behind the front door
for i, (word, xw) in enumerate((('VÝSTUP', -4.17), ('VÝSTUP', 0.44), ('VSTUP', 4.16))):
    p = surf(xw, u_of_y(1.45), 1) + Vector((0, 0, 0.0012))
    text_mesh(f'sign.{i}', word, 0.17, p, Vector((1, 0, 0)), Vector((0, 1, 0)), 'lettering',
              extrude=0.0005, font=FONT_SIGN)
# rain gutter above the windows along the whole side, its ends turned down (side elevation: 2.584 m,
# from above the rear corner window to the front door)
for side in (1, -1):
    f = side_pt(side)
    xr, xf = (-6.12, 5.99) if side == 1 else (-5.877, 5.91)      # left elevation: hooks at -5.93 / +5.94 m
    pts = [(xr - 0.035, 2.505), (xr - 0.03, 2.55), (xr, 2.584)] + [(x, 2.584) for x in np.linspace(xr + 0.25, xf - 0.25, 12)] + \
          [(xf, 2.584), (xf + 0.03, 2.55), (xf + 0.035, 2.505)]
    rod(f'gutter.{side}', [f(x, y) + normal(f, x, y) * 0.011 for x, y in pts], 0.011, 'rubber', seg=8)
# side route board behind the top of the third window from the front (right side, coloured elevation)
p0 = surf(3.775, u_of_y(2.372), 1) - Vector((0, 0, 0.03))
box('side_route_board', p0, (1.07, 0.205, 0.02), 'darkmetal', bevel=0.006)
# four small outlined plates just ahead of the middle door, two at window height and two in the red band
# (side elevation; door push buttons or signal lamps)
for k, yc in enumerate((2.375, 1.925, 1.445, 0.995)):
    f = side_pt(1)
    for (x0_, x1_, y0_, y1_) in ((0.006, 0.066, yc + 0.069, yc + 0.075), (0.006, 0.066, yc - 0.075, yc - 0.069),
                                 (0.006, 0.012, yc - 0.075, yc + 0.075), (0.060, 0.066, yc - 0.075, yc + 0.075)):
        add_grid(acc('rubber'), f, [x0_, x1_], [y0_, y1_], 0.0042)
    box(f'door_signal.{k}', surf(0.036, u_of_y(yc), 1) + Vector((0, 0, 0.002)), (0.046, 0.136, 0.003),
        'lettering' if yc > 1.6 else 'redpaint', bevel=0.001)
# amber indicator on the window pillar nearest x = 3.0 m on both sides; seen from the front they stand
# out ~50 mm beyond the body side, centred 2.23 m up, ~155 mm tall (front and side elevations)
for side in (1, -1):
    wins = sorted(WIN_R if side == 1 else WIN_L)
    gaps = [((wins[i][1] + wins[i + 1][0]) / 2) for i in range(len(wins) - 1) if wins[i + 1][0] - wins[i][1] < 0.3]
    xl = min(gaps, key=lambda g: abs(g - 3.0))
    p = surf(xl, u_of_y(2.225 if side == 1 else 2.240), side)       # left elevation: 2.24 m
    n = Vector((0, 0, side))
    bm = bmesh.new()
    cyl_into(bm, p - n * 0.004, p + n * 0.050, 0.04, 16, bevel=0.012)
    bmesh.ops.scale(bm, vec=Vector((1.0, 1.95, 1.0)), space=Matrix.Translation(-p), verts=bm.verts)
    finish(bm, f'side_lamp.{side}', 'amber', sharp=50)
# upright oval marker lamp in the red band under each cab window, just below the colour line (both side
# elevations: ~6.20 m, 1.39-1.40 m up)
for side_, ym in ((-1, 1.395), (1, 1.39)):
    _f = side_pt(side_)
    p = surf(6.20, u_of_y(ym), side_)
    n = normal(_f, 6.20, ym)
    # photo of 6149: a plain light lens, no dark ring; ~60 x 140 mm, upright
    bm = bmesh.new()
    cyl_into(bm, p - n * 0.004, p + n * 0.016, 0.030, 20, bevel=0.006)
    bmesh.ops.scale(bm, vec=Vector((1.0, 2.33, 1.0)), space=Matrix.Translation(-p), verts=bm.verts)
    finish(bm, f'side_marker.{side_}.lens', 'lamp', sharp=50)
# grey boxes hung under the left skirt, set back from it (left elevation: 0.27-0.32 m up, under the
# panels at x 0.18-1.31, -1.50..0.05 and -5.74..-4.86 m)
for k, (x0, x1) in enumerate(((0.175, 1.305), (-1.50, 0.05))):    # (none by the rear corner: photo of 6102)
    box(f'underframe_box.{k}', ((x0 + x1) / 2, 0.296, -1.13), (x1 - x0, 0.054, 0.04), 'grey', bevel=0.006)
# louvred grille in the red band on the left side (coloured elevation): cooling for the equipment
# under the floor. Left elevation: frame 1.55 m long outside (x -1.50 .. +0.045 m), top 1.35 m, an upside-
# down U open at the bottom; three bays of louvres run down to the aluminium strip.
GX0, GX1, GY1, GYB = -1.49, 0.035, 1.34, 0.832
bm_back, bm_slat = bmesh.new(), bmesh.new()
_bw = (GX1 - GX0 - 0.02 - 2 * 0.03) / 3
for k in range(3):
    gx0 = GX0 + 0.01 + k * (_bw + 0.03)
    gx1 = gx0 + _bw
    zc = -(1.25 + 0.006)
    box_into(bm_back, ((gx0 + gx1) / 2, (GYB + GY1 - 0.02) / 2, zc), (gx1 - gx0, GY1 - 0.02 - GYB, 0.008), 0.003)
    for j in range(13):
        yj = GYB + 0.020 + j * 0.035
        box_into(bm_slat, ((gx0 + gx1) / 2, yj, zc - 0.009), (gx1 - gx0 - 0.03, 0.012, 0.016), 0.0,
                 rot=Matrix.Rotation(math.radians(35), 3, 'X'))
finish(bm_back, 'grille.back', 'under', sharp=50)
fixed_uv(finish(bm_slat, 'grille.slats', 'paint', sharp=50), (0.5, 0.40 + 0.30 * 0.40))
_g = [(GX1, GYB, -(1.25 + 0.010))]
for (cx, cy, a0) in ((GX1 - 0.05, GY1 - 0.05, 0), (GX0 + 0.05, GY1 - 0.05, 90)):
    for k in range(7):
        a = math.radians(a0 + 15 * k)
        _g.append((cx + 0.05 * math.cos(a), cy + 0.05 * math.sin(a), -(1.25 + 0.010)))
_g.append((GX0, GYB, -(1.25 + 0.010)))
sweep('grille.frame', _g, 0.010, 'chrome', seg=8, closed=False)

# ============================================================================== glass (one object)
part('glass', explode=(0, 1.3, 0))
for side in (1, -1):
    f = side_pt(side)
    for (a0, a1, b0, b1) in (WIN_R if side == 1 else WIN_L):
        pane(f, (a0 - 0.012, a1 + 0.012, b0 - 0.012, WIN_SPLIT - 0.012))
        xm = a0 + (0.47 if side == 1 else 0.494) * (a1 - a0)   # two sliding vent panes on separate tracks
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
    opens = list(holes)
    ka = 1.249 * E.q
    if E is FRONT:
        gap = (LOBE_GAP[0] / (1.25 * 0.98 * E.q), LOBE_GAP[1] / (1.25 * 0.98 * E.q), Y_SIDE0 - 0.01, LOBE_GAP[3])
        opens.append(gap)
    if E is REAR:
        opens.append((LOBE_GAP_R[0] / (1.25 * 0.98 * E.q), LOBE_GAP_R[1] / (1.25 * 0.98 * E.q), Y_SIDE0 - 0.01, LOBE_GAP_R[3]))
    s_hi = 1.0
    cells(f, -1.0, 1.0, Y_SIDE0, H_ROOF, opens, 0.04,
          lambda y: 0.025 if y < 0.80 else 0.015 if y > 2.85 else 0.03 if y > 2.45 else 0.07,
          lambda y: 'paint', extra_b=(SKIRT, BELT, SIDE_TOP, 1.592, 1.638, 1.656), uvf=uv_nose(E))
    if E is REAR:                                     # rounded corners on the rear windows (side elevation)
        for h in holes:
            corner_fillets(f, h, RW_RC[0], RW_RC[1], ka, uv_nose(E), rl=RW_RC_L if h[0] < -0.99 else None,
                           rr=RW_RC_R if h[1] > 0.9 else None)
        S = breaks(-1.0, s_hi, [], lambda m: 0.07)    # bright beading over the red, as on the sides
        bright_strip(f, S, 1.608, 1.656)
        # the recess between the skirt lobes, as at the front (rear elevation): walls, ceiling, back wall
        XBR = -6.55
        zg = LOBE_GAP_R[1]
        YB = breaks(Y_SIDE0, LOBE_GAP_R[3], [], lambda m: 0.05)
        for sz in (-1, 1):
            add_grid(acc('grey'), lambda t, y, sz=sz: Vector((XBR + t * (nose(E, sz * zg, y).x + 0.005 - XBR), y, sz * zg)),
                     [0.0, 1.0], YB, orient=lambda c, sz=sz: Vector((0, 0, -sz)))
        xf = nose(E, 0.0, LOBE_GAP_R[3] - 0.01).x + 0.005
        box('recess_r.ceiling', ((XBR + xf) / 2, LOBE_GAP_R[3] - 0.005, 0), (abs(xf - XBR), 0.01, 2 * zg), 'grey', bevel=0.002)
        box('recess_r.back', (XBR, (Y_SIDE0 + LOBE_GAP_R[3]) / 2, 0), (0.01, LOBE_GAP_R[3] - Y_SIDE0, 2 * zg), 'grey', bevel=0.002)
        for yy in (0.46, 0.54, 0.62, 0.69):
            box(f'recess_r.slat.{yy}', (XBR - 0.012, yy, 0), (0.012, 0.025, 2 * zg - 0.04), 'darkmetal', bevel=0.003)
        # two thin bars dividing the rear window, and the one bar across the left corner window
        for zc in RW_BARS:
            add_grid(acc('rubber'), lambda zz, y: nose(E, zz, y), [zc - 0.015, zc + 0.015],
                     breaks(REAR_HOLES[0][2], REAR_HOLES[0][3], [], lambda m: 0.1), -0.006)

    if E is FRONT:
        for h in holes:
            corner_fillets(f, h, WS_RC[0], WS_RC[1], ka, uv_nose(E), rl=WS_RC_L if h[0] < -0.99 else None)
        # the recess between the skirt lobes (front elevation): flat inner walls, a ceiling under the
        # bumper and a grey back wall with horizontal slats, about 0.28 m deep
        XB = 6.55
        zg = LOBE_GAP[1]
        YB = breaks(Y_SIDE0, LOBE_GAP[3], [], lambda m: 0.05)
        for sz in (-1, 1):
            add_grid(acc('grey'), lambda t, y, sz=sz: Vector((XB + t * (nose(E, sz * zg, y).x - 0.005 - XB), y, sz * zg)),
                     [0.0, 1.0], YB, orient=lambda c, sz=sz: Vector((0, 0, -sz)))
        xf = nose(E, 0.0, LOBE_GAP[3] - 0.01).x - 0.005
        box('recess.ceiling', ((XB + xf) / 2, LOBE_GAP[3] - 0.005, 0), (xf - XB, 0.01, 2 * zg), 'grey', bevel=0.002)
        box('recess.back', (XB, (Y_SIDE0 + LOBE_GAP[3]) / 2, 0), (0.01, LOBE_GAP[3] - Y_SIDE0, 2 * zg), 'grey', bevel=0.002)
        for yy in (0.46, 0.54, 0.62, 0.69):
            box(f'recess.slat.{yy}', (XB + 0.012, yy, 0), (0.012, 0.025, 2 * zg - 0.04), 'darkmetal', bevel=0.003)
        # bright beading across the face at the top of the cream band (front elevation, centres of the
        # outline strokes: 1,590-1,638 mm)
        S = breaks(-1.0, 1.0, [], lambda m: 0.04)
        bright_strip(f, S, 1.592, 1.638)
    for side in (1, -1):
        add_grid(acc('under'), lambda x, u, s=side: surf(x, u, s), eased(E), lin(0.0, u_of_y(Y_SIDE0), 3))
    S = breaks(-1.0, s_hi, [], lambda m: 0.04)
    bm = acc('alu')
    add_grid(bm, f, S, [SKIRT - 0.008, SKIRT - 0.004], 0.0015)
    add_grid(bm, f, S, [SKIRT - 0.004, SKIRT + 0.010], 0.0035)
    add_grid(bm, f, S, [SKIRT + 0.010, SKIRT + 0.014], 0.0015)
    for h in holes:
        reveal(f, h, 'rubber', astep=0.03, start=0.006)
        rcl = None if h[0] > -0.99 else (WS_RC_L if E is FRONT else RW_RC_L)
        rcr = RW_RC_R if (E is REAR and h[1] > 0.9) else None
        gasket(f, h, ka=ka, rc=WS_RC if E is FRONT else RW_RC, rcl=rcl, rcr=rcr)

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
def tilted_pill(name, w, h, d, push, material, r=None):
    """Rounded-end (pill) slab in the plane of the dome, like the sign in the front elevation."""
    r = h / 2 * 0.92 if r is None else r
    bm = bmesh.new()
    ring = []
    for (cx, cy, a0) in ((w / 2 - r, h / 2 - r, 0), (-(w / 2 - r), h / 2 - r, 90), (-(w / 2 - r), -(h / 2 - r), 180), (w / 2 - r, -(h / 2 - r), 270)):
        for k in range(9):
            a = math.radians(a0 + 90 * k / 8)
            ring.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    back = [bm.verts.new(_RT @ Vector((x, y, -d / 2))) for x, y in ring]
    front = [bm.verts.new(_RT @ Vector((x, y, d / 2))) for x, y in ring]
    bm.faces.new(front)
    bm.faces.new(list(reversed(back)))
    m = len(ring)
    for k in range(m):
        bm.faces.new([back[k], back[(k + 1) % m], front[(k + 1) % m], front[k]])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.translate(bm, vec=_dc + _tilt * push, verts=bm.verts)
    return finish(bm, name, material, sharp=40)
def _rounded_poly(V, R, n=6):
    """Closed outline of polygon V (counter-clockwise) with each corner i rounded to radius R[i]."""
    out = []
    m = len(V)
    for i in range(m):
        P_, C_, N_ = Vector((*V[i - 1], 0)), Vector((*V[i], 0)), Vector((*V[(i + 1) % m], 0))
        u1, u2 = (P_ - C_).normalized(), (N_ - C_).normalized()
        th = u1.angle(u2) / 2
        d = R[i] / math.tan(th)
        ctr = C_ + (u1 + u2).normalized() * (R[i] / math.sin(th))
        a0, a1 = C_ + u1 * d - ctr, C_ + u2 * d - ctr
        ang0, ang1 = math.atan2(a0.y, a0.x), math.atan2(a1.y, a1.x)
        while ang1 > ang0: ang1 -= 2 * math.pi        # corners of a counter-clockwise outline turn left
        while ang1 < ang0 - 2 * math.pi: ang1 += 2 * math.pi
        if ang0 - ang1 > math.pi: ang1 += 2 * math.pi
        for k in range(n + 1):
            a = ang0 + (ang1 - ang0) * k / n
            out.append((ctr.x + R[i] * math.cos(a), ctr.y + R[i] * math.sin(a)))
    return out

def surface_pill(name, w, h, off, material, yc=2.70, r=None, rings=6, wb=None, rt=None, rb=None, inward=False):
    """Panel laid on the curved face of the dome (it follows the coachwork instead of standing proud of it
    at its ends): a pill, or with wb (bottom width) a trapezoid with rounded corners (top rt, bottom rb)."""
    ring = _pill_ring(w, h, r, wb, rt, rb)
    zb = 1.249 * FRONT.q
    def P(z, y):
        return nose(FRONT, z, y) + normal(fz, z / zb, y) * off
    bm = bmesh.new()
    levels = [1.0 - k / rings for k in range(rings)]
    vs = [[bm.verts.new(P(z * t, yc + y * t)) for z, y in ring] for t in levels]
    c = bm.verts.new(P(0.0, yc))
    m = len(ring)
    for i in range(len(levels) - 1):
        for k in range(m):
            bm.faces.new([vs[i][k], vs[i][(k + 1) % m], vs[i + 1][(k + 1) % m], vs[i + 1][k]])
    for k in range(m):
        bm.faces.new([vs[-1][k], vs[-1][(k + 1) % m], c])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for fc in bm.faces:
        fc.normal_update()
        if (fc.normal.x < 0) != inward:
            fc.normal_flip()
    return finish(bm, name, material, smooth=True, sharp=60)

_PILL_COUNTS = []
def _pill_ring(w, h, r=None, wb=None, rt=None, rb=None, counts=None):
    _PILL_COUNTS.clear()
    r = h / 2 * 0.92 if r is None else r
    ring = []
    if wb is not None:
        ring = _rounded_poly([(w / 2, h / 2), (-w / 2, h / 2), (-wb / 2, -h / 2), (wb / 2, -h / 2)], [rt, rt, rb, rb])
    else:
      for (cx, cy, a0) in ((w / 2 - r, h / 2 - r, 0), (-(w / 2 - r), h / 2 - r, 90), (-(w / 2 - r), -(h / 2 - r), 180), (w / 2 - r, -(h / 2 - r), 270)):
        for k in range(9):
            a = math.radians(a0 + 90 * k / 8)
            ring.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    full = []
    for i, q in enumerate(ring):                      # insert points along every long straight edge
        full.append(q)
        q2 = ring[(i + 1) % len(ring)]
        n_ = int(math.hypot(q2[0] - q[0], q2[1] - q[1]) / 0.04) if counts is None else counts[i]   # (also the sloping sides)
        _PILL_COUNTS.append(n_)
        for j in range(1, n_):
            t = j / n_
            full.append((q[0] + (q2[0] - q[0]) * t, q[1] + (q2[1] - q[1]) * t))
    return full

def surface_band(name, ring_a, off_a, ring_b, off_b, material, yc=2.70, flip=False):
    """Strip between two outlines of equal point count laid on the dome at two offsets along its normal
    (a flat ring when the offsets match, the walls of a recess when the outlines match)."""
    zb = 1.249 * FRONT.q
    P = lambda z, y, o: nose(FRONT, z, yc + y) + normal(fz, z / zb, yc + y) * o
    bm = bmesh.new()
    A = [bm.verts.new(P(z, y, off_a)) for z, y in ring_a]
    if all(q == ring_b[0] for q in ring_b):           # collapsed outline: a cap fanned from one point
        c_ = bm.verts.new(P(*ring_b[0], off_b)); B = [c_] * len(ring_b)
    else:
        B = [bm.verts.new(P(z, y, off_b)) for z, y in ring_b]
    m = len(A)
    for k in range(m):
        q = [A[k], A[(k + 1) % m], B[(k + 1) % m], B[k]]
        if q[2] is q[3]:
            q = q[:3]
        f_ = bm.faces.new(q)
        if flip: f_.normal_flip()
    return finish(bm, name, material, smooth=True, sharp=60)
# front elevation: dark grey panel 1,056 x 200 mm with strongly rounded ends, 2.60-2.80 m up; it lies on the
# dome, inside its outline (both side elevations and the photo: it does not stand out of the coachwork)
# photo of 6149: the box is a trapezoid with rounded corners, wider at the top, its sides leaning in by
# ~4.5 % of the width each; small radii at the top, larger at the bottom. Size and height as before.
RS_W, RS_H, RS_TAPER = 1.095, 0.215, 0.09            # RS_W: width at mid-height, as the old box
# (turned upside down at the owner's request: the top edge is the shorter one; corners another ~20 % rounder)
_wb = RS_W * (1 + RS_TAPER / 2)                       # bottom width
_wt = RS_W * (1 - RS_TAPER / 2)                       # top width
RS_RT, RS_RB = 0.077, 0.060                           # corner radii at the top and bottom
RS_LIP = 0.018                                        # dark frame round the opening, at the coachwork surface
RS_DEPTH = 0.050                                      # the blind and lettering sit ~50 mm behind the glass
_out = _pill_ring(_wt, RS_H, wb=_wb, rt=RS_RT, rb=RS_RB)
_RS_COUNTS = list(_PILL_COUNTS)                        # inner outlines reuse these splits: same point count
def _sign_outline(inset):
    k = inset * 2 * math.sqrt(1 + (RS_TAPER * RS_W / 2 / RS_H) ** 2)    # the sloping sides move in a little more
    return _pill_ring(_wt - k, RS_H - 2 * inset, wb=_wb - k, rt=RS_RT - inset, rb=RS_RB - inset, counts=_RS_COUNTS)
_open = _sign_outline(RS_LIP)
assert len(_out) == len(_open)
surface_band('route_box.frame', _out, 0.0015, _open, 0.0015, 'darkmetal')                       # lip at the surface
surface_band('route_box.reveal', _open, 0.0015, _open, -RS_DEPTH - 0.002, 'darkmetal', flip=True)  # walls of the recess
surface_pill('route_box.back', _wt - 0.04, RS_H - 0.04, -RS_DEPTH - 0.002, 'darkmetal',          # back of the recess
             wb=_wb - 0.04, rt=RS_RT - RS_LIP, rb=RS_RB - RS_LIP, rings=4)
_ib = _sign_outline(RS_LIP + 0.010)                   # the blind: a slightly raised panel at the back
surface_band('route_box.blind_edge', _open, -RS_DEPTH - 0.002, _ib, -RS_DEPTH + 0.001, 'darkmetal')
# cutter for the opening in the painted skin of the dome (applied after the build, then removed)
_cut = _sign_outline(RS_LIP - 0.004)
surface_band('route_box.cut_side', _cut, 0.03, _cut, -RS_DEPTH - 0.03, 'cutter')   # past the back of the box
surface_band('route_box.cut_a', _cut, 0.03, [(0.0, 0.0)] * len(_cut), 0.03, 'cutter')
surface_band('route_box.cut_b', _cut, -RS_DEPTH - 0.03, [(0.0, 0.0)] * len(_cut), -RS_DEPTH - 0.03, 'cutter')
# clear glass over the whole box, flush with the coachwork: a thin sheet with an outer and an inner face
# (parallel faces: it reflects like glass but does not shift what is behind it)
for nm_, off_, inw_ in (('glass', 0.0048, False), ('glass_in', 0.0039, True)):
    surface_pill(f'route_box.{nm_}', _wt - 0.004, RS_H - 0.004, off_, 'signglass',
                 wb=_wb - 0.004, rt=RS_RT - 0.002, rb=RS_RB - 0.002, inward=inw_, rings=4)
# edge of the raised panel round the sign: arches over it and runs down to the windscreen corners
_arch = [(-0.635, 2.760), (-0.640, 2.820), (-0.620, 2.865), (-0.570, 2.885), (-0.40, 2.893), (0.0, 2.896)]
_arch = _arch + [(-z, y) for z, y in reversed(_arch[:-1])]
rod('route_box.panel_edge', [nose(FRONT, z, y) + normal(fz, z / (1.249 * FRONT.q), y) * 0.002 for z, y in _arch], 0.006, 'lettering', seg=8)
# rain gutter over the windscreen: straight across just above the glass, ends hooked down outside it
_gut = [(-0.930, 2.390), (-0.934, 2.450), (-0.920, 2.495), (-0.885, 2.519), (-0.84, 2.524), (0.0, 2.524)]
_gut = [(-z, y) for z, y in reversed(_gut[:-1])]                 # right half with its hook (front elevation)
_ka = 1.249 * FRONT.q
_pts = [nose(FRONT, z, y) + normal(fz, z / _ka, y) * 0.012 for z, y in _gut]
# left half (left elevation): over the top of the large cab side window and its rounded rear corner,
# ending in a short hook just behind the mask joint (~5.98 m, 2.37 m)
_cA, _cy = -(0.988 * _ka - WS_RC_L[1][0]), WS_Y1 - WS_RC_L[1][1]
_left = [(0.0, WS_Y1 + 0.005)] + [(-0.84 * (1 - t) - (-_cA) * t, WS_Y1 + 0.005) for t in (0.25, 0.5, 0.75, 1.0)]
for th in np.linspace(0, 64, 11)[1:]:
    t = math.radians(th)
    _left.append((_cA - (WS_RC_L[1][0] + 0.02) * math.sin(t), _cy + (WS_RC_L[1][1] + 0.005) * math.cos(t)))
_pts += [nose_s(FRONT, a / _ka, y) + normal(fz, max(-0.999, a / _ka), y) * 0.012 for a, y in _left]
rod('windscreen.gutter', _pts, 0.013, 'rubber', seg=8)
# one line, as on 6149: the route number a little taller than the destination, plain DIN-style lettering
def _text_line(name, parts, center, bx, by, material, gap=0.05):
    bms = []
    for s_, size, font, spacing in parts:
        cu = bpy.data.curves.new(name, 'FONT')
        cu.body, cu.size, cu.extrude, cu.font = s_, size, 0.0005, font
        cu.align_x, cu.align_y, cu.space_character, cu.resolution_u = 'LEFT', 'BOTTOM_BASELINE', spacing, 3
        ob = bpy.data.objects.new(name + '_tmp', cu); COL.objects.link(ob)
        bpy.context.view_layer.update()
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
        bpy.data.objects.remove(ob)
        bm = bmesh.new(); bm.from_mesh(me); bpy.data.meshes.remove(me)
        xs = [v.co.x for v in bm.verts]
        bms.append((bm, min(xs), max(xs), max(v.co.y for v in bm.verts)))
    total = sum(b[2] - b[1] for b in bms) + gap * (len(bms) - 1)
    cap = max(b[3] for b in bms)
    out = bmesh.new()
    x = -total / 2
    for bm, x0_, x1_, _ in bms:
        bmesh.ops.translate(bm, vec=Vector((x - x0_, -cap / 2, 0)), verts=bm.verts)
        me = bpy.data.meshes.new('tmp'); bm.to_mesh(me); out.from_mesh(me); bpy.data.meshes.remove(me)
        x += (x1_ - x0_) + gap
    # lay the lettering on the curved dome (it stays under the glass across its whole width)
    zb_ = 1.249 * FRONT.q
    for v in out.verts:
        z_, y_ = -v.co.x, 2.70 + v.co.y
        v.co = nose(FRONT, z_, y_) + normal(fz, z_ / zb_, y_) * (-RS_DEPTH + 0.0015 + v.co.z)   # on the blind
    return finish(out, name, material, smooth=False)
_text_line('route_box.text', [('4', 0.150, FONT_ROUTE, 1.0), ('LIHOVAR', 0.122, FONT_ROUTE, 1.10)],
           _dc, _bx, _up, 'signtext', gap=0.065)

def on_glass(z, y, out):
    zb = prof(u_of_y(y))[0] * taper(y) * FRONT.q
    return nose_s(FRONT, z / zb, y) + normal(fz, z / zb, y) * out
# centre post dividing the windscreen (front elevation), on the outside of the glass
_pb, _pt = on_glass(0.0, WS_Y0 + 0.01, -0.004), on_glass(0.0, WS_Y1 - 0.01, -0.004)
_d = (_pt - _pb).normalized()
bm = bmesh.new()
box_into(bm, (0, 0, 0), (0.022, (_pt - _pb).length, 0.036), 0.004,
         rot=Vector((0, 1, 0)).rotation_difference(_d).to_matrix())
bmesh.ops.translate(bm, vec=(_pb + _pt) / 2, verts=bm.verts)
finish(bm, 'windscreen.centre_post', 'chrome', sharp=50)
for s in (1, -1):                                     # wipers: pivot at the bottom, arm, blade parked upright
    sp = on_glass(s * 0.30, 1.650, 0.012)
    b0, b1 = on_glass(s * 0.042, 1.76, 0.022), on_glass(s * 0.042, 2.19, 0.022)
    bmid = (b0 + b1) / 2
    cyl(f'wiper.spindle.{s}', sp - Vector((0.02, 0, 0)), sp + Vector((0.015, 0, 0)), 0.016, 'darkmetal', seg=12, bevel=0.002)
    rod(f'wiper.arm.{s}', [sp + Vector((0.012, 0, 0)), bmid], 0.011, 'darkmetal', seg=8)
    d = (b1 - b0).normalized()
    bm = bmesh.new()
    box_into(bm, (0, 0, 0), (0.030, (b1 - b0).length, 0.014), 0.003, rot=Vector((0, 1, 0)).rotation_difference(d).to_matrix())
    bmesh.ops.translate(bm, vec=bmid, verts=bm.verts)
    finish(bm, f'wiper.blade.{s}', 'rubber', sharp=50)

def bumper(name, E, y=0.69, half=1.20, out=0.035, th=0.05, h=0.08, n=72, material='alu', r=0.012, zr=None, stripe=0.008):
    """Bumper bar with a rounded section, following the mask."""
    bm = bmesh.new()
    f = nose_pt(E)
    zb = prof(u_of_y(y))[0] * taper(y) * E.q
    sec = []
    for (cx, cy, a0) in ((th / 2 - r, h / 2 - r, 0), (-(th / 2 - r), h / 2 - r, 90), (-(th / 2 - r), -(h / 2 - r), 180), (th / 2 - r, -(h / 2 - r), 270)):
        for k in range(4):
            a = math.radians(a0 + 30 * k)
            sec.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    rings, band = [], []
    z_lo, z_hi = zr if zr else (-half, half)
    for i in range(n + 1):
        z = z_lo + (z_hi - z_lo) * i / n
        s = z / zb
        p = f(s, y)
        nn = normal(f, max(-0.999, min(0.999, s)), y)
        nn.y = 0
        nn.normalize()
        up = Vector((0, 1, 0))
        rings.append([bm.verts.new(p + nn * (out + dx) + up * dy) for dx, dy in sec])
        band.append((p + nn * (out + th / 2 + 0.0008), up.copy(), nn.copy()))
    m = len(sec)
    for i in range(n):
        for k in range(m):
            bm.faces.new([rings[i][k], rings[i + 1][k], rings[i + 1][(k + 1) % m], rings[i][(k + 1) % m]])
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if stripe:                                        # two red lines along the outer face (photos of 6149)
        sb = bmesh.new()
        sp = (h - 2 * r) / 4                              # two lines, splitting the flat face into thirds
        for yc in (-sp * 0.85, sp * 0.85):
            vv = [(sb.verts.new(c + u * (yc - stripe / 2)), sb.verts.new(c + u * (yc + stripe / 2))) for c, u, _ in band]
            for i in range(n):
                f_ = sb.faces.new([vv[i][0], vv[i + 1][0], vv[i + 1][1], vv[i][1]])
                f_.normal_update()
                if f_.normal.dot(band[i][2]) < 0:
                    f_.normal_flip()
        finish(sb, name + '.stripe', 'stripe', sharp=60)
    return finish(bm, name, material, sharp=60)

bumper('bumper_front', FRONT, y=0.8015, out=0.045, th=0.045, h=0.133, r=0.015, zr=(-1.04, 1.04), n=48)   # 0.735-0.868 m; right end to the corner, meeting the side strip at the front door

def flap(name, E, f):
    """Small rectangular flap in the middle of the end, just under the belt (front and rear elevations)."""
    p = nose(E, 0.0, 1.505)
    n = normal(f, 0.0, 1.505)
    for nm, size, push, mt in ((f'{name}.seam', (0.004, 0.135, 0.305), 0.0, 'under'), (f'{name}.lid', (0.006, 0.125, 0.295), 0.002, 'redpaint')):
        bm = bmesh.new()
        box_into(bm, (0, 0, 0), size, 0.002)
        bmesh.ops.transform(bm, matrix=Matrix.Translation(p + n * push) @ Vector((1, 0, 0)).rotation_difference(n).to_matrix().to_4x4(), verts=bm.verts)
        finish(bm, nm, mt, sharp=50)
def seam_outline(name, E, f, zc, yc, w, h, line=0.008):
    for k, (z0, z1, y0, y1) in enumerate(((zc - w / 2, zc + w / 2, yc + h / 2 - line, yc + h / 2),
                                           (zc - w / 2, zc + w / 2, yc - h / 2, yc - h / 2 + line),
                                           (zc - w / 2, zc - w / 2 + line, yc - h / 2, yc + h / 2),
                                           (zc + w / 2 - line, zc + w / 2, yc - h / 2, yc + h / 2))):
        zb = prof(u_of_y(yc))[0] * taper(yc) * E.q
        add_grid(acc('rubber'), f, breaks(z0 / zb, z1 / zb, [], lambda m: 0.02), breaks(y0, y1, [], lambda m: 0.01), 0.0012)   # follows the face's kink at 1.45 m
seam_outline('front_flap', FRONT, fz, 0.0, 1.485, 0.316, 0.130)

FACE_X, FACE_Y = Vector((0, 0, -1)), Vector((0, 1, 0))
p = nose(FRONT, 0.0, 1.068)                           # front elevation: digits 137 mm high, 385 mm wide
_up = (nose(FRONT, 0.0, 1.20) - nose(FRONT, 0.0, 0.95)).normalized()      # the face leans back towards the bottom
text_mesh('fleet_number', NUMBER, 0.277, p + _up.cross(FACE_X).normalized() * -0.0012, FACE_X * 0.812, _up, 'gold', extrude=0.0008, font=FONT_NUM)
if '--tatra' in ARGS:
    p = nose(FRONT, 0.0, 0.92)
    text_mesh('tatra_lettering', 'TATRA', 0.11, p + Vector((0.004, 0, 0)), FACE_X, FACE_Y, 'chrome', extrude=0.006)

# horn under the front end, door side (ASSUMED position)
cyl('horn.bell', (5.12, 0.40, 0.55), (5.39, 0.40, 0.55), 0.022, 'chrome', seg=20, r2=0.055)   # behind the guard plate
cyl('horn.body', (4.99, 0.40, 0.55), (5.12, 0.40, 0.55), 0.03, 'darkmetal', seg=16, bevel=0.004)

# ============================================================================== headlights, mirror
part('headlights', explode=(1.5, 1.3, 0))
for s in (1, -1):
    z, y = s * 0.50, 1.073                           # front elevation: 1.00 m apart, centres 1.073 m up, ~200 mm
    p = nose(FRONT, z, y)
    n = normal(fz, z / 1.2, y)
    fixed_uv(cyl(f'headlamp.body.{s}', p - n * 0.02, p + n * 0.05, 0.10, 'paint', seg=40, bevel=0.01), (0.5, 0.82))
    cyl(f'headlamp.rim.{s}', p + n * 0.048, p + n * 0.064, 0.099, 'chrome', seg=40, bevel=0.006)
    cyl(f'headlamp.lens.{s}', p + n * 0.064, p + n * 0.080, 0.062, 'lens', seg=40, r2=0.050)
    # (no position lamps on the front face: photo of 6149, which shows only the side markers)

part('mirror', explode=(1.0, 1.3, 0.6))
# photo of 6149: a round light head on a short stalk outside the body at each lower corner of the
# windscreen; the stalk rises from the corner just above the colour line. Head ~95 mm across, ~510 mm
# behind the front end, ~1.82 m up (side elevation and the measured values).
_ff = side_pt(1)
for s in (1, -1):
    xm = 6.49
    root = surf(xm + 0.03, u_of_y(1.665), s)
    nr = normal(_ff, xm + 0.03, 1.665); nr = Vector((nr.x, nr.y, s * abs(nr.z)))
    sd = surf(xm, u_of_y(1.825), s)
    hc = Vector((xm, 1.825, sd.z + s * 0.085))
    rod(f'mirror.stalk.{s}', [root - nr * 0.01, root + nr * 0.03 + Vector((0, 0.035, 0)),
                              Vector((xm + 0.005, 1.765, hc.z - s * 0.012)), hc + Vector((0, -0.035, 0))], 0.008, 'chrome', seg=8)
    cyl(f'mirror.foot.{s}', root - nr * 0.004, root + nr * 0.012, 0.018, 'chrome', seg=12, bevel=0.003)
    ax = Vector((1, 0, 0))                           # light back faces forwards, glass looks back along the side
    cyl(f'mirror.head.{s}', hc - ax * 0.010, hc + ax * 0.014, 0.048, 'lamp', seg=24, bevel=0.006)
    cyl(f'mirror.glass.{s}', hc - ax * 0.013, hc - ax * 0.010, 0.041, 'mirror', seg=24)

# ============================================================================== rear mask
part('rear_mask', explode=(-1.0, 1.3, 0))
mask(REAR, REAR_HOLES)
flush('rear_mask')
# rear elevation: bumper strip 746-857 mm, stopping at the door strip on the right
_zb08 = prof(u_of_y(0.8015))[0] * taper(0.8015) * REAR.q
bumper('bumper_rear', REAR, y=0.8015, out=0.012, th=0.03, h=0.111, r=0.010, zr=(-0.997 * _zb08, 0.997 * _zb08), n=48)
# plate frame drawn as a flat outline (rear elevation: 304 x 133 mm, centre 1.49 m up)
seam_outline('rear_flap', REAR, rz, 0.0, 1.4925, 0.304, 0.133)
def _basis(n):
    """Rotation taking local x to the surface normal n with local y kept upright (no roll)."""
    ex = n.normalized()
    ey = (Vector((0, 1, 0)) - ex * ex.y).normalized()
    ez = ex.cross(ey)
    M = Matrix((ex, ey, ez)).transposed()
    return M.to_4x4()
# pill-shaped lamp (photo of 6149): a stadium outline with fully round ends, local x out of the face,
# y up, z along the lamp. Bright metal rim round a slightly domed lens.
TL_L, TL_H = 0.262, 0.100                                  # outline 262 x 100 mm (rear elevation; photo's 2.6 : 1)

def _stadium(L, H, n=10):
    r, ls = H / 2, (L - H) / 2
    mid = [z for z in (0.075, 0.032, -0.032, -0.075) if abs(z) < ls - 0.005]
    pts = [(r * math.sin(a), ls + r * math.cos(a)) for a in np.linspace(math.pi / 2, -math.pi / 2, n + 1)]
    pts += [(-r, z) for z in mid]                          # cuts along the straight edges, so the red middle has its own faces
    pts += [(r * math.sin(a), -ls + r * math.cos(a)) for a in np.linspace(-math.pi / 2, -3 * math.pi / 2, n + 1)]
    pts += [(r, z) for z in reversed(mid)]
    return pts                                             # (y, z), closed loop, clockwise seen from outside

def _pill_rim(name, TM):
    rings = [(TL_L, TL_H, 0.000), (TL_L, TL_H, 0.008), (TL_L - 0.006, TL_H - 0.006, 0.0125),
             (TL_L - 0.016, TL_H - 0.016, 0.0125), (TL_L - 0.016, TL_H - 0.016, 0.006)]
    bm = bmesh.new()
    loops = [[bm.verts.new((x, y, z)) for y, z in _stadium(L, H)] for L, H, x in rings]
    N = len(loops[0])
    for a, b in zip(loops, loops[1:]):
        for i in range(N):
            bm.faces.new((a[i], a[(i + 1) % N], b[(i + 1) % N], b[i]))
    bm.faces.new(list(reversed(loops[0])))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.transform(bm, matrix=TM, verts=bm.verts)
    finish(bm, name, 'chrome', sharp=60)

def _pill_lens(name, TM, keep, mat):
    L, H = TL_L - 0.020, TL_H - 0.020
    ls = (L - H) / 2
    bm = bmesh.new()
    loops = []
    for t in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        ring = []
        for y, z in _stadium(L, H):
            zc = max(-ls, min(ls, z))
            ring.append(bm.verts.new((0.010 + 0.007 * (1 - t * t), t * y, zc + t * (z - zc))))
        loops.append(ring)
    N = len(loops[0])
    for a, b in zip(loops, loops[1:]):
        for i in range(N):
            f = bm.faces.new((a[i], a[(i + 1) % N], b[(i + 1) % N], b[i]))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    drop = [f for f in bm.faces if not keep(f.calc_center_median().z) or f.calc_area() < 1e-9]
    bmesh.ops.delete(bm, geom=drop, context='FACES')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        if f.normal.x < 0: f.normal_flip()
    bmesh.ops.transform(bm, matrix=TM, verts=bm.verts)
    finish(bm, name, mat, sharp=60)

# tail-light units: pills with round ends, amber - red - amber in a bright rim (photo of 6149), ~262 x 90 mm, 0.95 m up (rear
# elevation). Photo of 6102 (the prototype this car follows): low in the red band and out near the corners,
# but wholly on the rear face, symmetric about the centre line; no lamps on the side walls. Each unit sits
# as far out as it can while its outer end still faces within ~33 deg of straight back.
def _face_ang(z, y):
    n = normal(rz, z / (1.25 * REAR.q), y)
    return math.degrees(math.acos(max(-1.0, min(1.0, -n.x / n.length))))
TL_Z = max(z for z in np.linspace(0.39, 0.80, 83) if _face_ang(z + 0.133, 0.95) <= 33.0)
print('TAIL', round(TL_Z, 3), round(_face_ang(TL_Z, 0.95), 1), round(_face_ang(TL_Z + 0.133, 0.95), 1))
for s in (1, -1):
    p = nose(REAR, s * TL_Z, 0.950)
    n = normal(rz, s * TL_Z / (1.25 * REAR.q), 0.950)
    R = _basis(n)
    TM = Matrix.Translation(p + n * 0.004) @ R
    _pill_rim(f'tail_unit.{s}.rim', TM)
    _pill_lens(f'tail_unit.{s}.amber', TM, lambda zc: abs(zc) > 0.032, 'amber')
    _pill_lens(f'tail_unit.{s}.red', TM, lambda zc: abs(zc) <= 0.032, 'redlens')
p = nose(REAR, 0.0, 1.077)                              # rear elevation: same digits as the front, 136 x 382 mm
_upR = (nose(REAR, 0.0, 1.16) - nose(REAR, 0.0, 1.00)).normalized()        # lies in the leaning face
text_mesh('rear_number', NUMBER, 0.277, p + normal(rz, 0.0, 1.077) * 0.0012, Vector((0, 0, 1)) * 0.812, _upR, 'gold', extrude=0.0008, font=FONT_NUM)
# raised panel on the rear dome, outlined as at the front (rear elevation: 1.21 m wide, top ~2.89 m)
_arch_r = [(-0.605, 2.790), (-0.610, 2.835), (-0.590, 2.868), (-0.540, 2.884), (-0.40, 2.890), (0.0, 2.892)]
_arch_r = _arch_r + [(-z, y) for z, y in reversed(_arch_r[:-1])]
rod('rear_panel_edge', [nose(REAR, z, y) + normal(rz, z / (1.25 * REAR.q), y) * 0.002 for z, y in _arch_r], 0.006, 'darkmetal', seg=8)

# ============================================================================== doors (four-leaf folding)
LEAF_T = 0.050                                       # leaf thickness: a hollow frame ~50 mm deep (photos of open doors)
LEAF_Y0 = 0.350                                      # leaf bottom, level with the body's bottom edge
OPEN_DEG = 85                                        # leaves open to about 85 deg (not quite square to the side)
DW_HALF = 0.100                                      # door windows: 200 mm wide
DW_UP, DW_LO = (1.613, 2.468), (0.654, 1.541)
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
        # photos of 6102 and 7807: two tall windows per leaf, centred in it, with fully rounded ends,
        # ~200 mm wide; the upper one starts ~72 mm under the door top, ~72 mm between them, the lower
        # one ends ~320 mm above the door bottom
        cx_ = (a + b) / 2
        glass = [(cx_ - DW_HALF, cx_ + DW_HALF, DW_LO[0], DW_LO[1]), (cx_ - DW_HALF, cx_ + DW_HALF, DW_UP[0], DW_UP[1])]
        front, back = leaf_pt(0.0), leaf_pt(LEAF_T)
        cells(front, a, b, LEAF_Y0, 2.54, glass, 0.4, lambda y: 0.2, lambda y: 'door', uvf=uvf)
        cells(back, a, b, LEAF_Y0, 2.54, glass, 0.4, lambda y: 0.2, lambda y: 'door', inward=True, uvf=uvf)
        for rect in glass:
            G0 = [rect[0] - 0.01, rect[1] + 0.01]
            G1 = breaks(rect[2] - 0.01, rect[3] + 0.01, [], lambda m: 0.3)
            add_grid(acc('glass'), leaf_pt(LEAF_T / 2 - 0.003), G0, G1)
            add_grid(acc('glass'), leaf_pt(LEAF_T / 2 + 0.003), G0, G1, inward=True)
            reveal(front, rect, 'door', depth=LEAF_T, uvf=uvf)
            rr_ = DW_HALF - 0.002
            corner_fillets(front, rect, rr_, rr_, 1.0, uvf, n=6, material='door')
            corner_fillets(back, rect, rr_, rr_, 1.0, uvf, n=6, material='door', inward=True)
            gasket(front, rect, rc=rr_, step=0.10)
        if i in (1, 2):
            # handle on the inside of each inner leaf (one per wing): a painted rod running diagonally across
            # the lower window, its upper end towards the leaf's hinge side, bent into the frame above and
            # below the window (photos of 7807 and 6102)
            sg = -1 if i == 1 else 1
            def _in(x, y, d):
                q = back(x, y)
                return q - normal(f, x, y) * d
            top, bot = (cx_ + sg * 0.068, DW_LO[1] - 0.05), (cx_ - sg * 0.068, DW_LO[0] + 0.05)
            pts = [_in(top[0], DW_LO[1] + 0.034, -0.004), _in(top[0], DW_LO[1] + 0.034, 0.030),
                   _in(top[0], top[1], 0.045), _in(bot[0], bot[1], 0.045),
                   _in(bot[0], DW_LO[0] - 0.034, 0.030), _in(bot[0], DW_LO[0] - 0.034, -0.004)]
            for p0, p1 in zip(pts, pts[1:]):
                cyl_into(acc('handle'), p0 - (p1 - p0).normalized() * 0.006, p1 + (p1 - p0).normalized() * 0.006, 0.0125, 8)
        reveal(front, (a, b, LEAF_Y0, 2.54), 'door', depth=LEAF_T, outward=True, uvf=uvf)
        hx = a if i in (0, 1) else b                # hinge knuckles on this leaf's hinge edge
        zf = f(hx, 1.5).z + 0.009
        for yk in (0.413, 1.627, 2.506):                # hinge marks at the top, middle and bottom (side elevation)
            cyl_into(acc('steel'), (hx, yk - 0.035, zf + 0.004), (hx, yk + 0.035, zf + 0.004), 0.011, 12, bevel=0.002)
        if i in (1, 2):                              # handle bar across the fold joint at mid-height
            jx = a if i == 1 else b
            box_into(acc('steel'), (jx + (0.03 if i == 1 else -0.03), 1.627, zf + 0.012), (0.06, 0.014, 0.012), 0.004)
        if i in (1, 2):                              # rubber edge seal where the pairs meet
            ex = b if i == 1 else a
            box_into(acc('rubber'), (ex + (0.004 if i == 1 else -0.004), 1.44, zf - LEAF_T / 2), (0.012, 2.19, LEAF_T - 0.012), 0.004)
        objs = flush(f'{pid}.leaf{i}')
        leaves.append((objs, a, b))
    hinge = [x0, x0 + w + gap / 2, x1 - w - gap / 2, x1]
    zh_ = lambda x: f(x, 1.5).z + 0.009 - LEAF_T / 2      # each hinge on the body side at its own x (the front door is on the taper)
    parents = [None, 0, 3, None]
    anchors = {}
    for i in (0, 3, 1, 2):
        objs, a, b = leaves[i]
        pv = (hinge[i], LEAF_Y0, zh_(hinge[i]))
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
    # step well from the plan of the coloured type drawing: the upper ("middle") step is an irregular hexagon
    # between the floor edge and the edge of the lower step; the lower step runs out to the door. Corners are
    # traced in plan, placed about the model's door centre and kept at the drawing's distances from the door.
    # The lower step lies just under the bottom of the leaves, so the folded wings pass over it.
    hx, dl, posts = STAIRS[pid]
    xc_d = (dl[0][0] + dl[1][0]) / 2
    zdoor = lambda x: f(x, 1.5).z + 0.009 - LEAF_T
    zdl = lambda x: dl[0][1] + (dl[1][1] - dl[0][1]) * (x - dl[0][0]) / (dl[1][0] - dl[0][0])
    def P(x, z):                                      # drawing plan point -> model (x, z)
        xm_ = x - xc_d + (x0 + x1) / 2
        return (xm_, z + zdoor(xm_) - zdl(x))
    H = [P(*q) for q in hx]
    D = [P(*q) for q in dl]
    # move the whole well inward (shape unchanged) until the folded leaf pairs, standing at right angles to
    # the side, clear the upper step with 20 mm to spare
    wl = (x1 - x0 - 3 * 0.012) / 4
    def _inside(pt, poly):
        c = False
        for (ax, az), (bx, bz) in zip(poly, poly[1:] + poly[:1]):
            if (az > pt[1]) != (bz > pt[1]) and pt[0] < ax + (pt[1] - az) * (bx - ax) / (bz - az):
                c = not c
        return c
    t = Vector((x1 - x0, 0, zdoor(x1) - zdoor(x0))).normalized()
    n_in = Vector((t.z, 0, -t.x))
    pocket = []
    ca, sa = math.cos(math.radians(OPEN_DEG)), math.sin(math.radians(OPEN_DEG))
    for xj, into in ((x0, 1), (x1, -1)):
        base = Vector((xj, 0, zdoor(xj)))
        d_open = n_in * sa + t * into * ca              # direction of the first leaf of the pair when open
        for k in range(13):
            for o_ in (-0.03, 0.02, 0.06, 0.10, 0.13):
                q = base + d_open * (wl + 0.02) * k / 12 + t * into * o_
                pocket.append((q.x, q.z))
    shift = 0.0
    while any(_inside((px, pz + shift), H) for px, pz in pocket) and shift < 0.3:
        shift += 0.005
    STAIR_SHIFT[pid] = shift
    P0 = P
    P = lambda x, z: (P0(x, z)[0], P0(x, z)[1] - shift)
    H = [(hx_, hz_ - shift) for hx_, hz_ in H]
    STAIR_FLOOR[pid] = (D[0][0], H[0], H[1], D[1][0])
    def prism(name, poly, y0, y1, mat):
        bm = bmesh.new()
        bot = [bm.verts.new((x, y0, z)) for x, z in poly]
        top = [bm.verts.new((x, y1, z)) for x, z in poly]
        bm.faces.new(top); bm.faces.new(list(reversed(bot)))
        n = len(poly)
        for i in range(n):
            bm.faces.new((bot[i], bot[(i + 1) % n], top[(i + 1) % n], top[i]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bmesh.ops.triangulate(bm, faces=[fc for fc in bm.faces if len(fc.verts) > 4])
        finish(bm, name, mat, smooth=False)
    def wall(name, line, y0, y1, mat, t=0.02):
        for k, (p0, p1) in enumerate(zip(line, line[1:])):
            dx, dz = p1[0] - p0[0], p1[1] - p0[1]
            L = math.hypot(dx, dz)
            box(f'{name}.{k}', ((p0[0] + p1[0]) / 2, (y0 + y1) / 2, (p0[1] + p1[1]) / 2), (L, y1 - y0, t), mat, bevel=0,
                rot=Matrix.Rotation(-math.atan2(dz, dx), 3, 'Y'))
    Y_UP, Y_LO = 0.640, 0.325
    prism(f'{pid}.tread.upper', H, Y_UP - 0.04, Y_UP, 'floor')
    # below the upper step the well's side walls stand behind the folded leaf pairs (100 mm outward)
    tx, tz = t.x, t.z
    def _out(q, sg): return (q[0] - sg * tx * 0.10, q[1] - sg * tz * 0.10)
    H5o, H2o = _out(H[5], 1), _out(H[2], -1)
    D0o, D1o = _out((D[0][0], zdoor(D[0][0])), 1), _out((D[1][0], zdoor(D[1][0])), -1)
    lower = [H5o, H[5], H[4], H[3], H[2], H2o, D1o, D0o]
    prism(f'{pid}.tread.lower', lower, Y_LO - 0.02, Y_LO, 'floor')        # (20 mm: stays behind the brown bar outside)
    wall(f'{pid}.riser.upper', [H[0], H[1]], Y_UP, FLOOR, 'under')                     # floor edge down to the upper step
    wall(f'{pid}.riser.lower', [H[2], H[3], H[4], H[5]], Y_LO, Y_UP - 0.04, 'under')  # upper step down to the lower one
    wall(f'{pid}.side.l', [D0o, H5o, H[5], H[0]], Y_LO - 0.02, FLOOR, 'under')
    wall(f'{pid}.side.r', [D1o, H2o, H[2], H[1]], Y_LO - 0.02, FLOOR, 'under')
    for nm, line, yy in (('upper', [H[2], H[3], H[4], H[5]], Y_UP), ('floor', [H[0], H[1]], FLOOR)):
        wall(f'{pid}.nosing.{nm}', line, yy - 0.006, yy + 0.006, 'steel', t=0.03)
    for k, q in enumerate(posts):                     # grab poles at the corners of the well (plan)
        xx, zz = P(*q)
        cyl(f'{pid}.pole.{k}', (xx, FLOOR, zz), (xx, 2.54, zz), 0.016, 'chrome', seg=14)

STAIRS = {   # plan of the coloured type drawing: upper-step hexagon (floor edge first), door line, pole feet
    'door_middle': ([(-1.382, 0.692), (-0.063, 0.692), (-0.063, 0.893), (-0.376, 0.997), (-1.106, 0.997), (-1.382, 0.893)],
                    [(-1.382, 1.184), (-0.063, 1.184)], [(-1.434, 0.71), (-0.010, 0.71)]),
    'door_front': ([(4.693, 0.650), (5.803, 0.505), (5.892, 0.721), (5.624, 0.848), (4.998, 0.933), (4.663, 0.882)],
                   [(4.655, 1.146), (6.019, 0.997)], [(4.581, 0.69)]),
    'door_rear': ([(-5.787, 0.513), (-4.661, 0.662), (-4.639, 0.888), (-4.967, 0.938), (-5.593, 0.856), (-5.891, 0.721)],
                  [(-5.980, 0.982), (-4.631, 1.161)], [(-4.557, 0.68), (-5.891, 0.564)]),
}
STAIR_FLOOR = {}
STAIR_SHIFT = {}
for pid, x0, x1 in DOOR_SPANS:
    door(pid, x0, x1)

# ============================================================================== interior
part('interior', explode=(0, 1.3, 0))
def lining_mat(y):
    return 'umakart' if y < WIN_Y0 else 'lining' if y < SIDE_TOP else 'ceiling'
for side in (1, -1):
    f = side_pt(side)
    cells(f, -XR, XS, FLOOR, SIDE_TOP, OPEN[side], 0.30, ystep, lining_mat, off=-WALL, inward=True, extra_b=(WIN_Y0,))
    add_grid(acc('ceiling'), lambda x, u, s=side: surf(x, u, s), breaks(-XR, XS, [], lambda m: 0.3), lin(u_of_y(SIDE_TOP), 1.0, 16), -WALL, inward=True)
for E, holes in ((FRONT, FRONT_HOLES), (REAR, REAR_HOLES)):
    opens_i = list(holes)
    cells(nose_pt(E), -1.0, 1.0, FLOOR, H_ROOF, opens_i, 0.08, lambda y: 0.03 if y > 2.70 else 0.12,
          lining_mat, off=-WALL, inward=True, extra_b=(WIN_Y0, SIDE_TOP))
INNER = 1.25 - WALL
pts = [(-XR, INNER)]
for _, x0, x1 in sorted(DOOR_SPANS, key=lambda d: d[1]):
    xa_, h0_, h1_, xb_ = STAIR_FLOOR[_]
    pts += [(xa_, INNER), h0_, h1_, (xb_, INNER)]
    if x1 < -TAPER_X0 < 0:
        pts += [(-TAPER_X0, INNER)]
def inner_w(x):
    return plan_w(x) * 1.25 - WALL
pts = [(x, inner_w(x) if z == INNER else z) for x, z in pts]
pts.append((XS, inner_w(XS)))
pts += [(x, inner_w(x)) for x in np.linspace(TAPER_X0, XS, 6)[1:]] if pts[-1][0] < TAPER_X0 else []
for E in (FRONT,):
    zb_f = prof(u_of_y(FLOOR))[0] * taper(FLOOR) * E.q
    for k in range(1, 24):
        s = 1 - 2 * k / 24
        q = nose_s(E, s * (zb_f - WALL) / zb_f, FLOOR)
        pts.append((q.x - WALL, q.z))
pts += [(x, -inner_w(x)) for x in np.linspace(XS, TAPER_X0, 6)] + [(x, -inner_w(x)) for x in np.linspace(-TAPER_X0, -XR, 6)]
zb_f = prof(u_of_y(FLOOR))[0] * taper(FLOOR)
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

# Single seats on both sides (1 + 1) as in the plan of the coloured type drawing, red leatherette on
# chrome tube frames; the heaters sit in the seat pedestals (Wikipedia).
SEAT_Y = FLOOR + 0.42
def seat(name, x, zc, width, material):
    box(f'{name}.cushion', (x, SEAT_Y, zc), (0.44, 0.09, width), material, bevel=0.03)
    box(f'{name}.back', (x - 0.235, SEAT_Y + 0.30, zc), (0.07, 0.50, width), material, bevel=0.03, rot=rot_z(-8))
    box(f'{name}.heater', (x - 0.03, FLOOR + 0.19, zc), (0.30, 0.36, width - 0.10), 'darkmetal', bevel=0.01)
    for dz in (-width / 2 + 0.03, width / 2 - 0.03):
        rod(f'{name}.frame.{dz:.2f}', [(x + 0.2, SEAT_Y - 0.05, zc + dz), (x - 0.25, SEAT_Y - 0.05, zc + dz),
                                       (x - 0.28, SEAT_Y + 0.56, zc + dz)], 0.011, 'chrome', seg=8)
    cyl(f'{name}.grab', (x - 0.28, SEAT_Y + 0.56, zc - width / 2 + 0.03), (x - 0.28, SEAT_Y + 0.56, zc + width / 2 - 0.03), 0.011, 'chrome', seg=8)
# Seat positions scaled off the plan (front to the right, 130 px per metre): 14 along the left side,
# 9 on the door side (4 between the rear and middle door, 2 + 3 either side of the conductor's place).
SEATS_L = (-5.70, -4.94, -4.20, -3.45, -2.72, -1.97, -1.21, -0.47, 0.28, 1.03, 1.78, 2.53, 3.27, 4.03)
SEATS_R = (-4.20, -3.47, -2.74, -1.98, 0.38, 1.09, 2.58, 3.34, 4.24)
for i, x in enumerate(SEATS_L):
    seat(f'seat.L{i}', x, -0.86 if i else -0.82, 0.44, 'leather_red')
for i, x in enumerate(SEATS_R):
    seat(f'seat.R{i}', x, 0.86, 0.44, 'leather_red')
# conductor's place on the door side between the middle and front door (plan): a seat facing the
# aisle behind a small desk, framed by chrome rails
XC = 1.86
box('conductor.cushion', (XC, SEAT_Y + 0.08, 0.92), (0.42, 0.09, 0.42), 'leather_red', bevel=0.03)
box('conductor.back', (XC, SEAT_Y + 0.40, 1.14), (0.42, 0.50, 0.07), 'leather_red', bevel=0.03)
cyl('conductor.post', (XC, FLOOR, 0.92), (XC, SEAT_Y + 0.03, 0.92), 0.035, 'darkmetal', seg=16)
box('conductor.desk', (XC, FLOOR + 0.50, 0.50), (0.62, 1.00, 0.22), 'umakart', bevel=0.012)
box('conductor.desk_top', (XC, FLOOR + 1.015, 0.52), (0.66, 0.03, 0.28), 'umakart', bevel=0.008)
for k in range(4):                                    # ticket holders on the desk (plan)
    box(f'conductor.tickets.{k}', (XC - 0.225 + k * 0.15, FLOOR + 1.06, 0.48), (0.11, 0.06, 0.10), 'darkmetal', bevel=0.006)
for xx in (XC - 0.33, XC + 0.33):
    rod(f'conductor.rail.{xx:.2f}', [(xx, FLOOR, 0.40), (xx, FLOOR + 1.05, 0.40), (xx, FLOOR + 1.05, 1.18)], 0.014, 'chrome', seg=12)
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
CZ = -0.20                                            # driver's seat and desk (plan)
box('cab.desk', (6.40, FLOOR + 0.33, CZ + 0.10), (0.40, 0.66, 1.25), 'cab', bevel=0.02)
box('cab.desk_top', (6.34, FLOOR + 0.70, CZ + 0.10), (0.44, 0.06, 1.29), 'cab', bevel=0.015, rot=rot_z(-12))
cyl('cab.controller', (6.32, FLOOR + 0.73, CZ + 0.28), (6.32, FLOOR + 0.80, CZ + 0.28), 0.06, 'darkmetal', seg=24, bevel=0.006)
cyl('cab.controller_handle', (6.32, FLOOR + 0.80, CZ + 0.28), (6.27, FLOOR + 0.86, CZ + 0.34), 0.012, 'chrome', seg=10)
for i, dz in enumerate((-0.30, -0.14, 0.02)):
    box(f'cab.dial.{i}', (6.46, FLOOR + 0.82, CZ + dz), (0.02, 0.10, 0.10), 'darkmetal', bevel=0.004)
box('cab.seat.cushion', (5.74, FLOOR + 0.50, CZ), (0.46, 0.09, 0.48), 'leather_grey', bevel=0.03)
box('cab.seat.back', (5.48, FLOOR + 0.82, CZ), (0.07, 0.56, 0.46), 'leather_grey', bevel=0.03)
cyl('cab.seat.post', (5.74, FLOOR, CZ), (5.74, FLOOR + 0.46, CZ), 0.04, 'darkmetal', seg=16)
for i, dz in enumerate((-0.20, 0.0, 0.20)):          # the T3 is driven with three pedals
    box(f'cab.pedal.{i}', (6.16, FLOOR + 0.05, CZ + dz), (0.22, 0.03, 0.10), 'rubber', bevel=0.006, rot=rot_z(-15))
box('cab.partition', (5.12, FLOOR + 0.55, -0.42), (0.03, 1.10, 1.62), 'lining', bevel=0.005,
    rot=Matrix.Rotation(math.radians(31), 3, 'Y'))   # diagonal, from the left wall towards the front door (plan)
flush('interior')

# ============================================================================== roof
part('roof_vents', explode=(0, 1.9, 0))
# three raised cream covers about 1.18 m long, standing ~60 mm above the roof (side elevation), with a
# dark air gap under the edge
# (rear ends: the right elevation puts them at -5.808 / -1.322 / 4.125 m, the left ~25 mm further forward)
for i, (x0, x1) in enumerate(((-5.796, -4.624), (-1.310, -0.138), (4.137, 5.300))):
    box(f'vent.{i}', ((x0 + x1) / 2, 3.052, 0), (x1 - x0, 0.06, 0.66), 'lettering', bevel=0.022)
    box(f'vent.{i}.gap', ((x0 + x1) / 2, 3.012, 0), (x1 - x0 - 0.04, 0.03, 0.62), 'darkmetal', bevel=0.004)

part('roof_equipment', explode=(0, 2.1, 0))
box('roof_hatch', (0.163, 3.040, 0.10), (0.430, 0.10, 0.62), 'darkmetal', bevel=0.02)            # top of the box over the middle door
def shoulder_box(name, x0, x1, side, y_lo, z_top, t_lo, t_hi, material):
    """A box lying on the curved roof shoulder: from the side wall at y_lo up over the edge to |z| = z_top,
    standing t_lo proud at the bottom and t_hi at the top."""
    u0 = u_of_y(y_lo)
    us = [u for u in np.linspace(u0, 1.0, 400) if prof(u)[0] >= z_top]
    u1 = us[-1]
    U = list(np.linspace(u0, u1, 12))
    X = list(np.linspace(x0, x1, 2))
    t = lambda u: t_lo + (t_hi - t_lo) * (u - u0) / (u1 - u0)
    g = lambda x, u: surf(x, u, side) + normal(lambda a, b: surf(a, b, side), x, u) * t(u)
    bm = acc(material)
    add_grid(bm, g, X, U)
    for xe in (x0, x1):                               # end faces down to the roof
        add_grid(bm, lambda d, u, xe=xe: surf(xe, u, side) + normal(lambda a, b: surf(a, b, side), xe, u) * t(u) * d,
                 [0.0, 1.0], U, orient=lambda c, xe=xe: Vector((1 if xe == x1 else -1, 0, 0)))
    add_grid(bm, lambda x, d: surf(x, u1, side) + normal(lambda a, b: surf(a, b, side), x, u1) * t(u1) * d, X, [0.0, 1.0],
             orient=lambda c: Vector((0, 1, -side)).normalized())
# Three dark boxes on each roof shoulder along the pantograph, seen in the side elevation as dark blocks
# 2.83-3.01 m up (x 1.48-2.49, 2.52-3.55, 3.58-4.09 m); their contents are not stated in the sources.
for side in (1, -1):
    boxes = ((1.477, 2.494), (2.524, 3.547), (3.577, 4.091)) if side == 1 else ((1.505, 2.474), (2.504, 3.558), (3.588, 4.090))
    for k, (x0, x1) in enumerate(boxes):              # (left elevation: joints at 2.489 and 3.573 m)
        shoulder_box(f'roof_box.{side}.{k}', x0, x1, side, 2.83, 0.25, 0.004, 0.008, 'darkmetal')   # nearly flush; up to the roof top line (left elevation)
# dark box on the right roof shoulder over the front edge of the middle door, from the gutter up over the
# roof edge to ~3.09 m (side elevation: x -0.06..0.37 m)
shoulder_box('roof_box.door', -0.060, 0.365, 1, 2.60, 0.15, 0.006, 0.012, 'darkmetal')
box('roof_box.door.foot', surf(0.1525, u_of_y(2.605), 1) + Vector((0, 0, 0.012)), (0.465, 0.03, 0.024), 'darkmetal', bevel=0.006)
flush('roof_equipment')
# low cover behind the pantograph, high at the front and sloping down to the rear (both elevations,
# front and rear view: about 0.6 m wide, top ~3.38 m above rail); contents not stated in the sources
bm = bmesh.new()
# side elevation: from x 1.00 m (rounded, sloping rear end) to 2.45 m, top rising to 3.36 m at the front
_prof = [(2.45, 3.005), (2.45, 3.335), (2.41, 3.362), (1.25, 3.30), (1.08, 3.27), (1.02, 3.20), (1.00, 3.10), (1.00, 3.005)]
_vs = [[bm.verts.new((x, y, z)) for x, y in _prof] for z in (-0.30, 0.30)]
bm.faces.new(list(reversed(_vs[0])))
bm.faces.new(_vs[1])
for k in range(len(_prof)):
    bm.faces.new([_vs[0][k], _vs[0][(k + 1) % len(_prof)], _vs[1][(k + 1) % len(_prof)], _vs[1][k]])
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
bmesh.ops.bevel(bm, geom=[e for e in bm.edges if e.calc_face_angle(0) > math.radians(20)], offset=0.025, segments=3, affect='EDGES', profile=0.5)
finish(bm, 'roof_cover', 'grey', sharp=50)

part('pantograph', explode=(0, 2.6, 0))
# Large rhombic (diamond) pantograph as drawn in both elevations: two diamond frames side by side,
# bottom corners on a common base, top corners carrying the pan, X-bracing between the frames
# (front elevation). Shown partly raised.
X0 = XP                                              # over the front bogie pivot: 3,800 mm behind the front end (labelled)
# side elevation: a pale platform plate on the roof (x 2.46-3.58 m, top 3.09 m), short insulators, a dark
# base frame (x 2.76-3.50 m, 3.13-3.28 m) with a ribbed spring block on top; the lower arms start at the
# two ends of the frame
box('platform', (X0 - 0.096, 3.060, 0), (1.115, 0.06, 1.24), 'grey', bevel=0.008)   # left elevation: 1.115 m long, centre 96 mm behind the pivot
for sx in (-1, 1):
    for sz in (-1, 1):
        bm = bmesh.new()
        for k in range(2):
            yk = 3.090 + k * 0.02
            cyl_into(bm, (X0 + sx * 0.30, yk, sz * 0.45), (X0 + sx * 0.30, yk + 0.02, sz * 0.45), 0.040 if k == 0 else 0.032, 16, bevel=0.003)
        finish(bm, f'insulator.{sx}{sz}', 'insulator', sharp=60)
box('base.frame', (X0, 3.190, 0), (0.74, 0.12, 1.12), 'darkmetal', bevel=0.012)
box('base.block', (X0, 3.262, 0), (0.66, 0.025, 0.70), 'darkmetal', bevel=0.004)
bm = bmesh.new()
for k in range(22):                                   # ribs of the spring block
    box_into(bm, (X0 - 0.315 + k * 0.03, 3.282, 0), (0.008, 0.020, 0.66), 0.002)
finish(bm, 'base.ribs', 'steel', sharp=50)
Y_BASE = 3.225
XB_ = 0.35                                            # half-distance between the arms' bottom pivots
# Scaled off the coloured elevations, where the pantograph is drawn raised to the wire at ~5.0 m:
# knees ~3.99 m above rail and 1.52 m either side of the centre, the two frames 1.06 m apart.
HW = 1.503
Y_MID, Y_TOP = 3.92, 4.905
ZF = 0.53
for sz in (-1, 1):
    zz = sz * ZF
    rod(f'diamond.{sz}', [(X0 - XB_, Y_BASE, zz), (X0 - HW, Y_MID, zz), (X0, Y_TOP, zz), (X0 + HW, Y_MID, zz), (X0 + XB_, Y_BASE, zz)],
        0.020, 'panto', seg=12)
    for sx in (-1, 1):
        cyl(f'hinge.base.{sz}{sx}', (X0 + sx * XB_, Y_BASE, zz - 0.04), (X0 + sx * XB_, Y_BASE, zz + 0.04), 0.03, 'darkmetal', seg=14, bevel=0.003)
    for sx in (-1, 1):
        cyl(f'hinge.knee.{sz}{sx}', (X0 + sx * HW, Y_MID, zz - 0.035), (X0 + sx * HW, Y_MID, zz + 0.035), 0.026, 'darkmetal', seg=14, bevel=0.003)
# X cross-bracing between the left and right frames (photos of 6149 and 6102, front and rear elevations):
# one X in the plane of each pair of upper arms, front and rear, running across the car from one frame to
# the other. Seen from the side it lies on the arms, so the inside of each diamond stays open.
for sx in (-1, 1):
    A, B, C = (X0 + sx * XB_, Y_BASE), (X0 + sx * HW, Y_MID), (X0, Y_TOP)
    for nm, P, Q in (('upper', B, C),):               # photo of 6102: no bracing between the lower arms
        p0 = (P[0] + (Q[0] - P[0]) * 0.10, P[1] + (Q[1] - P[1]) * 0.10)
        p1 = (P[0] + (Q[0] - P[0]) * 0.90, P[1] + (Q[1] - P[1]) * 0.90)
        for sz in (-1, 1):
            rod(f'brace.{nm}.{sx:+d}{sz:+d}', [(p0[0], p0[1], sz * (ZF - 0.02)), (p1[0], p1[1], -sz * (ZF - 0.02))], 0.008, 'panto', seg=8)
cyl('pan.tube', (X0, Y_TOP, -ZF - 0.05), (X0, Y_TOP, ZF + 0.05), 0.016, 'panto', seg=12)
for sx in (-1, 1):                                    # pan head: two bows with downturned horns, carbon strips
    xb = X0 + sx * 0.13
    # (rear elevation: bow 1.455 m across over the horns)
    cyl(f'pan.bow.{sx}', (xb, Y_TOP + 0.05, -0.66), (xb, Y_TOP + 0.05, 0.66), 0.016, 'panto', seg=12)
    box(f'pan.strip.{sx}', (xb, Y_TOP + 0.075, 0), (0.035, 0.02, 1.24), 'copper', bevel=0.004)
    for sz in (-1, 1):
        rod(f'pan.horn.{sx}{sz}', [(xb, Y_TOP + 0.05, sz * 0.66), (xb, Y_TOP + 0.03, sz * 0.695), (xb, Y_TOP - 0.05, sz * 0.712), (xb, Y_TOP - 0.15, sz * 0.70)], 0.015, 'panto')
    rod(f'pan.strut.{sx}', [(X0, Y_TOP, -0.25), (xb, Y_TOP + 0.05, -0.30)], 0.010, 'panto', seg=8)
    rod(f'pan.strut2.{sx}', [(X0, Y_TOP, 0.25), (xb, Y_TOP + 0.05, 0.30)], 0.010, 'panto', seg=8)
for sz in (-1, 1):
    cyl(f'pan.end.{sz}', (X0 - 0.13, Y_TOP + 0.05, sz * 0.66), (X0 + 0.13, Y_TOP + 0.05, sz * 0.66), 0.012, 'panto', seg=10)

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
    yc = 0.56                                         # side elevation: coupler 0.56 m up, head 600 mm beyond the body
    cyl(f'{pid}.bar', (s * 6.55, yc, 0), (tip - s * 0.08, yc, 0), 0.04, 'grey', seg=16, bevel=0.004)
    box(f'{pid}.head', (tip - s * 0.05, yc, 0), (0.10, 0.07, 0.14), 'grey', bevel=0.012)
    cyl(f'{pid}.pin', (tip - s * 0.05, yc - 0.07, 0), (tip - s * 0.05, yc + 0.07, 0), 0.014, 'steel', seg=12)
    box(f'{pid}.mount', (s * 6.55, yc, 0), (0.12, 0.16, 0.24), 'grey', bevel=0.015)   # on the back wall of the recess
    box(f'{pid}.head_cap', (tip - s * 0.05, yc, 0), (0.13, 0.09, 0.17), 'grey', bevel=0.025)
    chain = [(tip - s * (0.1 + 0.025 * k), yc - 0.03 - 0.035 * math.sin(math.pi * k / 12), 0.08) for k in range(13)]
    sweep(f'{pid}.chain', chain, 0.006, 'steel', seg=6)

# ============================================================================== plate under the front end
# Door-side elevation: a grey plate under the front door, a trapezoid from ~4.69 m (top, 0.29 m up) /
# 4.80 m (bottom, 0.14 m) to ~6.23 m (top) / 6.41 m (bottom). Front elevation: ~1.78 m wide. Photo of 6149:
# below the skirt only dark parts show, the plate tucked under the skirt bottom (~0.33 m) and set well
# back from the front, nothing light and nothing on the track. Built as a shallow tray: grey side plates
# (the step plate seen under the front door), a dark sloping front and a dark bottom.
part('guard_plate', explode=(0.8, -0.25, 0))
GP_W = 0.89                                           # half-width: 1.78 m
GP_TOP, GP_BOT = 0.280, 0.110
GP_XT0, GP_XT1, GP_XB0, GP_XB1 = 4.648, 6.288, 4.815, 6.42
bm = bmesh.new()
for sz in (-1, 1):
    vs = [bm.verts.new((x, y, sz * GP_W + dz)) for dz in (-0.006, 0.006)
          for x, y in ((GP_XT0, GP_TOP), (GP_XT1, GP_TOP), (GP_XB1, GP_BOT), (GP_XB0, GP_BOT))]
    for f in ((0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
        bm.faces.new([vs[k] for k in f])
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
finish(bm, 'guard_plate.side', 'grey', sharp=30)
# dark sloping front: from just under the skirt bottom down to the front bottom corner of the side plates
bm = bmesh.new()
_ft = (GP_XT1 - 0.03, 0.335)
_fb = (GP_XB1, GP_BOT)
vs = [bm.verts.new((x + dx, y, z)) for z in (-GP_W - 0.006, GP_W + 0.006) for (x, y), dx in ((_ft, 0), (_fb, 0), (_fb, -0.012), (_ft, -0.012))]
for f in ((0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
    bm.faces.new([vs[k] for k in f])
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
finish(bm, 'guard_plate.front', 'darkmetal', sharp=30)
box('guard_plate.bottom', ((GP_XB0 + GP_XB1) / 2, GP_BOT + 0.006, 0), (GP_XB1 - GP_XB0, 0.012, 2 * GP_W), 'darkmetal', bevel=0.003)
for xb in (4.85, 6.10):
    for zb in (-0.80, 0.80):
        box(f'guard_plate.hanger.{xb}.{zb}', (xb, (GP_BOT + BODY_BOTTOM) / 2 + 0.01, zb),
            (0.05, BODY_BOTTOM - GP_BOT + 0.02, 0.012), 'darkmetal', bevel=0.003)

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
# route sign: cut its opening through the painted skin of the dome, then drop the cutter
_cutters = [o for o in bpy.data.objects if o.type == 'MESH' and o.data.materials and o.data.materials[0] and o.data.materials[0].name == 'cutter']
_skins = [bpy.data.objects.get(n) for n in ('front_mask.paint', 'interior.lining', 'interior.ceiling')]
if _cutters and all(_skins):
    _cm = bpy.data.meshes.new('cutter_all'); _bm = bmesh.new()
    for o in _cutters:
        t_ = bpy.data.meshes.new('t'); t_ = o.data.copy(); t_.transform(o.matrix_world); _bm.from_mesh(t_); bpy.data.meshes.remove(t_)
    bmesh.ops.remove_doubles(_bm, verts=_bm.verts, dist=1e-5); _bm.to_mesh(_cm); _bm.free()
    _co = bpy.data.objects.new('cutter_all', _cm); COL.objects.link(_co)
    _co.matrix_world = Matrix.Identity(4)
    for _skin in _skins:                              # the painted skin and the lining/ceiling just behind it
        _md = _skin.modifiers.new('cut', 'BOOLEAN'); _md.operation = 'DIFFERENCE'; _md.object = _co; _md.solver = 'EXACT'
        _f0 = len(_skin.data.polygons)
        with bpy.context.temp_override(object=_skin, active_object=_skin):
            bpy.ops.object.modifier_apply(modifier='cut')
        print('SIGNCUT', _skin.name, _f0, '->', len(_skin.data.polygons))
    for o in _cutters + [_co]:
        bpy.data.objects.remove(o)
if '--exec' in ARGS:
    exec(open(ARGS[ARGS.index('--exec') + 1]).read())
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

if '--idpass' in ARGS:
    # Flat-colour orthographic views for the numeric comparison with the coloured drawing: every
    # material emits its base colour (or base-colour texture), glass is light blue, nothing is lit.
    # The clear cover over the route blind is left out: what shows through it is the blind and lettering.
    for o in [o for o in bpy.data.objects if o.type == 'MESH' and o.data.materials and o.data.materials[0] and o.data.materials[0].name == 'signglass']:
        bpy.data.objects.remove(o)
    for m in list(bpy.data.materials):
        if not m.use_nodes:
            continue
        nt = m.node_tree
        p = nt.nodes.get('Principled BSDF')
        out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if p is None or out is None:
            continue
        em = nt.nodes.new('ShaderNodeEmission')
        bc = p.inputs['Base Color']
        if m.name in ('alu', 'stripe'):                  # strips read as one bright strip, as in the drawings
            em.inputs['Color'].default_value = (0.95, 0.95, 0.95, 1)
        elif p.inputs['Transmission Weight'].default_value > 0.5 if 'Transmission Weight' in p.inputs else False:
            em.inputs['Color'].default_value = (0.55, 0.75, 0.90, 1)
        elif bc.is_linked:
            nt.links.new(bc.links[0].from_socket, em.inputs['Color'])
        else:
            em.inputs['Color'].default_value = bc.default_value
        for l in list(out.inputs['Surface'].links):
            nt.links.remove(l)
        nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    w = bpy.data.worlds.new('idw'); w.use_nodes = True
    w.node_tree.nodes['Background'].inputs['Color'].default_value = (0, 0, 0, 1)
    scene.world = w
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 8
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.filter_width = 0.5
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.render.film_transparent = True
    for name, loc, look, scale, res in (('id_front.png', (40, 1.7, 0), (0, 1.7, 0), 3.4, (3000, 3400)),
                                        ('id_side.png', (5.85, 1.7, 40), (5.85, 1.7, 0), 3.7, (3700, 3400)),
                                        ('id_right.png', (0.0, 2.5, 40), (0.0, 2.5, 0), 16.2, (8100, 2600)),
                                        ('id_left.png', (0.0, 2.5, -40), (0.0, 2.5, 0), 16.2, (8100, 2600)),
                                        ('id_rear.png', (-40, 2.6, 0), (0, 2.6, 0), 5.2, (3000, 5200))):
        if '--ids' in ARGS and name[3:-4] not in ARGS[ARGS.index('--ids') + 1].split(','):
            continue
        co = camera(loc, look, 50, res)
        co.data.type = 'ORTHO'
        co.data.ortho_scale = scale
        scene.render.filepath = os.path.join(OUT, name)
        bpy.ops.render.render(write_still=True)

if '--preexec' in ARGS:
    exec(open(ARGS[ARGS.index('--preexec') + 1]).read())
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
        if '--dooropen' in ARGS:                       # check renders only: leaves opened as on the web page
            for o in bpy.data.objects:
                mm = re.search(r'\.leaf(\d)$', o.name)
                if mm and o.get('hinge'):
                    o.rotation_euler[2] += {0: 1, 1: -2, 2: 2, 3: -1}[int(mm.group(1))] * math.radians(OPEN_DEG)
        if 'q34hi' in shots: shot('check_34hi.png', (11.0, 4.6, 7.8), (1.2, 1.1, 0.0), lens=36)
        if 'fdoor' in shots: shot('check_fdoor.png', (6.6, 1.6, 4.2), (5.3, 1.1, 0.9), lens=30)
        if 'sign' in shots: shot('check_sign.png', (9.6, 2.85, 1.6), (6.9, 2.68, 0.0), lens=60)
        if 'signside' in shots: shot('check_signside.png', (8.2, 2.95, 1.9), (6.95, 2.70, 0.25), lens=55)
        if 'jr' in shots: shot('check_jr.png', (RD1 + 0.35, 1.2, 3.2), (RD1 + 0.05, 1.2, 1.2), lens=40)
        if 'jf' in shots: shot('check_jf.png', (FD0 - 0.1, 1.2, 3.2), (FD0 - 0.05, 1.2, 1.2), lens=40)
        if 'jn' in shots: shot('check_jn.png', (FD1 + 0.15, 1.0, 3.0), (FD1 + 0.05, 0.85, 1.1), lens=40)
        if 'side' in shots: shot('check_side.png', (0, 1.6, 30), (0, 1.6, 0), ortho=16.0)
        if 'left' in shots: shot('check_left.png', (0, 1.6, -30), (0, 1.6, 0), ortho=16.0)
        if 'fortho' in shots: shot('check_fortho.png', (40, 1.6, 0), (0, 1.6, 0), ortho=6.4)
        if 'rortho' in shots: shot('check_rortho.png', (-40, 1.6, 0), (0, 1.6, 0), ortho=6.4)
        if 'preview' in shots: shot('preview.png', (12.6, 1.6, 6.6), (2.4, 2.05, -0.3), lens=34)   # README picture: whole car incl. pantograph
        if 'q34' in shots: shot('check_34.png', (12.2, 1.45, 6.4), (2.6, 1.55, -0.3), lens=38)
        if 'lq34' in shots: shot('check_34_left.png', (12.2, 1.45, -6.4), (2.6, 1.55, 0.3), lens=38)
        if 'rq34' in shots: shot('check_rear.png', (-12.6, 1.9, 5.2), (-4.2, 1.6, 0.0), lens=38)
        if 'corner' in shots: shot('check_corner.png', (11.0, 1.75, 3.9), (6.2, 1.25, 0.2), lens=40)
        if 'steprear' in shots: shot('check_steprear.png', (-8.8, 0.75, 1.55), (-3.6, 0.45, 1.10), lens=35)
        if 'lrear' in shots: shot('check_lrear.png', (-7.0, 3.6, -13.0), (-1.8, 1.5, 0.0), lens=42)
        if 'lrearc' in shots: shot('check_lrearc.png', (-10.6, 1.9, -4.3), (-6.5, 1.45, -0.5), lens=38)
        if 'lrearlow' in shots: shot('check_lrearlow.png', (-9.6, 0.55, -3.6), (-6.2, 0.45, -0.6), lens=35)
        if 'tails' in shots: shot('check_tails.png', (-9.2, 1.05, 1.1), (-6.9, 0.95, 0.0), lens=55)
        _mc = (MD0 + MD1) / 2
        if 'stairsout' in shots: shot('check_stairsout.png', (_mc + 0.3, 1.25, 3.0), (_mc, 0.45, 1.0), lens=28)
        if 'stairs' in shots: shot('check_stairs.png', (_mc - 0.2, 2.35, -0.15), (_mc - 0.05, 0.45, 1.0), lens=24)
        if 'stairsf' in shots: shot('check_stairsf.png', (5.3, 2.35, -0.15), (5.3, 0.45, 0.95), lens=24)
        if 'stairsr' in shots: shot('check_stairsr.png', (-5.3, 2.35, -0.15), (-5.3, 0.45, 1.0), lens=24)
        if 'doorc' in shots: shot('check_doorc.png', (_mc + 1.2, 1.45, 4.6), (_mc, 1.45, 1.2), lens=40)
        if 'doorin' in shots: shot('check_doorin.png', (_mc - 1.9, 1.55, -0.2), (_mc + 0.2, 1.15, 1.2), lens=30)
        if 'stepside' in shots: shot('check_stepside.png', (8.8, 0.75, 1.55), (3.6, 0.45, 1.10), lens=35)
        if 'panto' in shots: shot('check_panto.png', (9.5, 4.6, 3.2), (3.2, 4.0, 0.0), lens=45)
        if 'front' in shots: shot('check_front.png', (9.5, 1.7, 0.6), (6.5, 1.5, 0.0), lens=40)
        if 'bogie' in shots: shot('check_bogie.png', (4.6, 0.55, 2.6), (3.2, 0.40, 0.6), lens=28)
        if 'detail' in shots: shot('check_detail.png', (7.6, 2.0, 2.6), (5.6, 1.9, 1.1), lens=35)
        if 'interior' in shots: shot('check_interior.png', (-5.6, 1.62, 0.25), (3.0, 1.35, -0.05), lens=20)
