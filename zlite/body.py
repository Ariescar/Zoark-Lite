"""Parametric quadruped body.

A species is described by a handful of external measurements (shoulder height, head-body
length, tail length). From these a generic quadruped template is laid out: a capsule trunk,
a neck/head capsule, a tail capsule and four three-segment legs. The template uses fixed,
textbook-style segment fractions; it is intentionally simple and is not a skeletal
reconstruction.

Coordinates: metres, x forward, y left, z up, ground at z = 0.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, asdict

DENSITY = 1000.0   # kg/m^3

# share of the standing leg height taken by each segment (upper, lower, foot) and the
# fore-aft offsets that give the legs their resting zig-zag (fraction of leg height)
FORE = {"split": (0.40, 0.38, 0.22), "offset": (0.00, -0.10, 0.03)}
HIND = {"split": (0.36, 0.40, 0.24), "offset": (0.00, 0.12, -0.05)}


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


@dataclass
class Limb:
    name: str
    joints: list          # from the girdle joint down to the foot
    radius: float


@dataclass
class Body:
    name: str
    front: list           # trunk front point (between the shoulders)
    rear: list            # trunk rear point (between the hips)
    trunk_r: float
    head: list
    head_r: float
    tail_len: float
    tail_r: float
    limbs: list = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def capsules(self):
        out = [("trunk", self.rear, self.front, self.trunk_r), ("head", self.front, self.head, self.head_r)]
        if self.tail_len > 0:
            droop = math.radians(30)
            end = [self.rear[0] - self.tail_len * math.cos(droop), 0.0, self.rear[2] - self.tail_len * math.sin(droop)]
            out.append(("tail", self.rear, end, self.tail_r))
        for l in self.limbs:
            for i in range(len(l.joints) - 1):
                out.append((f"{l.name}{i}", l.joints[i], l.joints[i + 1], l.radius))
        return out

    def mass(self):
        return sum(DENSITY * (math.pi * r * r * _dist(a, b) + 4 / 3 * math.pi * r ** 3) for _, a, b, r in self.capsules())

    def com(self):
        tot, c = 0.0, [0.0, 0.0, 0.0]
        for _, a, b, r in self.capsules():
            m = DENSITY * (math.pi * r * r * _dist(a, b) + 4 / 3 * math.pi * r ** 3)
            tot += m
            for i in range(3):
                c[i] += m * (a[i] + b[i]) / 2
        return [x / tot for x in c]

    def feet(self):
        return [l.joints[-1] for l in self.limbs]


def _leg(name, x, y, height, tmpl, radius):
    pts, z, xx = [[x, y, height]], height, x
    for frac, off in zip(tmpl["split"], tmpl["offset"]):
        z -= frac * height
        xx += off * height
        pts.append([xx, y, max(0.0, z)])
    return Limb(name, pts, radius)


def make_body(sp: dict, legs=1.0, trunk=1.0, tail=1.0, front=1.0, size=1.0) -> Body:
    """Build a body from a species record and optional scale factors."""
    h = sp["shoulder_height_m"] * legs * size
    hb = sp["head_body_length_m"] * size
    trunk_len = 0.55 * hb * trunk
    width = 0.16 * hb
    girth = 0.11 * hb
    hip_h = h * sp.get("hip_to_shoulder", 1.0)
    front_pt = [0.0, 0.0, h + 0.4 * girth]
    rear_pt = [-trunk_len, 0.0, hip_h + 0.4 * girth]
    head = [0.30 * hb, 0.0, front_pt[2] + 0.12 * hb]
    r_leg = 0.035 * h + 0.012
    limbs = [_leg("fl", 0.0, width / 2, h, FORE, r_leg), _leg("fr", 0.0, -width / 2, h, FORE, r_leg),
             _leg("hl", -trunk_len, width / 2, hip_h, HIND, r_leg * 1.1), _leg("hr", -trunk_len, -width / 2, hip_h, HIND, r_leg * 1.1)]
    return Body(name=sp["name"], front=front_pt, rear=rear_pt, trunk_r=girth, head=head, head_r=0.55 * girth * front,
                tail_len=sp.get("tail_length_m", 0.0) * tail * size, tail_r=0.2 * girth, limbs=limbs,
                params={"legs": legs, "trunk": trunk, "tail": tail, "front": front, "size": size})


def balance(body: Body) -> dict:
    """Static balance: centre of mass over the support polygon of the feet."""
    pts = sorted({(f[0], f[1]) for f in body.feet()})
    hull = _hull(pts)
    c = body.com()
    margin, inside = float("inf"), True
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        ex, ey = b[0] - a[0], b[1] - a[1]
        L = math.hypot(ex, ey) or 1e-9
        d = (ex * (c[1] - a[1]) - ey * (c[0] - a[0])) / L
        inside &= d >= 0
        margin = min(margin, abs(d))
    m = margin if inside else -margin
    area = abs(sum(hull[i][0] * hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0] * hull[i][1] for i in range(len(hull)))) / 2
    return {"com": c, "margin_m": m, "tip_deg": math.degrees(math.atan2(m, max(c[2], 1e-6))), "support_m2": area}


def _hull(p):
    if len(p) < 3:
        return p
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(up) >= 2 and cr(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return lo[:-1] + up[:-1]
