"""Export a body plan as an MJCF model (MuJoCo XML), the format robot-learning stacks load directly.
Same structure as the Newton stand test: free trunk, hinge joints on every leg segment,
position actuators that hold the standing pose."""
from __future__ import annotations

import math
from xml.sax.saxutils import quoteattr

from .body import Body, _dist

G = 9.81


def _v(p):
    return " ".join(f"{x:.5f}" for x in p)


def to_mjcf(plan: Body, name: str | None = None) -> str:
    name = name or plan.name.replace(" ", "_")
    mass = plan.mass()
    leg_len = sum(_dist(l.joints[i], l.joints[i + 1]) for l in plan.limbs for i in range(len(l.joints) - 1)) / max(1, len(plan.limbs))
    kp = mass * G * leg_len / (max(1, len(plan.limbs)) * 0.05)
    o = [(plan.rear[i] + plan.front[i]) / 2 for i in range(3)]
    rel = lambda p, base: [p[i] - base[i] for i in range(3)]
    out = [f'<mujoco model={quoteattr(name)}>',
           '  <compiler angle="radian"/>',
           '  <option timestep="0.005" gravity="0 0 -9.81"/>',
           '  <default><geom density="1000" friction="0.75 0.005 0.0001"/><joint damping="1" limited="true" range="-1.2 1.2"/></default>',
           '  <worldbody>',
           '    <geom name="ground" type="plane" size="20 20 0.1"/>',
           f'    <body name="trunk" pos="{_v(o)}">',
           '      <freejoint name="root"/>']
    for nm, a, b, r in plan.capsules():
        if nm in ("trunk", "head", "tail"):
            out.append(f'      <geom name="{nm}" type="capsule" fromto="{_v(rel(a, o))} {_v(rel(b, o))}" size="{r:.5f}"/>')
    acts = []

    def chain(leg, i, parent_pos, indent):
        if i >= len(leg.joints) - 1:
            return []
        p0, p1 = leg.joints[i], leg.joints[i + 1]
        j = f"{leg.name}_j{i}"
        acts.append(j)
        lines = [f'{indent}<body name="{leg.name}_{i}" pos="{_v(rel(p0, parent_pos))}">',
                 f'{indent}  <joint name="{j}" type="hinge" axis="0 1 0"/>',
                 f'{indent}  <geom type="capsule" fromto="0 0 0 {_v(rel(p1, p0))}" size="{leg.radius:.5f}"/>']
        lines += chain(leg, i + 1, p0, indent + "  ")
        lines.append(f'{indent}</body>')
        return lines

    for leg in plan.limbs:
        out += chain(leg, 0, o, "      ")
    out += ['    </body>', '  </worldbody>', '  <actuator>']
    out += [f'    <position name="{j}_act" joint="{j}" kp="{kp:.1f}" kv="{kp / 20:.1f}"/>' for j in acts]
    out += ['  </actuator>', '</mujoco>']
    return "\n".join(out) + "\n"
