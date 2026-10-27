"""Stand test in NVIDIA Newton.


The body plan becomes an articulated rigid body: one free-floating trunk (trunk, neck/head and
tail capsules) and one chain of capsule segments per leg, linked by hinge joints that hold the
standing pose with a PD drive. The body is dropped onto a ground plane and simulated; it
"stands" if the trunk stays up and level.

Runs on CPU or GPU (NVIDIA Warp picks the device; on a Nebius AI Cloud GPU instance it uses CUDA).
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp
import newton

from .body import Body, _dist, balance

G = 9.81


def _quat_z_to(d):
    """Quaternion that turns +Z onto direction d (capsules in Newton lie along their local Z axis)."""
    d = np.asarray(d, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return wp.quat_identity()
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 0.999999:
        return wp.quat_identity()
    if c < -0.999999:
        return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    return wp.quat_from_axis_angle(wp.vec3(*axis.tolist()), math.acos(c))


def _capsule(builder, body, origin, a, b, r, cfg):
    a = np.asarray(a) - origin
    b = np.asarray(b) - origin
    mid = (a + b) / 2
    hh = max(1e-3, np.linalg.norm(b - a) / 2)
    builder.add_shape_capsule(body, xform=wp.transform(wp.vec3(*mid.tolist()), _quat_z_to(b - a)),
                              radius=float(r), half_height=float(hh), cfg=cfg)


def build_model(plan: Body, drop=0.02, device=None):
    """Articulated model of the plan. Returns (model, trunk body index, initial trunk height)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    builder.default_shape_cfg.ke = 2.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.75, ke=2.0e3, kd=1.0e2, kf=1.0e3)

    trunk_o = np.array([(plan.rear[i] + plan.front[i]) / 2 for i in range(3)])
    trunk_o[2] += drop
    trunk = builder.add_link(xform=wp.transform(wp.vec3(*trunk_o.tolist()), wp.quat_identity()), label="trunk")
    lift = np.array([0, 0, drop])
    for name, a, b, r in plan.capsules():
        if name in ("trunk", "head", "tail"):
            _capsule(builder, trunk, trunk_o, np.asarray(a) + lift, np.asarray(b) + lift, r, cfg)
    joints = [builder.add_joint_free(trunk, parent_xform=wp.transform(wp.vec3(*trunk_o.tolist()), wp.quat_identity()), label="root")]

    # hold the pose: stiffness scaled to the weight it has to carry
    mass = plan.mass()
    height = max(1e-3, plan.front[2])
    # drive stiffness from statics: each leg carries ~1/n of the weight over its own length;
    # sized so the joints hold the pose within ~0.05 rad (a standing animal holds its posture actively)
    leg_len = sum(_dist(l.joints[i], l.joints[i + 1]) for l in plan.limbs for i in range(len(l.joints) - 1)) / max(1, len(plan.limbs))
    ke = mass * G * leg_len / (max(1, len(plan.limbs)) * 0.05)
    kd = ke / 20.0
    for leg in plan.limbs:
        parent, parent_o = trunk, trunk_o
        for i in range(len(leg.joints) - 1):
            p0 = np.asarray(leg.joints[i]) + lift
            p1 = np.asarray(leg.joints[i + 1]) + lift
            seg = builder.add_link(xform=wp.transform(wp.vec3(*p0.tolist()), wp.quat_identity()), label=f"{leg.name}_{i}")
            _capsule(builder, seg, p0, p0, p1, leg.radius, cfg)
            joints.append(builder.add_joint_revolute(
                parent, seg,
                parent_xform=wp.transform(wp.vec3(*(p0 - parent_o).tolist()), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                axis=wp.vec3(0.0, 1.0, 0.0), target_pos=0.0, target_ke=ke, target_kd=kd,
                limit_lower=-1.2, limit_upper=1.2, label=f"{leg.name}_j{i}"))
            parent, parent_o = seg, p0
    builder.add_articulation(joints, label=plan.name)
    for dof in range(6, builder.joint_dof_count):              # every leg hinge holds its standing angle
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    return model, trunk, float(trunk_o[2])


def stand_test(plan: Body, seconds=2.0, fps=50, substeps=4, device=None):
    """Simulate the plan standing. Returns metrics (all from the Newton run plus the static check)."""
    model, trunk, z0 = build_model(plan, device=device)
    solver = newton.solvers.SolverMuJoCo(model, cone="elliptic", impratio=100, iterations=100, ls_iterations=50,
                                         nconmax=64, njmax=200, use_mujoco_contacts=False)
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    dt = 1.0 / fps / substeps
    traj = []
    for f in range(int(seconds * fps)):
        for _ in range(substeps):
            s0.clear_forces()
            pipe.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        q = s0.body_q.numpy()[trunk]
        traj.append(q.copy())
    traj = np.array(traj)
    z = traj[:, 2]
    # tilt of the trunk: angle between its local x axis and the ground plane
    qx, qy, qz, qw = traj[-1, 3:7]
    fwd = np.array([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)])
    up = np.array([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)])
    roll_pitch = math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))
    final_drop = (z0 - z[-1]) / max(z0, 1e-6)
    stands = bool(final_drop < 0.25 and roll_pitch < 25.0)
    st = balance(plan)
    return {
        "engine": "NVIDIA Newton " + getattr(newton, "__version__", ""),
        "device": str(wp.get_device()),
        "mass_kg": round(plan.mass(), 2),
        "stands": stands,
        "trunk_height_start_m": round(z0, 4),
        "trunk_height_end_m": round(float(z[-1]), 4),
        "height_drop_ratio": round(float(final_drop), 3),
        "trunk_tilt_deg": round(roll_pitch, 2),
        "com_m": [round(x, 4) for x in st["com"]],
        "tip_margin_m": round(st["margin_m"], 4),
        "tip_angle_deg": round(st["tip_deg"], 2),
        "support_area_m2": round(st["support_m2"], 4),
        "seconds": seconds,
        "trunk_z_trace": [round(float(v), 4) for v in z[:: max(1, len(z) // 30)]],
    }
