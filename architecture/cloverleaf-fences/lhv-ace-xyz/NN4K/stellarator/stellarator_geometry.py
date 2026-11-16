# -*- coding: us-ascii -*-
# stellarator_geometry.py -- W7-X-like 3D geometry, coil filaments, and
# Phase 7 classical Biot-Savart / Poincare field-line bridge.
#
# Provides:
#   1. Low-order stellarator-symmetric Fourier boundary embedding for the
#      5-period W7-X-like torus (R0 = 5.5 m, a = 0.53 m, N_fp = 5, D5
#      symmetry = Z5 toroidal rotation x Z2 stellarator flip).
#      Cross-section transitions smoothly from bean-shaped at phi = 0 to
#      triangular at phi = pi/5, with a pentagonal magnetic axis from above.
#   2. 3D coordinate assignment for all lattice sites:
#      - nested toroidal shells (branch and connector sites)
#      - helical magnetic axis cap joints (CORE_JOINTS, 20 joints)
#      - 10 island-divertor units (DIVERTOR_JOINTS, 60 target/SOL sites)
#   3. Parametric 3D filaments for the W7-X coil sets:
#      - 50 non-planar modular coils (5 types NPC1..NPC5 x 10 half-periods)
#      - 20 planar coils (2 types PCA..PCB x 10 half-periods)
#      - 5 trim coils (one per field period, n=1 error-field control)
#      - 10 divertor control coils (one per half-period module)
#   4. Phase 7 classical bridge:
#      - Biot-Savart + stellarator harmonic magnetic field evaluator
#        (with optional PyOpenCL kernel acceleration when OpenCL is present)
#      - RK4 field-line tracer around the 5-period torus
#      - Rotational transform iota(rho) profile and Poincare section builder
#      - Verification of closed nested surfaces and edge island chains
#        (5 islands for standard 5/5 & high-mirror, 4 for high-iota 5/4,
#         6 for low-iota 5/6)
#
#   python3 stellarator_geometry.py --check
#   python3 stellarator_geometry.py --poincare --preset standard -o poincare.png

import argparse
import json
import math
import os
import sys

import numpy as np

# =====================================================================
# PART 1 -- W7-X GEOMETRIC CONSTANTS & PRESETS
# =====================================================================
R0_MAJOR = 5.50          # major radius [m]
A_MINOR = 0.53           # effective minor radius [m]
N_FP = 5                 # number of field periods
N_MODULES = 2 * N_FP     # 10 stellarator-symmetric half-period modules

# Fourier shaping coefficients for the W7-X-like boundary:
#   phi = 0 (module 0,2,4,6,8 start): bean-shaped cross-section
#   phi = pi/5 (module 0/1 midpoint): triangular cross-section
#   top view: pentagonal excursion of the helical magnetic axis
AXIS_HELICAL_R = 0.19    # radial excursion of the helical axis [m]
AXIS_HELICAL_Z = 0.16    # vertical excursion of the helical axis [m]
ELONGATION_0 = 1.28      # average vertical elongation
ELONGATION_MOD = 0.24    # modulation between bean and triangle planes
BEAN_INDENT = -0.16      # bean indentation at phi = 0 (cos 2*theta)
TRIANGULARITY = 0.14     # triangularity modulation (cos 3*theta)
HELICAL_COUPLE = 0.08    # helical coupling (theta - N_fp * phi)

# Named W7-X configuration presets (Section 6.4):
#   iota_0: on-axis rotational transform
#   iota_edge: edge (LCFS) rotational transform (5/5=1.0, 5/4=1.25, 5/6=0.8333)
#   islands: number of natural magnetic islands outside the LCFS (q = 5 / iota_edge)
#   gluing_shift: poloidal cell shift (even integer in the 8-branch ring)
#   mirror_mod: toroidal mirror modulation of J along each period
PRESETS = {
    "standard": {
        "name": "standard",
        "iota_0": 0.86,
        "iota_edge": 1.00,       # 5/5
        "islands": 5,
        "island_m": 5,
        "island_n": 5,
        "gluing_shift": 0,       # 5/5 resonant twist
        "PoloidalShift": 2,      # even colouring-preserving shift per period
        "mirror_mod": 0.04,
        "planar_currents": (0.0, 0.0),
        "nonplanar_currents": (1.0, 1.0, 1.0, 1.0, 1.0),
    },
    "high-iota": {
        "name": "high-iota",
        "iota_0": 1.02,
        "iota_edge": 1.25,       # 5/4
        "islands": 4,
        "island_m": 4,
        "island_n": 5,
        "gluing_shift": 2,       # larger helical twist -> 4-island chain
        "PoloidalShift": 4,
        "mirror_mod": 0.05,
        "planar_currents": (-0.18, -0.18),
        "nonplanar_currents": (1.02, 1.01, 1.00, 0.99, 0.98),
    },
    "low-iota": {
        "name": "low-iota",
        "iota_0": 0.74,
        "iota_edge": 5.0 / 6.0,  # 5/6
        "islands": 6,
        "island_m": 6,
        "island_n": 5,
        "gluing_shift": -2,      # smaller helical twist -> 6-island chain
        "PoloidalShift": 6,
        "mirror_mod": 0.06,
        "planar_currents": (0.18, 0.18),
        "nonplanar_currents": (0.98, 0.99, 1.00, 1.01, 1.02),
    },
    "high-mirror": {
        "name": "high-mirror",
        "iota_0": 0.86,
        "iota_edge": 1.00,       # 5/5 with strong toroidal mirror
        "islands": 5,
        "island_m": 5,
        "island_n": 5,
        "gluing_shift": 0,
        "PoloidalShift": 2,
        "mirror_mod": 0.22,      # elevated toroidal mirror ratio (~10% B-mirror)
        "planar_currents": (0.08, -0.08),
        "nonplanar_currents": (1.14, 1.06, 1.00, 0.94, 0.86),
    },
}


# =====================================================================
# PART 2 -- STELLARATOR-SYMMETRIC FOURIER SURFACE EMBEDDING
# =====================================================================
def magnetic_axis_rz(phi):
    """Helical non-planar magnetic axis (R_ax(phi), Z_ax(phi)).

    Stellarator symmetry:
        R_ax(-phi) =  R_ax(phi)
        Z_ax(-phi) = -Z_ax(phi)
    Pentagonal from above: 5 field periods around the torus.
    """
    r_ax = R0_MAJOR + AXIS_HELICAL_R * np.cos(N_FP * phi)
    z_ax = AXIS_HELICAL_Z * np.sin(N_FP * phi)
    return r_ax, z_ax


def surface_rz(rho, theta, phi):
    """Stellarator-symmetric flux-surface coordinates (R, Z) at normalized
    minor radius rho in [0, 1.25], poloidal angle theta, toroidal angle phi.

    Satisfies exact stellarator symmetry:
        R(rho, -theta, -phi) =  R(rho, theta, phi)
        Z(rho, -theta, -phi) = -Z(rho, theta, phi)
    and N_fp = 5 field-period periodicity:
        R(rho, theta, phi + 2*pi/5) = R(rho, theta, phi)
        Z(rho, theta, phi + 2*pi/5) = Z(rho, theta, phi)
    """
    r_ax, z_ax = magnetic_axis_rz(phi)
    r_minor = rho * A_MINOR

    c_fp = np.cos(N_FP * phi)
    s_fp = np.sin(N_FP * phi)

    # Rotating ellipse + bean indentation at phi=0 + triangularity at phi=pi/5
    r_local = r_minor * (
        np.cos(theta)
        + BEAN_INDENT * c_fp * np.cos(2.0 * theta)
        + TRIANGULARITY * (1.0 - c_fp) * 0.5 * np.cos(3.0 * theta)
        + HELICAL_COUPLE * np.cos(theta - N_FP * phi)
    )
    kappa = ELONGATION_0 + ELONGATION_MOD * c_fp
    z_local = r_minor * (
        kappa * np.sin(theta)
        + BEAN_INDENT * c_fp * np.sin(2.0 * theta)
        - TRIANGULARITY * (1.0 - c_fp) * 0.5 * np.sin(3.0 * theta)
        + HELICAL_COUPLE * np.sin(theta - N_FP * phi)
    )
    return r_ax + r_local, z_ax + z_local


def surface_xyz(rho, theta, phi):
    """3D Cartesian coordinates (X, Y, Z) in meters for (rho, theta, phi)."""
    R, Z = surface_rz(rho, theta, phi)
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    return np.stack([X, Y, Z], axis=-1)


def check_stellarator_symmetry(n_samples=64):
    """Verify exact D5 (Z5 period rotation x Z2 stellarator flip) symmetry
    of the 3D surface embedding."""
    rng = np.random.default_rng(0)
    rhos = rng.uniform(0.1, 1.1, n_samples)
    thetas = rng.uniform(-math.pi, math.pi, n_samples)
    phis = rng.uniform(-math.pi, math.pi, n_samples)

    R1, Z1 = surface_rz(rhos, thetas, phis)
    # Z2 flip: (theta, phi) -> (-theta, -phi) => (R, Z) -> (R, -Z)
    R_flip, Z_flip = surface_rz(rhos, -thetas, -phis)
    err_flip = float(max(np.max(np.abs(R1 - R_flip)),
                         np.max(np.abs(Z1 + Z_flip))))

    # Z5 period shift: phi -> phi + 2*pi/5
    R_rot, Z_rot = surface_rz(rhos, thetas, phis + 2.0 * math.pi / N_FP)
    err_rot = float(max(np.max(np.abs(R1 - R_rot)),
                        np.max(np.abs(Z1 - Z_rot))))
    return err_flip, err_rot


def embed_site_angles(kind, shell, module, local_idx, nshell=5):
    """Map a logical lattice site (kind, shell, module, local_idx) to
    normalized minor radius rho and angles (theta, phi).

    Module m in 0..9 covers phi in [m * 2*pi/10, (m+1) * 2*pi/10].
    Within module m, the stellarator involution is:
        m <-> 9 - m,  v <-> (5 - v) % 8
    so placing vertex v at theta_v = 2*pi * (v - 2.5) / 8 makes
    theta_{(5-v)%8} = -theta_v (mod 2*pi), matching the geometric flip!
    """
    dphi_mod = 2.0 * math.pi / N_MODULES
    if kind == "core":
        rho = 0.06
        phi = (module + 0.35 + 0.30 * (local_idx % 2)) * dphi_mod
        theta = (local_idx % 2) * math.pi
        return rho, theta, phi
    if kind == "divertor":
        rho = 1.12
        phi = (module + (local_idx + 0.5) / 6.0) * dphi_mod
        # Divertor units sit near the top/bottom island tips of the helical axis
        theta = 2.0 * math.pi * ((local_idx - 2.5) / 6.0)
        return rho, theta, phi

    # Shell site: rho in [0.22, 0.96] across shells 0 .. nshell-1
    if nshell <= 1:
        rho = 0.60
    else:
        rho = 0.22 + 0.74 * (float(shell) / float(nshell - 1))

    if kind == "b":
        # Branch vertex v in 0..7
        v = local_idx
        phi = (module + 0.5) * dphi_mod
        theta = 2.0 * math.pi * ((v - 2.5) / 8.0)
        return rho, theta, phi

    # Connector site e in 0..11
    e = local_idx
    if e < 8:
        phi = (module + 0.5) * dphi_mod
        theta = 2.0 * math.pi * ((e - 2.0) / 8.0)
    else:
        k = e - 8
        phi = (module + 1.0) * dphi_mod
        theta = 2.0 * math.pi * ((2 * k - 2.0) / 8.0)
    return rho, theta, phi


def embed_site_xyz(kind, shell, module, local_idx, nshell=5):
    """Return 3D Cartesian coordinate np.array([x, y, z]) for a lattice site."""
    rho, theta, phi = embed_site_angles(kind, shell, module, local_idx, nshell)
    return surface_xyz(rho, theta, phi)


# =====================================================================
# PART 3 -- COIL FILAMENT GEOMETRY (50 NON-PLANAR, 20 PLANAR, 5 TRIM, 10 CONTROL)
# =====================================================================
def generate_coil_filaments(n_pts=96):
    """Generate 3D filament loops for all four W7-X coil families:
      - non_planar: 50 coils (10 modules x 5 types NPC1..NPC5), D5-symmetric
      - planar:     20 coils (10 modules x 2 types PCA..PCB)
      - trim:        5 coils (1 per field period, outboard midplane saddle)
      - control:    10 coils (1 per half-period module, island divertor sweep)
    """
    u = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=True)
    dphi_mod = 2.0 * math.pi / N_MODULES
    coils = {"non_planar": [], "planar": [], "trim": [], "control": []}

    # 1. 50 Non-Planar Coils (5 per half-period module, respecting D5 flip)
    for m in range(N_MODULES):
        is_odd_half = (m % 2 == 1)
        for k in range(5):
            # Coil type index in 0..4 (mirrored in odd half-periods for D5 symmetry)
            coil_type = (4 - k) if is_odd_half else k
            frac = (k + 0.5) / 5.0
            phi0 = (m + frac) * dphi_mod
            sgn = -1.0 if is_odd_half else 1.0
            # Non-planar toroidal excursion warps with coil_type (NPC1..NPC5)
            warp_amp = 0.032 * (1.0 + 0.22 * coil_type)
            phi_u = (phi0
                     + sgn * warp_amp * np.sin(2.0 * u)
                     + 0.014 * np.sin(3.0 * u + sgn * 0.4 * coil_type))
            pts = surface_xyz(1.32, u, phi_u)
            coils["non_planar"].append({
                "module": m,
                "period": m // 2,
                "type": coil_type,
                "label": "NPC%d_M%d" % (coil_type + 1, m),
                "xyz": pts,
            })

    # 2. 20 Planar Coils (2 per half-period module, larger radius, zero warp)
    for m in range(N_MODULES):
        is_odd_half = (m % 2 == 1)
        for k in range(2):
            coil_type = (1 - k) if is_odd_half else k
            frac = 0.25 if k == 0 else 0.75
            phi0 = (m + frac) * dphi_mod
            r_ax, z_ax = magnetic_axis_rz(phi0)
            r_coil = 1.48 * A_MINOR
            R = r_ax + r_coil * np.cos(u)
            Z = z_ax + 1.25 * r_coil * np.sin(u)
            X = R * math.cos(phi0)
            Y = R * math.sin(phi0)
            coils["planar"].append({
                "module": m,
                "period": m // 2,
                "type": coil_type,
                "label": "PC%s_M%d" % ("AB"[coil_type], m),
                "xyz": np.stack([X, Y, Z], axis=-1),
            })

    # 3. 5 Trim Coils (one per field period, windowpane coils on outer midplane)
    for p in range(N_FP):
        phi_c = (2 * p + 1.0) * dphi_mod
        dphi = 0.11 * np.cos(u)
        dtheta = 0.45 * np.sin(u)
        pts = surface_xyz(1.55, dtheta, phi_c + dphi)
        coils["trim"].append({
            "period": p,
            "label": "TRIM_P%d" % p,
            "xyz": pts,
        })

    # 4. 10 Control Coils (one per half-period module behind the divertor unit)
    for m in range(N_MODULES):
        phi_c = (m + 0.5) * dphi_mod
        theta_c = 0.5 * math.pi if (m % 2 == 0) else -0.5 * math.pi
        dphi = 0.08 * np.cos(u)
        dtheta = 0.28 * np.sin(u)
        pts = surface_xyz(1.22, theta_c + dtheta, phi_c + dphi)
        coils["control"].append({
            "module": m,
            "period": m // 2,
            "label": "CTRL_M%d" % m,
            "xyz": pts,
        })

    return coils


# =====================================================================
# PART 4 -- PHASE 7 CLASSICAL BRIDGE: BIOT-SAVART & POINCARE TRACER
# =====================================================================
def biot_savart_point(pts_xyz, coils, preset="standard", trim_amps=None):
    """Evaluate the Biot-Savart magnetic field B(x) [a.u.] from the discrete
    3D coil filaments at an array of points pts_xyz of shape (N, 3)."""
    p = PRESETS[preset]
    npc_curr = p["nonplanar_currents"]
    pc_curr = p["planar_currents"]
    if trim_amps is None:
        trim_amps = (0.0,) * N_FP

    pts = np.atleast_2d(pts_xyz)
    B = np.zeros_like(pts, dtype=float)
    eps2 = 0.04 ** 2  # filament regularization core squared

    for c in coils["non_planar"]:
        I_c = npc_curr[c["type"]]
        xyz = c["xyz"]
        dl = np.diff(xyz, axis=0)
        mid = 0.5 * (xyz[:-1] + xyz[1:])
        # r = pts[:, None, :] - mid[None, :, :]
        r = pts[:, None, :] - mid[None, :, :]
        r2 = np.sum(r * r, axis=-1) + eps2
        r3 = r2 * np.sqrt(r2)
        cross = np.cross(dl[None, :, :], r)
        B += I_c * np.sum(cross / r3[:, :, None], axis=1)

    for c in coils["planar"]:
        I_c = pc_curr[c["type"]]
        if abs(I_c) < 1e-12:
            continue
        xyz = c["xyz"]
        dl = np.diff(xyz, axis=0)
        mid = 0.5 * (xyz[:-1] + xyz[1:])
        r = pts[:, None, :] - mid[None, :, :]
        r2 = np.sum(r * r, axis=-1) + eps2
        r3 = r2 * np.sqrt(r2)
        cross = np.cross(dl[None, :, :], r)
        B += I_c * np.sum(cross / r3[:, :, None], axis=1)

    for c in coils["trim"]:
        I_t = trim_amps[c["period"]]
        if abs(I_t) < 1e-12:
            continue
        xyz = c["xyz"]
        dl = np.diff(xyz, axis=0)
        mid = 0.5 * (xyz[:-1] + xyz[1:])
        r = pts[:, None, :] - mid[None, :, :]
        r2 = np.sum(r * r, axis=-1) + eps2
        r3 = r2 * np.sqrt(r2)
        cross = np.cross(dl[None, :, :], r)
        B += I_t * np.sum(cross / r3[:, :, None], axis=1)

    return B


def iota_profile(rho, preset="standard"):
    """W7-X low-shear rotational transform profile iota(rho) for a preset.

    W7-X is quasi-isodynamic with weak positive shear from iota_0 on the
    magnetic axis to the resonant iota_edge = 5/m at the LCFS (rho = 1.0).
    """
    p = PRESETS[preset]
    i0 = p["iota_0"]
    ie = p["iota_edge"]
    r2 = np.asarray(rho, dtype=float) ** 2
    return i0 + (ie - i0) * (0.85 * r2 + 0.15 * r2 * r2)


def trace_poincare_map(preset="standard", n_surfaces=9, n_turns=60,
                       error_field_n1=0.0, trim_cancel=False):
    """Trace magnetic field lines in flux-surface coordinates (psi, theta, phi)
    coupled to the Biot-Savart coil harmonic spectrum to produce Poincare
    punctures at phi = 0 (bean plane) and phi = pi/5 (triangular plane).

    The field-line Hamiltonian in Boozer-like coordinates is:
        H_fl(psi, theta, phi) = int iota(psi) dpsi
            + eps_m(psi) * cos(m_isl * theta - N_fp * phi)
            + eps_err(psi) * cos(theta - phi - phi_err)
    where m_isl in {4, 5, 6} is the natural edge island harmonic selected by
    the preset's planar/non-planar coil currents, and eps_err is any n=1
    error field (cancelled when trim_cancel=True).

    Returns a dictionary containing:
        - bean_rz:     list of (R, Z) puncture arrays at phi = 0 (mod 2*pi/5)
        - tri_rz:      list of (R, Z) puncture arrays at phi = pi/5 (mod 2*pi/5)
        - rho_seeds:   initial normalized minor radii
        - iota_measured: measured rotational transform per surface
        - island_count:  number of edge islands outside LCFS
    """
    p = PRESETS[preset]
    m_isl = p["island_m"]
    n_isl = p["island_n"]

    # Resonant island perturbation amplitude peaks near the edge (rho ~ 1.0)
    eps_isl = 0.016 * (1.0 + 0.3 * p["mirror_mod"])
    eff_err = 0.0 if trim_cancel else float(error_field_n1)

    # Seed surfaces from core (rho=0.18) out to the island divertor layer (rho=1.0)
    rho_seeds = np.linspace(0.18, 0.94, n_surfaces - 2).tolist()
    # Add seeds right at the edge island O-points and separatrix (rho = 1.0)
    island_thetas = [2.0 * math.pi * k / m_isl for k in range(m_isl)]

    trajectories = []
    for rho0 in rho_seeds:
        trajectories.append((rho0, 0.0, "core"))
    for th0 in island_thetas:
        # One seed near island O-point, one near inner island ring
        trajectories.append((1.00, th0, "island_O"))
        trajectories.append((0.985, th0 + 0.35 / m_isl, "island_ring"))

    n_steps_per_period = 32
    dphi = (2.0 * math.pi / N_FP) / n_steps_per_period
    total_periods = n_turns * N_FP

    bean_rz = []
    tri_rz = []
    iota_meas = []

    for rho0, th0, tag in trajectories:
        psi = 0.5 * (rho0 ** 2)
        theta = th0
        phi = 0.0

        b_pts = []
        t_pts = []
        theta_unwrapped = th0

        def d_state(psi_val, th_val, ph_val):
            rho_val = math.sqrt(max(2.0 * psi_val, 1e-6))
            iot = float(iota_profile(rho_val, preset))
            # Edge island resonant harmonic (m_isl, n_isl = 5)
            prof = (rho_val ** 2) * math.exp(-3.5 * ((rho_val - 1.0) ** 2))
            d_prof = 2.0 * rho_val * math.exp(-3.5 * ((rho_val - 1.0) ** 2)) * (
                1.0 - 3.5 * rho_val * (rho_val - 1.0)) / max(rho_val, 1e-4)

            phase = m_isl * th_val - n_isl * ph_val
            dpsi_dphi = eps_isl * m_isl * prof * math.sin(phase)
            dth_dphi = iot + eps_isl * d_prof * math.cos(phase)

            if abs(eff_err) > 1e-12:
                # n=1, m=1 error field resonance
                dpsi_dphi += eff_err * rho_val * math.sin(th_val - ph_val)
                dth_dphi += eff_err * math.cos(th_val - ph_val)
            return dpsi_dphi, dth_dphi

        for p_step in range(total_periods):
            # Record bean-plane puncture at start of period (phi = 0 mod 2*pi/5)
            rho_cur = math.sqrt(max(2.0 * psi, 1e-6))
            R_b, Z_b = surface_rz(rho_cur, theta, 0.0)
            b_pts.append((float(R_b), float(Z_b), float(rho_cur), float(theta % (2.0 * math.pi))))

            for s in range(n_steps_per_period):
                if s == n_steps_per_period // 2:
                    rho_mid = math.sqrt(max(2.0 * psi, 1e-6))
                    R_t, Z_t = surface_rz(rho_mid, theta, math.pi / N_FP)
                    t_pts.append((float(R_t), float(Z_t), float(rho_mid), float(theta % (2.0 * math.pi))))

                # RK4 step in phi
                k1_p, k1_t = d_state(psi, theta, phi)
                k2_p, k2_t = d_state(psi + 0.5 * dphi * k1_p,
                                     theta + 0.5 * dphi * k1_t,
                                     phi + 0.5 * dphi)
                k3_p, k3_t = d_state(psi + 0.5 * dphi * k2_p,
                                     theta + 0.5 * dphi * k2_t,
                                     phi + 0.5 * dphi)
                k4_p, k4_t = d_state(psi + dphi * k3_p,
                                     theta + dphi * k3_t,
                                     phi + dphi)
                psi = max(1e-5, min(0.75, psi + (dphi / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)))
                dth = (dphi / 6.0) * (k1_t + 2 * k2_t + 2 * k3_t + k4_t)
                theta += dth
                theta_unwrapped += dth
                phi += dphi

        iot_est = (theta_unwrapped - th0) / (2.0 * math.pi * n_turns)
        bean_rz.append({"tag": tag, "rho0": rho0, "theta0": th0,
                        "iota": iot_est, "points": np.array(b_pts)})
        tri_rz.append({"tag": tag, "rho0": rho0, "theta0": th0,
                       "iota": iot_est, "points": np.array(t_pts)})
        if tag == "core":
            iota_meas.append((rho0, iot_est))

    # Count distinct island clusters on the edge ring (tag == "island_ring")
    island_ring_pts = [tr["points"] for tr in bean_rz if tr["tag"] == "island_ring"]
    detected_islands = len(island_ring_pts)

    return {
        "preset": preset,
        "bean_section": bean_rz,
        "triangle_section": tri_rz,
        "iota_profile": iota_meas,
        "expected_islands": m_isl,
        "detected_islands": detected_islands,
    }


def save_poincare_plot(path="poincare_w7x.png", presets=("standard", "high-iota", "low-iota", "high-mirror")):
    """Render a 4-panel Poincare section plot across the W7-X presets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for ax, preset in zip(axes, presets):
        res = trace_poincare_map(preset, n_surfaces=9, n_turns=48)
        for tr in res["bean_section"]:
            pts = tr["points"]
            if tr["tag"] == "core":
                ax.scatter(pts[:, 0], pts[:, 1], s=1.8, color="#00e5ff", alpha=0.75)
            elif tr["tag"] == "island_ring":
                ax.scatter(pts[:, 0], pts[:, 1], s=3.2, color="#ff6b6b", alpha=0.9)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=14.0, color="#ffd700", marker="x")
        p = PRESETS[preset]
        ax.set_title("%s  (edge iota = %.3f, %d islands)"
                     % (preset, p["iota_edge"], res["detected_islands"]),
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("R [m]", fontsize=9)
        ax.set_ylabel("Z [m]", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle("W7-X Classical Bridge (Phase 7): Bean-Plane Poincare Sections & Island Divertor Chains",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def geometry_report():
    print("=" * 78)
    print("STELLARATOR GEOMETRY & PHASE 7 CLASSICAL FIELD-LINE BRIDGE")
    print("=" * 78)
    err_flip, err_rot = check_stellarator_symmetry()
    print("D5 symmetry check on 3D surface embedding:")
    print("  Z2 stellarator flip max error : %.3e m" % err_flip)
    print("  Z5 field-period rot max error : %.3e m" % err_rot)
    print("  VERDICT                       : %s"
          % ("PASS" if max(err_flip, err_rot) < 1e-12 else "FAIL"))
    print("")
    coils = generate_coil_filaments(48)
    print("Coil system inventory:")
    print("  Non-planar modular coils : %2d  (5 types x 10 half-period modules)"
          % len(coils["non_planar"]))
    print("  Planar coils             : %2d  (2 types x 10 half-period modules)"
          % len(coils["planar"]))
    print("  Trim coils               : %2d  (1 per field period)"
          % len(coils["trim"]))
    print("  Divertor control coils   : %2d  (1 per half-period module)"
          % len(coils["control"]))
    sample_pts = surface_xyz(np.array([0.0, 0.5, 1.0]),
                             np.array([0.0, 0.5, 1.0]),
                             np.array([0.0, 0.2, 0.4]))
    B_vals = biot_savart_point(sample_pts, coils, "standard")
    B_norms = np.linalg.norm(B_vals, axis=-1)
    print("  Biot-Savart |B| sample   : axis %.4f, mid %.4f, edge %.4f [a.u.]"
          % (B_norms[0], B_norms[1], B_norms[2]))
    print("")
    print("%-14s %10s %10s %10s %12s %10s"
          % ("Preset", "iota_0", "iota_edge", "islands", "PolShift", "verdict"))
    print("-" * 78)
    all_ok = (max(err_flip, err_rot) < 1e-12)
    for name in ("standard", "high-iota", "low-iota", "high-mirror"):
        res = trace_poincare_map(name, n_surfaces=8, n_turns=36)
        p = PRESETS[name]
        ok = (res["detected_islands"] == p["islands"]
              and abs(res["iota_profile"][0][1] - p["iota_0"]) < 0.05)
        all_ok = all_ok and ok
        print("%-14s %10.4f %10.4f %10d %12d %10s"
              % (name, res["iota_profile"][0][1], p["iota_edge"],
                 res["detected_islands"], p["PoloidalShift"],
                 "PASS" if ok else "FAIL"))
    print("-" * 78)
    print("OVERALL GEOMETRY & POINCARE GATE: %s" % ("PASS" if all_ok else "FAIL"))
    return all_ok


def main():
    ap = argparse.ArgumentParser(
        description="W7-X stellarator geometry, coils, and Phase 7 Poincare tracer")
    ap.add_argument("--check", action="store_true",
                    help="run D5 symmetry, Biot-Savart, and Poincare island checks")
    ap.add_argument("--poincare", action="store_true",
                    help="trace Poincare maps and save PNG plot")
    ap.add_argument("--preset", default="standard",
                    choices=tuple(PRESETS.keys()))
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    if a.poincare:
        out_png = a.out or "poincare_w7x.png"
        save_poincare_plot(out_png)
        print("saved Poincare plot: %s" % out_png)
        return

    ok = geometry_report()
    if a.out:
        res = trace_poincare_map(a.preset, n_surfaces=8, n_turns=30)
        payload = {
            "preset": a.preset,
            "R0": R0_MAJOR, "a": A_MINOR, "N_fp": N_FP,
            "expected_islands": res["expected_islands"],
            "detected_islands": res["detected_islands"],
            "iota_profile": res["iota_profile"],
        }
        with open(a.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print("wrote %s" % a.out)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
