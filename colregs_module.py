"""
COLREGs Collision Avoidance Module for Multi-USV Environment

Implements:
1. Dynamic Ship Domain (Tam-Bucknall model) — speed-dependent safety zone
2. DCPA/TCPA computation (Formulas 6)
3. Collision Risk Index (CRI) with membership functions (Formulas 7-12)
4. COLREGs encounter type classification (E1-E6)
5. COLREGs compliance penalty computation (R_C = k_C * u_CRI)

References:
- Tam & Bucknall (2010): Ship domain model
- COLREGs Rules 13-17: Encounter situations and obligations
- Formulas (6)-(12): CRI computation model
"""

import numpy as np
from enum import IntEnum


# ============================================================================
# ========================= Constants ========================================
# ============================================================================

class EncounterType(IntEnum):
    """COLREGs encounter type classification (E1-E6 + SAFE)."""
    SAFE = 0                    # No collision risk
    OVERTAKING_GIVE_WAY = 1     # E1: Own ship overtaking, must give way
    OVERTAKING_STAND_ON = 2     # E2: Target overtaking own, maintain course
    HEAD_ON = 3                 # E3: Head-on, both turn starboard
    CROSSING_GIVE_WAY = 4       # E4: Target on starboard, own must give way (turn starboard)
    CROSSING_STAND_ON = 5       # E5: Target on port, own maintains course
    RULE_VIOLATION = 6          # E6: Target violating rules, own must give way


NUM_ENCOUNTER_TYPES = 7

# Per-other-USV observation feature dimension:
# rel_pos(2) + rel_vel(2) + heading_diff(2) + dcpa(1) + tcpa(1) + cri(1) + encounter_onehot(7) = 16
OTHER_USV_OBS_DIM = 16


# ============================================================================
# ========================= Utilities ========================================
# ============================================================================

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _usv_global_velocity(u, v, heading):
    """Convert body-frame velocity to global-frame."""
    vx = u * np.cos(heading) - v * np.sin(heading)
    vy = u * np.sin(heading) + v * np.cos(heading)
    return vx, vy


# ============================================================================
# ========================= Ship Domain ======================================
# ============================================================================

class ShipDomain:
    """
    Dynamic Ship Domain based on Tam-Bucknall (2010) model.

    Shape: front half = semi-ellipse (major axis r_a), rear half = semi-circle (radius r_b).
    Side axis r_c = r_b.  All axes scale with USV speed.

    The Dynamic Area (DA) is a scaled-up version used for CRI transition (R = da_scale * r).
    """

    def __init__(self, r_min=4.0, t1=1.2, t2=0.6,
                 v_cr1=2.5, v_cr2=6.0, da_scale=2.5):
        """
        Args:
            r_min:    Minimum safety radius (m), ≈ USV bounding diameter.
            t1:       Time parameter for front domain growth (s).
            t2:       Time parameter for rear domain at moderate speed (s).
            v_cr1:    First critical velocity threshold (m/s).
            v_cr2:    Second critical velocity threshold (m/s).
            da_scale: Dynamic Area = da_scale × domain radius for CRI.
        """
        self.r_min = float(r_min)
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.v_cr1 = float(v_cr1)
        self.v_cr2 = float(v_cr2)
        self.da_scale = float(da_scale)

    def compute_domain(self, speed):
        """
        Compute ship domain dimensions for given speed.

        Returns:
            r_a: front semi-ellipse major axis
            r_b: rear semi-circle radius
            r_c: side semi-axis (= r_b)
        """
        speed = max(0.0, abs(speed))

        # Front axis (Formula 3): r_a = V * t1 + r_min
        r_a = speed * self.t1 + self.r_min

        # Rear radius (Formula 4): piecewise by speed
        if speed < self.v_cr1:
            r_b = speed * self.t1 + self.r_min
        elif speed < self.v_cr2:
            r_b = speed * self.t2 - self.v_cr2 * self.t2 + self.r_min
        else:
            r_b = self.r_min

        r_b = max(r_b, self.r_min)
        r_c = r_b  # Formula 5

        return r_a, r_b, r_c

    def get_domain_radius_at_bearing(self, speed, bearing_rel):
        """
        Get the ship domain radius at a specific relative bearing.

        Args:
            speed:       USV speed (m/s).
            bearing_rel: Bearing relative to heading, [-pi, pi].  0 = ahead.

        Returns:
            r: domain radius at this bearing (m).
        """
        r_a, r_b, r_c = self.compute_domain(speed)
        bearing = abs(bearing_rel)

        if bearing <= np.pi / 2:
            # Front half: semi-ellipse  r = a*c / sqrt((c*cos)^2 + (a*sin)^2)
            cos_b = np.cos(bearing)
            sin_b = np.sin(bearing)
            denom = np.sqrt((r_c * cos_b) ** 2 + (r_a * sin_b) ** 2)
            r = r_a * r_c / max(denom, 1e-8)
        else:
            # Rear half: semi-circle
            r = r_b

        return r

    def get_dynamic_area_radius(self, speed, bearing_rel):
        """Dynamic Area radius R = da_scale × domain_radius at given bearing."""
        return self.da_scale * self.get_domain_radius_at_bearing(speed, bearing_rel)


# ============================================================================
# ========================= DCPA / TCPA ======================================
# ============================================================================

def compute_dcpa_tcpa(own_x, own_y, own_vx, own_vy,
                      target_x, target_y, target_vx, target_vy):
    """
    Compute DCPA and TCPA between two vessels (Formula 6, vector form).

    Args:
        own_x, own_y:       Own ship position.
        own_vx, own_vy:     Own ship global velocity.
        target_x, target_y: Target ship position.
        target_vx, target_vy: Target ship global velocity.

    Returns:
        dcpa:  Distance at Closest Point of Approach (m).
        tcpa:  Time to CPA (s). Negative ⇒ CPA in the past.
        v_rel: Magnitude of relative velocity (m/s).
    """
    rel_x = target_x - own_x
    rel_y = target_y - own_y
    rel_vx = target_vx - own_vx
    rel_vy = target_vy - own_vy

    dist = np.sqrt(rel_x ** 2 + rel_y ** 2)
    v_rel_sq = rel_vx ** 2 + rel_vy ** 2
    v_rel = np.sqrt(v_rel_sq)

    if v_rel < 1e-8:
        return dist, float('inf'), 0.0

    tcpa = -(rel_x * rel_vx + rel_y * rel_vy) / v_rel_sq

    cpa_x = rel_x + rel_vx * tcpa
    cpa_y = rel_y + rel_vy * tcpa
    dcpa = np.sqrt(cpa_x ** 2 + cpa_y ** 2)

    return dcpa, tcpa, v_rel


# ============================================================================
# ========================= Membership Functions (CRI) =======================
# ============================================================================

def compute_u_dcpa(dcpa, r, R):
    """
    DCPA membership function → spatial risk in [0, 1]  (Formula 7).

    Args:
        dcpa: Absolute DCPA value (m).
        r:    Ship domain radius at the CPA bearing (m).
        R:    Dynamic area radius at the CPA bearing (m).
    """
    abs_dcpa = abs(dcpa)

    if abs_dcpa <= r:
        return 1.0
    elif abs_dcpa <= R:
        if R - r < 1e-8:
            return 1.0
        # Smooth sine transition (boundary-verified: u(r)=1, u(R)=0)
        arg = np.pi / (R - r) * abs_dcpa - np.pi * (R + r) / (2.0 * (R - r))
        return 0.5 - 0.5 * np.sin(arg)
    else:
        return 0.0


def compute_time_thresholds(dcpa, r, R, v_rel):
    """
    Compute time thresholds T_r and T_R for TCPA membership (Formulas 10-11).

    Args:
        dcpa:  Absolute DCPA (m).
        r:     Ship domain radius (m).
        R:     Dynamic area radius (m).
        v_rel: Relative speed (m/s).

    Returns:
        T_r: Inner time threshold (crossing domain boundary).
        T_R: Outer time threshold (crossing dynamic area boundary).
    """
    abs_dcpa = abs(dcpa)

    if v_rel < 1e-8:
        return float('inf'), float('inf')

    # Formula 10
    if r >= abs_dcpa:
        T_r = np.sqrt(r ** 2 - abs_dcpa ** 2) / v_rel
    else:
        T_r = 0.0

    # Formula 11
    if R >= abs_dcpa:
        T_R = np.sqrt(R ** 2 - abs_dcpa ** 2) / v_rel
    else:
        T_R = 0.0

    return T_r, T_R


def compute_u_tcpa(tcpa, T_r, T_R):
    """
    TCPA membership function → temporal risk in [0, 1]  (Formulas 8-9).

    Args:
        tcpa: Time to CPA (can be negative if CPA is in the past).
        T_r:  Inner time threshold.
        T_R:  Outer time threshold.
    """
    if T_R <= T_r + 1e-8:
        return 1.0 if abs(tcpa) <= max(T_r, 0.1) else 0.0

    if tcpa > 0:
        # Ships approaching each other (Formula 8)
        if tcpa <= T_r:
            return 1.0
        elif tcpa <= T_R:
            return ((T_R - tcpa) / (T_R - T_r)) ** 2
        else:
            return 0.0
    else:
        # Ships have passed or are separating (Formula 9)
        abs_tcpa = abs(tcpa)
        if abs_tcpa <= T_r:
            return 1.0
        elif abs_tcpa <= T_R:
            return ((T_R - abs_tcpa) / (T_R - T_r)) ** 2
        else:
            return 0.0


def compute_cri(u_dcpa, u_tcpa):
    """
    Collision Risk Index (Formula 12).

    CRI = max(u_DCPA, u_TCPA) when both are non-zero, else 0.
    """
    if u_dcpa == 0.0 or u_tcpa == 0.0:
        return 0.0
    return max(u_dcpa, u_tcpa)


# ============================================================================
# ========================= Encounter Classification =========================
# ============================================================================

# Sector thresholds (COLREGs standard)
_HEAD_ON_BEARING = np.radians(15)       # Dead-ahead sector half-width
_HEAD_ON_HEADING_DIFF = np.radians(170) # Nearly reciprocal courses
_STERN_ARC = np.radians(112.5)          # Forward arc boundary (22.5° abaft beam)


def classify_encounter(own_heading, target_heading, bearing_rel):
    """
    Classify one of six COLREGs encounter types (E1-E6).

    Coordinate convention (standard maths — same as USV.py):
      - heading:     angle from +x axis, counter-clockwise.
      - bearing_rel: angle of target relative to own heading.
                     0 = dead ahead,  negative = starboard (right),
                     positive = port (left).

    Returns:
        EncounterType enum value.
    """
    heading_diff = normalize_angle(target_heading - own_heading)
    abs_bearing = abs(bearing_rel)

    # ---- 1. Head-on (E3, Rule 14) ----
    # Both vessels are roughly ahead of each other and on nearly reciprocal courses.
    if abs_bearing < _HEAD_ON_BEARING and abs(heading_diff) > _HEAD_ON_HEADING_DIFF:
        return EncounterType.HEAD_ON

    # ---- 2. Overtaking check (Rules 13) ----
    # "Overtaking" = approaching from more than 22.5° abaft the other's beam.
    # bearing of own from target (relative to target heading)
    bearing_own_from_target = normalize_angle(
        bearing_rel + own_heading + np.pi - target_heading)

    # Own ship is in target's stern arc ⇒ own is overtaking (E1, give way)
    if abs(bearing_own_from_target) > _STERN_ARC:
        return EncounterType.OVERTAKING_GIVE_WAY

    # Target is in own ship's stern arc ⇒ target is overtaking us (E2, stand on)
    if abs_bearing > _STERN_ARC:
        return EncounterType.OVERTAKING_STAND_ON

    # ---- 3. Crossing situations (Rules 15) ----
    # Negative bearing_rel = target on starboard ⇒ own is give-way (E4)
    if bearing_rel < 0:
        return EncounterType.CROSSING_GIVE_WAY
    else:
        return EncounterType.CROSSING_STAND_ON


# ============================================================================
# ========================= Full CRI Assessment ==============================
# ============================================================================

def compute_full_colregs_assessment(own_usv, target_usv, ship_domain):
    """
    Compute complete COLREGs assessment between two USVs.

    Args:
        own_usv:     Object with attributes (x, y, u, v, r, heading, max_u).
        target_usv:  Same as above.
        ship_domain: ShipDomain instance.

    Returns:
        dict with: dcpa, tcpa, v_rel, bearing_rel,
                   u_dcpa, u_tcpa, cri, encounter_type.
    """
    # Global velocities
    own_vx, own_vy = _usv_global_velocity(own_usv.u, own_usv.v, own_usv.heading)
    tgt_vx, tgt_vy = _usv_global_velocity(target_usv.u, target_usv.v,
                                           target_usv.heading)
    own_speed = np.sqrt(own_usv.u ** 2 + own_usv.v ** 2)

    # DCPA / TCPA (Formula 6)
    dcpa, tcpa, v_rel = compute_dcpa_tcpa(
        own_usv.x, own_usv.y, own_vx, own_vy,
        target_usv.x, target_usv.y, tgt_vx, tgt_vy)

    # Bearing of target relative to own heading
    dx = target_usv.x - own_usv.x
    dy = target_usv.y - own_usv.y
    bearing_abs = np.arctan2(dy, dx)
    bearing_rel = normalize_angle(bearing_abs - own_usv.heading)

    # Ship domain & dynamic area radii at target's bearing
    r = ship_domain.get_domain_radius_at_bearing(own_speed, bearing_rel)
    R = ship_domain.get_dynamic_area_radius(own_speed, bearing_rel)

    # Membership functions (Formulas 7-11)
    u_dcpa = compute_u_dcpa(dcpa, r, R)
    T_r, T_R = compute_time_thresholds(dcpa, r, R, v_rel)
    u_tcpa = compute_u_tcpa(tcpa, T_r, T_R)

    # CRI (Formula 12)
    cri = compute_cri(u_dcpa, u_tcpa)

    # Encounter type (E1-E6)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if dist < 1e-3 or cri < 1e-6:
        encounter_type = EncounterType.SAFE
    else:
        encounter_type = classify_encounter(
            own_usv.heading, target_usv.heading, bearing_rel)

    return {
        'dcpa': dcpa,
        'tcpa': tcpa,
        'v_rel': v_rel,
        'bearing_rel': bearing_rel,
        'dist': dist,
        'r': r,
        'R': R,
        'u_dcpa': u_dcpa,
        'u_tcpa': u_tcpa,
        'cri': cri,
        'encounter_type': encounter_type,
    }


# ============================================================================
# ========================= COLREGs Reward Penalty ===========================
# ============================================================================

def compute_colregs_penalty(encounter_type, cri, rudder_action,
                            k_C=5.0, starboard_threshold=-0.05):
    """
    Compute COLREGs compliance penalty (R_C from Formula 27).

    Penalty logic:
      - E3 (Head-on) / E4 (Crossing give-way):
          Must turn starboard. If rudder_action > starboard_threshold
          (i.e. NOT a starboard turn), penalty = -k_C × CRI.
      - E1 (Overtaking give-way):
          Must take evasive action. If |rudder| < small threshold,
          penalty = -0.5 × k_C × CRI.
      - E2, E5 (Stand-on): no directional penalty.
      - SAFE: no penalty.

    Coordinate convention:
      rudder_action < 0  →  starboard (right) turn.
      rudder_action > 0  →  port (left) turn.

    Args:
        encounter_type:      EncounterType enum value.
        cri:                 Collision Risk Index in [0, 1].
        rudder_action:       Rudder action value (action[1]).
        k_C:                 Base penalty coefficient.
        starboard_threshold: Threshold below which rudder counts as starboard.

    Returns:
        penalty: Non-positive float.
    """
    if cri < 1e-6:
        return 0.0

    if encounter_type == EncounterType.HEAD_ON or \
       encounter_type == EncounterType.CROSSING_GIVE_WAY:
        # Must turn starboard (action[1] < starboard_threshold)
        if rudder_action > starboard_threshold:
            return -k_C * cri
        return 0.0

    if encounter_type == EncounterType.OVERTAKING_GIVE_WAY:
        # Must take some evasive action (either direction)
        if abs(rudder_action) < 0.1:
            return -0.5 * k_C * cri
        return 0.0

    # E2, E5, SAFE, RULE_VIOLATION — no directional penalty
    return 0.0


# ============================================================================
# ========================= Observation Features =============================
# ============================================================================

def get_other_usv_obs_features(own_usv, target_usv, colregs_info,
                               x_max=100.0, y_max=100.0):
    """
    Build the 16-dim observation feature vector for one (own, target) pair.

    Layout:
      [0:2]   rel_pos_norm     (rel_x / x_max, rel_y / y_max)
      [2:4]   rel_vel_norm     (rel_vx, rel_vy) / max_rel_speed
      [4:6]   heading_diff     (sin, cos)
      [6]     dcpa_norm        dcpa / diag
      [7]     tcpa_norm        clip(tcpa, -50, 50) / 50
      [8]     cri              [0, 1]
      [9:16]  encounter_onehot  7 dims

    Args:
        own_usv:      Own USV object.
        target_usv:   Target USV object.
        colregs_info: Dict from compute_full_colregs_assessment.
        x_max, y_max: Environment bounds for normalization.

    Returns:
        np.ndarray of shape (16,).
    """
    diag = np.sqrt(x_max ** 2 + y_max ** 2)
    max_rel_speed = own_usv.max_u * 2.0

    # Relative position (normalized)
    rel_x = (target_usv.x - own_usv.x) / x_max
    rel_y = (target_usv.y - own_usv.y) / y_max

    # Relative velocity (global, normalized)
    own_vx, own_vy = _usv_global_velocity(own_usv.u, own_usv.v, own_usv.heading)
    tgt_vx, tgt_vy = _usv_global_velocity(target_usv.u, target_usv.v,
                                           target_usv.heading)
    rel_vx = (tgt_vx - own_vx) / max(max_rel_speed, 1e-8)
    rel_vy = (tgt_vy - own_vy) / max(max_rel_speed, 1e-8)

    # Heading difference
    hd = target_usv.heading - own_usv.heading
    hd_sin = np.sin(hd)
    hd_cos = np.cos(hd)

    # COLREGs quantities
    dcpa_norm = colregs_info['dcpa'] / diag
    tcpa_norm = np.clip(colregs_info['tcpa'], -50.0, 50.0) / 50.0
    cri = colregs_info['cri']

    # Encounter type one-hot
    enc_onehot = np.zeros(NUM_ENCOUNTER_TYPES, dtype=np.float32)
    enc_onehot[int(colregs_info['encounter_type'])] = 1.0

    return np.array([
        rel_x, rel_y, rel_vx, rel_vy,
        hd_sin, hd_cos,
        dcpa_norm, tcpa_norm, cri,
        *enc_onehot
    ], dtype=np.float32)
