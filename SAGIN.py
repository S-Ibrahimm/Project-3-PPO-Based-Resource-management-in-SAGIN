"""
SAGIN PPO: RIS-Assisted Drone Communication with PPO


Project summary
----------------
This project trains a PPO (Proximal Policy Optimization) reinforcement
learning agent to control a moving drone's trajectory, transmit power,
RIS (Reconfigurable Intelligent Surface) activation level, and satellite
offload fraction, in order to keep a ground user connected across a mix of
clear-signal (LoS) and blocked-signal (NLoS, "shadow zone") regions.

This builds on two earlier, simpler projects:
  - Project 1 used a Deep Q-Network (DQN) with a fixed RIS + fixed drone
    position, and found that RIS gives negligible benefit at long,
    clear-signal ranges ("double path loss": a two-hop reflected path loses
    more signal than it gains from reflecting).
  - Project 2 extended this to a moving drone and added a sensing task.
  - This project (3) moves to a continuous action space (PPO instead of
    DQN), a full 3D moving drone, mixed LoS/NLoS geometry, and a satellite
    backhaul option.

Key finding
------------
Running the RIS contribution analysis below (see analyze_ris_contribution)
shows that, in this project's specific geometry, activating the RIS changes
the total signal by only a tiny fraction of a decibel - even with all 128
elements active. The trained agent learns to never activate the RIS at all,
which is the mathematically correct response given the reward function
charges a small cost for RIS use and the physical benefit is negligible.
This is the same "double path loss" effect identified in Project 1, now
confirmed in a moving-drone, mixed LoS/NLoS setting.

Running this file
-------------------
python sagin_ppo.py

If a previously trained model file (sagin_ppo_trained_model.pt) exists in
the working directory, it will be loaded instead of retraining from
scratch. Otherwise, training runs from scratch (400,000 environment steps)
and the trained model is saved at the end for future runs.

Dependencies: torch, gymnasium, numpy, scipy, matplotlib. See
requirements.txt.
"""

import random
import time
import os  # used to check whether a previously saved trained model exists on disk
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.special import erfc
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. ENVIRONMENT DEFINITION

class SAGINEnv(gym.Env):
    """
    Custom Gymnasium Environment for PPO-based joint control of:
    drone 3D trajectory, transmit power, RIS activation level, and satellite
    offload fraction, in a mixed LoS/NLoS SAGIN (Satellite-Air-Ground Integrated
    Network) scenario. Builds directly on the physics style of Project 2's
    WirelessEnv (FSPL + Rician fading + RIS sub-array gain), but action space is
    now continuous (PPO) instead of discrete (DQN), and geometry is full 3D.
    """

    def __init__(self):
        super().__init__()

        # ------------------------------------------------------------------
        # Drone 3D movement bounds and per-step limits
        # ------------------------------------------------------------------
        self.x_bound = 5000.0          # horizontal range: -5000 to +5000 m (same span as Project 2's x-axis)
        self.y_bound = 5000.0          # second horizontal axis, drone can now also move sideways
        self.alt_min = 100.0           # minimum flight altitude (m)
        self.alt_max = 2000.0          # maximum flight altitude (m)


        self.max_horizontal_step = 50.0  # max metres the drone can move in x or y per step
        self.max_vertical_step = 20.0    # max metres the drone can climb/descend per step

        # ------------------------------------------------------------------
        # Continuous action space: [dx, dy, dz, power, ris, offload]
        # All 6 outputs are normalized to [-1, 1] (standard PPO practice — keeps
        # the policy network's output distribution well-behaved). We rescale
        # each dimension to its real physical range inside step().
        # ------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        # Index guide (kept here so step() reads clearly later):
        # 0 -> dx (horizontal move, x-axis)
        # 1 -> dy (horizontal move, y-axis)
        # 2 -> dz (vertical move, altitude change)
        # 3 -> transmit power level (continuous, rescaled to a dBm range)
        # 4 -> RIS activation level (continuous, rounded to 0-4 sub-arrays internally)
        # 5 -> satellite offload fraction (continuous, 0-1 split between direct/sat link)

        # ------------------------------------------------------------------
        # Power range — kept continuous now instead of Project 2's
        # discrete power_lvls list. Same realistic ceiling reasoning as Project 2
        # (directional small-drone link, ~33-36 dBm max).
        # ------------------------------------------------------------------
        self.power_min_dbm = -10.0
        self.power_max_dbm = 36.0

        # RIS still physically has 4 sub-arrays of 32 elements each
        # (128 elements total) — same hardware as Project 1/2. The agent's continuous
        # output gets rounded to the nearest integer in {0,1,2,3,4} inside step().
        self.num_ris_subarrays = 4
        self.elements_per_subarray = 32
        self.ris_x = -500.0            #x position of ris - ris is placed near shadow zones
        self.ris_y = 500.0

        # ------------------------------------------------------------------
        # Fixed geographic shadow zones (mixed LoS/NLoS).
        # Each zone is a rectangle in the horizontal (x, y) plane: if the user
        # falls inside one of these, the direct drone-to-user link is treated
        # as NLoS (heavily degraded) regardless of drone altitude; outside all
        # zones it's normal LoS. Using the USER's position (not the drone's)
        # to decide LoS/NLoS, since the user is the one whose connectivity we
        # care about — this choice is flagged for you to confirm before we
        # write the geometry-check function.
        # ------------------------------------------------------------------
        self.shadow_zones = [
            # (x_min, x_max, y_min, y_max) in metres
            (-3000.0, -1000.0, -1500.0, 1500.0),
            (500.0, 2500.0, 1000.0, 3000.0),
        ]
        self.nlos_extra_attenuation_db = 25.0  # extra loss applied to direct path inside a shadow zone

        # ------------------------------------------------------------------
        # Satellite backhaul "pipe" constants
        # ------------------------------------------------------------------
        self.sat_fixed_delay_penalty = 0.5     # reward cost per unit offload fraction, per step
        self.sat_budget_capacity = 100.0       # total rolling budget available per episode (abstract units)
        self.sat_budget_drain_rate = 5.0       # how many budget units 1.0 offload fraction drains per step
        self.sat_congestion_penalty_weight = 2.0  # penalty scaling once budget gets low/exceeded

        # ------------------------------------------------------------------
        # Carried over from Project 2 (unchanged physics constants)
        # ------------------------------------------------------------------
        self.target_snr = 5.0
        self.noise_floor = -90.0
        self.lam = 0.01                     # power penalty factor
        self.carrier_freq_hz = 2.4e9
        self.rician_k_factor_db = 6.0
        self.csi_noise_std_db = 2.0
        self.eta_ris = 2.0
        # Small reward-shaping constants needed by step()
        self.target_ber = 1e-3                 # BER threshold used both for reward bonus AND automated modulation selection
        self.ris_activation_cost_weight = 0.1  # small per-subarray cost, so RIS is no longer "free" like in Project 1
        self.energy_cost_weight = 0.001        # tiny cost per metre moved, discourages pointless drone thrashing

        # Episode length extended from 50 to 100 steps, to give the
        # drone enough runway to actually pass through both LoS and NLoS zones in 3D.
        self.max_steps = 100
        self.current_step = 0

        # ------------------------------------------------------------------
        # Observation space — placeholder dimensionality for now.
        # We'll finalize the exact variable list together in the next chunk (reset()),
        # but reserving 10 dims here based on our planning summary:
        # [drone_x, drone_y, drone_z, user_dist_3d, noisy_gain_db,
        #  sat_budget_remaining, ris_to_user_dist, los_nlos_flag,
        #  steps_remaining, (1 spare slot for now)]
        # ------------------------------------------------------------------
        obs_dim = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

            # ------------------------------------------------------------------
    # Helper: checks if a given (x, y) point falls inside
    # any of the fixed shadow zones defined in __init__. Returns True if
    # NLoS (blocked/degraded), False if LoS (clear). Used for the user's
    # position, per our earlier discussion — the receiver's surroundings
    # are what matters for blockage, not the drone's altitude.
    # ------------------------------------------------------------------
    def _check_los_status(self, pos_x, pos_y):
        for (x_min, x_max, y_min, y_max) in self.shadow_zones:
            if x_min <= pos_x <= x_max and y_min <= pos_y <= y_max:
                return True  # inside a shadow zone -> NLoS
        return False  # not inside any zone -> LoS

    # ------------------------------------------------------------------
    # reset(): starts a new episode. Mirrors Project 2's
    # reset() structure (random user position fixed for the episode, drone
    # starts at trajectory edge) but extended to 3D and adds satellite
    # budget refill + LoS/NLoS status for the first observation.
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Drone starts at a corner of its 3D operating volume:
        # far horizontal edge, mid-altitude (arbitrary reasonable starting point,
        # flag if you'd prefer a different/randomized start later).
        self.drone_x = -self.x_bound
        self.drone_y = 0.0
        self.drone_z = (self.alt_min + self.alt_max) / 2.0

        # User position now has both x and y, randomized once
        # per episode and then fixed — same "Option B" pattern as Project 2.
        self.user_x = np.random.uniform(-self.x_bound, self.x_bound)
        self.user_y = np.random.uniform(-self.y_bound, self.y_bound)

        # Satellite budget refills to full capacity at the
        # start of every episode — this is the "data plan resets each month"
        # analogy, just per-episode instead of per-month.
        self.sat_budget_remaining = self.sat_budget_capacity

        # Determine LoS/NLoS status for the user's starting
        # position — needed for both the reward logic later and the state.
        self.is_nlos = self._check_los_status(self.user_x, self.user_y)

        # Placeholder initial channel gain, just to have
        # something to build the very first state vector from — same
        # reasoning as Project 2's reset() (a "seed" value, real physics
        # happens in step()). Full 3D channel gain function comes in the
        # next chunk, so this is temporarily a rough placeholder.
        dist_3d = np.sqrt(
            (self.user_x - self.drone_x) ** 2
            + (self.user_y - self.drone_y) ** 2
            + self.drone_z ** 2
        )
        self.chan_gain_db = -20.0 * np.log10(max(dist_3d, 1.0))  # rough placeholder, replaced properly in step()

        # Return normalized state vector and empty info dict.
        # _get_normalized_state() itself will be written in the next chunk.
        return self._get_normalized_state(), {}


        # ------------------------------------------------------------------
    # Full 3D channel gain function — same overall recipe
    # as Project 2's _compute_channel_gain_db() (FSPL + Rician fading for
    # the direct path, FSPL + terrestrial path loss + array gain for the
    # RIS-reflected path, then superposition), but:
    #   1. Distances are true 3D (x, y, z) instead of 1D + fixed altitude.
    #   2. Direct path gets extra attenuation automatically when the
    #      object (user) is inside a shadow zone (NLoS), via _check_los_status().
    #   3. No separate 'sensing' link_type here — Project 3 has no sensing
    #      task (dropped per our scoping decision), only comms.
    # ------------------------------------------------------------------
    def _compute_channel_gain_db(self, object_x, object_y, ris_mode, fixed_fading_linear=None):
        c = 3e8

        # --- Direct path: drone -> object, true 3D distance ---
        dist_horiz_direct = np.sqrt((object_x - self.drone_x) ** 2 + (object_y - self.drone_y) ** 2)
        d_3d_direct = np.sqrt(dist_horiz_direct ** 2 + self.drone_z ** 2)

        fspl_direct_db = (
            20.0 * np.log10(max(d_3d_direct, 1.0))
            + 20.0 * np.log10(self.carrier_freq_hz)
            + 20.0 * np.log10((4.0 * np.pi) / c)
        )

        # NLoS check — if the user is inside a shadow zone,
        # pile on extra attenuation to represent the blockage. This is what
        # gives the RIS a genuine, situational job (see concept discussion).
        is_nlos = self._check_los_status(object_x, object_y)
        if is_nlos:
            fspl_direct_db = fspl_direct_db + self.nlos_extra_attenuation_db

        # --- Rician fading (unchanged formula from Project 1/2) ---
        if fixed_fading_linear is not None:
            fading_linear = fixed_fading_linear
        else:
            k_linear = 10.0 ** (self.rician_k_factor_db / 10.0)
            los_component = np.sqrt(k_linear / (k_linear + 1.0))
            scatter_component = np.sqrt(1.0 / (k_linear + 1.0)) * (
                np.random.randn() + 1j * np.random.randn()
            ) / np.sqrt(2.0)
            fading_linear = np.abs(los_component + scatter_component) ** 2

        gain_direct_linear = (10.0 ** (-fspl_direct_db / 10.0)) * max(fading_linear, 1e-4)

        # --- RIS-reflected path: drone -> RIS (3D) -> object (ground-level) ---
        dist_horiz_bs_ris = np.sqrt((self.ris_x - self.drone_x) ** 2 + (self.ris_y - self.drone_y) ** 2)
        d_3d_drone_to_ris = np.sqrt(dist_horiz_bs_ris ** 2 + self.drone_z ** 2)
        d_ris_to_object = max(np.sqrt((object_x - self.ris_x) ** 2 + (object_y - self.ris_y) ** 2), 1.0)

        fspl_drone_to_ris_db = (
            20.0 * np.log10(max(d_3d_drone_to_ris, 1.0))
            + 20.0 * np.log10(self.carrier_freq_hz)
            + 20.0 * np.log10((4.0 * np.pi) / c)
        )

        pl_0_db = 38.5  # reference terrestrial path loss at 1m (unchanged from Project 2)
        path_loss_ris_to_object_db = pl_0_db + (10.0 * self.eta_ris * np.log10(d_ris_to_object))
        total_ris_path_loss_db = fspl_drone_to_ris_db + path_loss_ris_to_object_db

        # ris_mode arrives here already rounded to an integer 0-4 by step()
        active_elements = ris_mode * self.elements_per_subarray
        array_gain = (active_elements ** 2) if active_elements > 0 else 1.0
        gain_ris_linear = (10.0 ** (-total_ris_path_loss_db / 10.0)) * array_gain

        # --- Superposition of direct + reflected paths (unchanged idea) ---
        total_gain_linear = gain_direct_linear + gain_ris_linear
        return 10.0 * np.log10(total_gain_linear), is_nlos


    # ------------------------------------------------------------------
    # Normalized state vector — 10 dimensions as reserved
    # in __init__: drone (x,y,z), 3D distance to user, noisy channel gain,
    # satellite budget remaining, RIS-to-user distance, LoS/NLoS flag,
    # steps remaining, and one spare slot reserved for later use.
    # ------------------------------------------------------------------
    def _get_normalized_state(self):
        noisy_chan_gain = self.chan_gain_db + np.random.normal(0.0, self.csi_noise_std_db)

        norm_drone_x = (self.drone_x + self.x_bound) / (2.0 * self.x_bound)
        norm_drone_y = (self.drone_y + self.y_bound) / (2.0 * self.y_bound)
        norm_drone_z = (self.drone_z - self.alt_min) / (self.alt_max - self.alt_min)

        dist_3d_user = np.sqrt(
            (self.user_x - self.drone_x) ** 2
            + (self.user_y - self.drone_y) ** 2
            + self.drone_z ** 2
        )
        # Rough normalizing constant: the largest possible 3D distance in this world
        max_possible_dist = np.sqrt((2 * self.x_bound) ** 2 + (2 * self.y_bound) ** 2 + self.alt_max ** 2)
        norm_dist_user = np.clip(dist_3d_user / max_possible_dist, 0.0, 1.0)

        norm_gain = np.clip((noisy_chan_gain - (-150.0)) / (-30.0 - (-150.0)), 0.0, 1.0)

        norm_sat_budget = np.clip(self.sat_budget_remaining / self.sat_budget_capacity, 0.0, 1.0)

        dist_ris_user = np.sqrt((self.user_x - self.ris_x) ** 2 + (self.user_y - self.ris_y) ** 2)
        max_ris_dist = np.sqrt((2 * self.x_bound) ** 2 + (2 * self.y_bound) ** 2)
        norm_ris_dist = np.clip(dist_ris_user / max_ris_dist, 0.0, 1.0)

        los_flag = 1.0 if self.is_nlos else 0.0
        steps_remaining_norm = (self.max_steps - self.current_step) / self.max_steps
        spare_slot = 0.0  # reserved for later (e.g. last offload fraction used)

        state = np.array([
            norm_drone_x, norm_drone_y, norm_drone_z,
            norm_dist_user, norm_gain, norm_sat_budget,
            norm_ris_dist, los_flag, steps_remaining_norm, spare_slot
        ], dtype=np.float32)

        return np.clip(state, 0.0, 1.0)

        # ------------------------------------------------------------------
    # Carried over unchanged from Project 2 (BER and data-rate formulas).
    # ------------------------------------------------------------------
    def _compute_ber(self, snr_db, modulation):
        snr_linear = 10.0 ** (snr_db / 10.0)

        def q_func(x):
            return 0.5 * erfc(x / np.sqrt(2.0))

        if modulation == 'BPSK':
            return float(q_func(np.sqrt(2.0 * snr_linear)))
        elif modulation == 'QPSK':
            return float(q_func(np.sqrt(snr_linear)))
        elif modulation == '16-QAM':
            return float(0.75 * q_func(np.sqrt(0.2 * snr_linear)))
        return 0.5

    def _compute_data_rate(self, modulation):
        rate_map = {'BPSK': 1.0, 'QPSK': 2.0, '16-QAM': 4.0}
        return rate_map.get(modulation, 1.0)

    # ------------------------------------------------------------------
    # Automated modulation selection, replacing the agent-chosen
    # modulation from Projects 1-2. Rather than fixed SNR thresholds (Project 1/2's
    # greedy baseline style), this tries the highest-throughput modulation first and
    # only backs off if it can't meet the BER target at the current SNR. This keeps
    # the choice physically justified by the same BER formula used in the reward,
    # instead of arbitrary hand-picked cutoffs.
    # ------------------------------------------------------------------
    def _select_modulation(self, snr_db):
        # Ordered from highest throughput to most robust
        for modulation in ['16-QAM', 'QPSK', 'BPSK']:
            ber = self._compute_ber(snr_db, modulation)
            if ber <= self.target_ber:
                return modulation, ber
        # Nothing met the target at this SNR -> fall back to the most robust
        # option anyway (BPSK has the lowest BER of the three at any given SNR),
        # so the agent still gets the least-bad outcome rather than an error.
        fallback_ber = self._compute_ber(snr_db, 'BPSK')
        return 'BPSK', fallback_ber

    # ------------------------------------------------------------------
    # step(): the core simulation tick. Rescales the agent's
    # 6 continuous [-1,1] outputs into real physical actions, moves the drone,
    # recomputes physics, determines modulation automatically, drains the
    # satellite budget, and assembles the full multi-term reward.
    # ------------------------------------------------------------------
    def step(self, action):
        self.current_step += 1

        # --------------------------------------------------------------
        # 1. Rescale raw [-1,1] action vector into real physical units.
        #    This is pure unit conversion — no physics happens here yet.
        # --------------------------------------------------------------
        dx = action[0] * self.max_horizontal_step
        dy = action[1] * self.max_horizontal_step
        dz = action[2] * self.max_vertical_step

        # Power: [-1,1] -> [power_min_dbm, power_max_dbm]
        power_dbm = self.power_min_dbm + (action[3] + 1.0) / 2.0 * (self.power_max_dbm - self.power_min_dbm)

        # RIS level: [-1,1] -> [0,4] continuous, then rounded to nearest integer
        # sub-array count (can't activate a fractional sub-array in hardware).
        ris_level_continuous = (action[4] + 1.0) / 2.0 * self.num_ris_subarrays
        ris_mode = int(np.clip(round(ris_level_continuous), 0, self.num_ris_subarrays))

        # Offload fraction: [-1,1] -> [0,1]
        offload_fraction = np.clip((action[5] + 1.0) / 2.0, 0.0, 1.0)

        # --------------------------------------------------------------
        # 2. Move the drone and clamp to the allowed 3D operating volume.
        #    Clamping prevents the agent from "escaping" into positions
        #    where the physics formulas would produce meaningless distances.
        # --------------------------------------------------------------
        self.drone_x = np.clip(self.drone_x + dx, -self.x_bound, self.x_bound)
        self.drone_y = np.clip(self.drone_y + dy, -self.y_bound, self.y_bound)
        self.drone_z = np.clip(self.drone_z + dz, self.alt_min, self.alt_max)

        # --------------------------------------------------------------
        # 3. Recompute channel gain at the new drone position, for the user's
        #    fixed (x,y). This also tells us the current LoS/NLoS status.
        # --------------------------------------------------------------
        self.chan_gain_db, self.is_nlos = self._compute_channel_gain_db(
            self.user_x, self.user_y, ris_mode=ris_mode
        )

        # --------------------------------------------------------------
        # 4. Comms quality: SNR, then automated modulation + BER + data rate.
        # --------------------------------------------------------------
        p_rx_dbm = power_dbm + self.chan_gain_db
        self.snr_db = p_rx_dbm - self.noise_floor

        chosen_modulation, ber = self._select_modulation(self.snr_db)
        data_rate = self._compute_data_rate(chosen_modulation)

        # --------------------------------------------------------------
        # 5. Comms reward term (same two-branch structure as Projects 1-2):
        #    a bonus for meeting the BER target, otherwise a continuous
        #    SNR-gap gradient so the agent always has something to climb.
        # --------------------------------------------------------------
        if ber <= self.target_ber:
            comms_reward = data_rate + 5.0 - (self.lam * max(0.0, power_dbm))
        else:
            snr_gap = np.clip(self.snr_db - self.target_snr, -20.0, 20.0)
            comms_reward = snr_gap - (self.lam * max(0.0, power_dbm))

        # --------------------------------------------------------------
        # 6. RIS activation cost — small penalty proportional to how many
        #    sub-arrays are active, so unlike Project 1, turning RIS on is
        #    never "free." This is what should make the agent learn to use
        #    RIS mainly when the NLoS flag makes it actually worth the cost.
        # --------------------------------------------------------------
        ris_cost = self.ris_activation_cost_weight * ris_mode

        # --------------------------------------------------------------
        # 7. Satellite budget: drain proportional to offload fraction, plus
        #    a fixed delay penalty (using satellite always costs latency)
        #    and a congestion penalty that ramps up as the budget runs low.
        # --------------------------------------------------------------
        drain = offload_fraction * self.sat_budget_drain_rate
        self.sat_budget_remaining = max(0.0, self.sat_budget_remaining - drain)

        sat_delay_penalty = self.sat_fixed_delay_penalty * offload_fraction

        low_budget_threshold = 0.2 * self.sat_budget_capacity  # congestion kicks in below 20% remaining
        if self.sat_budget_remaining < low_budget_threshold:
            congestion_ratio = (low_budget_threshold - self.sat_budget_remaining) / low_budget_threshold
            sat_congestion_penalty = self.sat_congestion_penalty_weight * congestion_ratio
        else:
            sat_congestion_penalty = 0.0

        # --------------------------------------------------------------
        # 8. Small movement/energy cost — discourages pointless thrashing
        #    around the sky when staying put (or moving purposefully) would
        #    do just as well. Deliberately tiny relative to comms/sat terms.
        # --------------------------------------------------------------
        energy_cost = self.energy_cost_weight * (abs(dx) + abs(dy) + abs(dz))

        # --------------------------------------------------------------
        # 9. Total reward — all terms combined. Power penalty is already
        #    folded into comms_reward above (lam * power), so it isn't
        #    repeated here.
        # --------------------------------------------------------------
        reward = comms_reward - ris_cost - sat_delay_penalty - sat_congestion_penalty - energy_cost

        # --------------------------------------------------------------
        # 10. Build next state and end-of-episode flags.
        # --------------------------------------------------------------
        next_state = self._get_normalized_state()
        terminated = False
        truncated = self.current_step >= self.max_steps

        # info dict carries useful diagnostics for evaluation/plotting later,
        # without cluttering the reward or the state vector itself.
        info = {
            'power_dbm': power_dbm,
            'modulation': chosen_modulation,
            'ber': ber,
            'ris_mode': ris_mode,
            'offload_fraction': offload_fraction,
            'sat_budget_remaining': self.sat_budget_remaining,
            'is_nlos': self.is_nlos,
        }

        return next_state, reward, terminated, truncated, info

    # ==========================================
# 2. PPO ACTOR-CRITIC NETWORK
# ==========================================

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO. Unlike DQN's single QNetwork, PPO needs
    two outputs:
      - Actor: outputs the MEAN of a Gaussian distribution for each of the
        6 continuous action dimensions. Actions are SAMPLED from this
        distribution during training (mean + noise scaled by log_std), not
        just read off directly - this sampling IS the exploration mechanism,
        replacing DQN's epsilon-greedy.
      - Critic: outputs a single number estimating how good the current
        state is (the value function), used later to compute advantages.

    The actor and critic each have their OWN independent hidden layers
    (no shared trunk). This matters because the critic's loss can be large
    and poorly scaled (raw episode rewards here range roughly from -2000 to
    +600), and if the actor and critic shared layers, the critic's large
    gradients would push around the same features the actor relies on to
    understand the state. Separate trunks mean a badly-scaled critic loss
    can only ever affect the critic's own weights. This is standard
    practice for continuous-control PPO.

    A learnable log_std parameter (one per action dimension, not
    state-dependent) controls how "spread out"/exploratory the policy
    currently is. It typically starts wide and narrows as training
    progresses - this is PPO's equivalent of DQN's epsilon decay, except
    it's learned rather than manually scheduled.

    log_std is clamped to a fixed numeric range every time it's used (see
    LOG_STD_MIN/MAX below) instead of being allowed to grow without bound.
    An unclamped log_std can in principle drift arbitrarily high, which is
    mathematically valid but physically meaningless for an action space
    living in [-1,1]. The clamp is a hard safety rail: even if the entropy
    bonus keeps pushing log_std upward, the actual std used for
    sampling/log-probs can never exceed a sane ceiling.
    """

    # std = exp(log_std), so these correspond to std in roughly
    # [exp(-3), exp(1)] ~= [0.05, 2.7] - wide enough to explore the full
    # [-1,1] action range early on, but bounded to a sane ceiling.
    LOG_STD_MIN = -3.0
    LOG_STD_MAX = 1.0

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Actor's own independent trunk - not shared with the critic.
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # Actor head: outputs the mean action for each of the 6 dimensions.
        # No activation on the final layer since actions are later clipped to
        # [-1,1] at the sampling stage, not forced by the network architecture.
        self.actor_mean = nn.Linear(64, action_dim)

        # Learnable log standard deviation - starts at 0.0 (std = 1.0), a
        # deliberately wide/exploratory starting point, one value per action
        # dimension. This is a free-standing parameter, not derived from the
        # state, which is standard practice for continuous-action PPO.
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic's own independent trunk - a completely separate set of
        # weights from the actor's, so a big critic loss can only ever
        # update the critic's own layers.
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # Critic head: outputs ONE number (state value estimate).
        self.critic_head = nn.Linear(64, 1)

    def forward(self, state):
        action_mean = self.actor_mean(self.actor_net(state))
        state_value = self.critic_head(self.critic_net(state))
        return action_mean, state_value

    def get_clamped_std(self):
        """Central place that applies the log_std clamp before
        exponentiating - used both when sampling/evaluating actions and when
        logging the diagnostic action_std, so there's only one source of truth."""
        clamped_log_std = torch.clamp(self.actor_log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return torch.exp(clamped_log_std)

    def get_action_and_value(self, state, action=None):
        """
        Given a state, returns a sampled action (or evaluates a given one),
        its log-probability under the current policy, the distribution's
        entropy (used for the entropy bonus), and the critic's value estimate.
        Reused both during rollout collection (action=None -> sample fresh)
        and during the PPO update step (action=<the one taken during rollout>
        -> just re-evaluate its log-prob/entropy under the CURRENT policy).

        Actions are bounded to [-1,1] using tanh squashing rather than a hard
        clamp. The reason: once the policy becomes confident and pushes its
        mean close to -1 or +1, a large share of raw Gaussian samples would
        land outside [-1,1] and need clamping. A hard clamp creates a
        mismatch, because the Gaussian's log-probability formula doesn't
        know clamping happened - it just measures "how likely is this exact
        number under a smooth bell curve", which is the wrong question once
        many different raw draws all get squashed into the same clamped
        value. That mismatch gets worse the more confident the policy gets,
        and can destabilize training (KL divergence estimates blow up).

        tanh squashing avoids this: any raw (unbounded) Gaussian sample is
        passed through tanh(), a smooth S-shaped curve that maps any real
        number into (-1, 1) without a hard cutoff. Because it's smooth
        everywhere, the probability math stays well-behaved no matter how
        confident the policy becomes. Using tanh requires one extra
        correction term in the log-probability calculation - a standard,
        well-known adjustment (the same one used in the SAC algorithm),
        included below.
        """
        action_mean, state_value = self.forward(state)
        action_std = self.get_clamped_std()  # log_std -> std, clamped to a sane range

        # Independent Gaussian per action dimension (diagonal covariance) -
        # simplest standard choice for continuous-action PPO. Note: this
        # Gaussian lives in an UNBOUNDED space now (not yet squashed to
        # [-1,1]) - the squashing happens below, after sampling.
        dist = torch.distributions.Normal(action_mean, action_std)

        # A tiny number to avoid ever taking log(0) or dividing by zero below.
        eps = 1e-6

        if action is None:
            # --- Collecting a fresh action during rollout ---
            # 1. Draw a raw sample from the plain, unbounded Gaussian.
            raw_action = dist.sample()
            # 2. Squash it into (-1, 1) using tanh. This is the action that
            #    actually gets sent to the environment.
            action = torch.tanh(raw_action)
        else:
            # --- Re-evaluating a past action during the PPO update step ---
            # `action` here is the SQUASHED value we stored in the rollout
            # buffer (already inside (-1, 1)). To ask "how likely was this
            # under the current policy's raw Gaussian?", we first need to
            # undo the squashing and recover the raw number. tanh's inverse
            # is atanh. We clamp the input slightly inside (-1, 1) first
            # because atanh(-1) or atanh(1) would blow up to infinity.
            safe_action = torch.clamp(action, -1.0 + eps, 1.0 - eps)
            raw_action = torch.atanh(safe_action)

        # Plain Gaussian log-probability of the RAW (pre-squash) number.
        log_prob_raw = dist.log_prob(raw_action)

        # Correction term for having squashed the sample through tanh.
        # (This is the standard SAC-style correction - it accounts for how
        # tanh stretches/compresses probability density near the edges.)
        log_prob_correction = torch.log(1.0 - action.pow(2) + eps)

        # Subtract the correction, then sum across the 6 action dimensions
        # to get one single log-probability number for the whole action.
        log_prob = (log_prob_raw - log_prob_correction).sum(axis=-1)

        # Entropy bonus: we use the plain Gaussian's entropy (before
        # squashing) as a simple, standard stand-in. The exact entropy of a
        # tanh-squashed Gaussian has no clean formula, and this approximation
        # is what's commonly used in practice - it still rewards the policy
        # for staying spread out / exploring.
        entropy = dist.entropy().sum(axis=-1)

        return action, log_prob, entropy, state_value.squeeze(-1)


class RunningMeanStd:
    """
    Tracks a running mean/variance of a stream of numbers using Welford's
    online algorithm (numerically stable, doesn't need to store the whole
    history). This is used to track the running variance of the raw
    rewards across the whole training run, so they can be rescaled into a
    small, consistent range before the critic ever sees them.

    Why this matters: episode rewards in this environment can swing from
    roughly -2000 to +600 depending on the state, so if the critic tries to
    fit those raw values directly, its loss starts very large and produces
    correspondingly large, destabilizing gradients. Dividing rewards by a
    running estimate of their standard deviation keeps the numbers the
    critic has to predict in a small, roughly-consistent range regardless of
    how the raw reward function is scaled - this is the same technique used
    by Stable-Baselines3's VecNormalize(norm_reward=True).

    Simplification worth knowing: a fully rigorous version normalizes by
    the running std of the DISCOUNTED RETURN stream, not the raw per-step
    reward. This normalizes the raw reward directly, which is simpler to
    reason about and is sufficient to keep the critic's loss well-scaled,
    at the cost of being a slightly less precise theoretical match to the
    literature.
    """

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = len(x)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


# ==========================================
# 3. PPO AGENT (ROLLOUT BUFFER + UPDATE LOGIC)
# ==========================================

class PPOAgent:
    """
    PPO agent. Structurally different from Project 1/2's
    DQNAgent in two important ways:
      1. ON-POLICY: collects a fixed-size batch of fresh experience (a
         "rollout") using the CURRENT policy, learns from exactly that batch
         for a few epochs, then discards it and collects new data. No sliding
         replay buffer of old transitions like DQN had.
      2. CLIPPED UPDATES: the core "Proximal" trick - restricts how far the
         updated policy's action probabilities can drift from the policy that
         actually collected the data, preventing one bad batch from wrecking
         a decent policy in a single update.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,                 # standard PPO learning rate (lower than DQN's 1e-3 - PPO updates are more sensitive)
        gamma=0.99,               # discount factor, same meaning as in DQN
        gae_lambda=0.95,          # GAE smoothing parameter (see concept explanation above)
        clip_epsilon=0.2,         # PPO's clipping range - standard value from the original paper
        rollout_steps=2048,       # steps collected per batch before each update
        update_epochs=4,          # reduced from 10 - see note below
        minibatch_size=64,        # rollout batch is split into minibatches of this size per epoch
        entropy_coef=0.01,        # weight of the entropy bonus (encourages continued exploration)
        value_coef=0.5,           # weight of the critic's loss relative to the actor's loss
        max_grad_norm=0.5,        # gradient clipping, same purpose as DQN's clip_grad_norm_
        target_kl=0.02,           # see note below - early-stopping threshold for update()
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.rollout_steps = rollout_steps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # update_epochs and target_kl work together to keep the policy from
        # drifting too far from the data it was collected with. If too many
        # epochs are run over the same batch, or the policy is allowed to
        # change too much per update, a large share of samples end up
        # outside PPO's clip range - meaning the actor is trying to move
        # further from its starting point than the "proximal" trust region
        # is meant to allow. Two settings guard against this together:
        #   1. update_epochs (default 4, not the more aggressive 10 some
        #      PPO implementations use): fewer repeated passes over the SAME
        #      batch of data means less chance to drift far from the policy
        #      that collected it before moving on to fresh data.
        #   2. target_kl: after each epoch, the average KL divergence (a
        #      measure of "how different is the new policy from the old
        #      one") across that epoch's updates is checked. If it exceeds
        #      this threshold, no further epochs run on this batch - a
        #      standard PPO safety valve (used in, e.g., OpenAI's Spinning
        #      Up PPO implementation) that reacts directly to policy drift
        #      instead of waiting for a fixed epoch count.
        self.target_kl = target_kl

        # Tracks a running estimate of reward variance so we can
        # rescale rewards before the critic ever has to fit them - see
        # RunningMeanStd's docstring above for the full reasoning.
        self.reward_rms = RunningMeanStd()

        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Rollout buffer - plain Python lists that accumulate
        # exactly ONE batch of on-policy data, then get wiped after each
        # update() call. This replaces DQN's ReplayBuffer entirely; there is
        # no long-term storage of old experience in PPO.
        self._reset_buffer()

    def _reset_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        """
        Samples an action from the current policy for a single state (used
        during rollout collection). Returns the action, its log-prob, and
        value estimate, which get stored in the rollout buffer alongside it.

        The action returned by get_action_and_value()
        is now ALREADY inside (-1, 1) internally (squashed with tanh before
        its log_prob was computed - see that method's docstring), so no
        separate clamping/squashing step is needed here anymore. Whatever
        method get_action_and_value() uses internally, this function just
        passes it straight through, so this line never needs to change again
        even if the squashing method changes.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_t)

        return (
            action.squeeze(0).numpy(),
            log_prob.item(),
            value.item(),
        )

    def store_transition(self, state, action, log_prob, reward, value, done):
        """Appends one step's worth of rollout data to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def _compute_gae(self, last_value, rewards):
        """
        Generalized Advantage Estimation - walks BACKWARDS
        through the collected rollout, blending immediate rewards with the
        critic's value estimates to produce a smoothed advantage for every
        step. last_value is the critic's estimate of the state AFTER the
        rollout ends (needed to bootstrap the very last step's advantage if
        the episode didn't happen to end exactly at the rollout boundary).

        `rewards` is now passed in explicitly
        (instead of always reading self.rewards directly) so the caller can
        supply the NORMALIZED reward stream instead of the raw one - see
        RunningMeanStd's docstring for why. self.dones/self.values are still
        read directly since only the reward scale needed changing.
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            # TD-error: how surprising was this step's outcome vs the critic's prediction
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            # Blend this step's TD-error with the running smoothed advantage from later steps
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def update(self, last_state):
        """
        The core PPO learning step. Called once the rollout
        buffer has accumulated `rollout_steps` transitions. Runs several
        epochs of minibatch gradient updates over the SAME batch of data,
        then wipes the buffer clean for the next round of collection.
        """
        # Bootstrap value for the state right after the rollout ends
        with torch.no_grad():
            last_state_t = torch.FloatTensor(last_state).unsqueeze(0)
            _, last_value = self.network(last_state_t)
            last_value = last_value.item()

        # Update our running reward-variance
        # estimate with this rollout's RAW rewards, then rescale those raw
        # rewards by the running std before computing GAE. This keeps the
        # numbers the critic has to predict (the `returns` below) in a small,
        # roughly-consistent range across the whole training run, instead of
        # whatever raw scale our reward function happens to produce.
        raw_rewards = np.array(self.rewards, dtype=np.float32)
        self.reward_rms.update(raw_rewards)
        normalized_rewards = raw_rewards / np.sqrt(self.reward_rms.var + 1e-8)

        advantages, returns = self._compute_gae(last_value, normalized_rewards)

        # Normalize advantages - standard PPO trick, keeps gradient scale
        # consistent across batches regardless of the raw reward magnitude.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(self.states))
        actions_t = torch.FloatTensor(np.array(self.actions))
        old_log_probs_t = torch.FloatTensor(self.log_probs)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        batch_size = len(self.states)
        indices = np.arange(batch_size)

        # Accumulators so we can report the AVERAGE
        # of each loss term across every minibatch/epoch in this update() call,
        # instead of only ever seeing the final minibatch's numbers. This lets
        # us plot how these quantities evolve rollout-by-rollout across the
        # whole training run, useful for diagnosing training issues (policy
        # loss/entropy/std trends are invisible from the reward number alone).
        diag_actor_losses = []
        diag_critic_losses = []
        diag_entropies = []
        diag_clip_fractions = []  # fraction of samples where PPO's clip actually activated
        diag_approx_kls = []      # per-minibatch approximate KL divergence from the old policy
        epochs_run = 0            # how many epochs actually completed before any early stop

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)  # reshuffle minibatches every epoch, standard practice
            epoch_approx_kls = []  # collected across this epoch's minibatches only

            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                # Re-evaluate the SAME actions under the CURRENT (being-updated)
                # policy, to compare against the log-probs recorded when they
                # were originally taken by the OLD policy during collection.
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    states_t[mb_idx], actions_t[mb_idx]
                )

                # Ratio of new policy's probability to old policy's probability
                # for the same action - this is the heart of PPO's clipping trick.
                log_ratio = new_log_probs - old_log_probs_t[mb_idx]
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence between the old policy
                # (that collected this data) and the current, being-updated
                # policy. This is the standard low-variance "k3" estimator
                # (see http://joschu.net/blog/kl-approx.html, also used in
                # CleanRL's PPO implementation) - computed under no_grad since
                # it's purely a diagnostic/stopping signal, not part of the loss.
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                epoch_approx_kls.append(approx_kl.item())

                # Unclipped and clipped versions of the objective - PPO takes
                # whichever is WORSE (more pessimistic), which is what prevents
                # the update from moving too far in one step.
                surr1 = ratio * advantages_t[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t[mb_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss - simple mean-squared error against the GAE returns,
                # same idea as DQN's MSE loss but against returns instead of Q-targets.
                critic_loss = nn.MSELoss()(new_values, returns_t[mb_idx])

                # Entropy bonus - SUBTRACTED from the loss (i.e. added to the
                # objective) so the optimizer is rewarded for keeping some
                # exploration alive, fighting premature convergence.
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Record this minibatch's numbers.
                # .item() detaches from the graph, so this is cheap/safe to do
                # every minibatch without affecting training itself at all.
                diag_actor_losses.append(actor_loss.item())
                diag_critic_losses.append(critic_loss.item())
                diag_entropies.append(entropy.mean().item())
                diag_approx_kls.append(approx_kl.item())
                # clip_fraction: how often |ratio - 1| exceeded clip_epsilon,
                # i.e. how often the clipping actually had to kick in. High
                # values (e.g. >0.3-0.5) mean the policy is trying to move
                # further per update than PPO's "proximal" trust region
                # comfortably allows - a classic sign of instability.
                clipped = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
                diag_clip_fractions.append(clipped.item())

            epochs_run += 1

            # If this epoch's average policy drift already exceeds the
            # threshold, stop running further epochs on this same batch of
            # data entirely - continuing would only push the policy further
            # from the one that actually collected the data, which is what
            # keeps clip_fraction from staying pinned near 1.0.
            if np.mean(epoch_approx_kls) > self.target_kl:
                break

        # Current action std, averaged across the 6
        # action dimensions - this is the single most direct number for
        # catching policy collapse: if this trends toward ~0 over rollouts,
        # the policy has stopped exploring/reacting to the state and is
        # emitting nearly the same action regardless of the input.
        with torch.no_grad():
            # Read the std through the same clamp used during training,
            # so this number always reflects what the policy is ACTUALLY using.
            current_std = self.network.get_clamped_std().mean().item()

        diagnostics = {
            'actor_loss': float(np.mean(diag_actor_losses)),
            'critic_loss': float(np.mean(diag_critic_losses)),
            'entropy': float(np.mean(diag_entropies)),
            'clip_fraction': float(np.mean(diag_clip_fractions)),
            'action_std': current_std,
            # approx_kl and epochs_run let us confirm the early
            # stop is actually engaging (epochs_run < update_epochs) and see
            # how far policy drift got before it did.
            'approx_kl': float(np.mean(diag_approx_kls)),
            'epochs_run': epochs_run,
        }

        # Wipe the buffer clean - this batch of data is now stale (it was
        # collected by the OLD policy, before the updates just performed).
        self._reset_buffer()

        # Return diagnostics so the training loop
        # can print/store them per rollout, instead of these numbers being
        # computed and immediately thrown away as before.
        return diagnostics

        # ==========================================
# 4. BASELINE HEURISTICS (adapted for continuous action space)
# ==========================================

def baseline_random(env):
    """Baseline 1: samples a completely random action from [-1,1]^6."""
    return env.action_space.sample()


def baseline_fixed(env):
    """
    Baseline 2: a constant, sensible action every step - no movement,
    moderate power, RIS off, no satellite offload. Mirrors the "always
    20dBm/QPSK/RIS-off" fixed baseline from Projects 1-2, just expressed
    in the new 6-dim continuous format.
    """
    # [dx, dy, dz, power, ris, offload] all in [-1,1]
    # power=0.0 -> midpoint of power range; ris=-1.0 -> rounds to mode 0; offload=-1.0 -> 0 fraction
    return np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)


def baseline_greedy(env, snr_db):
    """
    Baseline 3: simple SNR-based rule, translated into continuous outputs.
    No movement (greedy heuristics historically don't plan trajectory),
    power/RIS scale with how bad the current SNR is, and satellite offload
    is used as a fallback crutch when the link is poor.
    """
    if snr_db >= 15.0:
        power_action, ris_action, offload_action = -0.5, -1.0, -1.0   # good link -> save power, RIS off, no offload
    elif snr_db >= 5.0:
        power_action, ris_action, offload_action = 0.5, 0.0, -0.5     # medium link -> more power, some RIS, light offload
    else:
        power_action, ris_action, offload_action = 1.0, 1.0, 0.5      # poor link -> max power, RIS on, lean on satellite

    return np.array([0.0, 0.0, 0.0, power_action, ris_action, offload_action], dtype=np.float32)


def baseline_classical_grid(env, grid_size=5):
    """
    Baseline 4: a fine-grid search over [power, ris, offload] combinations
    (movement held at zero, since this baseline evaluates instantaneous
    link quality rather than trajectory planning). This REPLACES the
    exhaustive search from Projects 1-2 - the action space is now
    continuous, so a true exhaustive search is impossible. This is a
    "best of a fine sampling," not a true mathematical optimum - worth
    keeping in mind when writing this up, so it isn't oversold as an
    exact upper bound the way Project 1/2's baseline could honestly claim.

    Same fair-comparison principle as Project 2's frozen-fading bugfix:
    ALL candidates in this grid are judged against the SAME frozen fading
    draw, so the comparison is apples-to-apples.
    """
    best_action = None
    max_reward = -float('inf')

    # Freeze one fading realization so every grid candidate below is judged
    # under identical channel conditions (same principle as Project 2's fix).
    k_linear = 10.0 ** (env.rician_k_factor_db / 10.0)
    los_component = np.sqrt(k_linear / (k_linear + 1.0))
    scatter_component = np.sqrt(1.0 / (k_linear + 1.0)) * (
        np.random.randn() + 1j * np.random.randn()
    ) / np.sqrt(2.0)
    frozen_fading = np.abs(los_component + scatter_component) ** 2

    power_grid = np.linspace(-1.0, 1.0, grid_size)
    ris_grid = np.linspace(-1.0, 1.0, grid_size)
    offload_grid = np.linspace(-1.0, 1.0, grid_size)

    for p_act in power_grid:
        for r_act in ris_grid:
            for o_act in offload_grid:
                power_dbm = env.power_min_dbm + (p_act + 1.0) / 2.0 * (env.power_max_dbm - env.power_min_dbm)
                ris_mode = int(np.clip(round((r_act + 1.0) / 2.0 * env.num_ris_subarrays), 0, env.num_ris_subarrays))
                offload_fraction = np.clip((o_act + 1.0) / 2.0, 0.0, 1.0)

                test_gain_db, _ = env._compute_channel_gain_db(
                    env.user_x, env.user_y, ris_mode=ris_mode, fixed_fading_linear=frozen_fading
                )
                test_snr_db = power_dbm + test_gain_db - env.noise_floor
                modulation, ber = env._select_modulation(test_snr_db)
                data_rate = env._compute_data_rate(modulation)

                if ber <= env.target_ber:
                    comms_reward = data_rate + 5.0 - (env.lam * max(0.0, power_dbm))
                else:
                    snr_gap = np.clip(test_snr_db - env.target_snr, -20.0, 20.0)
                    comms_reward = snr_gap - (env.lam * max(0.0, power_dbm))

                ris_cost = env.ris_activation_cost_weight * ris_mode
                sat_delay_penalty = env.sat_fixed_delay_penalty * offload_fraction
                test_reward = comms_reward - ris_cost - sat_delay_penalty  # congestion/energy skipped: this baseline is instantaneous, not budget-aware

                if test_reward > max_reward:
                    max_reward = test_reward
                    best_action = np.array([0.0, 0.0, 0.0, p_act, r_act, o_act], dtype=np.float32)

    return best_action


# ==========================================
# 4B. NLoS / SHADOW-ZONE EVALUATION HELPERS 
# ==========================================
# The whole point of Project 3 is to see whether the trained agent learns to
# lean on the RIS more when the user is in a "shadow zone" (NLoS = blocked
# direct signal). The original 10-step evaluation above doesn't reliably show
# this, because the user's position is only picked ONCE per episode and never
# moves - so whether that one evaluation episode is LoS or NLoS is basically
# a coin flip decided the moment the episode starts. The functions below give
# us a proper, honest look at NLoS behaviour instead of hoping for lucky
# random placement.

def get_ppo_action(agent, state):
    """
    Small helper that returns the PPO agent's deterministic action for
    one state - i.e. "what would the trained agent actually do here, with no
    random exploration noise". This is the exact same tanh(mean) calculation
    used in the main evaluation loop below; pulled out into its own function
    here so we don't have to copy/paste it into every new evaluation helper.
    """
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action_mean, _ = agent.network(state_t)
        action = torch.tanh(action_mean).squeeze(0).numpy()
    return action


def force_user_position(env, user_x, user_y):
    """
    Resets the environment as normal, then OVERRIDES the randomly
    chosen user position with a specific (user_x, user_y) we pick by hand.
    This lets us deliberately place the user inside a shadow zone (or
    somewhere safely outside all of them) instead of waiting for the random
    reset() to happen to land there on its own.

    After overriding the position, we redo the same two small calculations
    reset() itself does right after picking user_x/user_y: figuring out
    whether this position is NLoS, and computing a rough starting channel
    gain just so the very first state vector has SOMETHING sensible in it
    (env.step() will recompute the real gain properly on the first actual
    step, exactly like it always does).
    """
    state, _ = env.reset()  # normal reset first (sets drone position, sat budget, etc.)

    # Overwrite the random user position with the one we chose on purpose.
    env.user_x = user_x
    env.user_y = user_y

    # Re-check LoS/NLoS status for this new, manually chosen position.
    env.is_nlos = env._check_los_status(env.user_x, env.user_y)

    # Recompute the same rough "placeholder" starting gain that reset() uses,
    # just so the first state vector reflects the new user position too.
    dist_3d = np.sqrt(
        (env.user_x - env.drone_x) ** 2
        + (env.user_y - env.drone_y) ** 2
        + env.drone_z ** 2
    )
    env.chan_gain_db = -20.0 * np.log10(max(dist_3d, 1.0))

    state = env._get_normalized_state()
    return state


def run_forced_episode(env, agent, user_x, user_y, label):
    """
    Runs one FULL episode (up to env.max_steps steps) with the user
    manually pinned at (user_x, user_y), using the trained PPO agent's
    deterministic policy the whole way through. Prints a short summary at
    the end: total reward, and - most importantly for this project - the
    AVERAGE RIS activation level the agent chose during the episode. If the
    RIS-in-NLoS idea is working, the NLoS-forced run's average RIS number
    should come out clearly higher than the LoS-forced run's.
    """
    state = force_user_position(env, user_x, user_y)
    is_nlos_this_run = env.is_nlos  # fixed for the whole episode, since the user doesn't move

    total_reward = 0.0
    ris_modes_used = []  # record the actual 0-4 sub-array count chosen each step

    for _ in range(env.max_steps):
        action = get_ppo_action(agent, state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        ris_modes_used.append(info['ris_mode'])
        if terminated or truncated:
            break

    avg_ris_mode = float(np.mean(ris_modes_used))

    print(f"--- Forced test: {label} ---")
    print(f"User pinned at ({user_x:.1f}, {user_y:.1f}) | NLoS: {is_nlos_this_run}")
    print(f"Total reward over episode: {total_reward:.2f}")
    print(f"Average RIS sub-arrays active (0-4 scale): {avg_ris_mode:.2f}\n")

    return total_reward, avg_ris_mode


def run_multi_episode_evaluation(env, agent, num_episodes=30):
    """
    Runs many FULL episodes back-to-back, each with a normal random
    reset() (i.e. NOT manually forced), using the trained PPO agent's
    deterministic policy throughout. For every episode we record whether it
    happened to be NLoS or LoS (decided once at reset, per the environment's
    own design), the total reward, and the average RIS activation level.

    At the end, episodes are split into two groups - "was NLoS" and "was
    LoS" - and we print the AVERAGE reward and AVERAGE RIS usage for each
    group separately. This is the fair, honest way to check the RIS-in-NLoS
    idea: nothing here is hand-picked, it's just many random episodes sorted
    by what they happened to be afterward.
    """
    nlos_rewards, nlos_ris_usage = [], []
    los_rewards, los_ris_usage = [], []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_is_nlos = env.is_nlos  # fixed for the whole episode

        total_reward = 0.0
        ris_modes_used = []

        for _ in range(env.max_steps):
            action = get_ppo_action(agent, state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            ris_modes_used.append(info['ris_mode'])
            if terminated or truncated:
                break

        avg_ris_mode = float(np.mean(ris_modes_used))

        if episode_is_nlos:
            nlos_rewards.append(total_reward)
            nlos_ris_usage.append(avg_ris_mode)
        else:
            los_rewards.append(total_reward)
            los_ris_usage.append(avg_ris_mode)

    print(f"--- Multi-episode evaluation ({num_episodes} random episodes) ---")
    print(f"NLoS episodes: {len(nlos_rewards)} | LoS episodes: {len(los_rewards)}\n")

    if nlos_rewards:
        print(f"NLoS group -> Avg reward: {np.mean(nlos_rewards):.2f} | "
              f"Avg RIS sub-arrays active: {np.mean(nlos_ris_usage):.2f}")
    else:
        print("NLoS group -> no episodes happened to land in a shadow zone this run.")

    if los_rewards:
        print(f"LoS group  -> Avg reward: {np.mean(los_rewards):.2f} | "
              f"Avg RIS sub-arrays active: {np.mean(los_ris_usage):.2f}")
    else:
        print("LoS group  -> no episodes happened to land outside all shadow zones this run.")

    print()  # blank line for readability before whatever prints next
    return nlos_rewards, nlos_ris_usage, los_rewards, los_ris_usage


def analyze_ris_contribution(env):
    """
    Answers the real question directly, using the environment's own
    physics formulas: "how much does the RIS-reflected path actually add to
    the total signal, in this project's real geometry?"

    This does NOT need a trained agent and does NOT need any random fading -
    it's a plain, repeatable physics calculation (same formulas env.step()
    uses internally, just computed separately for the direct path and the
    RIS path instead of only looking at their combined total). This is what
    lets us PROVE the "RIS barely helps here" finding from the code itself,
    instead of just eyeballing training results and guessing why.

    This is the same kind of check Project 1 did in its report (Section 5.4,
    the "double path loss" analysis) - we're doing the same thing here for
    Project 3's moving-drone, shadow-zone geometry.
    """
    c = 3e8

    def direct_path_gain_db(drone_x, drone_y, drone_z, obj_x, obj_y, is_nlos):
        # Same formula as SAGINEnv._compute_channel_gain_db's direct-path
        # section, but with fading fixed at "average" (no random boost/dip)
        # so every scenario below is compared on equal footing.
        d3d = np.sqrt((obj_x - drone_x) ** 2 + (obj_y - drone_y) ** 2 + drone_z ** 2)
        loss_db = (
            20.0 * np.log10(max(d3d, 1.0))
            + 20.0 * np.log10(env.carrier_freq_hz)
            + 20.0 * np.log10((4.0 * np.pi) / c)
        )
        if is_nlos:
            loss_db += env.nlos_extra_attenuation_db
        return -loss_db

    def ris_path_gain_db(drone_x, drone_y, drone_z, obj_x, obj_y, ris_mode):
        # Same formula as SAGINEnv._compute_channel_gain_db's RIS-path
        # section: drone -> RIS (free-space) + RIS -> object (terrestrial),
        # then the array gain from however many sub-arrays are active.
        d_bs_ris = np.sqrt((env.ris_x - drone_x) ** 2 + (env.ris_y - drone_y) ** 2 + drone_z ** 2)
        fspl_bs_ris = (
            20.0 * np.log10(max(d_bs_ris, 1.0))
            + 20.0 * np.log10(env.carrier_freq_hz)
            + 20.0 * np.log10((4.0 * np.pi) / c)
        )
        d_ris_obj = max(np.sqrt((obj_x - env.ris_x) ** 2 + (obj_y - env.ris_y) ** 2), 1.0)
        pl_ris_obj = 38.5 + (10.0 * env.eta_ris * np.log10(d_ris_obj))
        total_loss_db = fspl_bs_ris + pl_ris_obj

        active_elements = ris_mode * env.elements_per_subarray
        array_gain = (active_elements ** 2) if active_elements > 0 else 1.0
        gain_linear = (10.0 ** (-total_loss_db / 10.0)) * array_gain
        return 10.0 * np.log10(gain_linear)

    # A handful of representative (drone position, user position) scenarios,
    # covering: drone far from a shadow-zone user, drone close to a
    # shadow-zone user, and a user safely outside every shadow zone.
    test_points = [
        ("Drone far, user deep in shadow zone (NLoS)", -5000.0, 0.0, 1000.0, -2000.0, 0.0, True),
        ("Drone close, user deep in shadow zone (NLoS)", -2000.0, 0.0, 1000.0, -2000.0, 0.0, True),
        ("User safely outside all shadow zones (LoS)", 0.0, 0.0, 1000.0, 3000.0, 3000.0, False),
    ]

    print("--- RIS Contribution Analysis (does the RIS actually help here?) ---")
    print(f"{'Scenario':45s} | {'Direct only':>12s} | {'RIS adds':>12s}")
    for label, dx, dy, dz, ux, uy, is_nlos in test_points:
        direct_db = direct_path_gain_db(dx, dy, dz, ux, uy, is_nlos)
        ris_full_db = ris_path_gain_db(dx, dy, dz, ux, uy, ris_mode=4)  # RIS fully active, best case
        total_with_ris_db = 10.0 * np.log10(10.0 ** (direct_db / 10.0) + 10.0 ** (ris_full_db / 10.0))
        ris_contribution_db = total_with_ris_db - direct_db  # how much extra the RIS buys us, in dB
        print(f"{label:45s} | {direct_db:9.2f} dB | {ris_contribution_db:9.6f} dB")

    print(
        "\nInterpretation: even with ALL 4 RIS sub-arrays (128 elements) active, "
        "the RIS-reflected path changes the total signal by only a tiny fraction "
        "of a decibel in every scenario above. This is the same 'double path "
        "loss' effect found in Project 1: the RIS sits too far from both the "
        "drone and the user for its reflection to make up for the extra hop's "
        "path loss. Since the reward function charges a small cost for each "
        "active RIS sub-array, it is mathematically correct for the trained "
        "agent to leave the RIS off entirely - it has nothing meaningful to "
        "gain from turning it on.\n"
    )


# ==========================================
# 5. TRAINING LOOP & EVALUATION
# ==========================================

if __name__ == "__main__":
    env = SAGINEnv()
    state_dim = env.observation_space.shape[0]   # 10
    action_dim = env.action_space.shape[0]        # 6

    agent = PPOAgent(state_dim, action_dim)

    # ------------------------------------------------------------
    # RIS PHYSICS CHECK - runs immediately, no training needed
    # ------------------------------------------------------------
    # This answers "does the RIS actually help in this project's geometry?"
    # using pure physics formulas, before we even start training. It doesn't
    # depend on the trained agent at all, so it's safe to run every time.
    analyze_ris_contribution(env)

    # ------------------------------------------------------------
    # LOAD A PREVIOUSLY SAVED MODEL IF ONE EXISTS
    # ------------------------------------------------------------
    # Training takes a long time (400,000 steps). If we've already trained
    # once and saved the result, there's no need to do it again just to run
    # a new evaluation experiment - we can load the saved weights straight
    # into the network and skip the training loop entirely. If no saved file
    # exists yet (e.g. the very first time this script is run), we train
    # normally and save the result at the end for next time.
    model_save_path = "sagin_ppo_trained_model.pt"
    model_already_trained = os.path.exists(model_save_path)

    total_timesteps = 400_000   # total environment steps across all training, not "episodes" like DQN - PPO counts by steps since rollouts don't align to episode boundaries
    plot_rewards = []            # average episode reward, recorded once per rollout for the learning curve

    # One list per diagnostic, appended to once per
    # rollout (i.e. once per agent.update() call), so we end up with a full
    # rollout-by-rollout history of each quantity to plot alongside the reward
    # curve. This is purely observational - none of this feeds back into
    # training, it's just recording what update() already computes internally.
    plot_actor_losses = []
    plot_critic_losses = []
    plot_entropies = []
    plot_clip_fractions = []
    plot_action_stds = []
    plot_approx_kls = []      # 
    plot_epochs_run = []      # confirms whether/how often early-stopping engaged

    if model_already_trained:
        # Skip training entirely - just load the saved brain into the
        # network. All the plot_* lists above stay empty, which the plotting
        # code near the bottom checks for before trying to draw anything.
        print(f"\nFound a previously saved model at '{model_save_path}' - loading it "
              f"instead of retraining from scratch.")
        agent.network.load_state_dict(torch.load(model_save_path))
    else:
        print("=== TRAINING START (PPO) ===")

        state, _ = env.reset()
        episode_reward = 0.0
        episode_rewards_this_rollout = []   # completed episode totals collected during the current rollout, for logging

        timesteps_done = 0
        while timesteps_done < total_timesteps:

            # ------------------------------------------------------------
            # COLLECT ONE ROLLOUT (agent.rollout_steps transitions). Note this
            # does NOT respect episode boundaries - an episode can end partway
            # through a rollout, in which case we simply reset() and keep
            # filling the SAME rollout buffer. This is normal PPO behaviour;
            # unlike DQN's per-episode training loop structure.
            # ------------------------------------------------------------
            for _ in range(agent.rollout_steps):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, log_prob, reward, value, done)

                state = next_state
                episode_reward += reward
                timesteps_done += 1

                if done:
                    episode_rewards_this_rollout.append(episode_reward)
                    episode_reward = 0.0
                    state, _ = env.reset()

            # ------------------------------------------------------------
            # UPDATE: run PPO's clipped-objective gradient steps on the batch
            # just collected, then wipe the buffer for the next rollout.
            # `state` here is wherever the environment happens to be right now
            # (mid-episode or freshly reset) - that's the "last_state" GAE needs
            # to bootstrap the final advantage correctly.
            # ------------------------------------------------------------
            # update() now returns a dict of internal
            # training diagnostics instead of silently discarding them.
            diagnostics = agent.update(last_state=state)

            avg_reward = np.mean(episode_rewards_this_rollout) if episode_rewards_this_rollout else float('nan')
            plot_rewards.append(avg_reward)

            # Store this rollout's diagnostics for the
            # plots generated after training finishes.
            plot_actor_losses.append(diagnostics['actor_loss'])
            plot_critic_losses.append(diagnostics['critic_loss'])
            plot_entropies.append(diagnostics['entropy'])
            plot_clip_fractions.append(diagnostics['clip_fraction'])
            plot_action_stds.append(diagnostics['action_std'])
            plot_approx_kls.append(diagnostics['approx_kl'])      # 
            plot_epochs_run.append(diagnostics['epochs_run'])     # 

            print(f"Timesteps {timesteps_done:7d}/{total_timesteps} | "
                  f"Episodes completed this rollout: {len(episode_rewards_this_rollout):3d} | "
                  f"Avg Episode Reward: {avg_reward:.2f}")
            # Second print line with the internals -
            # kept separate from the line above so the existing reward-only log
            # format Projects 1/2 used is still intact and easy to scan.
            print(f"    [diagnostics] actor_loss: {diagnostics['actor_loss']:.4f} | "
                  f"critic_loss: {diagnostics['critic_loss']:.4f} | "
                  f"entropy: {diagnostics['entropy']:.4f} | "
                  f"clip_fraction: {diagnostics['clip_fraction']:.3f} | "
                  f"action_std: {diagnostics['action_std']:.4f} | "
                  # epochs_run < update_epochs confirms the KL
                  # early-stop actually fired for this rollout's update.
                  f"approx_kl: {diagnostics['approx_kl']:.4f} | "
                  f"epochs_run: {diagnostics['epochs_run']}/{agent.update_epochs}")
            episode_rewards_this_rollout = []

        # ------------------------------------------------------------
        # SAVE THE TRAINED MODEL
        # ------------------------------------------------------------
        # Training takes a long time (400,000 steps), so we save the trained
        # network's weights to a file once training finishes. This means any
        # FUTURE evaluation experiments can just load this file straight away
        # instead of retraining from scratch every time we want to test something
        # new. torch.save/load_state_dict is the standard, simple way to do this.
        torch.save(agent.network.state_dict(), model_save_path)
        print(f"\nTrained model saved to: {model_save_path}")

    # ------------------------------------------------------------
    # EVALUATION: run a fresh episode with the trained (deterministic-ish)
    # policy against the three lightweight baselines + the grid-search
    # baseline, logging step-by-step comparisons like Projects 1-2 did.
    # ------------------------------------------------------------
    print("\n=== EVALUATION RUN: PPO AGENT vs BASELINES ===")
    eval_state, _ = env.reset()
    print(f"User Position: ({env.user_x:.1f}, {env.user_y:.1f}) | NLoS: {env.is_nlos}\n")

    for step in range(1, 11):   # 10-step eval snapshot, long enough to see some trajectory movement
        current_snr = getattr(env, 'snr_db', 0.0)  # snr_db only exists after the first step() call

        # Baselines evaluated on the CURRENT state, for reference (not executed)
        rand_action = baseline_random(env)
        fixed_action = baseline_fixed(env)
        greedy_action = baseline_greedy(env, current_snr)

        start_grid = time.time()
        grid_action = baseline_classical_grid(env)
        grid_time_ms = (time.time() - start_grid) * 1000.0

        # PPO agent actually steps the environment forward
        start_ppo = time.time()
        with torch.no_grad():
            state_t = torch.FloatTensor(eval_state).unsqueeze(0)
            action_mean, _ = agent.network(state_t)   # use the MEAN directly at eval time (no sampling noise) - standard practice for reporting a deterministic policy
            # The network's raw mean is now unbounded (it lives in the
            # same pre-squash space as training), so we pass it through the
            # SAME tanh squashing used during training/collection, instead of
            # a hard clamp, to get the actual action sent to the environment.
            ppo_action = torch.tanh(action_mean).squeeze(0).numpy()
        ppo_time_ms = (time.time() - start_ppo) * 1000.0

        eval_state, reward, terminated, truncated, info = env.step(ppo_action)

        print(f"--- Step {step} | Drone: ({env.drone_x:.0f},{env.drone_y:.0f},{env.drone_z:.0f}) ---")
        print(f"Grid Search    : power={grid_action[3]:.2f} ris={grid_action[4]:.2f} offload={grid_action[5]:.2f} | Time: {grid_time_ms:.3f} ms")
        print(f"PPO Agent      : power={ppo_action[3]:.2f} ris={ppo_action[4]:.2f} offload={ppo_action[5]:.2f} | Time: {ppo_time_ms:.3f} ms")
        print(f"-> Result: SNR={env.snr_db:.1f}dB, Mod={info['modulation']}, BER={info['ber']:.4f}, "
              f"Reward={reward:.2f}, SatBudget={info['sat_budget_remaining']:.1f}, NLoS={info['is_nlos']}\n")

        if terminated or truncated:
            eval_state, _ = env.reset()

    # ------------------------------------------------------------
    # FORCED SHADOW-ZONE CHECK
    # ------------------------------------------------------------
    # Quick, hand-picked sanity check: manually place the user INSIDE one of
    # the shadow zones (NLoS) and run a full episode, then do the same with
    # the user placed safely OUTSIDE every shadow zone (LoS), and compare the
    # average RIS activation level between the two. If the RIS-in-NLoS idea
    # is working, the NLoS run's average should be clearly higher.
    #
    # Zone 1 from __init__ is (x: -3000 to -1000, y: -1500 to 1500), so
    # (-2000, 0) sits comfortably in the middle of it -> guaranteed NLoS.
    # (3000, 3000) sits far outside both zones -> guaranteed LoS.
    print("\n=== FORCED SHADOW-ZONE SANITY CHECK ===")
    run_forced_episode(env, agent, user_x=-2000.0, user_y=0.0, label="User INSIDE shadow zone (NLoS)")
    run_forced_episode(env, agent, user_x=3000.0, user_y=3000.0, label="User OUTSIDE all shadow zones (LoS)")

    # ------------------------------------------------------------
    # PROPER MULTI-EPISODE EVALUATION (LoS vs NLoS, no cherry-picking)
    # ------------------------------------------------------------
    # This is the fair version: many random episodes, sorted afterward into
    # "happened to be NLoS" vs "happened to be LoS" groups, with average
    # reward and average RIS usage reported for each group separately.
    print("\n=== MULTI-EPISODE EVALUATION (RANDOM, UNBIASED) ===")
    run_multi_episode_evaluation(env, agent, num_episodes=30)

    # ------------------------------------------------------------
    # LEARNING CURVE PLOT
    # ------------------------------------------------------------
    # Only makes sense if training actually ran THIS time (plot_rewards
    # stays empty when we loaded a saved model instead of retraining). If
    # you want to see the learning curve again after loading a saved model,
    # just look at the PNG file saved the last time training actually ran.
    if plot_rewards:
        print("\n=== GENERATING LEARNING CURVE PLOT ===")
        plt.figure(figsize=(10, 6))
        plt.plot(plot_rewards, label='PPO Agent (avg episode reward per rollout)', color='blue', linewidth=2)
        plt.title('PPO Learning Curve (SAGIN Scenario)')
        plt.xlabel('Rollout Number')
        plt.ylabel('Average Episode Reward')
        plt.grid(True)
        plt.legend()
        # Save an actual image file, not just a pop-up window, so we
        # have something permanent to keep/upload (a plt.show() window
        # disappears the moment you close it and leaves nothing on disk).
        plt.savefig("learning_curve.png", dpi=150, bbox_inches='tight')
        print("Saved plot to: learning_curve.png")
        plt.show()

        # ------------------------------------------------------------
        # Training internals plot, generated purely
        # for diagnosing the reward-collapse behaviour - not part of the actual
        # RL algorithm, this is only ever read by us, not by the agent.
        # 4 stacked subplots sharing the same x-axis (rollout number) as the
        # reward curve above, so we can visually line up WHEN action_std or
        # entropy starts collapsing against WHEN the reward starts falling.
        # ------------------------------------------------------------
        print("\n=== GENERATING TRAINING DIAGNOSTICS PLOT ===")
        fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)

        axes[0].plot(plot_action_stds, color='purple', linewidth=2)
        axes[0].set_ylabel('Action Std (avg over 6 dims)')
        axes[0].set_title('PPO Training Diagnostics (SAGIN Scenario)')
        axes[0].grid(True)
        # A std collapsing toward ~0 means the policy has stopped exploring and
        # is emitting an almost-fixed action regardless of state - this is the
        # single clearest signature of the "policy collapse" failure mode.

        axes[1].plot(plot_entropies, color='green', linewidth=2)
        axes[1].set_ylabel('Policy Entropy')
        axes[1].grid(True)
        # Entropy is directly tied to action_std for a Gaussian policy, so this
        # should track the plot above closely - included separately since it's
        # the actual quantity the entropy_coef term in the loss is fighting for.

        axes[2].plot(plot_actor_losses, color='red', linewidth=2, label='Actor loss')
        axes[2].plot(plot_critic_losses, color='orange', linewidth=2, label='Critic loss')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True)
        # A critic loss that grows rather than shrinks over training suggests the
        # value function is failing to keep up with a shifting reward landscape -
        # worth checking against the reward curve's decline phase.

        axes[3].plot(plot_clip_fractions, color='brown', linewidth=2)
        axes[3].set_ylabel('Clip Fraction')
        axes[3].grid(True)
        # High clip_fraction (rule of thumb: consistently above ~0.3-0.4) means
        # the policy is repeatedly trying to move further per update than PPO's
        # trust region wants to allow - a common early warning sign that shows
        # up before the reward curve itself visibly breaks down.

        # approx_kl: should now stay bounded much closer to
        # target_kl (0.02) than before, since we stop epochs early once it's
        # exceeded, instead of ploughing on for all 10 (now 4) epochs regardless.
        axes[4].plot(plot_approx_kls, color='teal', linewidth=2)
        axes[4].axhline(agent.target_kl, color='gray', linestyle='--', label=f'target_kl ({agent.target_kl})')
        axes[4].set_ylabel('Approx KL')
        axes[4].legend()
        axes[4].grid(True)

        # epochs_run: if this is consistently below update_epochs
        # (4), it confirms the early-stopping rule is actively engaging rollout
        # after rollout, rather than only theoretically existing in the code.
        axes[5].plot(plot_epochs_run, color='navy', linewidth=2, marker='o', markersize=3)
        axes[5].axhline(agent.update_epochs, color='gray', linestyle='--', label=f'update_epochs cap ({agent.update_epochs})')
        axes[5].set_ylabel('Epochs Run')
        axes[5].set_xlabel('Rollout Number')
        axes[5].legend()
        axes[5].grid(True)

        plt.tight_layout()
        # Same reasoning as above - save a permanent file alongside the
        # pop-up window.
        plt.savefig("training_diagnostics.png", dpi=150, bbox_inches='tight')
        print("Saved plot to: training_diagnostics.png")
        plt.show()
    else:
        # We loaded a previously saved model instead of training this
        # time, so there's no fresh rollout-by-rollout history to plot. This
        # is expected, not an error - the plots from the run that actually
        # did the training are still sitting on disk as PNG files.
        print("\n(No new training happened this run - skipping learning curve / "
              "diagnostics plots. See learning_curve.png and training_diagnostics.png "
              "from the run that originally trained this model.)")