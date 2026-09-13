# SAGIN PPO: RIS-Assisted Drone Communication

A PPO (Proximal Policy Optimization) reinforcement learning agent that
jointly controls a moving drone's 3D trajectory, transmit power, RIS
(Reconfigurable Intelligent Surface) activation level, and satellite
offload fraction, to keep a ground user connected across a mix of
clear-signal (LoS) and blocked-signal (NLoS / "shadow zone") regions.

This is the third in a series of three projects:

1. **DQN + fixed RIS + fixed drone** - found that RIS gives negligible
   benefit at long, clear-signal ranges ("double path loss").
2. **DQN + moving drone + sensing task(ISAC)** - extended the setup with drone
   mobility and an integrated sensing objective.
3. **This project** - moves to continuous control (PPO), a full 3D moving
   drone, mixed LoS/NLoS geometry, and a satellite backhaul option.

## Key finding

The RIS contribution analysis in this code (`analyze_ris_contribution`)
shows that, in this project's geometry, activating the RIS changes the
total signal by only a tiny fraction of a decibel - even with all 128
elements active. The trained agent correctly learns to never activate the
RIS, since the reward function charges a small cost for RIS use and the
physical benefit is negligible. This confirms the same "double path loss"
effect found in the earlier DQN project, now in a moving-drone, mixed
LoS/NLoS setting.

## Running it

```bash
pip install -r requirements.txt
python SAGIN.py
```

If a previously trained model file (`sagin_ppo_trained_model.pt`) is
present in the working directory, it will be loaded instead of retraining.
Otherwise, training runs from scratch (400,000 environment steps, roughly
150-200 PPO rollouts) and the trained model is saved at the end.

The script prints:
- A quick physics-based check of how much the RIS actually contributes to
  signal strength (no training required for this part).
- Training progress and internal PPO diagnostics (entropy, KL divergence,
  clip fraction, etc.), useful for confirming training stability.
- A step-by-step evaluation comparing the trained agent against a
  grid-search baseline.
- A forced test placing the user directly inside vs. outside a shadow
  zone, to directly check RIS usage in each case.
- A 30-episode randomized evaluation, split into LoS vs. NLoS groups.
- Two saved plots: `learning_curve.png` and `training_diagnostics.png`.

## Requirements

See `requirements.txt`. Tested with Python 3.10+.
