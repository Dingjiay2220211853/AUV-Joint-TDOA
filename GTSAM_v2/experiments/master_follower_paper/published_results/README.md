# Aggregate simulation results

The manuscript compares `D-only` (conventional dual-master TDOA) and `J-RR` (cyclic relay Joint TDOA). The tables also retain the three additional relay orders from the original run. All five methods use the same acoustic transmission count of 200 and the same 1,000 trial seeds.

| File | Contents |
| --- | --- |
| `frozen_design.json` | Experiment parameters, schedule definitions and source hashes |
| `directional_schedule_summary.csv` | Fleet and worst-follower estimates |
| `directional_schedule_agent_metrics.csv` | Per-follower estimates |
| `directional_schedule_rmse_comparisons.csv` | Planar and cross-track RMSE comparisons |
| `directional_schedule_paired_bootstrap.csv` | Paired ratios and bootstrap confidence intervals |
| `directional_schedule_consistency.csv` | NEES and 95% coverage |
| `directional_schedule_audit_summary.json` | Trial count and protocol audit |

Use the commands in the [main README](../../../../README.md) to regenerate the checkpoint, per-trial outputs and plots.
