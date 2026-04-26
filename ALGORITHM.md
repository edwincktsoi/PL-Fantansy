# FPL Squad Optimiser — Algorithm Design Paper

**Project:** PL-Fantasy  
**Date:** April 2026  
**Files:** `pl.py`, `backend.py`

---

## 1. Overview

This system automates Fantasy Premier League (FPL) squad selection using a multi-signal
points-forecasting model combined with integer linear programming (ILP) to pick the optimal
15-man squad and starting XI under the official FPL rules.

The pipeline has three stages:

```
FPL API  →  Forecast (projected_points per player)  →  ILP Optimiser  →  Squad + XI
```

All data is fetched live from the official FPL REST API
(`https://fantasy.premierleague.com/api/`). No external paid data sources are used.

---

## 2. Data Sources

All inputs are drawn from three FPL API endpoints:

| Endpoint | Used for |
|---|---|
| `bootstrap-static/` | Player list, team strength ratings, GW events, `ep_next` |
| `fixtures/` | Next-GW fixtures, home/away difficulty ratings (1–5) |
| `element-summary/{id}/` | Per-player GW-by-GW history: points, xG, xA, xGC, minutes |

Data is cached in Parquet files (`players.parquet`, `injuries.parquet`, `teams.parquet`)
to avoid redundant API calls within a session.

---

## 3. Player Filtering

Before any modelling, the player pool is filtered down to viable candidates.

### 3.1 Availability Filter

```
chance_of_playing_this_round >= 70%
```

Players with an injury probability above 30% are excluded. The `chance_of_playing_this_round`
field is provided directly by FPL and is `null` for fully fit players; nulls are treated as 100%.

### 3.2 Points History Filter

```
total_points >= 10   (mid-season)
total_points >= 0    (GW1–5, early season)
```

Mid-season, players with fewer than 10 cumulative points are almost certainly perennial
bench-warmers or squad fillers and are not worth the API call overhead. This threshold is
relaxed to zero for the first five gameweeks because no seasonal history exists yet —
keeping it at 10 would incorrectly eliminate every player.

---

## 4. Per-Gameweek Expected Points Model

### 4.1 Motivation

Raw FPL points are highly noisy. A striker who scores 12 points in one GW (goal + assist +
bonus) may have created only 0.3 xG of genuine opportunity. Blending raw points with an
expected-points (xP) estimate derived from underlying stats reduces the impact of lucky
finishes and unlucky blanks.

### 4.2 xP Formula

For each player in each of their last five gameweeks, the following vectorised estimate is
computed:

```
appearance  = 2.0  if minutes >= 60
            = 1.0  if 0 < minutes < 60
            = 0.0  if minutes == 0

attack      = xG × GOAL_POINTS[position] + xA × 3.0

cs_prob     = exp(-xGC)                 # Poisson P(0 goals conceded)
defense     = cs_prob × CS_POINTS[position] - (xGC / 2)   (GKP/DEF only)

xP          = (appearance + attack + defense).clip(lower=0)   if minutes > 0
            = 0                                                 otherwise
```

**Point values by position:**

| Position | Goal points | Clean sheet points |
|---|---|---|
| GKP (1) | 6 | 4 |
| DEF (2) | 6 | 4 |
| MID (3) | 5 | 1 |
| FWD (4) | 4 | 0 |

**Clean sheet probability** uses the Poisson distribution. If a team's expected goals
conceded in a match is `xGC`, the probability of a clean sheet is `P(X=0 | λ=xGC) = e^{-xGC}`.
This is a standard result from sports analytics and correctly handles fractional xGC values,
unlike the binary clean-sheet indicator that raw points use.

**Concession penalty** (`-xGC/2`, capped at −4) reflects the two points deducted per
two goals conceded for GKP/DEF. Dividing by two converts the expected-goals count into
an expected-deduction.

### 4.3 Per-GW Blend

For each of the five historical GWs, the raw points and xP estimate are blended:

```
blended_gw_r = (1 - xg_blend) × raw_pts_r  +  xg_blend × xP_r
```

The parameter `xg_blend` (default 0.35) controls how much weight the xP model gets
versus the actual scoreboard. A value of 0 uses raw points only; a value of 1 uses
pure xP. The default of 0.35 acknowledges that raw points contain useful information
(bonus points, set-piece goals) that the xP model cannot capture.

---

## 5. Decay-Weighted Form Score

### 5.1 Exponential Decay Weights

Recent gameweeks are more predictive of next-GW performance than older ones. Five
fixed weights are applied, most-recent first:

```
FORM_DECAY_WEIGHTS = (0.35, 0.25, 0.20, 0.12, 0.08)   # sums to 1.00
```

These follow a quasi-exponential decay: each weight is roughly 70–75% of the previous
one. The sum is exactly 1.0, so the resulting score is directly comparable in magnitude
to a single-GW point return.

### 5.2 Base Form Score

```
base_form_points = Σ  FORM_DECAY_WEIGHTS[r] × blended_gw_r
                   r=1..5
```

Missing GWs (player had no match, or fewer than five GWs played in the season) are
treated as zero — they neither inflate nor deflate the score.

---

## 6. Position-Aware Fixture Multipliers

### 6.1 Why Position Matters

A single fixture difficulty number does not affect all players equally. A centre-back
playing against a weak attack benefits from a good defensive fixture; a striker benefits
from facing a weak defence. Using the same multiplier for both roles misrepresents the
fixture advantage.

The system computes two separate multipliers per team per GW:

- **Attack multiplier** — used for FWD and MID
- **Defence multiplier** — used for GKP and DEF

### 6.2 Strength Ratio Formula

FPL provides four strength ratings per team: `strength_attack_home`, `strength_attack_away`,
`strength_defence_home`, `strength_defence_away`. For a fixture between home team H and
away team A:

```
h_atk = clip(strength_attack_home[H]  / strength_defence_away[A],  0.70, 1.30)
h_def = clip(strength_defence_home[H] / strength_attack_away[A],   0.70, 1.30)
a_atk = clip(strength_attack_away[A]  / strength_defence_home[H],  0.70, 1.30)
a_def = clip(strength_defence_away[A] / strength_attack_home[H],   0.70, 1.30)
```

A ratio above 1.0 means the team has a strength advantage in that dimension (favourable
fixture); below 1.0 means a disadvantage. The clip to [0.70, 1.30] prevents extreme
outlier fixtures from producing unrealistic projections.

### 6.3 Double Gameweek Handling

For Double Gameweek (DGW) teams, multipliers are **summed** across both fixtures.
A player with two favourable fixtures effectively gets approximately double the
multiplier of a single-game week, which is the correct behaviour — they have two
opportunities to score points.

For Blank Gameweek (BGW) teams, the sum is zero (no fixtures in the list), so
`projected_points` collapses to zero automatically.

### 6.4 Raw Multiplier for Display

A separate `fixture_multiplier` column (based on the 1–5 FPL difficulty scale mapped to
`{1:1.20, 2:1.10, 3:1.00, 4:0.85, 5:0.70}`) is preserved for UI display and BGW
detection (`fixture_multiplier == 0` indicates a blank). This is not used in the
final projection calculation.

---

## 7. Playing Time Factor

### 7.1 Problem with Binary Availability

The FPL `chance_of_playing` field is already used as a hard filter (≥70%). However,
among players who pass that filter, participation levels vary: a rotation risk starter
may average only 60 minutes per game, while a guaranteed starter averages 88.

Using `chance_of_playing / 100` as a multiplier double-counts injury risk and ignores
actual usage patterns.

### 7.2 Minutes-per-Start Model

```
avg_mins_per_start = total_season_minutes / season_starts
                   = 90.0    if starts == 0 but minutes > 0 (rotated-in player)
                   = 0.0     if both are zero

playing_time_factor = clip(avg_mins_per_start / 90.0,  0.0, 1.0)
```

This directly measures how many minutes a player typically plays per appearance,
expressed as a fraction of a full game. A player who averages 72 minutes per start
gets a factor of 0.80. A player who only ever comes on as a substitute averages ~20
minutes and gets ~0.22.

The factor is applied to the form-based projection only:

```
form_proj = base_form_points × playing_time_factor × _pos_fixture_mult
```

---

## 8. Season PPG Baseline

### 8.1 Motivation

Recent form can be misleading over a small sample. A midfielder who scored 20 points
in GW28 after a lucky hat-trick will have an inflated form score that poorly predicts
GW29. Blending in the season average anchors the projection to demonstrated long-run
performance.

### 8.2 PPG Projection

```
ppg_proj = points_per_game × _pos_fixture_mult
```

`points_per_game` is sourced directly from the FPL API and represents the player's
average points across all GWs played in the current season.

---

## 9. Adaptive Early-Season Strategy

### 9.1 Cold-Start Problem

The first 1–5 gameweeks of a new season present a structural data problem:

- `total_points` is 0 for all players at GW1 (the standard filter eliminates everyone)
- `points_per_game` resets at season start and is 0 until GW2
- Form history is sparse: 0, 1, or 2 GWs of actual data
- Decay weights designed for 5 GWs are applied to near-zero data

Without intervention, the model degrades to near-random squad selection for the
opening weeks of the season.

### 9.2 Layer 1 — Relaxed Filter

```python
pts_threshold = 0  if next_gw_id <= 5  else MIN_TOTAL_POINTS (10)
```

For GW1–5, the points-history gate is removed entirely. All available players
pass through to the modelling stage.

### 9.3 Layer 2 — Per-Player Data Availability Count

```python
n_avail[player] = number of GWs where minutes > 0  (0 to 5)
```

This is computed per player after the history fetch. A player signed mid-season at
GW15 may have only 1–2 GWs of data even though it is mid-season; they receive the
same protective treatment as an early-season player.

### 9.4 Layer 3 — Adaptive Baseline Blend

The `baseline_blend` parameter controls how much the season PPG anchors the
projection. Under sparse data, this is scaled up automatically per player:

```
adaptive_blend[player] = clip(
    baseline_blend + (1 - n_avail/5) × (1 - baseline_blend),
    baseline_blend, 1.0
)
```

**Blend values at different data depths (with default baseline_blend = 0.25):**

| GWs available | adaptive_blend | Interpretation |
|---|---|---|
| 0 | 1.00 | 100% season PPG (early season) |
| 1 | 0.81 | 81% PPG, 19% recent form |
| 2 | 0.63 | 63% PPG, 37% recent form |
| 3 | 0.44 | 44% PPG, 56% recent form |
| 4 | 0.31 | 31% PPG, 69% recent form |
| 5 | 0.25 | 25% PPG (normal mid-season mode) |

The blend transitions smoothly — there is no hard cutover — and operates per-player,
so a mid-season transfer target with no current-season data is handled identically to
a GW1 player.

### 9.5 Layer 4 — FPL ep_next Blend

FPL publishes its own next-GW estimate (`ep_next`) for every player. This signal is
already fixture-adjusted and incorporates FPL's proprietary model. During early season
it is the only signal with real predictive content, since it draws on pre-season
assessments, historical season data, and squad role information that the current-season
history model cannot yet produce.

```
ep_weight[player] = clip((3 - n_avail) / 3,  0.0, 1.0)
```

**ep_next weight at different data depths:**

| GWs available | ep_weight | Interpretation |
|---|---|---|
| 0 | 1.00 | Fully driven by FPL's own estimate |
| 1 | 0.67 | 67% ep_next, 33% own model |
| 2 | 0.33 | 33% ep_next, 67% own model |
| ≥ 3 | 0.00 | Own model takes over entirely |

Because `ep_next` is already fixture-adjusted by FPL, it is used **without** applying
the position-aware multiplier a second time.

---

## 10. Final Projection Formula

Bringing all components together, the full projection pipeline per player is:

```
# --- Form signal ---
base_form_points  = Σ FORM_DECAY_WEIGHTS[r] × [(1-xg_blend)×raw_pts_r + xg_blend×xP_r]
form_proj         = base_form_points × playing_time_factor × pos_fixture_mult

# --- Baseline signal ---
ppg_proj          = points_per_game × pos_fixture_mult

# --- Adaptive blend (data-depth aware) ---
model_proj        = (1 - adaptive_blend) × form_proj + adaptive_blend × ppg_proj

# --- Early-season correction ---
projected_points  = (1 - ep_weight) × model_proj + ep_weight × ep_next
```

**Default parameter values:**

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `xg_blend` | 0.35 | [0, 1] | Weight of xG/xA model vs raw points per GW |
| `baseline_blend` | 0.25 | [0, 1] | Floor weight of season PPG vs recent form |
| `adaptive_blend` | dynamic | [baseline_blend, 1.0] | Scales with data scarcity |
| `ep_weight` | dynamic | [0, 1] | Fades from 1 → 0 as GWs 0 → 3 |

---

## 11. Squad Optimisation (Integer Linear Programming)

### 11.1 Problem Formulation

Given the projected points for all eligible players, the system selects the
15-man squad that maximises total projected points subject to the FPL rules.

**Decision variable:**

```
x_i ∈ {0, 1}    for each player i  (1 = selected)
```

**Objective:**

```
Maximise  Σ x_i × projected_points_i
```

**Constraints:**

```
Σ x_i × now_cost_i  ≤  100.0              (budget)
Σ x_i               =  15                 (squad size)
Σ x_i [GKP]         =  2                  (exactly 2 GKPs)
Σ x_i [DEF]         =  5                  (exactly 5 DEFs)
Σ x_i [MID]         =  5                  (exactly 5 MIDs)
Σ x_i [FWD]         =  3                  (exactly 3 FWDs)
Σ x_i [team t]      ≤  3   ∀ team t       (max 3 per club)
x_j                 =  1   ∀ j ∈ force_include
x_k                 =  0   ∀ k ∈ force_exclude
```

The problem is solved using the CBC solver via the PuLP library. For a typical
player pool of 400–500 players, the ILP solves in under 1 second.

### 11.2 Blank Gameweek Handling

Teams with no fixture in the next GW have `pos_fixture_mult = 0`, which sets
`projected_points ≈ 0` for their players. The ILP naturally avoids selecting
BGW players without requiring any additional constraints.

---

## 12. Starting XI Optimisation

After the 15-man squad is selected, a second ILP picks the best 11 starters.

**Decision variable:**

```
s_i ∈ {0, 1}    for each squad player i  (1 = starts)
```

**Objective:**

```
Maximise  Σ s_i × projected_points_i
```

**Constraints:**

```
Σ s_i              =  11
Σ s_i [GKP]        =  1            (exactly 1 GK)
Σ s_i [DEF]        ∈  [3, 5]       (3–5 defenders)
Σ s_i [MID]        ∈  [2, 5]       (2–5 midfielders)
Σ s_i [FWD]        ∈  [1, 3]       (1–3 forwards)
```

The two unselected players become the bench. The highest-projected starter is
assigned Captain (×2 points), the second is assigned Vice-Captain.

---

## 13. Transfer Planner

The transfer planner solves a constrained version of the squad optimisation problem
where the number of player swaps is capped.

**Additional decision variables:**

```
buy_i ∈ {0, 1}    (player i is newly transferred in)
```

**Additional constraints:**

```
buy_i  ≥  x_i - is_current_i      (buy_i = 1 if selected but not in current squad)
buy_i  ≤  x_i
buy_i  ≤  1 - is_current_i
Σ buy_i  ≤  max_transfers
```

The objective remains the same (maximise projected points). The solver finds the
optimal squad achievable within the transfer budget, returning the recommended
transfers in and transfers out.

---

## 14. Fixture Information Pipeline

For each next-GW fixture, the system resolves human-readable opponent and venue
information:

```
next_match = "Chelsea (H)"              (single GW)
next_match = "Chelsea (H) + Arsenal (A)"  (DGW, concatenated)
difficulty = max(diff_1, diff_2)          (DGW takes the harder fixture's rating)
```

Teams absent from the next-GW fixture list (BGW) have no row in the output, which is
the intended signal used to display warnings and suppress projections in the UI.

---

## 15. Configurable Parameters Summary

| Constant / Parameter | Value | Where set |
|---|---|---|
| `BUDGET` | £100.0m | Constant |
| `SQUAD_SIZE` | 15 | Constant |
| `XI_SIZE` | 11 | Constant |
| `MAX_PER_TEAM` | 3 | Constant |
| `MIN_CHANCE_OF_PLAYING` | 70% | Constant |
| `MIN_TOTAL_POINTS` | 10 | Constant (relaxed to 0 at GW ≤ 5) |
| `FORM_DECAY_WEIGHTS` | (0.35, 0.25, 0.20, 0.12, 0.08) | Constant |
| `GOAL_POINTS` | {GKP:6, DEF:6, MID:5, FWD:4} | Constant |
| `CS_POINTS` | {GKP:4, DEF:4, MID:1, FWD:0} | Constant |
| `DIFFICULTY_MULTIPLIER` | {1:1.20, 2:1.10, 3:1.00, 4:0.85, 5:0.70} | Constant |
| `xg_blend` | 0.35 | UI slider (0–1) |
| `baseline_blend` | 0.25 | UI slider (0–1) |
| `max_transfers` | 1 | UI slider (1–3) |

---

## 16. Algorithm Execution Order

```
1.  Fetch bootstrap-static  →  player list, team strengths, events
2.  Fetch fixtures           →  compute pos_fixture_mult per team
3.  Filter players           →  availability + pts threshold
4.  Fetch element-summary    →  5-GW history per player (pts, xG, xA, xGC, mins)
5.  Compute xP per GW        →  _gw_xp_vec() vectorised
6.  Blend xP with raw pts    →  per-GW blended score
7.  Apply decay weights      →  base_form_points
8.  Compute playing_time_factor
9.  Apply pos_fixture_mult   →  form_proj
10. Compute ppg_proj         →  points_per_game × pos_fixture_mult
11. Count n_gw_available     →  per player
12. Compute adaptive_blend   →  scales with data scarcity
13. Compute ep_weight        →  fades out by GW3
14. Final blend              →  projected_points
15. ILP squad selection      →  15-man squad
16. ILP starting XI          →  11 starters + captain
17. (Optional) Transfer ILP  →  recommended transfers
```
