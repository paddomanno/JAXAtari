# DemonAttack Laser and Burst Research

## Wave-by-Wave Burst Mapping

| Wave | Burst behavior | Split demon |
|---|---|---|
| 1 | Sequential single projectile | No |
| 2 | Sequential single projectile | No |
| 3 | 2 columns × 4 rows | No |
| 4 | Tight multi-projectile / 2-column burst | No |
| 5 | Multi-segment burst with paired/two-column portion | Yes |
| 6 | Predominantly vertical single projectile | Yes |
| 7 | Sequential vertical projectile | No |
| 8 | Paired/parallel missiles | No |
| 9 | Multi-projectile / 2-column burst | Yes |
| 10 | Vertically stacked multi-segment burst | Yes |
| 11 | Sequential multi-segment burst | No |
| 12 | Vertical multi-segment burst | No |

## Reference

The original Atari 2600 DemonAttack game was used as the reference
for the JAXAtari implementation.

The reference game was run through ALE/OCAtari using:

`scripts/get_objects_patches.py -g DemonAttack`

ALE reported:

`DemonAttack-v4`

The research used the original game to inspect enemy missile
appearance, length, spacing, timing and burst formation.

## Laser / Enemy Missile Types

| Type | Visual characteristic | Description |
|---|---|---|
| Type 1 | Tiny single | Small square/very short projectile |
| Type 2 | Short single | Short vertical projectile |
| Type 3 | Long single | Tall vertical projectile |
| Type 4 | Wide 2-pixel | Wider projectile consisting of two adjacent pixels |

## Burst Types

### Tight 2-row burst

The tight two-row burst pattern was observed during the ALE
investigation and implemented in JAXAtari.

The formation consists of multiple projectile segments arranged in
two closely spaced horizontal columns.

### Long 1-row burst

The long burst uses a single centered horizontal position and a
vertically extended projectile representation.

This corresponds to the long single-column burst implemented in
JAXAtari.

------

## Wave 1

### Observation

- Laser/projectile type: sequential single projectile
- Individual missile size: 1 × 4 px
- Observed missile position: x=37, y=123
- Vertical movement: 8 px per observed position change
- Burst behavior: multiple short missile segments observed across successive frames
- Timing: a multi-frame gap was observed before the next burst
- Next burst observed at approximately x=102, y=123
- Projectile count: 1 missile observed in the recorded object output at a time
- Horizontal spacing: not applicable to a single observed projectile
- Split demon: no split observed in Wave 1
- Evidence: Wave 1 screenshot + ALE object-patch terminal output

------

## Wave 2

### Observation 1

- Laser/projectile type: sequential single projectile
- Individual missile size: 1 × 4 px
- First observed position: x=65, y=179
- Subsequent burst positions: x=40, y=115; x=37, y=115; x=37, y=123; x=38, y=131
- Vertical movement: 8 px per observed position change
- Burst behavior: multiple short missile segments observed across successive frames
- Timing: an initial missile was followed by a multi-frame gap before the next burst
- Projectile count: 1 missile observed in the object output at a time
- Horizontal spacing: not applicable to a single observed projectile
- Evidence: Wave 2 screenshot + ALE object-patch terminal output

### Observation 2

- Laser/projectile type: sequential single projectile
- Individual missile size: 1 × 4 px
- Spawn position: x=42, y=115
- X positions during travel: 42 → 39 → 40 → 41 → 40 → 39 → 40 → 39 → 40
- Y positions: 115 → 123 → 131 → 139 → 147 → 155 → 163 → 171 → 179
- Vertical spacing: 8 px
- Travel distance: 64 px
- Burst-to-burst gap: confirmed
- Projectile count: 1 missile observed during the recorded trajectory
- Horizontal spacing: not applicable to a single observed projectile
- Evidence: terminal object-patch output + screenshot

------

## Wave 3

### Observation

- Laser/projectile type: short single projectile
- Individual missile size: 1 × 4 px
- Active RAM rows: 4
- Projectiles per active row: 2
- Observed active projectile objects: 8
- Horizontal spacing: 7 px
- Vertical spacing: 8 px
- Burst pattern: 2 columns × 4 rows
- Timing/order: progressive activation/deactivation of projectile rows observed
- Split demon: no split observed in Wave 3
- Evidence: ALE screenshot + Projectile RAM 37-46 + OCAtari bitfield decoding

------

## Wave 4

### Observation 1

- Laser/projectile type: 1 × 4 px projectile
- Individual missile size: 1 × 4 px
- Projectile representation: bitfield-based
- Burst pattern: two horizontal columns
- Horizontal spacing: 7 px
- Vertical spacing: 8 px
- Initial RAM state:
  [0, 0, 129, 129, 129, 129, 0, 0, 0, 0]
- Later RAM state:
  [129, 129, 129, 0, 0, 0, 0, 0, 0, 0]
- Active rows: change progressively over time
- Timing/order: progressive change in active rows observed
- Evidence: Wave 4 screenshots + ALE object-patch output

### Observation 2

- Missiles visible: 3
- Missile dimensions: 1 × 4 px
- Pattern: tight multi-missile burst
- Horizontal spacing: 7 px
- Vertical spacing: 8 px
- RAM evidence:
  [0, 0, 0, 0, 34, 65, 33, 16, 0, 0]
- Visual evidence: 3 simultaneous EnemyMissiles

### Observation 3

- Pattern: tight 2-column burst
- Visual structure: two parallel missile columns
- Missile dimensions: 1 × 4 px
- Multiple projectile segments visible simultaneously: confirmed
- Horizontal spacing: 7 px
- Vertical spacing: 8 px
- Timing/order: progressive burst firing observed
- Evidence: game screenshot + Projectile RAM/object logging

### Split-demon observation

- Split demon: not observed in Wave 4

------

## Wave 5

### Laser observation

- Laser/projectile type: multi-segment burst with paired/two-column portion
- Individual missile size: 1 × 4 px
- Multiple missile segments visible simultaneously: confirmed
- Visual pattern: multiple vertically stacked missile segments with a paired/two-column portion visible in the lower part of the burst
- Missile movement: predominantly downward
- Vertical movement: approximately 8 px per logged position change
- Logged EnemyMissile positions: x=74, y=123 followed later by x=71, y=123
- Projectile RAM state at burst start:
  [0, 0, 0, 0, 0, 0, 0, 80, 0, 0]
- Later Projectile RAM state:
  [0, 0, 0, 0, 0, 0, 40, 128, 0, 0]
- Burst behavior: projectile RAM changes progressively while the missile segments remain active
- Horizontal behavior: paired/two-column portion is visible in the lower part of the burst
- Exact projectile count: not determined directly from the available object output
- Exact horizontal spacing: not measured from the recorded object output
- Exact burst duration: not measured as a frame count
- Evidence: Wave 5 ALE screenshot + object-patch terminal recording

### Split-demon observation

- Large demon hit: confirmed
- Result: splits into 2 smaller demons
- Split behavior: confirmed in Wave 5
- Evidence: Wave 5 ALE screenshot + object-patch terminal recording

------

## Wave 6

### Laser observation

- Laser/projectile type: predominantly vertical single projectile
- Individual missile size: 1 × 4 px
- Vertical movement: 8 px per observed position change
- Trajectory: predominantly vertical
- Horizontal movement: slight X variation observed during travel
- Burst pattern: predominantly vertical single projectile
- Projectile count: 1 missile observed at a time in the recorded object output
- Horizontal spacing: not applicable to a single observed projectile
- Timing/order: missile lifecycle observed during the recorded burst
- Evidence: Wave 6 ALE screenshots + EnemyMissile object output + Projectile RAM output

### Split-demon observation

- Large demon hit: confirmed
- Result: splits into 2 smaller demons
- Small demon size: 8 × 7 px
- Both small demons remain active: confirmed
- One small demon attacks/shoots at a time: confirmed by ALE observation
- After one small demon is killed, the remaining small demon can follow/chase the player: confirmed by ALE observation
- Player contact with the remaining small demon is lethal: confirmed by ALE observation
- Evidence: Wave 6 ALE screenshots + object-patch output

------

### Wave 7

### Laser observation

- Laser/projectile type: sequential vertical projectile
- Individual missile size: 1 × 4 px
- Full observed trajectory:
  y=115 → 123 → 131 → 139 → 147 → 155 → 163 → 171 → 179
- Vertical movement: 8 px per observed position
- Horizontal positions observed: approximately x=66–71 px during the recorded cycle
- Horizontal movement: slight X variation during travel
- Trajectory: predominantly vertical
- Projectile count: 1 missile observed during the recorded trajectory
- Burst pattern: sequential single projectile
- Timing/order: complete missile lifecycle recorded from y=115 to y=179
- Split demon: not observed in Wave 7
- Large demon remains a single large demon after being hit: confirmed
- Evidence: complete Wave 7 ALE terminal recording + screenshots

------

## Wave 8

### Laser observation

- Laser/projectile type: paired/parallel missiles
- Individual missile size: 1 × 4 px
- Two EnemyMissiles visible simultaneously: confirmed
- Visual appearance: long vertical missiles
- The two visible missiles are horizontally separated
- Both missiles move downward together
- Observed logged missile X position: approximately x=88
- Observed Y positions: 123 → 131 → 139 → 147 → 155 → 163 → 171 → 179
- Vertical movement: approximately 8 px per position
- Projectile count: at least 2 missiles visible simultaneously
- Horizontal spacing: confirmed as a paired/two-column formation; exact pixel spacing was not extracted from the available object output
- Timing/order: both visible missiles move downward together; exact frame interval was not extracted
- Burst pattern: paired/parallel missiles
- Evidence: Wave 8 ALE screenshots + object-patch terminal output + Projectile RAM output

### Projectile RAM Evidence

Observed RAM states:

```text
Projectile RAM 37-46:
[0, 0, 0, 0, 129, 129, 0, 129, 0, 0]

Projectile RAM 37-46:
[0, 129, 129, 0, 129, 0, 0, 0, 0, 0]

Projectile RAM 37-46:
[0, 129, 0, 0, 0, 0, 0, 0, 0, 0]
```

------

## Wave 9

### Laser observation

- Laser/projectile type: multi-projectile / 2-column burst
- Individual missile size: 1 × 4 px
- Multiple missiles visible simultaneously: confirmed
- Burst pattern: multi-projectile / paired pattern observed
- Two-column arrangement: observed
- Vertical movement: approximately 8 px per position
- Horizontal movement: missile X position changes during travel
- Horizontal spacing: 7 px between the two projectile columns
- Projectile count: 6 active projectile segments at the initial recorded burst state
- Timing/order: progressive reduction of active projectile segments observed during the burst
- Evidence: Wave 9 ALE screenshots + object-patch terminal output

### Split-demon observation

- Large demon hit: confirmed
- Result: splits into 2 smaller demons
- Split behavior: confirmed in Wave 9
- Evidence: Wave 9 ALE observation + screenshot

### Projectile RAM Evidence

Observed RAM states during the Wave 9 burst:

```text
[0, 0, 32, 64, 64, 128, 0, 0, 0, 0]
[0, 64, 128, 32, 64, 0, 0, 0, 0, 0]
[128, 128, 64, 128, 0, 0, 0, 0, 0, 0]
[128, 128, 64, 0, 0, 0, 0, 0, 0, 0]
[128, 128, 0, 0, 0, 0, 0, 0, 0, 0]
[128, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```
------


## Wave 10

### Laser observation

- Laser/projectile type: vertically stacked multi-segment burst
- Individual missile size: 1 × 4 px
- Multiple missile segments visible simultaneously: confirmed
- Visible missile count: 3 in the captured frame
- Burst pattern: vertically stacked/sequential projectile segments
- Initial active RAM positions: 4
- Vertical spacing: 8 px
- Horizontal movement: gradual X movement during travel
- Observed X progression: 102 → 103 → 104 → 106 → 107 → 108 → 109 → 111 → 112 → 113 → 114 → 115 → 116
- Observed Y progression: 139 → 147 → 155 → 163
- Burst progression: active RAM positions progressively decrease until the RAM state becomes zero
- Evidence: Wave 10 ALE screenshot + object-patch terminal output

### Split-demon observation

- Large demon hit: confirmed
- Result: splits into 2 smaller demons
- Split behavior: confirmed in Wave 10
- Evidence: Wave 10 ALE observation + screenshot

### Projectile RAM Evidence

Observed RAM states during the Wave 10 burst:

```text
[128, 128, 64, 128, 0, 0, 0, 0, 0, 0]
[128, 128, 64, 0, 0, 0, 0, 0, 0, 0]
[128, 128, 0, 0, 0, 0, 0, 0, 0, 0]
[128, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```
------

## Wave 11

### Laser observation

- Laser/projectile type: sequential multi-segment burst
- Individual missile size: 1 × 4 px
- Missile bursts: confirmed
- Missile movement: downward in approximately 8 px Y increments
- Horizontal movement: gradual X drift during travel
- First recorded burst: approximately x=95 → 100, y=171 → 179
- First burst terminates when Projectile RAM becomes all zero
- Subsequent burst observed at approximately x=101, y=123
- Projectile RAM activation for subsequent burst:
  [0, 0, 0, 0, 0, 0, 0, 129, 0, 0]
- Exact complete burst count: 3 projectile segments
- Exact horizontal spacing between simultaneous missiles: 0 px; segments form a single vertical column
- Timing/order: recorded from start to end
- Evidence: complete Wave 11 ALE terminal recording + screenshots

### Projectile RAM Evidence

Observed RAM states:

```text
[128, 128, 64, 0, 0, 0, 0, 0, 0, 0]
[128, 128, 0, 0, 0, 0, 0, 0, 0, 0]
[128, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Next burst:
[0, 0, 0, 0, 0, 0, 128, 0, 0, 0]
[0, 0, 0, 0, 0, 128, 128, 0, 0, 0]
```
------

## Wave 12

### Laser observation

- Laser/projectile type: vertical multi-segment burst
- Individual missile size: 1 × 4 px
- Burst pattern: 2 columns × 4 rows at burst start
- Multiple missile segments: confirmed
- Initial Projectile RAM:
  [0, 0, 129, 129, 129, 129, 0, 0, 0, 0]
- Initial active RAM rows: 4
- Initial active projectile segments: 8
- Horizontal spacing: 7 px
- Vertical spacing: 8 px
- Active rows progressively decrease from 4 → 3 → 2 → 1 → 0
- Missile Y progression:
  139 → 147 → 155 → 163 → 171 → 179
- Missile X progression:
  106 → 107 → 108 → 109 → 110 → 111 → 112 → 113 → 114 → 115 → 116 → 117 → 118 → 119 → 120 → 121
- Horizontal movement: gradual rightward drift during travel
- Burst termination: Projectile RAM becomes all zero
- Subsequent burst: confirmed
- Split demon: no split observed in Wave 12
- Evidence: complete Wave 12 ALE terminal recording + screenshots

### Projectile RAM sequence

```text
[0, 0, 129, 129, 129, 129, 0, 0, 0, 0]
[0, 129, 129, 129, 129, 0, 0, 0, 0, 0]
[129, 129, 129, 129, 0, 0, 0, 0, 0, 0]
[129, 129, 129, 0, 0, 0, 0, 0, 0, 0]
[129, 129, 0, 0, 0, 0, 0, 0, 0, 0]
[129, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```


## 40 Original-Game Enemy Missile Samples

The following 40 enemy missile sprite samples were collected from the
original Atari 2600 DemonAttack game through ALE/OCAtari.

These samples were used during the laser/projectile type investigation.

| Sample range | Files |
|---|---|
| 0–39 | `docs/demonattack_sprites/EnemyMissile_0.png` through `EnemyMissile_39.png` |

## ALE Evidence Screenshots

The `docs/demonattack_evidence/` directory contains the 44 screenshots
captured during the ALE/OCAtari investigation of the original Atari
2600 DemonAttack game.

These screenshots provide the visual evidence for the Wave 1–12
projectile, burst, RAM, and split-demon observations documented above.