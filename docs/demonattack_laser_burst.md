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

### Observation 1

- Laser/projectile type: sequential single projectile
- Individual missile size: 1 × 4 px
- Observed missile position: x=37, y=123
- Vertical movement: 8 px per observed position change
- Burst behavior: multiple short missile segments; burst persists across multiple frames
- Timing: a multi-frame gap was observed before the next burst
- Next burst observed at approximately x=102, y=123
- Exact projectile count: not determined from the available object logger
- Exact horizontal spacing: not determined from the available object logger
- Evidence: Wave 1 screenshot + ALE object-patch terminal output

------

## Wave 2

### Observation 1

- Laser/projectile type: sequential single projectile
- Individual missile size: 1 × 4 px
- First observed position: x=65, y=179
- Subsequent burst positions: x=40, y=115; x=37, y=115; x=37, y=123; x=38, y=131
- Vertical movement: 8 px per observed position change
- Burst behavior: multiple short missile segments; burst persists across multiple frames
- Timing: an initial missile was followed by a multi-frame gap before the next burst
- Exact projectile count: not determined from the available object logger
- Exact horizontal spacing: not determined from the available object logger
- Evidence: Wave 2 screenshot + ALE object-patch terminal output

### Observation 2

- Laser/projectile type: sequential single projectile
- Missile size: 1 × 4 px
- Spawn position: x=42, y=115
- X positions during travel: 42, 39, 40, 41, 40, 39, 40, 39, 40
- Y positions: 115 → 123 → 131 → 139 → 147 → 155 → 163 → 171 → 179
- Vertical spacing: 8 px
- Travel distance: 64 px
- Burst-to-burst gap: confirmed
- Simultaneous projectile count: not determined from the available object logger
- Horizontal projectile spacing: not determined from the available object logger
- Evidence: terminal object-patch output + screenshot

------

## Wave 3

### Observation 1

- Laser/projectile type: short single projectile
- Individual missile size: 1 × 4 px
- Active RAM rows: 4
- Projectiles per active row: 2
- Observed active projectile objects: 8
- Horizontal spacing: 7 px
- Vertical spacing: 8 px
- Burst pattern: 2 columns × 4 rows
- Timing/order: progressive row activation/deactivation observed
- Wave mapping: Wave 3
- Evidence: ALE screenshot + Projectile RAM 37-46 + OCAtari bitfield decoding

------

## Wave 4

### Observation 1

- Laser/projectile type: short single projectile
- Individual missile size: 1 × 4 px
- Projectile representation: bitfield-based
- Burst pattern: two horizontal columns
- Horizontal spacing: 7 px
- Vertical spacing: 8 px
- Initial RAM state: [0, 0, 129, 129, 129, 129, 0, 0, 0, 0]
- Later RAM state: [129, 129, 129, 0, 0, 0, 0, 0, 0, 0]
- Active rows: change over time
- Timing/order: progressive change in active rows observed
- Exact burst duration: not measured
- Evidence: Wave 4 screenshots + ALE object-patch output

### Burst #2

- Missiles visible: 3
- Missile dimensions: 1 × 4 px
- Pattern: tight multi-missile burst
- Horizontal spacing: not determined from the available object logger
- Timing: not measured
- RAM evidence: [0, 0, 0, 0, 34, 65, 33, 16, 0, 0]
- Visual evidence: 3 simultaneous EnemyMissiles

### Additional burst observation

- Pattern: tight 2-column burst
- Visual structure: two parallel missile columns
- Missile dimensions: 1 × 4 px
- Multiple projectile segments visible simultaneously: yes
- Horizontal spacing: not determined from the available object logger
- Timing: not measured
- Evidence: game screenshot + projectile RAM/object logging

> **Important:** The current object logger does not expose all
> simultaneously visible EnemyMissile objects, so exact horizontal
> spacing cannot always be derived from the object output alone.

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
- Burst behavior: RAM changes while the missile remains active
- Exact projectile count: not determined from the available object logger
- Exact horizontal spacing: not determined from the available object logger
- Exact burst duration: not measured

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
- Vertical movement observed: 8 px per position
- Trajectory: predominantly vertical
- Horizontal movement: slight X variation observed during travel
- Projectile count: not determined from the available object logger
- Horizontal spacing: not determined from the available object logger
- Timing/order: not measured precisely
- Burst pattern: predominantly vertical single projectile
- Evidence: ALE screenshots + EnemyMissile object output + Projectile RAM output

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

## Wave 7

### Laser observation

- Laser/projectile type: sequential vertical projectile
- Individual missile size: 1 × 4 px
- Full observed trajectory: y=115 → 123 → 131 → 139 → 147 → 155 → 163 → 171 → 179
- Vertical movement: 8 px per position
- Horizontal positions observed: approximately x=66–71 px during the recorded cycle
- Trajectory: predominantly vertical with slight horizontal variation
- Projectile count: not determined from the available object logger
- Horizontal spacing: not determined from the available object logger
- Timing/order: full missile lifecycle recorded; exact timing not measured
- Burst pattern: sequential vertical projectile
- Evidence: complete Wave 7 ALE terminal recording + screenshots

### Split-demon observation

- Split after being shot: not observed in Wave 7
- Large demon remains a single large demon after being hit: confirmed by ALE observation

------

## Wave 8

### Laser observation

- Laser/projectile type: paired/parallel missiles
- Two EnemyMissiles can be visible simultaneously: confirmed visually
- Individual logged missile size: 1 × 4 px
- Visual appearance: long vertical missiles
- The two visible missiles are horizontally separated
- Both missiles move downward together
- Observed logged missile X position: approximately x=88
- Observed Y positions: 123 → 131 → 139 → 147 → 155 → 163 → 171 → 179
- Vertical movement: approximately 8 px per position
- Projectile count: at least 2 visually simultaneous
- Horizontal spacing: not determined precisely from the available object logger
- Timing/order: not measured precisely
- Burst pattern: paired/parallel missiles observed
- Evidence: Wave 8 ALE screenshots + object-patch terminal output + Projectile RAM output

### Projectile RAM Evidence

Observed RAM states:

```text
Projectile RAM 37-46:
[0, 0, 0, 0, 129, 129, 0, 129, 0, 0]

Projectile RAM 37-46:
[0, 129, 129, 0, 129, 0, 0, 0, 0, 0]

'''Projectile RAM 37-46:
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
- Horizontal spacing: not determined from the available object logger
- Exact projectile count: not determined from the available object logger
- Timing/order: not measured precisely
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
- Vertical spacing: 8 px
- Horizontal movement: gradual X movement during travel
- Observed X progression: 102 → 103 → 104 → 106 → 107 → 108 → 109 → 111 → 112 → 113 → 114 → 115 → 116
- Observed Y progression: 139 → 147 → 155 → 163
- Projectile count per complete burst: TBD
- Horizontal spacing: TBD
- Timing/order: TBD
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
- Exact complete burst count: TBD
- Exact horizontal spacing between simultaneous missiles: TBD
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
- Burst pattern: vertical multi-segment burst
- Multiple missile segments: confirmed
- Initial Projectile RAM:
  [0, 0, 129, 129, 129, 129, 0, 0, 0, 0]
- Initial active rows: 4
- Active rows progressively decrease during the burst
- Missile Y progression:
  139 → 147 → 155 → 163 → 171 → 179
- Vertical movement: 8 px per position
- Missile X progression:
  106 → 107 → 108 → 109 → 110 → 111 → 112 → 113 → 114 → 115 → 116 → 117 → 118 → 119 → 120 → 121
- Horizontal movement: gradual rightward drift during travel
- Projectile RAM after burst:
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- Burst termination: confirmed when Projectile RAM becomes all zero
- Subsequent burst: confirmed
- Exact burst-to-burst timing: TBD
- Exact horizontal spacing: TBD
- Exact projectile count: TBD
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