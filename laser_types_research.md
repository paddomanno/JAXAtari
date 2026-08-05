# Demon Attack — Enemy Missile Types (ALE Research)

Observed across waves 1-16. 40 sprite samples collected using get_objects_patches.py.
Game loops after wave 12, so all unique patterns were captured.

## Type 1: Tiny single
- Very small square projectile
- Appears briefly at start of burst

## Type 2: Short single
- Small square projectile
- Most common in early-mid waves
- Standard burst pattern (BOMB_TYPE_STANDARD in code)

## Type 3: Medium-Long single
- Taller bar projectile
- Appears in later waves
- Long burst pattern (BOMB_TYPE_LONG in code)

## Type 4: Wide 2-pixel burst
- Two pixels side by side
- Part of tight 2-row burst
- Appears across multiple waves

## Conclusion
4 distinct visual laser types identified. Code already implements:
- BOMB_TYPE_STANDARD (covers Types 1 and 2)
- BOMB_TYPE_LONG (covers Type 3)
- Burst firing system (covers Type 4)
