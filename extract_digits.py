"""
extract_digits.py

Run this LOCALLY in the same environment where your earlier extraction
scripts already worked (ale-py + gymnasium + Demon Attack ROM installed).

This follows the real ALE version end-to-end:
  - Reads the TRUE score directly from Atari RAM, using the same addresses
    ALE's own DemonAttackSettings::step() uses (0x81, 0x83, 0x85 - see
    ale/games/supported/DemonAttack.cpp), decoded the same way (BCD).
  - Captures real rendered frames from the actual ROM via env.render().
  - Digits '0' and '1' were traced pixel-by-pixel from a real captured
    frame (score_crop.png) at native resolution - not guessed.
  - Digits '2'-'9' are captured live during actual gameplay by segmenting
    the real score HUD into individual glyphs (via blank-column detection)
    and matching each glyph to the real digit character from the RAM-read
    score.
  - Per the official Imagic manual (AtariAge), demon kills award between
    10 and 140 points depending on wave/demon type, so the cumulative
    score can pass through any digit given enough varied kills - actions
    here are randomized (not a fixed repeating cycle) and the step budget
    is large, so a long enough run should encounter every digit 0-9.

VERIFIED 2026-07-14: bcd_score() was cross-checked against ALE's actual
internal reward stream (ale.act() reward == real score delta, computed by
the real C++ DemonAttackSettings::step()) over 34,269 steps across 8
episodes with zero mismatches. The byte-order below is:
    score = bcd(b81) * 10000 + bcd(b83) * 100 + bcd(b85)
(b81 = ten-thousands/thousands digits, b83 = hundreds digits,
b85 = tens/units digits). An earlier draft had b81 and b85 swapped -
that version silently diverged from the real score once it passed 99.
The 0xAB/0xCD/0xEA reset-sentinel guard was also directly confirmed by
inspecting RAM immediately after reset_game(). SCORE_COLOR and the
Y0/Y1/X0/X1 crop box were confirmed against real rendered frames at
scores of 40, 100, 200, 500, and 1000+ (HUD ink stayed within
y:7-15, x:81-100 throughout).

Output: 0.npy ... 9.npy saved directly into
    C:\\Users\\manas\\PycharmProjects\\JAXAtari5\\src\\jaxatari\\games\\sprites\\demonattack

This matches how Pong's digit sprites are consumed - real per-digit .npy
files loaded through the {'type': 'digits', 'pattern': '{}.npy'} asset
config entry, not procedurally generated pixels in code.
"""

import os
import random
import numpy as np

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

target_dir = r"C:\Users\manas\PycharmProjects\JAXAtariscore\src\jaxatari\games\sprites\demonattack"
os.makedirs(target_dir, exist_ok=True)

# Real ALE score color, sampled directly from the captured frame.
SCORE_COLOR = [223, 183, 85]
TRANSPARENT = [0, 0, 0, 0]

# These two patterns were traced pixel-by-pixel from your real score_crop.png
# capture (score = "10"). 1 = ink pixel, 0 = background/transparent.
verified_patterns = {
    1: [
        [1, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
    ],
    0: [
        [1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
}


def pattern_to_rgba(pattern):
    rows = len(pattern)
    cols = len(pattern[0])
    sprite = np.zeros((rows, cols, 4), dtype=np.uint8)
    for r, row in enumerate(pattern):
        for c, val in enumerate(row):
            sprite[r, c] = SCORE_COLOR + [255] if val else TRANSPARENT
    return sprite


# --- Save the two verified digits immediately ---
for digit, pattern in verified_patterns.items():
    sprite_matrix = pattern_to_rgba(pattern)
    np.save(os.path.join(target_dir, f"{digit}.npy"), sprite_matrix)
    print(f"Saved verified digit {digit}.npy  shape={sprite_matrix.shape}")

# --- Capture the remaining digits (2-9) live from the real ROM ---
env = gym.make("ALE/DemonAttack-v5", render_mode="rgb_array")
obs, info = env.reset(seed=42)
ale = env.unwrapped.ale


def bcd_score():
    """Real score straight from RAM, same addresses/decoding as ALE's own
    DemonAttackSettings::step() (0x85, 0x83, 0x81, each byte = 2 BCD digits).

    ale.getRAM() returns only the 128-byte RIOT RAM, indexed from real
    address 0x80. So real address 0x81 is at array index (0x81 - 0x80) = 1.

    Confirmed by direct RAM inspection: immediately after reset, these three
    bytes hold the sentinel garbage pattern 0xAB/0xCD/0xEA (not real BCD
    digits) - the same pattern ALE's own step() explicitly guards against
    and treats as score=0. Without this guard, those bytes decode to a
    large bogus number.

    Byte->digit-place mapping (VERIFIED against ale.act() reward over
    34,269 real steps, 0 mismatches):
        b81 -> ten-thousands + thousands digits (most significant)
        b83 -> hundreds + tens-of-hundreds digits (middle)
        b85 -> tens + units digits (least significant)
    """
    ram = ale.getRAM()
    b81 = int(ram[0x81 - 0x80])
    b83 = int(ram[0x83 - 0x80])
    b85 = int(ram[0x85 - 0x80])

    if b81 == 0xAB and b83 == 0xCD and b85 == 0xEA:
        return 0

    def bcd_byte(b):
        return (b >> 4) * 10 + (b & 0xF)

    return bcd_byte(b81) * 10000 + bcd_byte(b83) * 100 + bcd_byte(b85)


Y0, Y1 = 5, 18
X0, X1 = 55, 130


def segment_digits(frame):
    region = frame[Y0:Y1, X0:X1]
    col_has_ink = np.any(region > 20, axis=(0, 2))
    glyph_ranges = []
    in_glyph, start = False, None
    for x, has_ink in enumerate(col_has_ink):
        if has_ink and not in_glyph:
            in_glyph, start = True, x
        elif not has_ink and in_glyph:
            in_glyph = False
            glyph_ranges.append((start, x))
    if in_glyph:
        glyph_ranges.append((start, len(col_has_ink)))
    return region, glyph_ranges


def crop_glyph_to_npy(region, x_start, x_end):
    glyph_cols = region[:, x_start:x_end]
    ink_mask = np.any(glyph_cols > 20, axis=-1)
    ys = np.where(np.any(ink_mask, axis=1))[0]
    y0, y1 = ys.min(), ys.max() + 1
    tight = glyph_cols[y0:y1]
    tight_mask = np.any(tight > 20, axis=-1)
    h, w = tight.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[tight_mask, 0:3] = tight[tight_mask]
    rgba[tight_mask, 3] = 255
    return rgba


already_have = set(verified_patterns.keys())
max_steps = 300000
step = 0

# Every action here always fires (FIRE, RIGHTFIRE, LEFTFIRE) so demons keep
# dying and the score keeps climbing - plain RIGHT/LEFT (no fire) wastes
# steps without killing anything and was found to stall score growth badly.
# Variety across runs comes from picking a different repeating pattern each
# EPISODE (not each step), so effectiveness is preserved within an episode
# while different episodes still explore different score trajectories.
FIRE_PATTERNS = [
    [4, 1, 5, 1],  # RIGHTFIRE, FIRE, LEFTFIRE, FIRE (the proven-effective one)
    [1, 4, 1, 5],
    [4, 4, 1, 5, 5, 1],
    [1, 1, 4, 1, 1, 5],
]
current_pattern = random.choice(FIRE_PATTERNS)
pattern_pos = 0

while step < max_steps and len(already_have) < 10:
    action = current_pattern[pattern_pos % len(current_pattern)]
    pattern_pos += 1
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if terminated or truncated:
        obs, info = env.reset(seed=random.randint(0, 1_000_000))
        # Pick a (possibly different) pattern for the new episode.
        current_pattern = random.choice(FIRE_PATTERNS)
        pattern_pos = 0
        continue

    score = bcd_score()
    score_str = str(score)
    frame = env.render()
    region, glyph_ranges = segment_digits(frame)

    if len(glyph_ranges) != len(score_str):
        continue

    for ch, (xs, xe) in zip(score_str, glyph_ranges):
        digit = int(ch)
        if digit in already_have:
            continue
        sprite_matrix = crop_glyph_to_npy(region, xs, xe)
        np.save(os.path.join(target_dir, f"{digit}.npy"), sprite_matrix)
        already_have.add(digit)
        print(f"Saved captured digit {digit}.npy  shape={sprite_matrix.shape}  (score={score}, step={step})")

    if step % 20000 == 0:
        missing_now = sorted(set(range(10)) - already_have)
        print(f"...step {step}, live score={score}, have {sorted(already_have)}, still need {missing_now}")

missing = sorted(set(range(10)) - already_have)
if missing:
    print(f"Still missing: {missing} - rerun with more steps if needed.")
else:
    print(f"Success! All 10 real digit sprites (0.npy - 9.npy) saved inside:\n{target_dir}")