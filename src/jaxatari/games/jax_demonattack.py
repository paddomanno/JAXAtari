import os
from functools import partial
from typing import Tuple

import numpy as np
import chex
import jax.lax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

def _create_explosion_sprite(consts: "DemonAttackConstants") -> jnp.ndarray:
    mask = jnp.array([
        [1, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0],
    ], dtype=jnp.uint8)

    color = jnp.array((*consts.PLAYER_COLOR, 255), dtype=jnp.uint8)
    mask_rgba = jnp.where(mask[:, :, None] == 1, color, jnp.zeros(4, dtype=jnp.uint8))
    sprite = jnp.zeros((*consts.PLAYER_SIZE, 4), dtype=jnp.uint8)
    sprite = sprite.at[:].set(mask_rgba)

    return sprite

def _create_projectile_sprite(size: Tuple[int, int], color_rgb: Tuple[int, int, int]) -> jnp.ndarray:
    sprite = np.zeros((*size, 4), dtype=np.uint8)
    sprite[:, :] = (*color_rgb, 255)
    return jnp.array(sprite)

def _create_digit_sprites(consts: "DemonAttackConstants") -> jnp.ndarray:
    digits = np.zeros((10, 8, 8, 4), dtype=np.uint8)
    color = np.array((*consts.SCORE_COLOR, 255), dtype=np.uint8)

    patterns = [
        [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
    ]

    for i, pattern in enumerate(patterns):
        for r, row in enumerate(pattern):
            for c, val in enumerate(row):
                if val:
                    digits[i, r + 1, c + 2] = color

    return jnp.array(digits)

def _get_default_asset_config() -> tuple:
    """
    Does not contain procedural assets. Those are generated in the init of the renderer.
    """
    return (
        {'name': 'background', 'type': 'background', 'file': 'Background.npy'},
        {'name': 'player', 'type': 'single', 'file': 'Player.npy'},
        {'name': 'player_missile', 'type': 'single', 'file': 'PlayerMissile.npy'},
        {'name': 'demon_1', 'type': 'group', 'files': [
            'Enemy_1/Enemy_1.npy',
            'Enemy_1/Enemy_2.npy',
            'Enemy_1/Enemy_3.npy',
            'Enemy_1/Enemy_4.npy',
        ]},
        {'name': 'demon_2', 'type': 'group', 'files': [
            'Enemy_2/Enemy_1.npy',
            'Enemy_2/Enemy_2.npy',
            'Enemy_2/Enemy_3.npy',
            'Enemy_2/Enemy_4.npy',
        ]},
        {'name': 'enemy_spawn_left', 'type': 'group', 'files': [
            'EnemySpawnAnimation/EnemySpawn_left_1.npy',
            'EnemySpawnAnimation/EnemySpawn_left_2.npy',
            'EnemySpawnAnimation/EnemySpawn_left_3.npy',
        ]},
        {'name': 'enemy_spawn_right', 'type': 'group', 'files': [
            'EnemySpawnAnimation/EnemySpawn_right_1.npy',
            'EnemySpawnAnimation/EnemySpawn_right_2.npy',
            'EnemySpawnAnimation/EnemySpawn_right_3.npy',
        ]},
        {'name': 'bunker', 'type': 'single', 'file': 'Bunker.npy'},
    )

def _add_demon_hpos(hpos: chex.Array) -> chex.Array:
    hpos = (hpos + 16) & 255
    return jnp.where(
        (hpos >= 128) & (hpos < 144),
        (hpos - 241) & 255,
        hpos,
    ).astype(jnp.int32)

def _sub_demon_hpos(hpos: chex.Array) -> chex.Array:
    hpos = (hpos - 16) & 255
    return jnp.where(
        (hpos < 128) & (hpos >= 112),
        (hpos + 241) & 255,
        hpos,
    ).astype(jnp.int32)

def _sub_demon_hpos_n(hpos: chex.Array, count: chex.Array, max_count: int) -> chex.Array:
    def body(value, step):
        return jnp.where(step < count, _sub_demon_hpos(value), value), None

    result, _ = jax.lax.scan(body, hpos, jnp.arange(max_count, dtype=jnp.int32))
    return result.astype(jnp.int32)

def _demon_hpos_to_x(
    hpos: chex.Array,
    start_hpos: int,
    start_x: int,
    fallback_hpos: int,
    fallback_x: int,
) -> chex.Array:
    target = hpos & 255

    def sub_body(carry, _):
        current, x, result, found = carry
        match = current == target
        result = jnp.where(match & ~found, x, result)
        return (
            _sub_demon_hpos(current),
            x + 1,
            result,
            found | match,
        ), None

    (_, _, result, found), _ = jax.lax.scan(
        sub_body,
        (
        jnp.full_like(target, start_hpos),
            jnp.full_like(target, start_x),
            jnp.full_like(target, start_x),
            jnp.zeros_like(target, dtype=jnp.bool_),
        ),
        jnp.arange(241, dtype=jnp.int32),
    )

    def add_body(carry, step):
        current, fallback_step = carry
        fallback_step = jnp.where(current == target, step, fallback_step)
        return (_add_demon_hpos(current), fallback_step), None

    (_, fallback_step), _ = jax.lax.scan(
        add_body,
        (jnp.full_like(target, fallback_hpos), jnp.zeros_like(target)),
        jnp.arange(241, dtype=jnp.int32),
    )
    fallback = jnp.where(
        fallback_step <= fallback_x,
        fallback_x - fallback_step,
        fallback_x + 241 - fallback_step,
    )
    return jnp.where(found, result, fallback).astype(jnp.int32)

def _demon_x_to_hpos(x: chex.Array, start_hpos: int, start_x: int) -> chex.Array:
    target = x.astype(jnp.int32)

    def body(carry, _):
        current_hpos, current_x, result = carry
        result = jnp.where(current_x == target, current_hpos, result)
        return (_sub_demon_hpos(current_hpos), current_x + 1, result), None

    (_, _, result), _ = jax.lax.scan(
        body,
        (
            jnp.full_like(target, start_hpos),
            jnp.full_like(target, start_x),
            jnp.full_like(target, start_hpos),
        ),
        jnp.arange(241, dtype=jnp.int32),
    )
    return result.astype(jnp.int32)

class DemonAttackConstants(struct.PyTreeNode):
    # Static Configuration
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=2)
    MAX_DEMONS: int = struct.field(pytree_node=False, default=3)
    LASER_SPEED: int = struct.field(pytree_node=False, default=4)
    BOMB_SPEED: int = struct.field(pytree_node=False, default=2)
    RESPAWN_DELAY: int = struct.field(pytree_node=False, default=30)
    MAX_LIVING_DEMONS: int = struct.field(pytree_node=False, default=3)
    SPAWN_ANIM_FRAMES: int = struct.field(pytree_node=False, default=3)
    SPAWN_ANIM_FRAME_DURATION: int = struct.field(pytree_node=False, default=6)
    SPAWN_ANIM_WIDTH: int = struct.field(pytree_node=False, default=32)
    WAVE_TOTAL_DEMONS: int = struct.field(pytree_node=False, default=8)
    DEMON_TELEPORT_DURATION: int = struct.field(pytree_node=False, default=32)
    DEMON_VERTICAL_MOTION_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(64, 128, 192, 240, 240, 192, 128, 64),
    )
    DEMON_HORIZONTAL_MOTION_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(255, 192, 160, 128, 128, 160, 192, 255),
    )
    DEMON_INITIAL_REGISTER: Tuple[int, int, int] = struct.field(
        pytree_node=False,
        default=(1, 0, 0),
    )
    DEMON_INITIAL_V_POSITION: Tuple[int, int, int] = struct.field(
        pytree_node=False,
        default=(150, 135, 120),
    )
    DEMON_INITIAL_RANDOM: int = struct.field(pytree_node=False, default=234)
    DEMON_INITIAL_TELEPORT: int = struct.field(pytree_node=False, default=2)
    DEMON_INITIAL_TELEPORT_TIMER: int = struct.field(pytree_node=False, default=10)
    DEMON_SPAWN_LEFT_H_POSITION: int = struct.field(pytree_node=False, default=112)
    DEMON_SPAWN_RIGHT_H_POSITION: int = struct.field(pytree_node=False, default=169)
    DEMON_NORMAL_REGISTER: int = struct.field(pytree_node=False, default=144)
    DEMON_SPAWN_REGISTER: int = struct.field(pytree_node=False, default=64)
    DEMON_RIGHT_BOUND_H_POSITION: int = struct.field(pytree_node=False, default=73)
    DEMON_LEFT_BOUND_H_POSITION: int = struct.field(pytree_node=False, default=113)
    DEMON_RIGHT_SIDE_OFFSET: int = struct.field(pytree_node=False, default=8)
    DEMON_V_MIN_DISTANCE: int = struct.field(pytree_node=False, default=12)
    DEMON_V_MIN: int = struct.field(pytree_node=False, default=72)
    DEMON_V_MAX: int = struct.field(pytree_node=False, default=151)

    # completing wave 84 freezes into a blank screen
    MAX_WAVES: int = struct.field(pytree_node=False, default=84)
    FREEZE_AFTER_MAX_WAVES: bool = struct.field(pytree_node=False, default=False)
    BLANK_SCREEN_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0))

    WAVE_SPRITE_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(0, 1, 1, 1, 1, 1)
    )

    WAVE_BOMB_SPEED_TABLE: Tuple[int, ...] = struct.field(pytree_node=False, default=(2, 2, 3, 3, 4, 4))
    WAVE_BOMB_DROP_PROB_TABLE: Tuple[float, ...] = struct.field(
        pytree_node=False,
        default=(0.025, 0.035, 0.045, 0.055, 0.065, 0.08)
    )
    WAVE_LASER_SPEED_TABLE: Tuple[int, ...] = struct.field(pytree_node=False, default=(3, 4, 5, 5, 6, 6))

    # Coordinates & Sizes. Sizes are (height, width).
    PLAYER_Y: int = struct.field(pytree_node=False, default=174)
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(12, 7))
    DEMON_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(9, 18))
    LASER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(4, 1))
    PLAYER_LASER_DEPTH: int = struct.field(pytree_node=False, default=2)
    BOMB_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(4, 1))
    MAX_BUNKERS: int = struct.field(pytree_node=False, default=3)
    BUNKER_X: int = struct.field(pytree_node=False, default=16)
    BUNKER_Y: int = struct.field(pytree_node=False, default=188)
    BUNKER_SPACING: int = struct.field(pytree_node=False, default=7)

    # Boundaries
    BOUNDARY = 16
    PLAYER_MIN_X: int = struct.field(pytree_node=False, default=BOUNDARY)
    PLAYER_MAX_X: int = struct.field(pytree_node=False, default=160 - BOUNDARY - 7) # WIDTH - boundary - player's width
    DEMON_MIN_X: int = struct.field(pytree_node=False, default=16)  # left boundary for demons
    DEMON_MAX_X: int = struct.field(pytree_node=False, default=136) # right boundary for demons
    DEMON_MIN_Y: int = struct.field(pytree_node=False, default=20)  # top boundary for demons
    DEMON_MAX_Y: int = struct.field(pytree_node=False, default=100) # bottom boundary for demons

    # Colors
    PLAYER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(206, 49, 173))
    BOMB_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(251, 135, 140))
    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(194, 169, 53))

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)

class DemonAttackState(struct.PyTreeNode):
    player_x: chex.Array
    laser_x: chex.Array
    laser_y: chex.Array
    laser_active: chex.Array

    demons_x: chex.Array
    demons_y: chex.Array  # Shape: (MAX_DEMONS,)
    demons_alive: chex.Array  # Shape: (MAX_DEMONS,) bool
    demon_left_h_parameters: chex.Array
    demon_left_h_position: chex.Array
    demon_right_h_position: chex.Array
    demon_v_position: chex.Array
    demon_v_parameters: chex.Array
    demon_register: chex.Array
    demon_teleport: chex.Array
    demon_teleport_timer: chex.Array
    appeared_demons: chex.Array
    demon_random: chex.Array

    bomb_x: chex.Array
    bomb_y: chex.Array
    bomb_active: chex.Array

    score: chex.Array
    lives: chex.Array
    player_exploding: chex.Array
    explosion_timer: chex.Array
    wave_number: chex.Array # Actual attack wave: 0, 1, 2, ...
    wave_pattern: chex.Array # Table/difficulty index: 0..5, reused after early waves.
    wave_total: chex.Array
    wave_spawned: chex.Array
    spawn_timer: chex.Array
    spawn_anim_timer: chex.Array
    game_frozen: chex.Array

    step_counter: chex.Array
    key: chex.PRNGKey

class DemonAttackObservation(struct.PyTreeNode):
    player: ObjectObservation
    demons: ObjectObservation
    laser: ObjectObservation
    bomb: ObjectObservation
    score: jnp.ndarray
    lives: jnp.ndarray

class DemonAttackInfo(struct.PyTreeNode):
    time: jnp.ndarray
    wave_number: jnp.ndarray
    wave_pattern: jnp.ndarray

class JaxDemonAttack(JaxEnvironment[DemonAttackState, DemonAttackObservation, DemonAttackInfo, DemonAttackConstants]):
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.FIRE, Action.RIGHT, Action.LEFT, Action.RIGHTFIRE, Action.LEFTFIRE],
        dtype=jnp.int32,
    )

    def __init__(self, consts: DemonAttackConstants = None):
        consts = consts or DemonAttackConstants()
        super().__init__(consts)
        self.renderer = DemonAttackRenderer(self.consts)

    def _wave_pattern(self, wave_number: chex.Array) -> chex.Array:
        # Waves 0..11 are unique; afterwards the game loops over the last four
        # harder difficulty patterns. The true wave counter remains wave_number.
        return jnp.minimum(wave_number, len(self.consts.WAVE_SPRITE_TABLE) - 1).astype(jnp.int32)

    def _wave_level_mod12(self, wave_number: chex.Array) -> chex.Array:
        return jnp.where(wave_number < 12, wave_number, 8 + jnp.mod(wave_number, 4)).astype(jnp.int32)

    def _wave_int_table(self, table: Tuple, wave_pattern: chex.Array) -> chex.Array:
        return jnp.asarray(table, dtype=jnp.int32)[wave_pattern]

    def _wave_float_table(self, table: Tuple, wave_pattern: chex.Array) -> chex.Array:
        return jnp.asarray(table, dtype=jnp.float32)[wave_pattern]

    def _wave_sprite_index(self, wave_pattern: chex.Array) -> chex.Array:
        return self._wave_int_table(self.consts.WAVE_SPRITE_TABLE, wave_pattern)

    def _initial_demon_values(self):
        zeros = jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.int32)
        return dict(
            demons_x=zeros,
            demons_y=zeros,
            demons_alive=jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.bool_),
            demon_left_h_parameters=zeros,
            demon_left_h_position=zeros,
            demon_right_h_position=zeros,
            demon_v_position=jnp.asarray(self.consts.DEMON_INITIAL_V_POSITION, dtype=jnp.int32),
            demon_v_parameters=zeros,
            demon_register=jnp.asarray(self.consts.DEMON_INITIAL_REGISTER, dtype=jnp.int32),
            demon_teleport=jnp.array(self.consts.DEMON_INITIAL_TELEPORT, dtype=jnp.int32),
            demon_teleport_timer=jnp.array(self.consts.DEMON_INITIAL_TELEPORT_TIMER, dtype=jnp.int32),
            appeared_demons=jnp.array(0, dtype=jnp.int32),
            demon_random=jnp.array(self.consts.DEMON_INITIAL_RANDOM, dtype=jnp.int32),
        )

    def _next_demon_random(self, random: chex.Array) -> chex.Array:
        shifted = (random * 2) & 255
        carry = ((shifted ^ random) // 64) & 1
        return (shifted | carry).astype(jnp.int32)

    def _new_demon_v_position(self, state: DemonAttackState, demon: chex.Array) -> chex.Array:
        def first():
            return (jnp.array(self.consts.DEMON_V_MAX, dtype=jnp.int32) + state.demon_v_position[1]) // 2

        def second():
            return (state.demon_v_position[0] + state.demon_v_position[2]) // 2

        def third():
            return (state.demon_v_position[1] + self._new_demon_v_shift(state.wave_number)) // 2

        return jax.lax.switch(demon, (first, second, third)).astype(jnp.int32)

    def _new_demon_v_shift(self, wave_number: chex.Array) -> chex.Array:
        return jnp.array(44, dtype=jnp.int32) - self._wave_level_mod12(wave_number) * 2

    def _sub_demon_h_n(self, hpos: chex.Array, count: int) -> chex.Array:
        return _sub_demon_hpos_n(hpos, jnp.array(count, dtype=jnp.int32), 16)

    def _demon_h_to_x(self, hpos: chex.Array) -> chex.Array:
        return _demon_hpos_to_x(
            hpos,
            self.consts.DEMON_SPAWN_LEFT_H_POSITION,
            self.consts.DEMON_MIN_X - self.consts.DEMON_SIZE[1],
            self.consts.DEMON_RIGHT_BOUND_H_POSITION,
            self.consts.DEMON_MAX_X,
        )

    def _demon_x_to_h(self, x: chex.Array) -> chex.Array:
        return _demon_x_to_hpos(
            x,
            self.consts.DEMON_SPAWN_LEFT_H_POSITION,
            self.consts.DEMON_MIN_X - self.consts.DEMON_SIZE[1],
        )

    def _spawn_target_x(self, ids: chex.Array) -> chex.Array:
        spacing = (self.consts.DEMON_MAX_X - self.consts.DEMON_MIN_X) // (self.consts.MAX_DEMONS + 1)
        return (self.consts.DEMON_MIN_X + (ids + 1) * spacing).astype(jnp.int32)

    def _sync_demons_from_internal_positions(self, state: DemonAttackState) -> DemonAttackState:
        ids = jnp.arange(self.consts.MAX_DEMONS)
        demons_x_from_h = jnp.clip(
            self._demon_h_to_x(state.demon_left_h_position),
            self.consts.DEMON_MIN_X,
            self.consts.DEMON_MAX_X,
        )
        demons_x = jnp.where(state.spawn_anim_timer > 0, self._spawn_target_x(ids), demons_x_from_h)
        demons_y = jnp.clip(
            jnp.array(self.consts.PLAYER_Y + self.consts.PLAYER_LASER_DEPTH, dtype=jnp.int32) - state.demon_v_position,
            self.consts.DEMON_MIN_Y,
            self.consts.DEMON_MAX_Y,
        )
        return state.replace(
            demons_x=demons_x,
            demons_y=demons_y,
            demons_alive=(state.demon_register & 192) != 0,
            wave_spawned=state.appeared_demons,
        )

    def _start_wave(self, state: DemonAttackState, wave_number: chex.Array) -> DemonAttackState:
        wave_pattern = self._wave_pattern(wave_number)
        wave_total = self.consts.WAVE_TOTAL_DEMONS
        zeros = jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.int32)

        state = state.replace(
            wave_number=wave_number,
            wave_pattern=wave_pattern,
            wave_total=wave_total,
            wave_spawned=jnp.array(0, dtype=jnp.int32),
            spawn_timer=jnp.array(0, dtype=jnp.int32),
            spawn_anim_timer=zeros,
            game_frozen=jnp.array(False, dtype=jnp.bool_),
            **self._initial_demon_values(),
            bomb_active=jnp.array(False, dtype=jnp.bool_),
        )
        return self._sync_demons_from_internal_positions(state)

    def _advance_wave(self, state: DemonAttackState) -> DemonAttackState:
        next_wave_number = state.wave_number + 1

        should_freeze = jnp.array(self.consts.FREEZE_AFTER_MAX_WAVES, dtype=jnp.bool_) & (
            next_wave_number >= self.consts.MAX_WAVES
        )

        return jax.lax.cond(
            should_freeze,
            lambda s: s.replace(
                wave_number=next_wave_number,
                demons_alive=jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.bool_),
                bomb_active=jnp.array(False, dtype=jnp.bool_),
                laser_active=jnp.array(False, dtype=jnp.bool_),
                game_frozen=jnp.array(True, dtype=jnp.bool_),
            ),
            lambda s: self._start_wave(s, next_wave_number),
            operand=state,
        )

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[DemonAttackObservation, DemonAttackState]:
        wave_number = jnp.array(0, dtype=jnp.int32)
        wave_pattern = self._wave_pattern(wave_number)
        wave_total = self.consts.WAVE_TOTAL_DEMONS

        state = DemonAttackState(
            player_x=jnp.array(76, dtype=jnp.int32),
            laser_x=jnp.array(0, dtype=jnp.int32),
            laser_y=jnp.array(0, dtype=jnp.int32),
            laser_active=jnp.array(False, dtype=jnp.bool_),
            **self._initial_demon_values(),
            bomb_x=jnp.array(0, dtype=jnp.int32),
            bomb_y=jnp.array(0, dtype=jnp.int32),
            bomb_active=jnp.array(False, dtype=jnp.bool_),
            score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_BUNKERS, dtype=jnp.int32),
            player_exploding=jnp.array(False, dtype=jnp.bool_),
            explosion_timer=jnp.array(0, dtype=jnp.int32),
            wave_number=wave_number,
            wave_pattern=wave_pattern,
            wave_total=wave_total,
            wave_spawned=jnp.array(0, dtype=jnp.int32),
            spawn_timer=jnp.array(0, dtype=jnp.int32),
            spawn_anim_timer=jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.int32),
            game_frozen=jnp.array(False, dtype=jnp.bool_),
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=key,
        )
        state = self._sync_demons_from_internal_positions(state)
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DemonAttackState, action: chex.Array) -> Tuple[
        DemonAttackObservation, DemonAttackState, float, bool, DemonAttackInfo]:
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        prev_state = state

        # Handle Explosion Timer
        def update_explosion(s):
            new_timer = s.explosion_timer - 1
            exploding = new_timer > 0
            # If timer reaches 0, player hit logic should have already reduced lives.
            # We just need to stop exploding.
            return s.replace(explosion_timer=new_timer, player_exploding=exploding)

        def normal_step(s, act):
            s = s.replace(demon_random=self._next_demon_random(s.demon_random))
            # 0. Spawn Animation Step
            s = self._spawn_animation_step(s)
            # 1. Player Step
            s = self._player_step(s, act)
            # 2. Laser Step
            s = self._laser_step(s, act)
            # 3. Demon Step
            s = self._demons_step(s)
            # 4. Bomb Step
            s = self._bomb_step(s)
            # 5. Collision Detection
            s = self._handle_collisions(s)
            return s

        state = jax.lax.cond(
            state.game_frozen,
            lambda s: s,
            lambda s: jax.lax.cond(
                s.player_exploding,
                update_explosion,
                lambda ss: normal_step(ss, atari_action),
                operand=s,
            ),
            operand=state,
        )

        key, next_key = jax.random.split(state.key)
        state = state.replace(key=next_key, step_counter=state.step_counter + 1)

        observation = self._get_observation(state)
        reward = self._get_reward(prev_state, state)
        done = self._get_done(state)
        info = self._get_info(state)
        return observation, state, reward, done, info

    def _player_step(self, state: DemonAttackState, action: chex.Array) -> DemonAttackState:
        move_right = (action == Action.RIGHT) | (action == Action.RIGHTFIRE)
        move_left = (action == Action.LEFT) | (action == Action.LEFTFIRE)

        dx = jax.lax.select(
            move_right,
            self.consts.PLAYER_SPEED,
            jax.lax.select(move_left, -self.consts.PLAYER_SPEED, 0),
        )

        new_x = jnp.clip(state.player_x + dx, self.consts.PLAYER_MIN_X, self.consts.PLAYER_MAX_X)
        return state.replace(player_x=new_x)

    def _spawn_animation_step(self, state: DemonAttackState) -> DemonAttackState:
        return state.replace(
            spawn_anim_timer=jnp.maximum(state.spawn_anim_timer - 1, 0),
        )

    def _laser_step(self, state: DemonAttackState, action: chex.Array) -> DemonAttackState:
        # Fire laser if not active and FIRE action
        fire = (action == Action.FIRE) | (action == Action.RIGHTFIRE) | (action == Action.LEFTFIRE)
        should_fire = fire & ~state.laser_active

        laser_x = jax.lax.select(
            should_fire,
            state.player_x + self.consts.PLAYER_SIZE[1] // 2,
            state.laser_x,
        )
        laser_y = jax.lax.select(
            should_fire,
            jnp.array(self.consts.PLAYER_Y + self.consts.PLAYER_LASER_DEPTH, dtype=jnp.int32),
            state.laser_y,
        )
        laser_active = should_fire | state.laser_active

        # Move laser
        laser_speed = self._wave_int_table(self.consts.WAVE_LASER_SPEED_TABLE, state.wave_pattern)
        laser_y = jax.lax.select(laser_active, laser_y - laser_speed, laser_y)

        # Deactivate if out of bounds
        laser_active = laser_active & (laser_y > 0)

        return state.replace(laser_x=laser_x, laser_y=laser_y, laser_active=laser_active)

    def _demons_step(self, state: DemonAttackState) -> DemonAttackState:
        ids = jnp.arange(self.consts.MAX_DEMONS)
        frame_mod4 = state.step_counter & 3
        selected = jnp.maximum(frame_mod4 - 1, 0)

        target_v = self._new_demon_v_position(state, selected)
        selected_mask = ids == selected
        selected_active = frame_mod4 != 0
        demon_v_position = jnp.where(
            selected_active & selected_mask,
            state.demon_v_position + jnp.where(target_v >= state.demon_v_position[selected], 1, -1),
            state.demon_v_position,
        )
        demon_register = jnp.where(
            selected_active & selected_mask & ((state.demon_random & 7) == 0),
            state.demon_register ^ 16,
            state.demon_register,
        )
        phase_mask = ids == frame_mod4
        demon_register = jnp.where(
            phase_mask,
            (demon_register & 240) | ((demon_register + 1) & 15),
            demon_register,
        )

        timer = jnp.maximum(state.demon_teleport_timer - 1, 0)
        tele_mask = ids == state.demon_teleport
        tele_kind = demon_register[state.demon_teleport] & 192
        can_appear = state.appeared_demons < state.wave_total
        start_spawn = (state.demon_teleport_timer > 0) & (timer == 0) & (tele_kind == 0) & can_appear
        finish_spawn = (state.demon_teleport_timer > 0) & (timer == 0) & (tele_kind != 0)
        can_schedule = (state.demon_teleport_timer == 0) & can_appear
        free = (demon_register & 192) == 0
        scheduled = self.consts.MAX_DEMONS - 1 - jnp.argmax(free[::-1].astype(jnp.int32))
        schedule = can_schedule & jnp.any(free)

        demon_teleport = jnp.where(schedule, scheduled, state.demon_teleport)
        schedule_mask = ids == scheduled
        demon_v_position = jnp.where(
            schedule & schedule_mask,
            self._new_demon_v_position(state.replace(demon_v_position=demon_v_position), scheduled),
            demon_v_position,
        )
        demon_register = jnp.where(start_spawn & tele_mask, demon_register | 64, demon_register)
        demon_register = jnp.where(finish_spawn & tele_mask, self.consts.DEMON_NORMAL_REGISTER, demon_register)

        spawn_target_x = self._spawn_target_x(ids)
        spawn_target_h = self._demon_x_to_h(spawn_target_x)
        demon_left_h_position = jnp.where(
            start_spawn & tele_mask,
            self.consts.DEMON_SPAWN_LEFT_H_POSITION,
            jnp.where(finish_spawn & tele_mask, spawn_target_h, state.demon_left_h_position),
        )
        demon_right_h_position = jnp.where(
            start_spawn & tele_mask,
            self.consts.DEMON_SPAWN_RIGHT_H_POSITION,
            jnp.where(
                finish_spawn & tele_mask,
                self._sub_demon_h_n(spawn_target_h, self.consts.DEMON_RIGHT_SIDE_OFFSET),
                state.demon_right_h_position,
            ),
        )

        spawning = (demon_register & 192) == self.consts.DEMON_SPAWN_REGISTER
        demon_left_h_position = jnp.where(spawning, _sub_demon_hpos(demon_left_h_position), demon_left_h_position)
        demon_right_h_position = jnp.where(spawning, _add_demon_hpos(demon_right_h_position), demon_right_h_position)

        normal = (demon_register & 192) == 128
        phase = demon_register & 7
        v_sum = state.demon_v_parameters + jnp.asarray(self.consts.DEMON_VERTICAL_MOTION_TABLE, dtype=jnp.int32)[phase]
        h_sum = state.demon_left_h_parameters + jnp.asarray(self.consts.DEMON_HORIZONTAL_MOTION_TABLE, dtype=jnp.int32)[phase]
        v_step = normal & (v_sum > 255)
        h_step = normal & (h_sum > 255)

        demon_v_position = jnp.where(
            v_step,
            demon_v_position + jnp.where((demon_register & 8) != 0, 1, -1),
            demon_v_position,
        )
        previous_left_h_position = demon_left_h_position
        demon_left_h_position = jnp.where(
            h_step & ((demon_register & 16) != 0),
            _sub_demon_hpos(demon_left_h_position),
            jnp.where(h_step, _add_demon_hpos(demon_left_h_position), demon_left_h_position),
        )
        moved_x = self._demon_h_to_x(demon_left_h_position)
        outside_x = (moved_x < self.consts.DEMON_MIN_X) | (moved_x > self.consts.DEMON_MAX_X)
        turn = normal & (
            (((demon_register & 16) != 0) & (moved_x >= self.consts.DEMON_MAX_X))
            | (((demon_register & 16) == 0) & (moved_x <= self.consts.DEMON_MIN_X))
        )
        demon_left_h_position = jnp.where(turn & outside_x, previous_left_h_position, demon_left_h_position)
        demon_register = jnp.where(
            turn,
            ((demon_register ^ 16) & 240) | 1,
            demon_register,
        )
        demon_right_h_position = jnp.where(
            normal,
            self._sub_demon_h_n(demon_left_h_position, self.consts.DEMON_RIGHT_SIDE_OFFSET),
            demon_right_h_position,
        )

        top = jnp.clip(demon_v_position[0], self.consts.DEMON_V_MIN, self.consts.DEMON_V_MAX)
        middle = jnp.minimum(demon_v_position[1], top - self.consts.DEMON_V_MIN_DISTANCE)
        bottom = jnp.minimum(demon_v_position[2], middle - self.consts.DEMON_V_MIN_DISTANCE)
        demon_v_position = jnp.stack((
            top,
            middle,
            jnp.maximum(bottom, self._new_demon_v_shift(state.wave_number)),
        )).astype(jnp.int32)

        state = state.replace(
            demon_left_h_parameters=jnp.where(normal, h_sum & 255, state.demon_left_h_parameters),
            demon_left_h_position=demon_left_h_position,
            demon_right_h_position=demon_right_h_position,
            demon_v_position=demon_v_position,
            demon_v_parameters=jnp.where(normal, v_sum & 255, state.demon_v_parameters),
            demon_register=demon_register,
            demon_teleport=demon_teleport,
            demon_teleport_timer=jnp.where(
                start_spawn,
                self.consts.DEMON_TELEPORT_DURATION,
                jnp.where(
                    finish_spawn,
                    0,
                    jnp.where(schedule, (state.demon_random & 31) | 1, timer),
                ),
            ),
            appeared_demons=state.appeared_demons + start_spawn.astype(jnp.int32),
            spawn_anim_timer=jnp.where(
                start_spawn & tele_mask,
                self.consts.DEMON_TELEPORT_DURATION,
                jnp.where(finish_spawn & tele_mask, 0, state.spawn_anim_timer),
            ),
        )
        return self._sync_demons_from_internal_positions(state)

    def _bomb_step(self, state: DemonAttackState) -> DemonAttackState:
        # Drop bomb from a random living demon if no bomb is active
        key, drop_key, demon_idx_key = jax.random.split(state.key, 3)
        can_drop_bomb = jnp.logical_and(state.demons_alive, state.spawn_anim_timer <= 0)

        drop_prob = self._wave_float_table(self.consts.WAVE_BOMB_DROP_PROB_TABLE, state.wave_pattern)
        should_drop = (~state.bomb_active) & jnp.any(can_drop_bomb) & (jax.random.uniform(drop_key) < drop_prob)

        # Pick a random demon index
        demon_idx = jax.random.randint(demon_idx_key, (), 0, self.consts.MAX_DEMONS)
        demon_idx = jnp.where(can_drop_bomb[demon_idx], demon_idx, jnp.argmax(can_drop_bomb))

        bomb_x = jax.lax.select(should_drop, state.demons_x[demon_idx] + self.consts.DEMON_SIZE[1] // 2, state.bomb_x)
        bomb_y = jax.lax.select(should_drop, state.demons_y[demon_idx] + self.consts.DEMON_SIZE[0], state.bomb_y)
        bomb_active = jnp.logical_or(should_drop, state.bomb_active)

        # Move bomb using the current wave speed
        bomb_speed = self._wave_int_table(self.consts.WAVE_BOMB_SPEED_TABLE, state.wave_pattern)
        bomb_y = jax.lax.select(bomb_active, bomb_y + bomb_speed, bomb_y)

        # Deactivate if out of bounds
        bomb_active = jnp.logical_and(bomb_active, bomb_y < self.consts.HEIGHT)

        return state.replace(bomb_x=bomb_x, bomb_y=bomb_y, bomb_active=bomb_active, key=key)

    def _handle_collisions(self, state: DemonAttackState) -> DemonAttackState:
        # Laser vs Demons
        def check_demon_collision(i, carry):
            s_alive, s_score, l_active = carry

            demon_hit = jnp.logical_and(
                s_alive[i],
                jnp.logical_and(
                    l_active,
                    jnp.logical_and(
                        state.laser_x + self.consts.LASER_SIZE[1] > state.demons_x[i],
                        jnp.logical_and(
                            state.laser_x < state.demons_x[i] + self.consts.DEMON_SIZE[1],
                            jnp.logical_and(
                                state.laser_y < state.demons_y[i] + self.consts.DEMON_SIZE[0],
                                state.laser_y + self.consts.LASER_SIZE[1] > state.demons_y[i]
                            )
                        )
                    )
                )
            )

            new_alive = s_alive.at[i].set(
                jnp.logical_and(s_alive[i], jnp.logical_not(demon_hit))
            )
            new_score = jnp.where(demon_hit, s_score + 10 + state.wave_pattern * 2, s_score)
            new_laser_active = jnp.logical_and(l_active, jnp.logical_not(demon_hit))
            return new_alive, new_score, new_laser_active

        init_carry = (
            state.demons_alive,
            state.score,
            state.laser_active,
        )
        demons_alive, score, laser_active = jax.lax.fori_loop(
            0,
            self.consts.MAX_DEMONS,
            check_demon_collision,
            init_carry,
        )

        demon_killed = jnp.any(
            jnp.logical_and(
                state.demons_alive,
                jnp.logical_not(demons_alive),
            )
        )

        spawn_timer = jnp.where(
            demon_killed,
            jnp.array(self.consts.RESPAWN_DELAY, dtype=jnp.int32),
            jnp.maximum(state.spawn_timer - 1, 0),
        )
        killed = state.demons_alive & ~demons_alive

        # Bomb vs Player
        player_hit = jnp.logical_and(
            state.bomb_active,
            jnp.logical_and(
                jnp.abs(state.bomb_x - state.player_x) < self.consts.PLAYER_SIZE[1],
                jnp.logical_and(
                    state.bomb_y < self.consts.PLAYER_Y + self.consts.PLAYER_SIZE[0],
                    state.bomb_y + self.consts.BOMB_SIZE[1] > self.consts.PLAYER_Y
                )
            )
        )

        lives = jnp.where(player_hit, jnp.maximum(state.lives - 1, 0), state.lives)
        bomb_active = jnp.logical_and(state.bomb_active, jnp.logical_not(player_hit))

        # If player hit, start explosion
        player_exploding = jnp.logical_or(state.player_exploding, player_hit)
        explosion_timer = jnp.where(player_hit, 20, state.explosion_timer)

        state = state.replace(
            demons_alive=demons_alive,
            demon_register=jnp.where(killed, 0, state.demon_register),
            demon_teleport=jnp.where(demon_killed, jnp.argmax(killed.astype(jnp.int32)), state.demon_teleport),
            demon_teleport_timer=jnp.where(demon_killed, spawn_timer, state.demon_teleport_timer),
            score=score,
            laser_active=laser_active,
            lives=lives,
            bomb_active=bomb_active,
            player_exploding=player_exploding,
            explosion_timer=explosion_timer,
            spawn_timer=spawn_timer,
        )

        return self._refill_or_advance_wave(state)

    def _refill_or_advance_wave(self, state: DemonAttackState) -> DemonAttackState:
        wave_finished = (state.appeared_demons >= state.wave_total) & ~jnp.any(state.demons_alive)

        return jax.lax.cond(
            wave_finished,
            lambda s: self._advance_wave(s),
            lambda s: s,
            operand=state,
        )

    def render(self, state: DemonAttackState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: DemonAttackState):
        player = ObjectObservation.create(
            x=state.player_x,
            y=jnp.array(self.consts.PLAYER_Y),
            width=jnp.array(self.consts.PLAYER_SIZE[1]),
            height=jnp.array(self.consts.PLAYER_SIZE[0]),
        )

        demons = ObjectObservation.create(
            x=state.demons_x,
            y=state.demons_y,
            width=jnp.array(self.consts.DEMON_SIZE[1]),
            height=jnp.array(self.consts.DEMON_SIZE[0]),
            active=state.demons_alive
        )

        laser = ObjectObservation.create(
            x=state.laser_x,
            y=state.laser_y,
            width=jnp.array(self.consts.LASER_SIZE[1]),
            height=jnp.array(self.consts.LASER_SIZE[0]),
            active=state.laser_active
        )

        bomb = ObjectObservation.create(
            x=state.bomb_x,
            y=state.bomb_y,
            width=jnp.array(self.consts.BOMB_SIZE[1]),
            height=jnp.array(self.consts.BOMB_SIZE[0]),
            active=state.bomb_active
        )

        return DemonAttackObservation(
            player=player,
            demons=demons,
            laser=laser,
            bomb=bomb,
            score=state.score,
            lives=state.lives
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        object_space = spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))
        demons_space = spaces.get_object_space(n=self.consts.MAX_DEMONS,
                                               screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        return spaces.Dict({
            "player": object_space,
            "demons": demons_space,
            "laser": object_space,
            "bomb": object_space,
            "score": spaces.Box(low=0, high=99999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=self.consts.MAX_BUNKERS, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: DemonAttackState) -> DemonAttackInfo:
        return DemonAttackInfo(
            time=state.step_counter,
            wave_number=state.wave_number,
            wave_pattern=state.wave_pattern,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: DemonAttackState, state: DemonAttackState):
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: DemonAttackState) -> bool:
        return state.lives <= 0

class DemonAttackRenderer(JAXGameRenderer):
    def __init__(self, consts: DemonAttackConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or DemonAttackConstants()

        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 2. Create procedural assets
        bomb_sprite = _create_projectile_sprite(self.consts.BOMB_SIZE, self.consts.BOMB_COLOR)
        explosion_sprite = _create_explosion_sprite(self.consts)
        digit_sprites = _create_digit_sprites(self.consts)

        # Update asset config with procedural data
        final_asset_config.append({'name': 'projectile_demon', 'type': 'procedural', 'data': bomb_sprite})
        final_asset_config.append({'name': 'explosion', 'type': 'procedural', 'data': explosion_sprite})
        final_asset_config.append({'name': 'score_digits', 'type': 'procedural', 'data': digit_sprites})

        # 3. Bake assets
        sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "demonattack")
        jax.debug.print(f"Using sprites from: {sprite_path}")
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: DemonAttackState):
        blank = jnp.ones((self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)
        blank = blank * jnp.asarray(self.consts.BLANK_SCREEN_COLOR, dtype=jnp.uint8)

        return jax.lax.cond(
            state.game_frozen,
            lambda: blank,
            lambda: self._render_gameplay(state),
        )

    def _render_gameplay(self, state: DemonAttackState):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render bunkers as the visible life count.
        bunker_mask = self.SHAPE_MASKS["bunker"]

        def render_bunker(i, r):
            return jax.lax.cond(
                i < state.lives,
                lambda: self.jr.render_at(
                    r,
                    self.consts.BUNKER_X + i * self.consts.BUNKER_SPACING,
                    self.consts.BUNKER_Y,
                    bunker_mask,
                ),
                lambda: r,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_BUNKERS, render_bunker, raster)

        # Render player or explosion
        player_mask = jax.lax.select(state.player_exploding, self.SHAPE_MASKS["explosion"], self.SHAPE_MASKS["player"])
        raster = self.jr.render_at(raster, state.player_x, self.consts.PLAYER_Y, player_mask)

        # Render demons
        raster = self._draw_demons(raster, state)

        # Render laser
        laser_mask = self.SHAPE_MASKS["player_missile"]
        laser_render_x = jax.lax.select(
            state.laser_active,
            state.laser_x,
            state.player_x + self.consts.PLAYER_SIZE[1] // 2
        )
        laser_render_y = jax.lax.select(
            state.laser_active,
            state.laser_y,
            self.consts.PLAYER_Y + self.consts.PLAYER_LASER_DEPTH
        )
        raster = self.jr.render_at(raster, laser_render_x, laser_render_y, laser_mask)

        # Render bomb
        bomb_mask = self.SHAPE_MASKS["projectile_demon"]
        raster = jax.lax.cond(
            state.bomb_active,
            lambda: self.jr.render_at(raster, state.bomb_x, state.bomb_y, bomb_mask),
            lambda: raster
        )

        # Render Score
        score_digits = self.jr.int_to_digits(state.score, max_digits=4)
        digit_masks = self.SHAPE_MASKS["score_digits"]

        is_single_digit = state.score < 10
        is_double_digit = jnp.logical_and(state.score >= 10, state.score < 100)
        is_triple_digit = jnp.logical_and(state.score >= 100, state.score < 1000)

        start_index = jax.lax.select(
            is_single_digit,
            3,
            jax.lax.select(is_double_digit, 2, jax.lax.select(is_triple_digit, 1, 0)),
        )
        num_to_render = jax.lax.select(
            is_single_digit,
            1,
            jax.lax.select(is_double_digit, 2, jax.lax.select(is_triple_digit, 3, 4)),
        )

        raster = self.jr.render_label_selective(
            raster,
            70,
            10,
            score_digits,
            digit_masks,
            start_index,
            num_to_render,
            spacing=8,
        )

        return self.jr.render_from_palette(raster, self.PALETTE)

    def _draw_demons(self, raster, state):
        demon_anim_idx = (state.step_counter % 32) // 8

        sprite_group_idx = jnp.asarray(
            self.consts.WAVE_SPRITE_TABLE,
            dtype=jnp.int32,
        )[state.wave_pattern]

        demon_masks = jax.lax.switch(
            sprite_group_idx,
            [
                lambda: self.SHAPE_MASKS["demon_1"],
                lambda: self.SHAPE_MASKS["demon_2"],
            ],
        )

        demon_mask = demon_masks[demon_anim_idx]

        spawn_anim_total = self.consts.DEMON_TELEPORT_DURATION
        ids = jnp.arange(self.consts.MAX_DEMONS)
        spacing = (self.consts.DEMON_MAX_X - self.consts.DEMON_MIN_X) // (self.consts.MAX_DEMONS + 1)
        spawn_target_x = self.consts.DEMON_MIN_X + (ids + 1) * spacing

        def render_demon(i, r):
            is_spawning = state.spawn_anim_timer[i] > 0

            elapsed = spawn_anim_total - state.spawn_anim_timer[i]
            spawn_frame = jnp.clip(
                (elapsed * self.consts.SPAWN_ANIM_FRAMES) // spawn_anim_total,
                0,
                self.consts.SPAWN_ANIM_FRAMES - 1,
            )

            spawn_left_mask = self.SHAPE_MASKS["enemy_spawn_left"][spawn_frame]
            spawn_right_mask = self.SHAPE_MASKS["enemy_spawn_right"][spawn_frame]

            def render_spawn():
                spawn_max_x = min(self.consts.DEMON_MAX_X, self.consts.WIDTH - self.consts.SPAWN_ANIM_WIDTH)
                target_x = jnp.clip(
                    spawn_target_x[i] - (self.consts.SPAWN_ANIM_WIDTH - self.consts.DEMON_SIZE[1]) // 2,
                    self.consts.DEMON_MIN_X,
                    spawn_max_x,
                )
                last_step = jnp.array(spawn_anim_total - 1, dtype=jnp.int32)
                left_render_x = (
                    self.consts.DEMON_MIN_X * (last_step - elapsed)
                    + target_x * elapsed
                ) // last_step
                right_render_x = (
                    spawn_max_x * (last_step - elapsed)
                    + target_x * elapsed
                ) // last_step

                spawn_raster = self.jr.render_at_clipped(
                    r,
                    left_render_x,
                    state.demons_y[i],
                    spawn_left_mask,
                )
                return self.jr.render_at_clipped(
                    spawn_raster,
                    right_render_x,
                    state.demons_y[i],
                    spawn_right_mask,
                )

            def render_normal():
                return self.jr.render_at_clipped(
                    r,
                    state.demons_x[i],
                    state.demons_y[i],
                    demon_mask,
                )

            return jax.lax.cond(
                state.demons_alive[i],
                lambda: jax.lax.cond(
                    is_spawning,
                    render_spawn,
                    render_normal,
                ),
                lambda: r,
            )

        return jax.lax.fori_loop(0, self.consts.MAX_DEMONS, render_demon, raster)
