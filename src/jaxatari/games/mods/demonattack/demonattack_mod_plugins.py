from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.games.jax_demonattack import DEMON_STATUS_NORMAL, DemonAttackState, DEMON_STATUS_SMALL
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin


def _clear_bombs(env, state: DemonAttackState) -> DemonAttackState:
    return state.replace(
        bomb_active=jnp.zeros_like(state.bomb_active, dtype=jnp.bool_),
        bomb_burst_step=jnp.array(env.consts.BOMB_BURST_RATES, dtype=jnp.int32),
        bomb_burst_length=jnp.array(0, dtype=jnp.int32),
        bomb_burst_timer=jnp.array(0, dtype=jnp.int32),
        bomb_action_counter=jnp.array(0, dtype=jnp.int32),
    )


class FastPlayerMod(JaxAtariInternalModPlugin):
    """Moves the player twice as fast."""

    constants_overrides = {
        "PLAYER_SPEED": 2,
    }


class FastLaserMod(JaxAtariInternalModPlugin):
    """Makes the player laser travel faster on every wave pattern."""

    constants_overrides = {
        "WAVE_LASER_SPEED_TABLE": (6, 7, 8, 8, 9, 9),
    }


class SlowEnemyShotsMod(JaxAtariInternalModPlugin):
    """Keeps enemy shots at the slowest speed for all difficulty levels."""

    constants_overrides = {
        "ENEMY_SHOT_SPEED_TABLE": (1, 1, 1, 1, 1, 1),
    }


class ShortWavesMod(JaxAtariInternalModPlugin):
    """Reduces the number of demons required to finish each wave."""

    constants_overrides = {
        "WAVE_TOTAL_DEMONS": 3,
    }


class NoEnemyShotsMod(JaxAtariPostStepModPlugin):
    """Removes active enemy shots after each reset and step."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        return _clear_bombs(self._env, new_state)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: DemonAttackState):
        state = _clear_bombs(self._env, state)
        return self._env._get_observation(state), state


class InfiniteLivesMod(JaxAtariPostStepModPlugin):
    """Restores the bunker/life count after each reset and step."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        return new_state.replace(
            lives=jnp.array(self._env.consts.MAX_BUNKERS, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: DemonAttackState):
        state = state.replace(
            lives=jnp.array(self._env.consts.MAX_BUNKERS, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
        )
        return self._env._get_observation(state), state


class RelentlessWavesMod(JaxAtariInternalModPlugin):
    """Increases pressure with quicker respawns and more frequent full bursts."""

    constants_overrides = {
        "RESPAWN_DELAY": 10,
        "SPAWN_MOVE_PAUSE": 4,
        "WAVE_TOTAL_DEMONS": 12,
        "ENEMY_SHOT_ACTION_TABLE": (4, 4, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2),
        "BOMB_PRE_FIRE_PAUSE": 8,
        "BOMB_BURST_LENGTH_OPTIONS": (5, 7, 7, 7),
        "BOMB_JITTER_X_TABLE": (-1, 0, 1, 0, -1, 0, 1),
    }


class LateWaveStartMod(JaxAtariPostStepModPlugin):
    """Starts each reset on wave 8 so tracking projectiles and late sprites are active."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: DemonAttackState):
        state = self._env._initialize_wave_state(
            state,
            jnp.array(self._env.consts.TRACKING_PROJECTILES_START_WAVE, dtype=jnp.int32),
        )
        return self._env._get_observation(state), state


class PlayerGuidedLaserMod(JaxAtariPostStepModPlugin):
    """Lets the player steer an active laser horizontally after firing."""

    conflicts_with = ["homing_laser"]

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        guided_x = new_state.player_x + self._env.consts.PLAYER_SIZE[1] // 2
        return new_state.replace(
            laser_x=jnp.where(new_state.laser_active, guided_x, new_state.laser_x),
        )


class HomingLaserMod(JaxAtariPostStepModPlugin):
    """Steers the active player laser toward the nearest hittable demon."""

    conflicts_with = ["player_guided_laser"]

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        hittable = jnp.logical_and(
            new_state.demons_alive,
            new_state.spawn_anim_timer <= 0,
        )
        demon_center_x = new_state.demons_x + self._env.consts.DEMON_SIZE[1] // 2
        target_idx = jnp.argmin(jnp.where(
            hittable,
            jnp.abs(demon_center_x - new_state.laser_x),
            10_000,
        ))
        target_x = demon_center_x[target_idx] - self._env.consts.LASER_SIZE[1] // 2
        laser_delta = jnp.clip(target_x - new_state.laser_x, -2, 2)
        laser_x = jnp.clip(
            new_state.laser_x + laser_delta,
            self._env.consts.DEMON_MIN_X,
            self._env.consts.DEMON_MAX_X,
        )
        should_home = jnp.logical_and(new_state.laser_active, jnp.any(hittable))
        return new_state.replace(
            laser_x=jnp.where(should_home, laser_x, new_state.laser_x),
        )


class TeleportingDemonsMod(JaxAtariPostStepModPlugin):
    """Blinks one active demon in place and teleport."""

    BLINK_FRAMES = 28
    TELEPORT_INTERVAL = 192

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        ids = jnp.arange(self._env.consts.MAX_DEMONS, dtype=jnp.int32)
        warning_active = new_state.spawn_pause_timer > self._env.consts.SPAWN_MOVE_PAUSE
        finish_warning = jnp.logical_and(
            prev_state.spawn_pause_timer > self._env.consts.SPAWN_MOVE_PAUSE,
            new_state.spawn_pause_timer <= self._env.consts.SPAWN_MOVE_PAUSE,
        )
        teleport_busy = jnp.logical_or(
            jnp.any(warning_active),
            jnp.any(prev_state.spawn_pause_timer > self._env.consts.SPAWN_MOVE_PAUSE),
        )
        eligible = jnp.logical_and(
            new_state.demons_alive,
            new_state.spawn_anim_timer <= 0,
        )
        eligible = jnp.logical_and(
            eligible,
            new_state.spawn_pause_timer <= 0,
        )
        eligible = jnp.logical_and(
            eligible,
            new_state.demon_status != DEMON_STATUS_SMALL,
        )
        eligible = jnp.logical_and(eligible, jnp.logical_not(teleport_busy))
        due = jnp.mod(new_state.step_counter, self.TELEPORT_INTERVAL) == 0
        desired_slot = jnp.mod(
            new_state.step_counter // self.TELEPORT_INTERVAL + new_state.wave_number,
            self._env.consts.MAX_DEMONS,
        )
        candidate_order = jnp.mod(desired_slot + ids, self._env.consts.MAX_DEMONS)
        ordered_eligible = eligible[candidate_order]
        first_ordered_idx = jnp.argmax(ordered_eligible.astype(jnp.int32))
        selected_slot = candidate_order[first_ordered_idx]
        has_target = jnp.any(eligible)
        start_warning = jnp.logical_and(
            due,
            jnp.logical_and(has_target, ids == selected_slot),
        )
        teleport = finish_warning
        x_span = (
            self._env.consts.DEMON_MAX_X
            - self._env.consts.DEMON_MIN_X
            - self._env.consts.DEMON_SIZE[1]
        )
        min_gap = self._env.consts.DEMON_MIN_VERTICAL_DISTANCE
        lane_min_y = jnp.asarray((
            self._env.consts.DEMON_MIN_Y,
            new_state.demons_y[0] + min_gap,
            new_state.demons_y[1] + min_gap,
        ), dtype=jnp.int32)
        lane_max_y = jnp.asarray((
            new_state.demons_y[1] - min_gap,
            new_state.demons_y[2] - min_gap,
            self._env.consts.DEMON_MAX_Y,
        ), dtype=jnp.int32)
        lane_center_y = jnp.asarray((
            (self._env.consts.DEMON_MIN_Y + new_state.demons_y[1]) // 2,
            (new_state.demons_y[0] + new_state.demons_y[2]) // 2,
            (new_state.demons_y[1] + self._env.consts.DEMON_MAX_Y) // 2,
        ), dtype=jnp.int32)
        seed = new_state.step_counter + new_state.wave_number * 13 + ids * 29
        target_x = self._env.consts.DEMON_MIN_X + jnp.mod(seed * 11, x_span)
        target_y = jnp.clip(
            lane_center_y + jnp.mod(seed * 7, 5) - 2,
            lane_min_y,
            lane_max_y,
        )
        return new_state.replace(
            demons_x=jnp.where(teleport, target_x, new_state.demons_x),
            demons_y=jnp.where(teleport, target_y, new_state.demons_y),
            spawn_pause_timer=jnp.where(
                start_warning,
                self._env.consts.SPAWN_MOVE_PAUSE + self.BLINK_FRAMES,
                jnp.where(teleport, 0, new_state.spawn_pause_timer),
            ),
            demon_moving_right=jnp.where(
                teleport,
                target_x < new_state.player_x,
                new_state.demon_moving_right,
            ),
            demon_moving_down=jnp.where(teleport, True, new_state.demon_moving_down),
        )


class SideStepLowestDemonsMod(JaxAtariPostStepModPlugin):
    """Makes the lowest normal demon sidestep away from near misses until no other demon is alive."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        ids = jnp.arange(self._env.consts.MAX_DEMONS, dtype=jnp.int32)
        active = jnp.logical_and(
            new_state.demons_alive,
            jnp.logical_and(
                new_state.spawn_anim_timer <= 0,
                new_state.spawn_pause_timer <= 0,
            ),
        )
        active = jnp.logical_and(active, new_state.demon_status == DEMON_STATUS_NORMAL)
        multiple_demons_alive = jnp.sum(new_state.demons_alive.astype(jnp.int32)) > 1
        bottom_slot = jnp.array(self._env.consts.MAX_DEMONS - 1, dtype=jnp.int32)
        moved_this_step = jnp.logical_or(
            prev_state.demons_x != new_state.demons_x,
            prev_state.demons_y != new_state.demons_y,
        )
        active = jnp.logical_and(
            active,
            jnp.logical_and(multiple_demons_alive, ids == bottom_slot),
        )
        active = jnp.logical_and(active, moved_this_step)

        demon_center = new_state.demons_x + self._env.consts.DEMON_SIZE[1] // 2
        laser_distance = jnp.abs(demon_center - new_state.laser_x)
        laser_near_x = laser_distance <= 16
        laser_in_lane = jnp.logical_and(
            new_state.laser_y <= new_state.demons_y + self._env.consts.DEMON_SIZE[0] + 12,
            new_state.laser_y >= new_state.demons_y - 28,
        )
        threatened = jnp.logical_and(
            active,
            jnp.logical_and(new_state.laser_active, jnp.logical_and(laser_near_x, laser_in_lane)),
        )
        dodge_active = threatened
        dodge_dir = jnp.where(
            new_state.laser_x < demon_center,
            2,
            -2,
        )
        demons_x = jnp.clip(
            new_state.demons_x + jnp.where(dodge_active, dodge_dir, 0),
            self._env.consts.DEMON_MIN_X,
            self._env.consts.DEMON_MAX_X - self._env.consts.DEMON_SIZE[1],
        )
        return new_state.replace(
            demons_x=demons_x,
            demon_moving_right=jnp.where(dodge_active, dodge_dir > 0, new_state.demon_moving_right),
        )


class ZigZagMovementDemonsMod(JaxAtariPostStepModPlugin):
    """Adds a zigzag movement pattern to normal demons."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        ids = jnp.arange(self._env.consts.MAX_DEMONS, dtype=jnp.int32)
        normal_active = jnp.logical_and(
            new_state.demon_status == DEMON_STATUS_NORMAL,
            jnp.logical_and(
                new_state.spawn_anim_timer <= 0,
                new_state.spawn_pause_timer <= 0,
            ),
        )
        source_firing = jnp.logical_and(
            ids == new_state.bomb_source_idx,
            jnp.logical_or(new_state.bomb_burst_length > 0, jnp.any(new_state.bomb_active)),
        )
        normal_active = jnp.logical_and(normal_active, jnp.logical_not(source_firing))
        phase = jnp.mod(new_state.step_counter + ids * 13, 48)
        zigzag = phase < 24
        x_step = jnp.where(zigzag, 1, -1)
        y_pulse = phase == 0
        y_step = jnp.where(new_state.demon_moving_down, 1, -1)
        demons_x = jnp.clip(
            jnp.where(normal_active, prev_state.demons_x + x_step, new_state.demons_x),
            self._env.consts.DEMON_MIN_X,
            self._env.consts.DEMON_MAX_X - self._env.consts.DEMON_SIZE[1],
        )
        demons_y = jnp.clip(
            jnp.where(
                jnp.logical_and(normal_active, y_pulse),
                prev_state.demons_y + y_step,
                new_state.demons_y,
            ),
            self._env.consts.DEMON_MIN_Y,
            self._env.consts.DEMON_MAX_Y - self._env.consts.DEMON_SIZE[0],
        )
        hit_x_edge = jnp.logical_or(
            demons_x <= self._env.consts.DEMON_MIN_X,
            demons_x >= self._env.consts.DEMON_MAX_X - self._env.consts.DEMON_SIZE[1],
        )
        return new_state.replace(
            demons_x=demons_x,
            demons_y=demons_y,
            demon_moving_right=jnp.where(
                jnp.logical_and(normal_active, hit_x_edge),
                jnp.logical_not(new_state.demon_moving_right),
                new_state.demon_moving_right,
            ),
            demon_moving_down=jnp.where(
                jnp.logical_and(normal_active, y_pulse),
                jnp.logical_not(new_state.demon_moving_down),
                new_state.demon_moving_down,
            ),
        )
