from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.games.jax_demonattack import DemonAttackState
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

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        guided_x = new_state.player_x + self._env.consts.PLAYER_SIZE[1] // 2
        return new_state.replace(
            laser_x=jnp.where(new_state.laser_active, guided_x, new_state.laser_x),
        )
