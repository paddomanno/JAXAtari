import os
from functools import partial
from typing import Tuple

import chex
import jax.lax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation


def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'player', 'type': 'single', 'file': 'player.npy'},
    )

class DemonAttackConstants(struct.PyTreeNode):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)

    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(144, 72, 17))
    PLAYER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(92, 186, 92))

    PLAYER_Y: int = struct.field(pytree_node=False, default=140)
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(4, 16))

    PLAYER_MAX_SPEED: float = struct.field(pytree_node=False, default=5.75)
    PLAYER_MIN_X: float = struct.field(pytree_node=False, default=24.0)
    PLAYER_MAX_X: float = struct.field(pytree_node=False, default=190.0)

class DemonAttackState(struct.PyTreeNode):
    player_x: chex.Array

class DemonAttackObservation(struct.PyTreeNode):
    player: ObjectObservation

class DemonAttackInfo(struct.PyTreeNode):
    pass

class JaxDemonAttack(JaxEnvironment[DemonAttackState, DemonAttackObservation, DemonAttackInfo, DemonAttackConstants]):
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.RIGHT, Action.LEFT, Action.RIGHTFIRE, Action.LEFTFIRE],
        dtype=jnp.int32,
    )
    def __init__(self, consts: DemonAttackConstants = None):
        consts = consts or DemonAttackConstants()
        super().__init__(consts)
        self.renderer = DemonAttackRenderer(self.consts)

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[DemonAttackObservation, DemonAttackState]:
        state = DemonAttackState(
            player_x=jnp.array(96.0, dtype=jnp.float32)
        )

        initial_obs = self._get_observation(state)

        return initial_obs, state

    def _player_step(self, state: DemonAttackState, action: chex.Array) -> DemonAttackState:
        right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
        left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)

        # 1. Determine Analog Target Speed
        target_speed = jax.lax.cond(
            right,
            lambda _: self.consts.PLAYER_MAX_SPEED,
            lambda _: jax.lax.cond(
                left,
                lambda _: -self.consts.PLAYER_MAX_SPEED,
                lambda _: 0.0,
                operand=None,
            ),
            operand=None,
        )

        # 2. player instantly has max speed
        new_speed = target_speed

        # 3. Apply position update and clip to physical bounds
        new_x = jnp.clip(
            state.player_x + new_speed,
            self.consts.PLAYER_MIN_X,
            self.consts.PLAYER_MAX_X,
        )

        return state.replace(
            player_x=new_x,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DemonAttackState, action: chex.Array) -> Tuple[DemonAttackObservation, DemonAttackState, float, bool, DemonAttackInfo]:
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        previous_state = state
        state = self._player_step(state, atari_action)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info

    def render(self, state: DemonAttackState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: DemonAttackState):
        player = ObjectObservation.create(
            y=jnp.array(self.consts.PLAYER_Y),
            x=state.player_x,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        # Use get_object_space helper to create standard ObjectObservation spaces
        object_space = spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        return spaces.Dict({
            "player": object_space,
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: DemonAttackState, ) -> DemonAttackInfo:
        return None

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: DemonAttackState, state: DemonAttackState):
        return 0

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: DemonAttackState) -> bool:
        return False


class DemonAttackRenderer(JAXGameRenderer):
    def __init__(self, consts: DemonAttackConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or DemonAttackConstants()

        # Use injected config if provided, else default
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

        # 2. Bake assets once
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "pong")
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        player_mask = self.SHAPE_MASKS["player"]
        raster = self.jr.render_at(
            raster,
            jnp.round(state.player_x).astype(jnp.int32),
            self.consts.PLAYER_Y,
            player_mask,
        )

        return self.jr.render_from_palette(raster, self.PALETTE)