"""
This script is used to simply play the Atari games manually.
"""
import imageio
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pygame
from ocatari.core import OCAtari
import os
from argparse import ArgumentParser
import cv2
import jax.numpy as jnp

parser = ArgumentParser()
parser.add_argument("-g", "--game", type=str, default="Pong")
parser.add_argument("-r", "--record", action="store_true")
parser.add_argument("-pr", "--print-reward", action="store_true")

args = parser.parse_args()

MAX_OBJECTS_PER_CATEGORY = 100
BACKGROUND_RGB = np.array([0, 0, 0])
IGNORED_OBJECT_CATEGORIES = ["NoObject", "Player", "PlayerMissile", "EnemyPart", "EnemyMissile"]

def save_rgb_array_as_png(rgb_array, filename):
    imageio.imwrite(filename, rgb_array)


def save_patch_as_png(patch, filename):
    (filename, patch)


def retrieve_objects_patch(object, frame):
    x_o, y_o, w, h = object.xywh
    # Expand detected object by 1 pixel but limit to frame boundaries
    x = np.maximum(0, x_o-1)
    y = np.maximum(0, y_o-1)
    # +2 normally, less if x or y is clipped
    w = np.minimum(np.minimum(x_o+w+1, x+w+2), frame.shape[1]) - x
    h = np.minimum(np.minimum(y_o+h+1, y+h+2), frame.shape[0]) - y
    patch = frame[y:y+h, x:x+w]
    if patch is None or patch.size == 0:
        return None
    patch_rgba = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)

    # Step 3: Remove the background
    # Set the alpha channel to 0 (transparent) where the color doesn't match the object RGB
    for i in range(h):
        for j in range(w):
            # Get the pixel color at (i, j)
            pixel_color = patch_rgba[i, j][:3]  # RGB part
            # Compare it to the object RGB color
            if np.all(pixel_color == BACKGROUND_RGB):
                # Set alpha to 0 if it doesn't match the object's RGB
                patch_rgba[i, j]= [0, 0, 0, 0]  # Set alpha to 0 (transparent)
            else:
                # Keep the alpha to 255 (opaque) if it matches
                patch_rgba[i, j][3] = 255
    return patch_rgba


def save_objects_patches(objects, frame, game):
    msg_given = False
    for object in objects:
        if object.category in IGNORED_OBJECT_CATEGORIES:
            continue
        patch = retrieve_objects_patch(object, frame)
        if patch is None:
            continue
        big_patch = np.repeat(np.repeat(patch, 20, axis=0), 20, axis=1)
        if not os.path.isdir(f'patches/{game}'):
            os.makedirs(f'patches/{game}', exist_ok=True)
        if not object.category:
            print("Skipping object with no category")
            continue
        i = 0
        while os.path.exists(f'patches/{game}/{object.category}_{i}.png'):
            i += 1
            if i == np.floor(MAX_OBJECTS_PER_CATEGORY*0.8) and not msg_given:
                print("Avoiding to save redundant objects > 0")
                msg_given = True
            if i > MAX_OBJECTS_PER_CATEGORY:
                break
        imageio.imwrite(f'patches/{game}/{object.category}_{i}.png', big_patch)
        jnp.save(f'patches/{game}/{object.category}_{i}.npy', patch)

    print(f"Objects patches saved in patches/{game}")


class Renderer:
    env: gym.Env
    current_wave = 0

    def __init__(self, env_name: str):
        self.env = OCAtari(env_name, mode="both", hud=False, render_mode="human",
                           render_oc_overlay=True, frameskip=1)
        self.env.reset()
        self.env.render()  # initialize pygame video system

        self.num_skipped_waves = 0
        self.paused = False
        self.frame_by_frame = False
        self.next_frame = False
        self.current_actions = set()
        self.keys2actions = {}
        for i, action in enumerate(self.env.get_action_meanings()):
            if action in ["RIGHT", "LEFT", "UP", "DOWN"]:
                self.keys2actions[eval(f'pygame.K_{action}')] = i
            elif action == "FIRE":
                self.keys2actions[eval(f'pygame.K_SPACE')] = i
            # elif action != "NOOP":
            #     import ipdb; ipdb.set_trace()
            #     print("ACTION NOT COVERED")
            #     raise NotImplementedError
        self.frame = 0
        # self.env.set_ram(16, 6)

    def run(self):
        self.skip_attract(self.env)

        self.running = True
        while self.running:
            self._handle_user_input()
            if not (self.frame_by_frame and not self.next_frame) and not self.paused:
                action = self._get_action()
                obs, reward, term, trunc, info = self.env.step(action)
                self.env.render()
                if args.record and self.frame % 4 == 0:
                    frame = self.env.unwrapped.ale.getScreenRGB()
                    save_rgb_array_as_png(
                        frame, f'frames/{args.game}_{self.frame}.png')
                if args.print_reward and reward != 0:
                    print(reward)
                self.frame += 1
                self.next_frame = False

        pygame.quit()

    def _get_action(self):
        for action in self.current_actions:
            try:
                action_int = int(action)
            except Exception:
                action_int = 0
            # Ensure action is within the valid range
            if hasattr(self.env, 'action_space') and hasattr(self.env.action_space, 'n'):
                if not (0 <= action_int < self.env.action_space.n):
                    action_int = 0  # fallback to NOOP if out of range
            return action_int
        return 0

    def _handle_user_input(self):
        self.current_mouse_pos = np.asarray(pygame.mouse.get_pos())

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                self.running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed                
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                elif event.key == pygame.K_f: self.frame_by_frame = not self.frame_by_frame; self.next_frame = False; print(f"{'Frame-by-frame' if self.frame_by_frame else 'Continuous'}")
                elif event.key == pygame.K_n and self.frame_by_frame: self.next_frame = True; print("Next Frame")
                if event.key == pygame.K_r:  # 'R': reset
                    self.env.reset()

                if event.key == pygame.K_l:  # L to go to next level/wave
                    self.goto_next_level()

                if event.key == pygame.K_o:  # 'O': Save objects
                    screen = self.env.unwrapped._ale.getScreenRGB()
                    save_objects_patches(
                        self.env.objects, screen, self.env.game_name)
                    # screen = np.repeat(np.repeat(screen, 6, axis=0), 6, axis=1)
                    # save_rgb_array_as_png(
                    #     screen, f'patches/{self.env.game_name}_{self.frame}.png')

                elif event.key in self.keys2actions:  # env action                    
                    self.current_actions.add(self.keys2actions[event.key])

            elif event.type == pygame.KEYUP:  # keyboard key released
                if event.key in self.keys2actions:
                    self.current_actions.remove(self.keys2actions[event.key])

    def ram_addr(asm_addr):
        """Convert a 2600 zero-page address ($80–$FF) to ALE RAM index (0–127)."""
        assert 0x80 <= asm_addr <= 0xFF, f"Address ${asm_addr:02X} is not in zero-page RAM"
        return asm_addr - 0x80


    # All addresses as ALE indices (asm_addr - 0x80)
    # LEVEL = ram_addr(0xBE)  # 62
    APPEARED_ENEMIES = ram_addr(0x9B)  # 27
    ENEMY_REG_1 = ram_addr(0xAF)  # 47 ← bits 7+6 = liveness, NOT position
    ENEMY_REG_2 = ram_addr(0xB0)  # 48
    ENEMY_REG_3 = ram_addr(0xB1)  # 49
    SMALL_DEMON_REG = ram_addr(0xB2)  # 50
    TELE_COUNTDOWN = ram_addr(0xBB)  # 59

    skip_spawning = True

    def goto_next_level(self):

        ale = self.env._env.unwrapped.ale

        # we don't know the actual wave, so if you manually play to a new wave this is not accurate anymore
        target = self.num_skipped_waves + 1 + 1
        print(f"Performing Skip to wave {target} (?)")
        self.num_skipped_waves += 1

        ale.setRAM(self.APPEARED_ENEMIES, 0)

        # Now end waves repeatedly until we're one wave before target
        self.end_current_wave(ale, self.env, settle_frames=100)

        # Step a frame to let the game register the change
        obs, _, _, _, _ = self.env.step(0)

        if self.skip_spawning:
            for _ in range(150):
                self.env.step(1)  # action 1 = FIRE

    def end_current_wave(self, ale, env, settle_frames):
        """
        Trick the game into ending the current wave by:
        1. Setting AppearedEnemies = 8 (wave-complete threshold)
        2. Clearing all three enemy registers (no live enemies)
        Then letting it run its normal wave-transition code.
        """
        ale.setRAM(self.APPEARED_ENEMIES, 8)
        ale.setRAM(self.ENEMY_REG_1, 0)
        ale.setRAM(self.ENEMY_REG_2, 0)
        ale.setRAM(self.ENEMY_REG_3, 0)
        ale.setRAM(self.SMALL_DEMON_REG, 0)  # no split-demon active
        ale.setRAM(self.TELE_COUNTDOWN, 0)  # no teleport in progress
        for _ in range(settle_frames):
            env.step(0)  # NOOP — let the game animate and spawn new wave

    def skip_attract(self, env, frames=200):
        """Fire through the attract/title screen to start the game."""
        for _ in range(frames):
            env.step(1)  # action 1 = FIRE

    def _get_ram_value_at(self, idx: int):
        if self.ram is not None and 0 <= idx < len(self.ram): return self.ram[idx]
        return 0

if __name__ == "__main__":
    # renderer = Renderer(args.game)
    # renderer = Renderer("Seaquest")
    renderer = Renderer(args.game)
    renderer.run()
