# !/usr/bin/python
# -*- coding: utf-8 -*-
import datetime
import os
import random
from os.path import dirname, join
from pathlib import Path
from dataclasses import dataclass
import time

import numpy as np
import pygame

from bci.Game.helper import *


# Global Constants

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)

PROJECT_ROOT: Path = Path()
ASSETS = join(os.getcwd(), "resources", "game_assets", "dino")


@dataclass(frozen=True)
class MentalConfig:
    mapping_table: dict
    cue_list: list


pygame.display.set_caption("Chrome Dino Runner")

Ico = pygame.image.load(os.path.join(ASSETS, "DinoWallpaper.png"))
pygame.display.set_icon(Ico)

BG = pygame.image.load(os.path.join(ASSETS, "Track.png"))

FONT_COLOR = (0, 0, 0)
game_speed_backup = 10
game_speed = game_speed_backup


# Classes
class Dinosaur:
    X_POS = 80
    Y_POS = SCREEN_HEIGHT // 2 - 50
    JUMP_VEL = 15

    def __init__(self, pygame):
        RUNNING = [
            pygame.image.load(os.path.join(ASSETS, "DinoRun1.png")),
            pygame.image.load(os.path.join(ASSETS, "DinoRun2.png")),
        ]
        JUMPING = pygame.image.load(os.path.join(ASSETS, "DinoJump.png"))
        FAIL = pygame.image.load(os.path.join(ASSETS, "dino_sad.png"))

        # resize images
        RUNNING = [pygame.transform.scale(img, (120, 130)) for img in RUNNING]
        JUMPING = pygame.transform.scale(JUMPING, (120, 130))
        FAIL = pygame.transform.scale(FAIL, (120, 130))
        self.fail_img = FAIL
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_run = True
        self.dino_jump = False
        self.dino_fail = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

        self.cnt = 0

        self.invulnerable = False
        self.health = 15

    def update(self, userInput):
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()
        if self.dino_fail:
            self.fail()

        if self.step_index >= 10:
            self.step_index = 0

        if userInput is None:
            return

        """
        if userInput.type == GameInputType.BINARY1 and not self.dino_jump:
            self.dino_run = False
            self.dino_jump = True
        elif userInput.type == GameInputType.BINARY2 and not self.dino_jump:
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput.type == GameInputType.CONTINUOUS2):
            self.dino_run = True
            self.dino_jump = False
        """

    def fail(self):
        """Handle the dinosaur's failure state."""
        self.image = self.fail_img

        if self.dino_fail:
            # cnt depending on the speed, to be showing the failure
            self.cnt += 1
            self.dino_rect.y = self.Y_POS + 5
            if self.cnt >= 2 * game_speed:
                self.cnt = 0
                self.dino_fail = False
                self.dino_run = True
                self.dino_jump = False
                self.jump_vel = self.JUMP_VEL
                self.image = self.run_img[0]
                self.dino_rect.y = self.Y_POS

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            # Determine the constant height we want to achieve
            TARGET_HEIGHT = 200  # You can adjust this value as needed

            # Constant factor for gravity (rate of velocity decrease)
            GRAVITY_FACTOR = 0.05

            # Calculate gravity based on game speed
            speed_factor = GRAVITY_FACTOR * game_speed

            # Initial jump velocity based on the desired height and game speed
            self.JUMP_VEL = (2.4 * TARGET_HEIGHT * speed_factor) ** 0.5

            # Apply jump velocity to the dinosaur's position
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= speed_factor

        if self.jump_vel < -self.JUMP_VEL:
            self.dino_jump = False
            self.dino_run = True
            self.jump_vel = self.JUMP_VEL
            print(self.jump_vel)

    # def jump(self):
    #     self.image = self.jump_img
    #     if self.dino_jump:
    #         # Adjust jump velocity based on game speed to ensure consistent jump behavior
    #         # when game speed is 10, jump velocity should be around 5 and decrease by 0.4 for each frame
    #         # when game speed is 20, jump velocity should be around 8.5 and decrease by 0.5 for each frame
    #         # when game speed is 30, jump velocity should be around 12 and decrease by 0.6 for each frame
    #         # self.JUMP_VEL = 3.5 + 70 / game_speed
    #         self.dino_rect.y -= self.jump_vel
    #         speed_factor = 0.3 + 0.1 * (game_speed**2 / 100)
    #         # speed_factor = 0.4 * (game_speed / 10)
    #         self.jump_vel -= speed_factor

    #     if self.jump_vel < -self.JUMP_VEL:
    #         self.dino_jump = False
    #         self.JUMP_VEL = 15 + (game_speed**2 / 100) * 0.1
    #         self.jump_vel = self.JUMP_VEL
    #         print(self.jump_vel)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


class DecayingBar:
    def __init__(
        self, x, y, width, height, color, max_value, decay_rate=0.1, bump_rate=0.01
    ):
        # decay_rate=0.01
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.max_value = max_value
        self.decay_rate = decay_rate
        self.bump_rate = bump_rate
        self.value = self.height

    def update(self):
        self.value += self.decay_rate
        if self.value > self.height:
            self.value = self.height

    def draw(self, SCREEN):
        pygame.draw.rect(SCREEN, self.color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(SCREEN, (0, 0, 0), (self.x, self.y, self.width, self.value))
        pygame.draw.line(
            SCREEN,
            (255, 0, 0),
            (self.x - self.width, self.y + self.height),
            (self.x + 2 * self.width, self.y + self.height),
            2,
        )
        pygame.draw.line(
            SCREEN,
            (255, 0, 0),
            (self.x - self.width, self.y + (self.height - self.max_value)),
            (self.x + 2 * self.width, self.y + (self.height - self.max_value)),
            2,
        )

    def bump(self):
        self.value -= self.bump_rate
        if self.value < 0:
            self.value = 0


class Task:
    def __init__(self, pygame):
        self.x = 1000
        self.y = 70
        self.ARROW_UP = pygame.image.load(os.path.join(ASSETS, "yellow_arrow_up.png"))
        self.ARROW_DOWN = pygame.image.load(
            os.path.join(ASSETS, "yellow_arrow_down.png")
        )
        self.ARROW_LEFT = pygame.image.load(
            os.path.join(ASSETS, "yellow_arrow_left.png")
        )
        self.ARROW_RIGHT = pygame.image.load(
            os.path.join(ASSETS, "yellow_arrow_right.png")
        )
        self.CIRCLE = pygame.image.load(os.path.join(ASSETS, "small_yellow_circle.png"))

    def draw(self, SCREEN, cue):
        if "ARROW LEFT" in cue:
            SCREEN.blit(self.ARROW_LEFT, (650, 150))
        elif "ARROW RIGHT" in cue:
            SCREEN.blit(self.ARROW_RIGHT, (650, 150))
        elif "ARROW DOWN" in cue:
            SCREEN.blit(self.ARROW_DOWN, (900, 70))
        elif "ARROW UP" in cue:
            SCREEN.blit(self.ARROW_UP, (900, 70))
        elif "CIRCLE" in cue:
            SCREEN.blit(self.CIRCLE, (860, 90))


class Cloud:
    def __init__(self, SCREEN_WIDTH, pygame):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = pygame.image.load(os.path.join(ASSETS, "Cloud.png"))
        self.width = self.image.get_width()

    def update(self, SCREEN_WIDTH):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, ImagePath, type, SCREEN_WIDTH, pygame):
        self.image = [pygame.image.load(ImagePath[i]) for i in range(len(ImagePath))]
        # if image too big resize height to 200
        if self.image[0].get_height() > 200:
            self.image = [
                pygame.transform.scale(
                    img, (int(img.get_width() * 200 / img.get_height()), 200)
                )
                for img in self.image
            ]
        # self.image = [pygame.transform.scale(img, (100, 100)) for img in self.image]
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH
        self.ideal_jump = self.rect.x - random.randint(50, 100)

    def update(self, game_speed):
        isoutside = False
        self.rect.x -= game_speed
        self.ideal_jump = self.rect.x - random.randint(50, 100)
        if self.rect.x < -self.rect.width:
            isoutside = True
        return isoutside

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, SCREEN_WIDTH, pygame):
        ImagePath = [
            os.path.join(ASSETS, "SmallCactus1.png"),
            os.path.join(ASSETS, "SmallCactus2.png"),
            os.path.join(ASSETS, "SmallCactus3.png"),
        ]
        self.type = random.randint(0, 2)
        super().__init__(ImagePath, self.type, SCREEN_WIDTH, pygame)
        self.rect.y = SCREEN_HEIGHT // 2
        print(f"SmallCactus rect.y: {self.rect.y}, type: {self.type}")


class LargeCactus(Obstacle):
    def __init__(self, SCREEN_WIDTH, pygame):
        ImagePath = [
            os.path.join(ASSETS, "LargeCactus1.png"),
            os.path.join(ASSETS, "LargeCactus2.png"),
            os.path.join(ASSETS, "LargeCactus3.png"),
            # os.path.join(ASSETS, "LargeCactus4.png"),
        ]

        self.type = random.randint(0, 2)
        super().__init__(ImagePath, self.type, SCREEN_WIDTH, pygame)
        self.rect.y = SCREEN_HEIGHT // 2  # Adjusted for larger cactus height
        print(f"LargeCactus rect.y: {self.rect.y}, type: {self.type}")


class Bird(Obstacle):
    BIRD_HEIGHT = [300, 350]

    def __init__(self, SCREEN_WIDTH, pygame):
        self.type = 0
        ImagePath = [
            os.path.join(ASSETS, "Bird1.png"),
            os.path.join(ASSETS, "Bird2.png"),
        ]
        super().__init__(ImagePath, self.type, SCREEN_WIDTH, pygame)
        self.rect.y = random.choice(self.BIRD_HEIGHT) + 20
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index // 5], self.rect)
        self.index += 1


class HighScoreTable:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scores = []
        self.load_scores()
        self.rank = -1

    def load_scores(self):
        try:
            with open(self.file_path, "r") as f:
                self.scores = []
                for line in f.readlines():
                    try:
                        name, score = line.strip().split(": ")
                        self.scores.append((name, int(score)))
                    except ValueError:
                        print(
                            f"Skipping invalid entry in high score file: {line.strip()}"
                        )
                self.scores.sort(key=lambda x: x[1], reverse=True)
        except FileNotFoundError:
            self.scores = []

    def save_scores(self):
        with open(self.file_path, "w") as f:
            for name, score in self.scores:
                f.write(f"{name}: {score}\n")  # Ensures only valid lines are saved

    def add_score(self, name, score):
        self.scores.append((name, score))
        self.scores.sort(key=lambda x: x[1], reverse=True)
        self.rank = self.scores.index((name, score)) + 1
        self.save_scores()

    def display(self, screen, font):
        # Draw background box
        box_width, box_height = 600, 530
        box_x, box_y = (SCREEN_WIDTH - box_width) // 2, 30
        pygame.draw.rect(
            screen,
            (50, 50, 50),
            (box_x, box_y, box_width, box_height),
            border_radius=15,
        )
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (box_x, box_y, box_width, box_height),
            3,
            border_radius=15,
        )

        # Add the title
        title = font.render("HIGH SCORES", True, (255, 255, 255))
        screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 50))
        # Add the Rank
        title = font.render(f"Your rank is: {self.rank}", True, (255, 255, 255))
        screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 80))

        # Add headers
        header_font = pygame.font.Font(None, 32)
        headers = ["Rank", "Name", "Score"]
        header_positions = [
            SCREEN_WIDTH // 2 - 200,
            SCREEN_WIDTH // 2,
            SCREEN_WIDTH // 2 + 200,
        ]
        for i, header in enumerate(headers):
            header_text = header_font.render(header, True, (200, 200, 200))
            screen.blit(
                header_text, (header_positions[i] - header_text.get_width() // 2, 135)
            )

        # Add scores
        row_font = pygame.font.Font(None, 28)
        for i, (name, score) in enumerate(self.scores[:10]):
            color = (
                (255, 215, 0) if i == self.rank - 1 else (255, 255, 255)
            )  # Highlight the top score
            rank_text = row_font.render(str(i + 1), True, color)
            name_text = row_font.render(name, True, color)
            score_text = row_font.render(str(score), True, color)

            row_y = 160 + i * 40
            screen.blit(
                rank_text, (header_positions[0] - rank_text.get_width() // 2, row_y)
            )
            screen.blit(
                name_text, (header_positions[1] - name_text.get_width() // 2, row_y)
            )
            screen.blit(
                score_text, (header_positions[2] - score_text.get_width() // 2, row_y)
            )

        # Add "Press Enter to continue" at the bottom of the box
        continue_text = font.render("Press Enter to continue", True, (12, 12, 12))
        continue_y = (
            box_y + box_height - 5
        )  # Position it 50 pixels from the bottom of the box
        screen.blit(
            continue_text,
            (SCREEN_WIDTH // 2 - continue_text.get_width() // 2, continue_y),
        )
        # press button to continue
