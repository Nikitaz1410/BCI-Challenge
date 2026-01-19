# -*- coding: utf-8 -*-
import os
import yaml
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(os.getcwd())
CONFIG_DIR = PROJECT_ROOT / "config"

@dataclass
class DinoConfig:
    countdown: int
    success_rate: float
    game_speed: int
    quicktime_duration: int
    num_reaction_times: int

    @property
    def max_time(self) -> int:
        return self.quicktime_duration


def load_dino_config() -> DinoConfig:
    cfg_path = CONFIG_DIR / "dino_config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return DinoConfig(
        countdown=cfg['Countdown'],
        success_rate=cfg['success_rate'],
        game_speed=cfg['game_speed'],
        quicktime_duration=cfg['quicktime_duration'],
        num_reaction_times=cfg['num_reaction_times'],
    )


def load_training_config():
    path = CONFIG_DIR / "training_config.yaml"
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)