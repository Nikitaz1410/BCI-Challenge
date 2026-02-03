# main.py
# -*- coding: utf-8 -*-
"""
Game entrypoint and main loop.
Gameplay identical; relies on helpers.py for config, UDP/blinks, and PyQt survey.
"""

from __future__ import annotations

# =============================================================================
# Standard Library
# =============================================================================
import csv
import datetime
import os
import random
import time
from collections import deque
from os.path import join
from typing import Optional
from pathlib import Path
import sys

# Ensure project `src` directory is on Python path (for `bci` imports)
src_dir = Path(__file__).parent.parent.parent  # .../src
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# =============================================================================
# Third-party
# =============================================================================
import numpy as np
import pygame

# =============================================================================
# Local modules
# =============================================================================
from bci.game.helper.dinosaur import (
    Bird,
    Cloud,
    DecayingBar,
    Dinosaur,
    HighScoreTable,
    LargeCactus,
    SmallCactus,
    Task,
)
from bci.game.helper.lsl_markerstream import LSLMarkerStream
from bci.game.helpers import (
    config,
    configure_udp_endpoint,
    get_current_marker,
    get_recent_cmd_details,
    get_udp_listener_status,
    get_udp_payload_snapshot,
    is_udp_thread_running,
    set_runtime_context,
    show_survey_pyqt_and_save,
    start_udp_thread,
    stop_udp_thread,
)

pygame.init()
np.random.seed(42)  # For reproducibility in simulations

# =============================================================================
# Window / Assets
# =============================================================================
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

# UPDATE PATHS TO FIT WITH OUR PROJECT STRUCTURE
ASSETS = join(os.getcwd(), "resources", "game_assets", "dino")
WINDOW_FLAGS = pygame.RESIZABLE  # | pygame.FULLSCREEN
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), WINDOW_FLAGS)

pygame.display.set_caption("Chrome Dino Runner")
pygame.display.set_icon(pygame.image.load(os.path.join(ASSETS, "DinoWallpaper.png")))
BG = pygame.image.load(os.path.join(ASSETS, "Track.png"))

# =============================================================================
# Config & thresholds
# =============================================================================
CONFIG = config()

# Thresholds / timing (override in YAML if desired)

EVIDENCE_WINDOW_S = CONFIG.evidence_window_s
IGNORE_FIRST_S = CONFIG.ignore_first_s
MIN_VOTES = CONFIG.min_votes
REFRACTORY_S = CONFIG.refractory_s
STALE_PRED_S = CONFIG.stale_pred_s
GATED_BUMP_MULTIPLIER = CONFIG.gated_bump_multiplier
# ---- Debug + control knobs ----
DEBUG_HUD = CONFIG.debug_hud
PROB_THRESHOLD = CONFIG.prob_threshold

EVIDENCE_WINDOW_S = CONFIG.evidence_window_s
MIN_VOTES = CONFIG.min_votes


# Optional explicit replay switch (so seeing any marker doesn't auto-force replay)
USE_REPLAY_MARKERS = CONFIG.use_replay_markers

SLEEP_TIME = CONFIG.sleep_time  # seconds to sleep between bumps
print(f"Using SLEEP_TIME: {SLEEP_TIME} seconds")


# =============================================================================
# Human-bot fallback planner
# =============================================================================
class _Evidence:
    """Sliding-window counter for recent predictions, with deduping."""

    def __init__(self):
        self.t = []
        self.y = []

    def add(self, t, label):
        self.t.append(t)
        self.y.append(int(label))
        cutoff = t - 3.0  # keep a few seconds
        while self.t and self.t[0] < cutoff:
            self.t.pop(0)
            self.y.pop(0)

    def stats(self, now, target_idx, win_s):
        start = now - win_s
        idxs = [i for i, ti in enumerate(self.t) if ti >= start]
        total = len(idxs)
        if total == 0:
            return 0, 0, 0.0
        target = sum(1 for i in idxs if self.y[i] == target_idx)
        return total, target, target / total


def new_session_simulation(success_rate: float = 0.6, num_trials: int = 10):
    # The above Python code is simulating a scenario where success is determined based on a success rate
    # and random chance.
    # get if success depending on success_rate and random chance
    min_success = PROB_THRESHOLD * num_trials
    random_num = np.random.rand()
    print(
        f"Random number generated: {random_num:.2f} (success rate threshold: {min_success:.2f}, success rate: {success_rate:.2f})"
    )
    random_success = random_num < success_rate
    print(
        f"Simulating session with success rate {success_rate:.2f}, min successes {min_success}, total trials {num_trials}"
    )
    print(f"Random success condition: {random_success} ")
    if random_success:
        # get a random number between 3 and 6
        num_success = np.random.randint(min_success, num_trials)
        trials = [-2] * num_success + [-1] * (
            num_trials - num_success
        )  # Simulate successes and failures
    else:
        num_success = np.random.randint(1, min_success)
        trials = [-2] * num_success + [-1] * (num_trials - num_success)
    print(
        f"Simulated {num_success} successes out of {num_trials} trials with success rate {success_rate:.2f}"
    )
    random.shuffle(trials)  # Shuffle the trials to randomize order
    # print(f"Trials: {trials}")
    # trials = [-2, -2, -2, 1, -2, -1, -1, -2, -1, -2]
    # trials = trials + [-1]*(num_trials - len(trials))  # Ensure we have exactly num_trials
    return trials


# =============================================================================
# Metrics logging & evidence
# =============================================================================
def log_trial_metrics(
    user_id: str,
    mode: str,
    cue: str | None,
    marker: str | None,
    success: int,
    t_success: float | None,
    total_preds: int,
    target_votes: int,
    frac_target: float,
    consensus: float,
    max_time: float,
) -> None:
    out_dir = os.path.join(os.getcwd(), "data", f"sub-{user_id}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dino_trial_metrics.csv")
    header = [
        "timestamp",
        "mode",
        "cue",
        "marker",
        "success",
        "time_to_success_s",
        "total_votes",
        "target_votes",
        "frac_target",
        "consensus",
        "max_time_s",
    ]
    row = [
        datetime.datetime.now().isoformat(timespec="milliseconds"),
        mode,
        cue,
        marker,
        success,
        (round(t_success, 3) if t_success is not None else ""),
        total_preds,
        target_votes,
        round(frac_target, 3),
        round(consensus, 3),
        max_time,
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


# =============================================================================
# Game state
# =============================================================================
GAME_MODE = "time"  # "health" or "time"
TIME_LIMIT = CONFIG.countdown

game_speed_backup = CONFIG.game_speed
game_speed = game_speed_backup
game_speed_change = False

FONT_COLOR = (0, 0, 0)
points = 0
death_count = 0

cwdPath = os.getcwd()
user_id = "P004"
gametype = "quicktime"

obstacles = []
marker_names = ["CIRCLE ONSET", "ARROW LEFT ONSET", "ARROW RIGHT ONSET"]


def _draw_moving_background() -> None:
    global x_pos_bg, y_pos_bg, game_speed
    image_width = BG.get_width()
    SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
    SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
    if x_pos_bg <= -image_width:
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        x_pos_bg = 0
    x_pos_bg -= game_speed


def parse_boolish(x: str) -> int:
    if isinstance(x, str):
        v = x.strip().lower()
        return 1 if v in ("1", "true", "t", "yes", "y") else 0
    return int(bool(x))


def success_rate_last_session(user_id):
    file_path = os.path.join(
        os.getcwd(), "data", f"sub-{user_id}", "dino_bar_values_and_jumps.csv"
    )
    rows = []
    with open(file_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # robust timestamp parsing
            ts_str = row["timestamp"]
            try:
                ts = datetime.datetime.fromisoformat(ts_str)
            except Exception:
                # fallback if needed (adjust if your format differs)
                ts = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
            rows.append((ts, parse_boolish(row["jump_success"])))

    if not rows:
        print("No data yet.")
        return

    last_date = rows[-1][0].date()
    session_successes = [s for ts, s in rows if ts.date() == last_date]
    rate = sum(session_successes) / len(session_successes)
    print(
        f"Session {last_date}: {rate:.1%} success over {len(session_successes)} jumps"
    )


# =============================================================================
# Main loop
# =============================================================================
def main() -> None:
    global \
        game_speed, \
        x_pos_bg, \
        y_pos_bg, \
        points, \
        obstacles, \
        death_count, \
        marker_names, \
        SCREEN_WIDTH, \
        SCREEN_HEIGHT
    global lsl_markerstream

    start_udp_thread()

    in_quicktime = False
    run = True
    isoutside = False
    lost = False
    pause = False

    sim_pred: list[int] = []
    session_use_udp: Optional[bool] = None

    num_tasks = CONFIG.num_tasks
    cues = marker_names * num_tasks
    cues_empty = False
    random.shuffle(cues)

    clock = pygame.time.Clock()
    player = Dinosaur(pygame)
    cloud = Cloud(SCREEN_WIDTH, pygame)
    task = Task(pygame)

    x_pos_bg = 0
    y_pos_bg = SCREEN_HEIGHT // 2
    points = 0

    font = pygame.font.Font("freesansbold.ttf", 20)

    if GAME_MODE == "time":
        game_start_time = time.time()

    def save_bar_values_and_jumps(decay_value: float, jump_success: int) -> None:
        file_path = os.path.join(
            os.getcwd(), "data", f"sub-{user_id}", "dino_bar_values_and_jumps.csv"
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "decay_value", "jump_success"])
            writer.writerow([datetime.datetime.now(), decay_value, jump_success])

    def draw_score() -> None:
        nonlocal font
        global points, game_speed, game_speed_change
        points += 1
        if points % 100 == 0 and game_speed_change:
            game_speed += 1

    def unpause_only() -> None:
        nonlocal pause, run
        # resume whatever loop we were in, but DO NOT touch in_quicktime
        pause = False
        run = True

    def unpause() -> None:
        nonlocal pause, run, in_quicktime
        pause = False
        run = True
        in_quicktime = False

    def paused() -> None:
        nonlocal pause
        pause = True
        _font = pygame.font.Font("freesansbold.ttf", 30)
        text = _font.render("Game Paused, Press 'u' to Unpause", True, FONT_COLOR)
        rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        SCREEN.blit(text, rect)
        pygame.display.update()

        while pause:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_u:
                    unpause_only()  # <-- was unpause()

    def quicktime() -> None:
        """Quicktime challenge with decaying bar."""
        nonlocal cues_empty, sim_pred, session_use_udp

        def visualize_marker(cue: str) -> None:
            # draw cue overlay; we will redraw each frame inside the loop
            task.draw(SCREEN, cue)

        def handle_decaying_bar(
            decay_bar: DecayingBar, cue_idx: int
        ) -> tuple[int, int, str | None]:
            nonlocal \
                in_quicktime, \
                player, \
                lost, \
                font, \
                sim_predicted, \
                cue, \
                time_now, \
                sim_pred, \
                session_use_udp
            """Handle the decaying bar and evidence gate logic, returning UDP stats."""
            trial_t0 = time.perf_counter()

            total_cmds = 0
            matching_cmds = 0
            last_cmd_label_local: str | None = None
            last_counted_mono = -1.0
            use_live_udp = bool(session_use_udp)

            # Decide target: replay only if the marker label matches our known set
            rm0 = get_current_marker()
            if isinstance(rm0, str) and rm0 in marker_names:
                mode = "replay"
                target_idx = marker_names.index(rm0)
            else:
                mode = "online"
                target_idx = cue_idx

            # NOTE: success_rate is the argument you pass into quicktime(success_rate)
            # If you want CONFIG-driven percent instead, convert: success_rate = CONFIG.fallback_success_rate_pct / 100.0

            while in_quicktime:
                # --- events / keys ---
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_u:
                            unpause()
                        elif e.key == pygame.K_SPACE:
                            decay_bar.bump()

                now = time.perf_counter()

                # --- read latest prediction (top-1 index) and de-duplicate ---
                cmd_idx, cmd_label, cmd_mono = get_recent_cmd_details()
                new_command = (
                    cmd_idx is not None
                    and cmd_mono is not None
                    and cmd_mono != last_counted_mono
                )
                if new_command:
                    last_counted_mono = float(cmd_mono)
                    last_cmd_label_local = cmd_label
                    if use_live_udp:
                        total_cmds += 1
                        if cmd_idx is not None and marker_names[cmd_idx] == cue:
                            matching_cmds += 1

                # # --- retarget in replay if marker changes mid-trial ---
                # if mode == "replay":
                #     rm_live = get_current_marker()
                #     if isinstance(rm_live, str) and rm_live in marker_names:
                #         target_idx = marker_names.index(rm_live)

                if not use_live_udp:
                    if sim_pred and not sim_predicted:
                        pred = sim_pred.pop(0)
                        if pred == -2:
                            decay_bar.bump()
                            time.sleep(0.1)

                        sim_predicted = True
                        time_now = time.perf_counter()
                    else:
                        if time.perf_counter() - time_now > SLEEP_TIME:
                            sim_predicted = False

                else:
                    if new_command:
                        sim_predicted = True
                        if cmd_idx is not None and marker_names[cmd_idx] == cue:
                            decay_bar.bump()
                            print(
                                f"UDP bump at {now:.2f} for cue {cue} (index {cmd_idx})"
                            )
                            time.sleep(0.1)  # debounce
                        time_now = time.perf_counter()
                    elif time.perf_counter() - time_now > SLEEP_TIME - 0.2:
                        sim_predicted = False

                # --- draw as before ---
                decay_bar.update()
                decay_bar.draw(SCREEN)

                # # --- small on-screen HUD for debugging ---
                # if DEBUG_HUD:
                #     y = 10
                #     def line(s):
                #         nonlocal y
                #         SCREEN.blit(font.render(s, True, (0, 0, 0)), (300, y)); y += 18
                #     line(f"mode={mode}  udp={udp_connected}  stale={no_pred_recently}  age={max(0.0, now-last_pred_t):.2f}s")
                #     line(f"votes={total}  target={tgt}  frac={frac:.2f}  thr={prob_thresh:.2f}  min={MIN_VOTES}")
                #     line(f"bar={decay_bar.value:.1f}")

                # --- success / timeout (your original logic) ---
                if decay_bar.value < decay_bar.height - decay_bar.max_value:
                    save_bar_values_and_jumps(decay_bar.value, 1)
                    player.dino_jump = True
                    player.dino_run = False
                    player.dino_fail = False
                    lsl_markerstream.send_marker("JUMP")
                    print("LSL marker sent: JUMP SUCCESS")
                    unpause()

                if (now - trial_t0) > CONFIG.max_time:
                    save_bar_values_and_jumps(decay_bar.value, 0)
                    lost = True
                    player.dino_fail = True
                    player.dino_run = False
                    player.dino_jump = False
                    lsl_markerstream.send_marker("JUMP FAIL")
                    print("LSL marker sent: JUMP FAIL")
                    unpause()

                clock.tick(30)
                pygame.display.update()

            return total_cmds, matching_cmds, last_cmd_label_local

        nonlocal in_quicktime
        in_quicktime = True
        time_now = 0.0

        cue = cues.pop(0)
        if cues == []:
            cues_empty = True
            print("No more cues left, ending quicktime.")

        # Send the cue ONCE (LSL), then draw it; drawing repeats each frame in the loop
        lsl_markerstream.send_marker(cue)
        print(f"LSL marker sent: {cue}")
        visualize_marker(cue)
        mode_text = "live UDP" if session_use_udp else "simulation"
        print(f"Quicktime mode: {mode_text}")
        sim_predicted = False
        height = 120
        max_value = height * PROB_THRESHOLD
        decay_bar = DecayingBar(
            x=SCREEN_WIDTH - 175,
            y=SCREEN_HEIGHT - 300,
            width=50,
            height=height,
            color=(240, 128, 128),
            max_value=max_value,
            decay_rate=0.4,
            bump_rate=18,
        )  # those are magic numbers, do not change, its based on how much needed to match bump and accuracy
        total_cmds, matching_cmds, last_cmd_label = handle_decaying_bar(
            decay_bar, marker_names.index(cue)
        )

        if session_use_udp:
            if total_cmds:
                accuracy_pct = (matching_cmds / total_cmds) * 100.0
                accuracy_text = f"{accuracy_pct:.1f}% ({matching_cmds}/{total_cmds})"
            else:
                accuracy_text = "no UDP commands"
        else:
            accuracy_text = "simulation mode"

        mode_text = "live UDP" if session_use_udp else "simulation"
        last_label_text = last_cmd_label or "none"
        if not session_use_udp and last_cmd_label:
            last_label_text = f"{last_label_text} (ignored)"
        print(
            f"Quicktime summary ({mode_text}) - accuracy {accuracy_text}, last UDP command: {last_label_text}"
        )

        sim_pred = []

    hit_time: datetime.datetime | None = None

    while run:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_p:
                run = False
                paused()

        SCREEN.fill((255, 255, 255) if CONFIG.day_time else (0, 0, 0))
        _draw_moving_background()
        player.draw(SCREEN)
        player.update(None)

        if len(obstacles) == 0:
            r = random.randint(0, 2)
            obstacles.append(
                SmallCactus(SCREEN_WIDTH, pygame)
                if r == 0
                else LargeCactus(SCREEN_WIDTH, pygame)
                if r == 1
                else Bird(SCREEN_WIDTH, pygame)
            )

        cloud.draw(SCREEN)
        cloud.update(SCREEN_WIDTH)
        draw_score()

        if GAME_MODE == "health":
            SCREEN.blit(
                font.render(f"Health: {player.health}", True, (240, 128, 128)), (10, 40)
            )
        else:
            time_remaining = max(0, TIME_LIMIT - (time.time() - game_start_time))
            SCREEN.blit(
                font.render(f"Time: {int(time_remaining)}s", True, (240, 128, 128)),
                (10, 40),
            )
            SCREEN.blit(
                font.render(f"Health: {player.health}", True, (240, 128, 128)), (10, 70)
            )
            if time_remaining <= 0 or cues_empty:
                death_count += 1
                menu(death_count)
                return

        for obstacle in obstacles[:]:
            obstacle.draw(SCREEN)
            isoutside = obstacle.update(game_speed)
            if isoutside:
                obstacles.pop()

            if (
                gametype == "quicktime"
                and not lost
                and not player.invulnerable
                and not cues_empty
            ):
                if (
                    obstacle.ideal_jump - player.dino_rect.x < 150
                ) and not player.dino_jump:
                    run = False
                    # paused()

                    in_quicktime = True
                    player.dino_run = False
                    player.dino_jump = False
                    player.dino_fail = False
                    num_trials = int(CONFIG.max_time // (SLEEP_TIME))
                    print(f"Starting quicktime with {num_trials} trials")
                    print(f"sleep_time: {SLEEP_TIME}")
                    sim_pred = []
                    listener_bound, has_data, _, _ = get_udp_listener_status()
                    _, last_payload_ts, last_payload_valid, _ = (
                        get_udp_payload_snapshot()
                    )
                    has_recent_valid = (
                        last_payload_valid
                        and last_payload_ts > 0
                        and (time.time() - last_payload_ts) <= STALE_PRED_S
                    )
                    if session_use_udp is None:
                        session_use_udp = listener_bound and has_data
                        mode_text = (
                            "live UDP commands"
                            if session_use_udp
                            else "simulated quicktime predictions"
                        )
                        print(f"Session input mode locked to {mode_text}.")
                        if listener_bound and not has_data and has_recent_valid:
                            print(
                                "Recent UDP packet detected but marked invalid; gameplay will stay in simulation until a recognized command arrives."
                            )
                    elif session_use_udp is False and listener_bound and has_data:
                        print(
                            "Live UDP commands detected but session is locked to simulation; restart the game to switch modes."
                        )
                    if session_use_udp:
                        sim_pred = []
                    else:
                        sim_pred = new_session_simulation(
                            success_rate=CONFIG.success_rate,
                            num_trials=num_trials,
                        )

                    quicktime()

            if not player.invulnerable and player.dino_rect.colliderect(obstacle.rect):
                player.health -= 1
                player.invulnerable = True
                hit_time = datetime.datetime.now()
                if gametype == "quicktime":
                    lost = False
                pygame.time.delay(350)
                if GAME_MODE == "health" and player.health == 0:
                    pygame.time.delay(2000)
                    death_count += 1
                    menu(death_count)
                    return

        if player.invulnerable and isinstance(hit_time, datetime.datetime):
            if datetime.datetime.now() - hit_time > datetime.timedelta(seconds=2):
                player.invulnerable = False

        pygame.display.update()
        clock.tick(30)


# =============================================================================
# Menu
# =============================================================================
def menu(initial_death_count: int) -> None:
    global points, FONT_COLOR, game_speed, user_id, gametype
    global lsl_markerstream, time_limit, death_count
    global SCREEN_WIDTH, SCREEN_HEIGHT

    death_count = initial_death_count
    run = True
    font = pygame.font.Font("freesansbold.ttf", 30)
    number_of_games = 3

    start_udp_thread()

    def get_user_id() -> str:
        global user_id, SCREEN_WIDTH, SCREEN_HEIGHT, survey_language, current_session
        user_id = ""
        language = "eng"
        session = 1
        input_active = True
        small_font = pygame.font.Font("freesansbold.ttf", 18)
        tiny_font = pygame.font.Font("freesansbold.ttf", 14)
        active_input = "user"
        _, _, default_port, _ = get_udp_listener_status()
        port_input = str(default_port)
        port_feedback = ""
        port_feedback_color = (200, 200, 200)
        port_feedback_time = 0.0

        def truncate_payload(text: str, max_len: int = 60) -> str:
            return text if len(text) <= max_len else text[: max_len - 3] + "..."

        def parse_port_input() -> tuple[bool, int]:
            nonlocal port_feedback, port_feedback_color, port_feedback_time
            try:
                port_val = int(port_input)
            except ValueError:
                port_feedback = "Enter a numeric port."
                port_feedback_color = (220, 100, 100)
                port_feedback_time = time.time()
                return False, 0
            if not (1 <= port_val <= 65535):
                port_feedback = "Port must be 1-65535."
                port_feedback_color = (220, 100, 100)
                port_feedback_time = time.time()
                return False, 0
            return True, port_val

        def apply_port_change() -> bool:
            nonlocal port_input, port_feedback, port_feedback_color, port_feedback_time
            valid, port_val = parse_port_input()
            if not valid:
                return False
            if configure_udp_endpoint(port_val):
                port_feedback = f"Port set to {port_val}"
                port_feedback_color = (80, 200, 160)
            else:
                port_feedback = "Stop listener before changing port."
                port_feedback_color = (220, 180, 80)
            port_feedback_time = time.time()
            port_input = str(port_val)
            return True

        def attempt_start_game() -> None:
            nonlocal \
                input_active, \
                port_input, \
                port_feedback, \
                port_feedback_color, \
                port_feedback_time
            global survey_language, current_session
            if not user_id:
                return
            if not is_udp_thread_running():
                valid, port_val = parse_port_input()
                if not valid:
                    return
                configure_udp_endpoint(port_val)
                port_input = str(port_val)
                start_udp_thread()
                port_feedback = f"Listening on port {port_val}"
                port_feedback_color = (80, 200, 160)
                port_feedback_time = time.time()
            SCREEN_WIDTH = SCREEN.get_width()
            SCREEN_HEIGHT = SCREEN.get_height()
            survey_language = language
            current_session = session
            set_runtime_context(user_id, os.getcwd(), survey_language)
            input_active = False

        def udp_config_screen() -> None:
            nonlocal port_input, port_feedback, port_feedback_color, port_feedback_time
            config_active = True
            udp_active_input = "port"

            while config_active:
                SCREEN_HEIGHT = SCREEN.get_height()
                SCREEN_WIDTH = SCREEN.get_width()

                SCREEN.fill((12, 14, 20))

                back_button_rect = pygame.Rect(20, 20, 160, 50)
                pygame.draw.rect(
                    SCREEN, (60, 90, 150), back_button_rect, border_radius=12
                )
                SCREEN.blit(
                    small_font.render("Back", True, (255, 255, 255)),
                    (back_button_rect.x + 54, back_button_rect.y + 14),
                )

                panel_width = min(680, SCREEN_WIDTH - 80)
                panel_height = min(520, SCREEN_HEIGHT - 80)
                panel_x = (SCREEN_WIDTH - panel_width) // 2
                panel_y = (SCREEN_HEIGHT - panel_height) // 2
                udp_panel_rect = pygame.Rect(
                    panel_x, panel_y, panel_width, panel_height
                )

                content_x = udp_panel_rect.x + 24
                content_width = udp_panel_rect.width - 48
                header_height = 68

                status_rect = pygame.Rect(
                    content_x, udp_panel_rect.y + header_height + 16, content_width, 76
                )
                command_rect = pygame.Rect(
                    content_x, status_rect.bottom + 14, content_width, 70
                )
                port_rect = pygame.Rect(
                    content_x, command_rect.bottom + 16, content_width, 150
                )
                marker_rect_height = 86
                marker_rect = pygame.Rect(
                    content_x,
                    udp_panel_rect.bottom - 24 - marker_rect_height,
                    content_width,
                    marker_rect_height,
                )
                log_rect = pygame.Rect(
                    content_x,
                    port_rect.bottom + 16,
                    content_width,
                    marker_rect.y - 24 - (port_rect.bottom + 16),
                )
                if log_rect.height < 48:
                    log_rect.height = 0

                port_input_rect = pygame.Rect(
                    port_rect.x + 12, port_rect.y + 46, 150, 48
                )
                apply_port_rect = pygame.Rect(
                    port_input_rect.right + 12, port_input_rect.y, 170, 48
                )
                listener_button_rect = pygame.Rect(
                    port_rect.x + 12,
                    port_input_rect.bottom + 12,
                    content_width - 24,
                    48,
                )

                listener_bound, udp_has_data, listener_port, packet_log = (
                    get_udp_listener_status()
                )
                listener_running = is_udp_thread_running()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            config_active = False
                        elif event.key == pygame.K_TAB:
                            udp_active_input = "port"
                        elif event.key == pygame.K_RETURN:
                            if udp_active_input == "port" and not listener_running:
                                apply_port_change()
                            else:
                                if listener_running:
                                    if stop_udp_thread():
                                        port_feedback = "Listener stopped."
                                        port_feedback_color = (200, 180, 80)
                                        port_feedback_time = time.time()
                                        listener_running = False
                                else:
                                    valid, port_val = parse_port_input()
                                    if valid:
                                        configure_udp_endpoint(port_val)
                                        port_input = str(port_val)
                                        start_udp_thread()
                                        port_feedback = f"Listening on port {port_val}"
                                        port_feedback_color = (80, 200, 160)
                                        port_feedback_time = time.time()
                                        listener_running = True
                        elif udp_active_input == "port":
                            if event.key == pygame.K_BACKSPACE:
                                port_input = port_input[:-1]
                            elif (
                                len(event.unicode) == 1
                                and event.unicode.isdigit()
                                and len(port_input) < 5
                            ):
                                port_input += event.unicode
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        if back_button_rect.collidepoint(x, y):
                            config_active = False
                        elif port_input_rect.collidepoint(x, y):
                            udp_active_input = "port"
                        elif apply_port_rect.collidepoint(x, y):
                            if listener_running:
                                port_feedback = "Stop listener to apply."
                                port_feedback_color = (220, 180, 80)
                                port_feedback_time = time.time()
                            else:
                                apply_port_change()
                        elif listener_button_rect.collidepoint(x, y):
                            if listener_running:
                                if stop_udp_thread():
                                    port_feedback = "Listener stopped."
                                    port_feedback_color = (200, 180, 80)
                                    port_feedback_time = time.time()
                                    listener_running = False
                            else:
                                valid, port_val = parse_port_input()
                                if valid:
                                    configure_udp_endpoint(port_val)
                                    port_input = str(port_val)
                                    start_udp_thread()
                                    port_feedback = f"Listening on port {port_val}"
                                    port_feedback_color = (80, 200, 160)
                                    port_feedback_time = time.time()
                                    listener_running = True

                listener_bound, udp_has_data, listener_port, packet_log = (
                    get_udp_listener_status()
                )
                listener_running = is_udp_thread_running()
                payload_text, payload_ts, payload_valid, payload_label = (
                    get_udp_payload_snapshot()
                )

                pygame.draw.rect(SCREEN, (33, 35, 46), udp_panel_rect, border_radius=20)
                pygame.draw.rect(
                    SCREEN, (80, 100, 140), udp_panel_rect, 2, border_radius=20
                )
                header_rect = pygame.Rect(
                    udp_panel_rect.x,
                    udp_panel_rect.y,
                    udp_panel_rect.width,
                    header_height,
                )
                pygame.draw.rect(
                    SCREEN,
                    (60, 90, 150),
                    header_rect,
                    border_top_left_radius=20,
                    border_top_right_radius=20,
                )
                SCREEN.blit(
                    small_font.render("UDP Configuration", True, (255, 255, 255)),
                    (header_rect.x + 22, header_rect.y + 18),
                )

                blink_on = (pygame.time.get_ticks() // 400) % 2 == 0
                pygame.draw.rect(SCREEN, (46, 52, 66), status_rect, border_radius=16)
                indicator_center = (
                    status_rect.x + 24,
                    status_rect.y + status_rect.height // 2,
                )
                if listener_bound and udp_has_data:
                    indicator_color = (60, 200, 160)
                    status_line = "Receiving commands"
                elif listener_bound:
                    indicator_color = (220, 180, 80)
                    status_line = "Listening..."
                elif listener_running:
                    indicator_color = (200, 140, 60) if blink_on else (150, 100, 40)
                    status_line = "Binding to port..."
                else:
                    indicator_color = (220, 70, 70)
                    status_line = "Listener stopped"
                pygame.draw.circle(SCREEN, indicator_color, indicator_center, 12)
                SCREEN.blit(
                    tiny_font.render("Status", True, (220, 220, 220)),
                    (indicator_center[0] + 20, status_rect.y + 12),
                )
                SCREEN.blit(
                    tiny_font.render(status_line, True, (200, 200, 200)),
                    (indicator_center[0] + 20, status_rect.y + 38),
                )
                if listener_bound:
                    port_status_line = f"Bound to 127.0.0.1:{listener_port}"
                else:
                    state = "running" if listener_running else "stopped"
                    port_status_line = (
                        f"Configured port {port_input or listener_port} ({state})"
                    )
                SCREEN.blit(
                    tiny_font.render(port_status_line, True, (180, 200, 220)),
                    (indicator_center[0] + 20, status_rect.y + 56),
                )

                if payload_ts > 0:
                    age = time.time() - payload_ts
                    if age <= 10:
                        pygame.draw.rect(
                            SCREEN, (45, 49, 63), command_rect, border_radius=16
                        )
                        SCREEN.blit(
                            small_font.render("Last command", True, (220, 220, 220)),
                            (command_rect.x + 12, command_rect.y + 10),
                        )
                        label_text = payload_label or truncate_payload(payload_text, 44)
                        validity = "valid" if payload_valid else "unknown"
                        val_color = (80, 200, 160) if payload_valid else (220, 180, 80)
                        SCREEN.blit(
                            tiny_font.render(label_text, True, (200, 200, 200)),
                            (command_rect.x + 12, command_rect.y + 30),
                        )
                        SCREEN.blit(
                            tiny_font.render(
                                f"{validity} command - {age:.1f}s", True, val_color
                            ),
                            (command_rect.x + 12, command_rect.y + 46),
                        )
                    else:
                        pygame.draw.rect(
                            SCREEN, (55, 30, 30), command_rect, border_radius=16
                        )
                        SCREEN.blit(
                            small_font.render("Last command", True, (255, 200, 200)),
                            (command_rect.x + 12, command_rect.y + 10),
                        )
                        SCREEN.blit(
                            tiny_font.render(
                                "No recent commands (>10s)", True, (255, 180, 180)
                            ),
                            (command_rect.x + 12, command_rect.y + 34),
                        )
                        SCREEN.blit(
                            tiny_font.render(
                                "Check your sender.", True, (220, 160, 160)
                            ),
                            (command_rect.x + 12, command_rect.y + 50),
                        )
                else:
                    pygame.draw.rect(
                        SCREEN, (45, 49, 63), command_rect, border_radius=16
                    )
                    SCREEN.blit(
                        small_font.render("Last command", True, (220, 220, 220)),
                        (command_rect.x + 12, command_rect.y + 10),
                    )
                    SCREEN.blit(
                        tiny_font.render("No packets yet.", True, (130, 130, 130)),
                        (command_rect.x + 12, command_rect.y + 32),
                    )

                pygame.draw.rect(SCREEN, (45, 49, 63), port_rect, border_radius=16)
                SCREEN.blit(
                    small_font.render("UDP Port", True, (220, 220, 220)),
                    (port_rect.x + 12, port_rect.y + 12),
                )
                hint_text = (
                    "Stop listener to edit."
                    if listener_running
                    else "Pick a port, then start."
                )
                SCREEN.blit(
                    tiny_font.render(hint_text, True, (170, 190, 210)),
                    (port_rect.x + 14, port_rect.y + 36),
                )

                border_color = (
                    (90, 180, 140)
                    if udp_active_input == "port" and not listener_running
                    else (90, 90, 90)
                )
                pygame.draw.rect(
                    SCREEN, border_color, port_input_rect, border_radius=12
                )
                pygame.draw.rect(
                    SCREEN,
                    (15, 15, 15),
                    port_input_rect.inflate(-6, -6),
                    border_radius=10,
                )
                SCREEN.blit(
                    small_font.render(port_input or "5005", True, (255, 255, 255)),
                    (port_input_rect.x + 12, port_input_rect.y + 12),
                )

                apply_color = (80, 140, 200) if not listener_running else (70, 70, 70)
                pygame.draw.rect(SCREEN, apply_color, apply_port_rect, border_radius=12)
                SCREEN.blit(
                    tiny_font.render("Apply", True, (255, 255, 255)),
                    (apply_port_rect.x + 64, apply_port_rect.y + 14),
                )

                listener_btn_color = (
                    (80, 200, 140) if not listener_running else (200, 120, 80)
                )
                pygame.draw.rect(
                    SCREEN, listener_btn_color, listener_button_rect, border_radius=14
                )
                listen_label = (
                    "Start Listener" if not listener_running else "Stop Listener"
                )
                SCREEN.blit(
                    small_font.render(listen_label, True, (255, 255, 255)),
                    (listener_button_rect.x + 18, listener_button_rect.y + 14),
                )

                if port_feedback and (time.time() - port_feedback_time) < 4.0:
                    SCREEN.blit(
                        tiny_font.render(port_feedback, True, port_feedback_color),
                        (port_rect.x + 14, listener_button_rect.bottom + 6),
                    )

                if log_rect.height >= 60:
                    pygame.draw.rect(SCREEN, (42, 45, 58), log_rect, border_radius=14)
                    SCREEN.blit(
                        small_font.render("Recent packets", True, (220, 220, 220)),
                        (log_rect.x + 12, log_rect.y + 10),
                    )
                    now_ts = time.time()
                    max_lines = max(1, (log_rect.height - 40) // 18)
                    entries = list(reversed(packet_log))[:max_lines]
                    line_y = log_rect.y + 36
                    if entries:
                        for payload_text_i, ts_i, recognized_i, label_i in entries:
                            age_i = now_ts - ts_i
                            flag = "OK" if recognized_i else "??"
                            label_display = label_i or truncate_payload(
                                payload_text_i, 44
                            )
                            line = f"[{flag}] {age_i:.1f}s {label_display}"
                            SCREEN.blit(
                                tiny_font.render(line, True, (200, 200, 200)),
                                (log_rect.x + 12, line_y),
                            )
                            line_y += 18
                    else:
                        SCREEN.blit(
                            tiny_font.render(
                                "No packets recorded.", True, (130, 130, 130)
                            ),
                            (log_rect.x + 12, line_y),
                        )

                pygame.draw.rect(SCREEN, (42, 45, 58), marker_rect, border_radius=14)
                SCREEN.blit(
                    small_font.render("Marker stream", True, (220, 220, 220)),
                    (marker_rect.x + 12, marker_rect.y + 10),
                )
                marker_info = ""
                if "lsl_markerstream" in globals():
                    marker_info = f"LSL: {lsl_markerstream.name} (UID {lsl_markerstream.marker_id})"
                SCREEN.blit(
                    tiny_font.render(
                        marker_info or "LSL stream initialising...",
                        True,
                        (200, 200, 200),
                    ),
                    (marker_rect.x + 14, marker_rect.y + 34),
                )
                SCREEN.blit(
                    tiny_font.render(
                        "Markers: " + " | ".join(marker_names), True, (200, 200, 200)
                    ),
                    (marker_rect.x + 14, marker_rect.y + 52),
                )

                pygame.display.update()

            pygame.event.clear()

        while input_active:
            SCREEN_HEIGHT = SCREEN.get_height()
            SCREEN_WIDTH = SCREEN.get_width()

            lang_button_rect = pygame.Rect(10, SCREEN_HEIGHT // 2 + 130, 300, 42)
            sess_minus_rect = pygame.Rect(10, SCREEN_HEIGHT // 2 + 220, 48, 42)
            sess_plus_rect = pygame.Rect(72, SCREEN_HEIGHT // 2 + 220, 48, 42)
            start_game_button = pygame.Rect(10, SCREEN_HEIGHT // 2 + 300, 340, 82)

            input_rect = pygame.Rect(10, SCREEN_HEIGHT // 2 - 50, 220, 88)

            summary_width = 280
            summary_height = 220
            summary_x = SCREEN_WIDTH - summary_width - 30
            summary_y = SCREEN_HEIGHT // 2 - summary_height - 20
            udp_summary_rect = pygame.Rect(
                summary_x, summary_y, summary_width, summary_height
            )
            udp_open_rect = pygame.Rect(
                summary_x, udp_summary_rect.bottom + 16, summary_width, 56
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        active_input = "user"
                    elif event.key == pygame.K_RETURN:
                        attempt_start_game()
                    elif (
                        event.key == pygame.K_BACKSPACE
                        and active_input == "user"
                        and user_id
                    ):
                        user_id = user_id[:-1]
                    elif (
                        active_input == "user"
                        and len(event.unicode) == 1
                        and event.unicode.isprintable()
                    ):
                        user_id += event.unicode
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if input_rect.collidepoint(x, y):
                        active_input = "user"
                    elif lang_button_rect.collidepoint(x, y):
                        language = "de" if language == "eng" else "eng"
                    elif sess_minus_rect.collidepoint(x, y):
                        session = max(1, session - 1)
                    elif sess_plus_rect.collidepoint(x, y):
                        session = min(999, session + 1)
                    elif start_game_button.collidepoint(x, y):
                        attempt_start_game()
                    elif udp_open_rect.collidepoint(x, y):
                        udp_config_screen()

            SCREEN.fill((0, 0, 0))
            label = (
                "Enter your user ID:"
                if language == "eng"
                else "Geben Sie Ihre Benutzer-ID ein:"
            )
            SCREEN.blit(
                font.render(label, True, (255, 255, 255)),
                (13, SCREEN_HEIGHT // 2 - 110),
            )

            input_border_color = (
                (90, 180, 140) if active_input == "user" else (60, 60, 60)
            )
            pygame.draw.rect(SCREEN, input_border_color, input_rect, border_radius=12)
            pygame.draw.rect(
                SCREEN, (25, 25, 25), input_rect.inflate(-6, -6), border_radius=10
            )
            placeholder = "type.." if language == "eng" else "tippen.."
            shown = user_id if user_id else placeholder
            color = (255, 255, 255) if user_id else (160, 160, 160)
            SCREEN.blit(
                font.render(shown, True, color), (input_rect.x + 10, input_rect.y + 28)
            )

            lang_label = "English" if language == "eng" else "Deutsch"
            lang_text = (
                "Selected Language: " if language == "eng" else "Ausgewählte Sprache: "
            )
            SCREEN.blit(
                font.render(f"{lang_text}{lang_label}", True, (255, 255, 255)),
                (10, SCREEN_HEIGHT // 2 + 90),
            )
            pygame.draw.rect(SCREEN, (60, 50, 50), lang_button_rect, border_radius=8)
            lang_btn_text = "Sprache ändern" if language == "eng" else "Change Language"
            SCREEN.blit(
                font.render(lang_btn_text, True, (255, 255, 255)),
                (lang_button_rect.x + 16, lang_button_rect.y + 6),
            )

            SCREEN.blit(
                font.render(f"Session: {session}", True, (255, 255, 255)),
                (10, SCREEN_HEIGHT // 2 + 180),
            )
            pygame.draw.rect(SCREEN, (80, 80, 80), sess_minus_rect, border_radius=8)
            pygame.draw.rect(SCREEN, (80, 80, 80), sess_plus_rect, border_radius=8)
            SCREEN.blit(
                font.render("-", True, (255, 255, 255)),
                (sess_minus_rect.x + 18, sess_minus_rect.y + 6),
            )
            SCREEN.blit(
                font.render("+", True, (255, 255, 255)),
                (sess_plus_rect.x + 16, sess_plus_rect.y + 6),
            )

            start_btn_color = (60, 200, 160) if user_id else (80, 80, 80)
            pygame.draw.rect(
                SCREEN, start_btn_color, start_game_button, border_radius=16
            )
            start_text = "Start Dino Game" if language == "eng" else "Starte Dino Spiel"
            SCREEN.blit(
                font.render(start_text, True, (255, 255, 255)),
                (start_game_button.x + 28, start_game_button.y + 28),
            )

            listener_bound, udp_has_data, listener_port, _ = get_udp_listener_status()
            listener_running = is_udp_thread_running()
            payload_text, payload_ts, payload_valid, payload_label = (
                get_udp_payload_snapshot()
            )
            pygame.draw.rect(SCREEN, (28, 30, 40), udp_summary_rect, border_radius=18)
            pygame.draw.rect(
                SCREEN, (70, 90, 140), udp_summary_rect, 2, border_radius=18
            )

            status_indicator_center = (udp_summary_rect.x + 24, udp_summary_rect.y + 36)
            blink_on = (pygame.time.get_ticks() // 400) % 2 == 0
            if listener_bound and udp_has_data:
                indicator_color = (60, 200, 160)
                summary_status = "Receiving commands"
            elif listener_bound:
                indicator_color = (220, 180, 80)
                summary_status = "Listening..."
            elif listener_running:
                indicator_color = (200, 140, 60) if blink_on else (150, 100, 40)
                summary_status = "Starting..."
            else:
                indicator_color = (220, 70, 70)
                summary_status = "Stopped"
            pygame.draw.circle(SCREEN, indicator_color, status_indicator_center, 10)
            SCREEN.blit(
                small_font.render("UDP status", True, (220, 220, 220)),
                (status_indicator_center[0] + 18, udp_summary_rect.y + 14),
            )
            SCREEN.blit(
                tiny_font.render(summary_status, True, (200, 200, 200)),
                (status_indicator_center[0] + 18, udp_summary_rect.y + 36),
            )

            configured_port = port_input or str(listener_port)
            port_line = (
                f"Port {listener_port} (running)"
                if listener_running
                else f"Port {configured_port}"
            )
            SCREEN.blit(
                tiny_font.render(port_line, True, (190, 200, 220)),
                (udp_summary_rect.x + 18, udp_summary_rect.y + 68),
            )

            if listener_bound and payload_ts > 0:
                age = time.time() - payload_ts
                if age <= 10:
                    label_text = payload_label or truncate_payload(payload_text, 42)
                    validity = "valid" if payload_valid else "unknown"
                    val_color = (80, 200, 160) if payload_valid else (220, 180, 80)
                    SCREEN.blit(
                        tiny_font.render("Last command", True, (200, 200, 200)),
                        (udp_summary_rect.x + 18, udp_summary_rect.y + 96),
                    )
                    SCREEN.blit(
                        tiny_font.render(label_text, True, (200, 200, 200)),
                        (udp_summary_rect.x + 18, udp_summary_rect.y + 114),
                    )
                    SCREEN.blit(
                        tiny_font.render(f"{validity} - {age:.1f}s", True, val_color),
                        (udp_summary_rect.x + 18, udp_summary_rect.y + 132),
                    )
                else:
                    SCREEN.blit(
                        tiny_font.render(
                            "Commands stale (>10s)", True, (220, 160, 160)
                        ),
                        (udp_summary_rect.x + 18, udp_summary_rect.y + 96),
                    )
                    SCREEN.blit(
                        tiny_font.render(
                            "Open setup for details", True, (200, 200, 200)
                        ),
                        (udp_summary_rect.x + 18, udp_summary_rect.y + 114),
                    )
            elif listener_bound:
                SCREEN.blit(
                    tiny_font.render("Waiting for commands...", True, (160, 160, 160)),
                    (udp_summary_rect.x + 18, udp_summary_rect.y + 96),
                )
            else:
                SCREEN.blit(
                    tiny_font.render("No packets yet", True, (160, 160, 160)),
                    (udp_summary_rect.x + 18, udp_summary_rect.y + 96),
                )

            pygame.draw.rect(SCREEN, (70, 120, 200), udp_open_rect, border_radius=14)
            SCREEN.blit(
                small_font.render("Open UDP setup", True, (255, 255, 255)),
                (udp_open_rect.x + 28, udp_open_rect.y + 14),
            )

            pygame.display.update()

        return user_id

    high_score_table = HighScoreTable("high_scores.txt")

    if death_count == 0:
        lsl_markerstream = LSLMarkerStream("MyDinoGameMarkerStream", "myuid123456")
        uid = get_user_id()
        print(uid)

    if 0 < death_count < number_of_games:
        print(f"Game {death_count} of {number_of_games} completed")
        lsl_markerstream.send_marker(f"GAME_START_{death_count}")
        SCREEN.fill((255, 255, 255))
        high_score_table.add_score(user_id, points)
        high_score_table.display(SCREEN, font)
        pygame.display.flip()

        success_rate_last_session(user_id)

        waiting_for_input = True
        while waiting_for_input:
            continue_rect = pygame.Rect(10, SCREEN_HEIGHT // 2 + 300, 360, 80)
            pygame.draw.rect(SCREEN, (60, 200, 160), continue_rect, border_radius=16)
            ctext = (
                "Click to continue"
                if "survey_language" in globals() and survey_language == "eng"
                else "Klicken Sie hier, um fortzufahren"
            )
            SCREEN.blit(
                font.render(ctext, True, (255, 255, 255)),
                (continue_rect.x + 20, continue_rect.y + 25),
            )
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN:
                    waiting_for_input = False

        SCREEN.fill((255, 255, 255))
        pygame.display.flip()

        print("Starting PyQt survey...")
        show_survey_pyqt_and_save()
        global game_speed
        game_speed = game_speed_backup

        SCREEN.fill((255, 255, 255))
        font = pygame.font.Font("freesansbold.ttf", 30)
        for i in range(5, 0, -1):
            SCREEN.fill((255, 255, 255))
            SCREEN.blit(
                font.render(f"Next game starts in {i} seconds...", True, (0, 0, 0)),
                (10, SCREEN_HEIGHT // 2),
            )
            pygame.display.flip()
            pygame.time.delay(1000)

        main()
        return

    if death_count >= number_of_games:
        SCREEN.fill((255, 255, 255))
        SCREEN.blit(
            font.render(
                f"End of game - Completed {number_of_games} games", True, (0, 0, 0)
            ),
            (10, SCREEN_HEIGHT // 2),
        )
        pygame.display.flip()

        waiting_for_input = True
        while waiting_for_input:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN:
                    waiting_for_input = False
                    pygame.quit()
                    quit()

    while run:
        global FONT_COLOR
        FONT_COLOR = (240, 128, 128)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                pygame.quit()
                exit()
            else:
                if death_count == 0:
                    main()
        pygame.display.update()


if __name__ == "__main__":
    menu(initial_death_count=0)
