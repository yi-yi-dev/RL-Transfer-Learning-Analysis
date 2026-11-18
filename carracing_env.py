import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d import car_racing

# Values chosen with pistas_interactivo.ipynb (case 1 = original, case 2 = new).
ORIGINAL = dict(
    TRACK_DETAIL_STEP=car_racing.TRACK_DETAIL_STEP * 1,
    TRACK_TURN_RATE=0.7,
    TRACK_WIDTH=car_racing.TRACK_WIDTH * 2.2,
    TRACK_RAD=car_racing.TRACK_RAD * 0.8,
)
NEW = dict(
    TRACK_DETAIL_STEP=car_racing.TRACK_DETAIL_STEP * 0.2,
    TRACK_TURN_RATE=0.1,
    TRACK_WIDTH=car_racing.TRACK_WIDTH * 0.5,
    TRACK_RAD=car_racing.TRACK_RAD * 0.5,
)
TRACKS = {
    "CarRacingOriginal-v0": [ORIGINAL],
    "CarRacingNew-v0": [NEW],
    "CarRacingMix-v0": [ORIGINAL, NEW],
}

# Grid of track distributions: TRACK_WIDTH multiplier x TRACK_TURN_RATE, with a fixed radius and step.
GRID_WIDTHS = np.linspace(0.5, 2.5, 4)
GRID_TURN_RATES = np.linspace(0.1, 0.4, 4)


def grid_env_id(width_index, turn_index):
    return f"CarRacingGrid-w{width_index}-t{turn_index}-v0"


for i, width in enumerate(GRID_WIDTHS):
    for j, turn_rate in enumerate(GRID_TURN_RATES):
        TRACKS[grid_env_id(i, j)] = [dict(
            TRACK_DETAIL_STEP=car_racing.TRACK_DETAIL_STEP * 0.4,
            TRACK_TURN_RATE=float(turn_rate),
            TRACK_WIDTH=car_racing.TRACK_WIDTH * float(width),
            TRACK_RAD=car_racing.TRACK_RAD * 0.5,
        )]


class CustomTrackCarRacing(car_racing.CarRacing):
    def __init__(self, track_options, **kwargs):
        super().__init__(**kwargs)
        self.track_options = track_options

    def _create_track(self):
        track_params = self.track_options[self.np_random.integers(len(self.track_options))]

        # The track generator reads module constants, so we change them only while it runs.
        saved = {name: getattr(car_racing, name) for name in track_params}
        vars(car_racing).update(track_params)
        try:
            return super()._create_track()
        finally:
            vars(car_racing).update(saved)


for env_id, track_options in TRACKS.items():
    gym.register(env_id, entry_point=CustomTrackCarRacing, max_episode_steps=1000, kwargs=dict(track_options=track_options))


class OffTrackTruncation(gym.Wrapper):
    """Ends the episode when the car has all its wheels off the road for `patience` frames in a row."""

    def __init__(self, env, patience=100):
        super().__init__(env)
        self.patience = patience

    def reset(self, **kwargs):
        self.off_track_frames = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        on_road = any(wheel.tiles for wheel in self.env.unwrapped.car.wheels)
        self.off_track_frames = 0 if on_road else self.off_track_frames + 1
        if self.off_track_frames >= self.patience:
            return obs, reward, terminated, True, info | {"off_track": True}
        return obs, reward, terminated, truncated, info


END_REASONS = ["lap", "off_map", "off_track", "time_limit"]

# Info keys that MultiAttempt adds at the end of an episode. Each "attempt_*" value has one item per attempt.
EPISODE_INFO = ["track_seed", "attempt_returns", "attempt_lengths", "attempt_tiles", "attempt_ends"]


def end_reason(terminated, info):
    if terminated:
        return END_REASONS.index("lap" if info.get("lap_finished") else "off_map")
    return END_REASONS.index("off_track" if info.get("off_track") else "time_limit")


class MultiAttempt(gym.Wrapper):
    """One episode is `attempts` laps on the same track. The car goes back to the start after each attempt.

    With `feedback`, the observation also holds a summary of the attempts so far.
    """

    def __init__(self, env, attempts, feedback=False):
        super().__init__(env)
        self.attempts = attempts
        self.feedback = feedback
        self.rng = np.random.default_rng()
        if feedback:
            self.observation_space = gym.spaces.Dict(
                image=env.observation_space,
                feedback=gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            )

    def observe(self, obs):
        if not self.feedback:
            return obs

        previous = self.returns[-2] if len(self.returns) > 1 else 0.0
        best = max(self.returns[:-1], default=0.0)
        attempt = (len(self.returns) - 1) / (self.attempts - 1)
        # 1000 is the reward for a full lap.
        return dict(image=obs, feedback=np.array([previous / 1000, best / 1000, attempt], dtype=np.float32))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.track_seed = int(self.rng.integers(2**31))
        self.returns, self.lengths, self.tiles, self.ends = [0.0], [0], [], []
        obs, info = self.env.reset(seed=self.track_seed, options=options)
        return self.observe(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.returns[-1] += reward
        self.lengths[-1] += 1
        if not (terminated or truncated):
            return self.observe(obs), reward, False, False, info

        track = self.env.unwrapped
        self.tiles.append(track.tile_visited_count / len(track.track))
        self.ends.append(end_reason(terminated, info))
        if len(self.returns) == self.attempts:
            episode_info = dict(
                track_seed=self.track_seed,
                attempt_returns=np.array(self.returns),
                attempt_lengths=np.array(self.lengths),
                attempt_tiles=np.array(self.tiles),
                attempt_ends=np.array(self.ends),
            )
            return self.observe(obs), reward, terminated, truncated, info | episode_info

        self.returns.append(0.0)
        self.lengths.append(0)
        obs, _ = self.env.reset(seed=self.track_seed)
        return self.observe(obs), reward, False, False, info


def make_env(env_id, attempts=1, feedback=False, normalize_reward=False, render_mode=None, max_episode_steps=None):
    def thunk():
        env = gym.make(env_id, continuous=False, render_mode=render_mode, max_episode_steps=max_episode_steps)
        env = OffTrackTruncation(env)
        env = gym.wrappers.MaxAndSkipObservation(env, skip=2)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, (64, 64))
        env = gym.wrappers.FrameStackObservation(env, 2)
        env = MultiAttempt(env, attempts, feedback)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env)
        return env

    return thunk
