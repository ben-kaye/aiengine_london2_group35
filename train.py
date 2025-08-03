import os

NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
        f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")
os.environ["MUJOCO_GL"] = "egl"
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import subprocess

import torch

if subprocess.run("nvidia-smi").returncode:
    raise RuntimeError(
        "Cannot communicate with GPU. "
        "Make sure you are using a GPU Colab runtime. "
        "Go to the Runtime menu and select Choose runtime type."
    )


import functools
import os
import time


import jax
import mediapy as media
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params

np.set_printoptions(precision=3, suppress=True, linewidth=100)


def get_env(env_name):
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)

    return env, env_cfg


def get_ppo_params(env_name, **overrides):
    ppo_params = manipulation_params.brax_ppo_config(env_name)

    for k, v in overrides.items():
        setattr(ppo_params, k, v)

    return ppo_params


def train(
    ppo_params: dict, save_path: str, seed: int = 1, progress_fn: callable | None = None
):
    network_factory = ppo_networks.make_ppo_networks

    ppo_training_params = dict(ppo_params)
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )

    train_fn = functools.partial(
        ppo.train,
        **ppo_training_params,
        network_factory=network_factory,
        progress_fn=progress_fn,
        seed=seed,
    )

    t0 = time.perf_counter()
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    t1 = time.perf_counter()
    print(f"time to train: {(t1 - t0) / 60:.2f} mins")

    torch.save(params, save_path)
    return make_inference_fn, params, metrics


def make_video(
    jit_inference_fn,
    jit_step,
    jit_reset,
    env,
    env_cfg,
    save_path: str,
    seed: int = 1,
):
    rng = jax.random.PRNGKey(seed)
    rollout = []
    n_episodes = 1

    for _ in range(n_episodes):
        state = jit_reset(rng)
        rollout.append(state)
        for i in range(env_cfg.episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state)

    render_every = 1
    frames = env.render(rollout[::render_every])
    rewards = [s.reward for s in rollout]
    with media.set_show_save_dir(save_path.parent):
        media.show_video(frames, fps=1.0 / env.dt / render_every, title=save_path.stem)


if __name__ == "__main__":
    env_name = "PandaPickCube"
    env, env_cfg = get_env(env_name)
    ppo_params = get_ppo_params(env_name)
    make_inference_fn, params, metrics = train(ppo_params, "params.pt")

    # test
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    make_video(jit_inference_fn, jit_step, jit_reset, env, env_cfg, "video.mp4")
