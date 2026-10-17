# UR Environment Installation

Auxiliary scripts, environment records, and verification tools live in this `support/` directory. [Chinese instructions](../安装命令.md).

## Install From the Current Checkout

Install the UR dependencies with the OpenPI model stack:

```bash
cd ~/RLinf
bash requirements/install.sh embodied --model openpi --env ur \
  --venv .venv --no-root --install-rlinf
```

Alternatively, run the wrapper to install dependencies and repair OpenCV display support:

```bash
bash rltoken-setup/support/install.sh
```

## Clone the Fork

The `rlt` branch contains the `--env ur` installation target. Clone that branch for a new installation:

```bash
git clone --depth 1 --branch rlt https://github.com/scy-v/RLinf.git ~/RLinf
cd ~/RLinf

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

bash requirements/install.sh embodied --model openpi --env ur \
  --venv .venv --no-root --install-rlinf
```

## OpenCV Display Support

After installing OpenPI, select the GUI OpenCV wheel:

```bash
source ~/RLinf/.venv/bin/activate
uv pip uninstall opencv-python opencv-python-headless
uv pip install --no-deps opencv-python==4.11.0.86
```

## Daily Use

Activate the environment:

```bash
source ~/RLinf/.venv/bin/activate
```

The UR LeRobot reader is pinned to `d6ea3bbce0fc8c75138f02639ac4ceaf12a67829`, matching your OpenPI environment and supporting this v3.0 dataset. The installer reuses RLinf's existing Torch/Datasets stack.

UR dependencies are listed in `requirements/embodied/envs/ur.txt`, including RTDE, the DH gripper, and the space mouse.
