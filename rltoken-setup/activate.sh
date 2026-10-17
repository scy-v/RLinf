# Usage: source /home/suchenyu/RLinf/rltoken-setup/activate.sh
source /home/suchenyu/RLinf/.venv/bin/activate
# The UR RTDE backend does not need inherited ROS/Python 3.10 packages.
export PYTHONPATH=/home/suchenyu/RLinf
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
# OpenPI uses JAX for data transforms; keep its GPU memory preallocation off.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /home/suchenyu/RLinf
