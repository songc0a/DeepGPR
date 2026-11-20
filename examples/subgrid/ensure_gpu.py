"""Make sure a CUDA context can be created: when the page cache has eaten the free RAM the
NVIDIA driver cannot allocate its system-memory buffers ("CUDA driver initialization
failed"); touching and releasing a large array makes the kernel drop the cache."""
import subprocess
import sys

import numpy as np

CHECK = "import torch; torch.zeros(1).cuda()"
for attempt in range(4):
    ok = subprocess.run([sys.executable, "-c", CHECK], capture_output=True).returncode == 0
    free_gb = int(open("/proc/meminfo").read().split("MemFree:")[1].split()[0]) / 2 ** 20
    if ok and free_gb > 6:
        sys.exit(0)
    a = np.ones((28 * 2 ** 30) // 8)
    a[:: 2 ** 20].sum()
    del a
print("GPU still not usable", file=sys.stderr)
sys.exit(1)
