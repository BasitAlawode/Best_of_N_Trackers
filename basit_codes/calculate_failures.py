import numpy as np

import sys
sys.path.append("toolkit")
from toolkit.utils import overlap_ratio

# My failure calculation
def calculate_failures(t_traj, g_traj, fail_iou=0.5):
    """ Calculate number of failures
    Args:
        trajectory: list of bbox
    Returns:
        num_failures: number of failures
        failures: failures point in trajectory, start with 0
    """
    failures = [i for i, x, y in zip(range(len(t_traj)), t_traj, g_traj)
            if overlap_ratio(np.expand_dims(np.array(x, dtype=np.float32), axis=0), \
                             np.expand_dims(np.array(y, dtype=np.float32),  axis=0)) < fail_iou]
    num_failures = len(failures)
    return num_failures, failures