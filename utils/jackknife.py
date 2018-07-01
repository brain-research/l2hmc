import numpy as np
from sklearn.model_selection import KFold

#  def block_resampling(data, num_blocks):
#      """Block resample data to return num_blocks samples of original data."""
#      if not isinstance(data, np.ndarray):
#          data = np.array(data)
#      num_samples = data.shape[0]
#      #  if num_samples < 1:
#      #      raise ValueError()
#      #      #raise ValueError("Data must have at least one sample.")
#      #  if num_blocks < 1:
#      #      raise ValueError("Number of resampled blocks must be greater than or"
#      #                       "equal to 1.")
#      kf = KFold(n_splits=num_blocks)
#      resampled_data = []
#      for i, j in kf.split(data):
#          resampled_data.append(data[i])]
#      return resampled data


def block_resampling(data, num_blocks):
    """ Block-resample data to return num_blocks samples of original data. """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    num_samples = data.shape[0]
    if num_samples < 1:
        raise ValueError("Data must have at least one sample.")
    if num_blocks < 1:
        raise ValueError("Number of resampled blocks must be greater than or"
                         "equal to 1.")
    kf = KFold(n_splits = num_blocks)
    resampled_data = []
    for i, j in kf.split(data):
        resampled_data.append(data[i])
    return resampled_data



def jackknife(x, func, num_blocks=100):
    """Jackknife estimate of the estimator function."""
    n = len(x)
    block_size = n // num_blocks
    idx = np.arange(0, n, block_size)
    return np.sum(func(x[idx!=i]) for i in range(n))/float(n)

def jackknife_var(x, func, num_blocks=100):
    """Jackknife estimate of the variance of the estimator function."""
    n = len(x)
    block_size = n // num_blocks
    idx = np.arange(0, n, block_size)
    j_est = jackknife(x, func)
    return (n - 1) / (n + 0.) * np.sum(
        func(x[idx!=i]) - j_est**2.0 for i in range(n)
    )

def jackknife_err(y_i, y_full, num_blocks):
    if isinstance(y_i, list):
        y_i = np.array(y_i)
    if isinstance(y_full, list):
        y_full = np.array(y_full)
    try:
        err = np.sqrt((num_blocks - 1) * np.sum((y_i - y_full)**2) / num_blocks)
    except ValueError:
        print(f"y_i.shape: {y_i.shape}, y_full.shape: {y_full.shape}")
        raise
    return err

