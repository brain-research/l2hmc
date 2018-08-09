import numpy as np

def distance(pt1, pt2):
    d = pt2 - pt1
    return np.sqrt(d.T.dot(d))

def calc_min_distance(means, cov):
    idxs = [(i, j) for i in range(len(means) - 1, 0, -1)
            for j in range(len(means) - 1) if i > j]
    min_dist = []
    for pair in idxs:
        _vec = means[pair[1]] - means[pair[0]]
        _dist = np.sqrt(_vec.T.dot(_vec))
        _unit_vec = np.sqrt(cov) * _vec / _dist
        p0 = means[pair[0]] + _unit_vec
        p1 = means[pair[1]] - _unit_vec
        _diff = p1 - p0
        _min_dist = np.sqrt(_diff.T.dot(_diff))
        min_dist.append(_min_dist)
    return min(min_dist)

def calc_tunneling_rate(trajectory, min_distance):
    idxs = [(i, i+1) for i in range(len(trajectory) - 1)]
    #min_distance = calc_min_dist(means, cov)
    tunneling_events = {}
    num_events = 0
    for pair in idxs:
        _distance = distance(trajectory[pair[0]], trajectory[pair[1]])
        if _distance >= min_distance:
            tunneling_events[pair] = _distance
            num_events += 1
    tunneling_rate = num_events / (len(trajectory) - 1)
    return tunneling_events, tunneling_rate

def match_distribution(x, means, num_distributions):
    """Given a point x and multiple distributions (each with their own
    respective mean, contained in `means`), try to identify which distribution
    the point x belongs to.

    Args:
        point (scalar or array-like):
            Point belonging to some unknown distribution.
        means (array-like):
            Array containing the mean vectors of different normal
            distributions.
    Returns:
        Index in `means` corresponding to the distribution `x` is closest  to.
    """
    norm_diff_arr = []
    #for mean in means:
    for row in range(num_distributions):
        #diff = x - meaan
        diff = x - means[row]
        norm_diff = np.sqrt(np.dot(diff.T, diff))
        norm_diff_arr.append(norm_diff)
    return np.argmin(np.array(norm_diff_arr))

def find_tunneling_events(trajectory, means, num_distributions):
    idxs = [(i, i+1) for i in range(len(trajectory) - 1)]
    #  tunneling_events = {}
    num_events = 0
    for pair in idxs:
        x0 = trajectory[pair[0]]
        x1 = trajectory[pair[1]]
        dist0 = match_distribution(x0, means, num_distributions)
        dist1 = match_distribution(x1, means, num_distributions)
        if dist1 != dist0:
            #  tunneling_events[pair] = [x0, x1]
            num_events += 1
    tunneling_rate = num_events / (len(trajectory) - 1)
    return tunneling_rate
