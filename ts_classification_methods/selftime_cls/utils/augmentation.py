import numpy as np
from tqdm import tqdm
import utils.helper as hlp



def slidewindow(ts, horizon=.2, stride=0.2):
    xf = []
    yf = []
    for i in range(0, ts.shape[0], int(stride * ts.shape[0])):
        horizon1 = int(horizon * ts.shape[0])
        if (i + horizon1 + horizon1 <= ts.shape[0]):
            xf.append(ts[i:i + horizon1,0])
            yf.append(ts[i + horizon1:i + horizon1 + horizon1, 0])

    xf = np.asarray(xf)
    yf = np.asarray(yf)

    return xf, yf




def cutout(ts, perc=.1):
    seq_len = ts.shape[0]
    new_ts = ts.copy()
    win_len = int(perc * seq_len)
    start = np.random.randint(0, seq_len-win_len-1)
    end = start + win_len
    start = max(0, start)
    end = min(end, seq_len)
    # print("[INFO] start={}, end={}".format(start, end))
    new_ts[start:end, ...] = 0
    # return new_ts, ts[start:end, ...]
    return new_ts


def cut_piece2C(ts, perc=.1):
    seq_len = ts.shape[0]
    win_class = seq_len/(2*2)

    if perc<1:
        win_len = int(perc * seq_len)
    else:
        win_len = perc

    start1 = np.random.randint(0, seq_len-win_len)
    end1 = start1 + win_len
    start2 = np.random.randint(0, seq_len - win_len)
    end2 = start2 + win_len

    if abs(start1-start2)<(win_class):
        label=0
    else:
        label=1
    return ts[start1:end1, ...], ts[start2:end2, ...], label


def cut_piece3C(ts, perc=.1):
    seq_len = ts.shape[0]
    win_class = seq_len/(2*3)

    if perc<1:
        win_len = int(perc * seq_len)
    else:
        win_len = perc

    start1 = np.random.randint(0, seq_len-win_len)
    end1 = start1 + win_len
    start2 = np.random.randint(0, seq_len - win_len)
    end2 = start2 + win_len

    if abs(start1-start2)<(win_class):
        label=0
    elif abs(start1-start2)<(2*win_class):
        label=1
    else:
        label=2

    return ts[start1:end1, ...], ts[start2:end2, ...], label


def cut_piece4C(ts, perc=.1):
    seq_len = ts.shape[0]
    win_class = seq_len / (2 * 4)

    if perc < 1:
        win_len = int(perc * seq_len)
    else:
        win_len = perc

    start1 = np.random.randint(0, seq_len - win_len)
    end1 = start1 + win_len
    start2 = np.random.randint(0, seq_len - win_len)
    end2 = start2 + win_len

    if abs(start1 - start2) < (win_class):
        label = 0
    elif abs(start1 - start2) < (2 * win_class):
        label = 1
    elif abs(start1 - start2) < (3 * win_class):
        label = 2
    else:
        label = 3

    return ts[start1:end1, ...], ts[start2:end2, ...], label


def cut_piece5C(ts, perc=.1):
    seq_len = ts.shape[0]
    win_class = seq_len / (2 * 5)

    if perc < 1:
        win_len = int(perc * seq_len)
    else:
        win_len = perc

    start1 = np.random.randint(0, seq_len - win_len)
    end1 = start1 + win_len
    start2 = np.random.randint(0, seq_len - win_len)
    end2 = start2 + win_len

    if abs(start1 - start2) < (win_class):
        label = 0
    elif abs(start1 - start2) < (2 * win_class):
        label = 1
    elif abs(start1 - start2) < (3 * win_class):
        label = 2
    elif abs(start1 - start2) < (4 * win_class):
        label = 3
    else:
        label = 4

    return ts[start1:end1, ...], ts[start2:end2, ...], label


def cut_piece6C(ts, perc=.1):
    seq_len = ts.shape[0]
    win_class = seq_len / (2 * 6)

    if perc < 1:
        win_len = int(perc * seq_len)
    else:
        win_len = perc

    start1 = np.random.randint(0, seq_len - win_len)
    end1 = start1 + win_len
    start2 = np.random.randint(0, seq_len - win_len)
    end2 = start2 + win_len

    if abs(start1 - start2) < (win_class):
        label = 0
    elif abs(start1 - start2) < (2 * win_class):
        label = 1
    elif abs(start1 - start2) < (3 * win_class):
        label = 2
    elif abs(start1 - start2) < (4 * win_class):
        label = 3
    elif abs(start1 - start2) < (5 * win_class):
        label = 4
    else:
        label = 5

    return ts[start1:end1, ...], ts[start2:end2, ...], label


def cut_piece7C(ts, perc=.1):
    seq_len = ts.shape[0]
    win_class = seq_len / (2 * 7)

    if perc < 1:
        win_len = int(perc * seq_len)
    else:
        win_len = perc

    start1 = np.random.randint(0, seq_len - win_len)
    end1 = start1 + win_len
    start2 = np.random.randint(0, seq_len - win_len)
    end2 = start2 + win_len

    if abs(start1 - start2) < (win_class):
        label = 0
    elif abs(start1 - start2) < (2 * win_class):
        label = 1
    elif abs(start1 - start2) < (3 * win_class):
        label = 2
    elif abs(start1 - start2) < (4 * win_class):
        label = 3
    elif abs(start1 - start2) < (5 * win_class):
        label = 4
    elif abs(start1 - start2) < (6 * win_class):
        label = 5
    else:
        label = 6

    return ts[start1:end1, ...], ts[start2:end2, ...], label


def cut_piece8C(ts, perc=.1):
    seq_len = ts.shape[0]
    win_class = seq_len / (2 * 8)

    if perc < 1:
        win_len = int(perc * seq_len)
    else:
        win_len = perc

    start1 = np.random.randint(0, seq_len - win_len)
    end1 = start1 + win_len
    start2 = np.random.randint(0, seq_len - win_len)
    end2 = start2 + win_len

    if abs(start1 - start2) < (win_class):
        label = 0
    elif abs(start1 - start2) < (2 * win_class):
        label = 1
    elif abs(start1 - start2) < (3 * win_class):
        label = 2
    elif abs(start1 - start2) < (4 * win_class):
        label = 3
    elif abs(start1 - start2) < (5 * win_class):
        label = 4
    elif abs(start1 - start2) < (6 * win_class):
        label = 5
    elif abs(start1 - start2) < (7 * win_class):
        label = 6
    else:
        label = 7

    return ts[start1:end1, ...], ts[start2:end2, ...], label


def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

def scaling_s(x, sigma=0.1, plot=False):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(1, x.shape[1]))
    x_ = np.multiply(x, factor[:, :])

    if plot:
        hlp.plot1d(x, x_, save_file='aug_examples/scal.png')

    return x_

def rotation_s(x, plot=False):
    flip = np.random.choice([-1], size=(1, x.shape[1]))
    rotate_axis = np.arange(x.shape[1])
    np.random.shuffle(rotate_axis)
    x_ = flip[:, :] * x[:, rotate_axis]
    if plot:
        hlp.plot1d(x, x_, save_file='aug_examples/rotation_s.png')
    return x_

def rotation2d(x, sigma=0.2):
    thetas = np.random.normal(loc=0, scale=sigma, size=(x.shape[0]))
    c = np.cos(thetas)
    s = np.sin(thetas)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        rot = np.array(((c[i], -s[i]), (s[i], c[i])))
        ret[i] = np.dot(pat, rot)
    return ret

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret

def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):

        li = []
        for dim in range(x.shape[2]):
            li.append(CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps))
        warper = np.array(li).T

        ret[i] = pat * warper

    return ret


def magnitude_warp_s(x, sigma=0.2, knot=4, plot=False):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[0])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(1, knot + 2, x.shape[1]))
    warp_steps = (np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0] - 1., num=knot + 2))).T

    li = []
    for dim in range(x.shape[1]):
        li.append(CubicSpline(warp_steps[:, dim], random_warps[0, :, dim])(orig_steps))
    warper = np.array(li).T

    x_ = x * warper

    if plot:
        hlp.plot1d(x, x_, save_file='aug_examples/magnitude_warp_s.png')
    return x_


def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret


def time_warp_s(x, sigma=0.2, knot=4, plot=False):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[0])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(1, knot + 2, x.shape[1]))
    warp_steps = (np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for dim in range(x.shape[1]):
        time_warp = CubicSpline(warp_steps[:, dim],
                                warp_steps[:, dim] * random_warps[0, :, dim])(orig_steps)
        scale = (x.shape[0] - 1) / time_warp[-1]
        ret[:, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[0] - 1),
                                   x[:, dim]).T
    if plot:
        hlp.plot1d(x, ret, save_file='aug_examples/time_warp_s.png')
    return ret


def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret


def window_slice_s(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio * x.shape[0]).astype(int)
    if target_len >= x.shape[0]:
        return x
    starts = np.random.randint(low=0, high=x.shape[0] - target_len, size=(1)).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for dim in range(x.shape[1]):
        ret[:, dim] = np.interp(np.linspace(0, target_len, num=x.shape[0]), np.arange(target_len),
                                   x[starts[0]:ends[0], dim]).T
    return ret


def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret


def window_warp_s(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, 1)
    warp_size = np.ceil(window_ratio * x.shape[0]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(low=1, high=x.shape[0] - warp_size - 1, size=(1)).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    pat=x
    for dim in range(x.shape[1]):
        start_seg = pat[:window_starts[0], dim]
        window_seg = np.interp(np.linspace(0, warp_size - 1,
                                           num=int(warp_size * warp_scales[0])), window_steps,
                               pat[window_starts[0]:window_ends[0], dim])
        end_seg = pat[window_ends[0]:, dim]
        warped = np.concatenate((start_seg, window_seg, end_seg))
        ret[:, dim] = np.interp(np.arange(x.shape[0]), np.linspace(0, x.shape[0] - 1., num=warped.size),
                                   warped).T
    return ret

def spawner(x, labels, sigma=0.05, verbose=0):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/

    import utils.dtw as dtw
    random_points = np.random.randint(low=1, high=x.shape[1]-1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:
            random_sample = x[np.random.choice(choices)]
            # SPAWNER splits the path into two randomly
            path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            combined = np.concatenate((np.vstack(path1), np.vstack(path2+random_points[i])), axis=1)
            if verbose:
                print(random_points[i])
                dtw_value, cost, DTW_map, path = dtw.dtw(pat, random_sample,
                                                         return_flag = dtw.RETURN_ALL,
                                                         slope_constraint=slope_constraint,
                                                         window=window)
                dtw.draw_graph1d(cost, DTW_map, path, pat, random_sample)
                dtw.draw_graph1d(cost, DTW_map, combined, pat, random_sample)
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=mean.shape[0]), mean[:,dim]).T
        else:
            print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = pat
    return jitter(ret, sigma=sigma)

def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True):
    # https://ieeexplore.ieee.org/document/8215569

    import utils.dtw as dtw

    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i in tqdm(range(ret.shape[0])):
        # get the same class as i
        choices = np.where(l == l[i])[0]
        if choices.size > 0:
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]

            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)

            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]

            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern
                    weighted_sums += np.ones_like(weighted_sums)
                else:
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(np.log(0.5)*dtw_value/dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight

            ret[i,:] = average_pattern / weighted_sums[:,np.newaxis]
        else:
            print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = x[i]
    return ret

# Proposed

def random_guided_warp(x, labels, slope_constraint="symmetric", use_window=True, dtw_type="normal"):
    import utils.dtw as dtw

    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:
            # pick random intra-class pattern
            random_prototype = x[np.random.choice(choices)]

            if dtw_type == "shape":
                path = dtw.shape_dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                path = dtw.dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)

            # Time warp
            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            print("There is only one pattern of class %d, skipping timewarping"%l[i])
            ret[i,:] = pat
    return ret

def discriminative_guided_warp(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, dtw_type="normal", use_variable_slice=True):
    import utils.dtw as dtw

    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)

    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)

        # remove ones of different classes
        positive = np.where(l[choices] == l[i])[0]
        negative = np.where(l[choices] != l[i])[0]

        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]

            # vector embedding and nearest prototype in one
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.shape_dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.shape_dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.shape_dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)

            # Time warp
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps-warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            print("There is only one pattern of class %d"%l[i])
            ret[i,:] = pat
            warp_amount[i] = 0.
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # unchanged
            ret = window_slice(ret, reduce_ratio=0.95)
        else:
            for i, pat in enumerate(ret):
                # Variable Sllicing
                ret[i] = window_slice(pat[np.newaxis,:,:], reduce_ratio=0.95+0.05*warp_amount[i]/max_warp)[0]
    return ret
