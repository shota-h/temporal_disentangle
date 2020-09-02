import random as rn
import numpy as np
import h5py
import copy
import torch


def data_review():
    with h5py.File('./data/colon.hdf5', 'r') as f:
        with h5py.File('./data/colon_renew.hdf5', 'w') as f_out:
            key = list(f.keys())
            df = pd.read_csv('./data/colon_data2label.csv')
            # flag = False
            for k in key:
                cat_df = df[df.sequence_num == int(k)]
                for i, fname in enumerate(cat_df.filename):
                    # print(f[k].attrs['mayo_label'][i])
                    p = f[k].attrs['part_label'][i]
                    m = f[k].attrs['mayo_label'][i]
                    f_out.create_dataset(name='img/{}/{}'.format(k, fname), data=f[k][i])
                    f_out['img/{}/{}'.format(k, fname)].attrs['part'] = p-1
                    f_out['img/{}/{}'.format(k, fname)].attrs['mayo'] = m-1


def get_flatted_data(data_dpath, trans=True):
    with h5py.File(data_dpath, 'r') as f:
        srcs = []
        targets1, targets2, targets3 = [], [], []
        for group_key in f.keys():
            for parent_key in f[group_key].keys():
                parent_group = '{}/{}'.format(group_key, parent_key)
                src = []
                target1, target2, target3 = [], [], []
                for child_key in f[parent_group].keys():
                    child_group = '{}/{}'.format(parent_group, child_key)
                    src.append(f[child_group][()])
                    target1.append(f[child_group].attrs['part'])
                    target2.append(f[child_group].attrs['mayo'])
                    # target3.append(f[child_group].attrs['mayo'])
                    if trans:
                        if 'colon' in data_dpath:
                            if target2[-1] > 1:
                                target2[-1] = 1
                            elif target2[-1] <= 1:
                                target2[-1] = 0
        
                srcs.extend(src)
                targets1.extend(target1)
                targets2.extend(target2)
                targets3.extend(target3)

        srcs = np.asarray(srcs)
        if srcs.max() > 1:
            srcs = srcs / 255
            # srcs = srcs / srcs.max()
        srcs = np.transpose(srcs, (0, 3, 1, 2))
        targets1 = np.asarray(targets1)
        targets2 = np.asarray(targets2)
        # targets3 = np.asarray(targets3)
        srcs = torch.from_numpy(srcs).float()
        targets1 = torch.from_numpy(targets1).long()
        targets2 = torch.from_numpy(targets2).long()
        # targets3 = torch.from_numpy(targets3).long()
        return srcs, targets1, targets2


def get_triplet_flatted_data(data_dpath):
    # srcs, srcs_p, srcs_n = [], [], []
    # targets1, targets2 = [], []
        
    src, p_src, n_src = [], [], []
    target1, target2 = [], []
    with h5py.File(data_dpath, 'r') as f:
        for group_key in f.keys():
            for parent_key in f[group_key].keys():
                parent_group = '{}/{}'.format(group_key, parent_key)
                child_key_list = list(f[parent_group].keys())
                for i, child_key in enumerate(child_key_list):
                    child_group = '{}/{}'.format(parent_group, child_key)
                    if child_key_list[i+1:i+2] and child_key_list[i+2:i+3]:
                        p_child_group = '{}/{}'.format(parent_group, child_key_list[i+1])
                        n_child_group = '{}/{}'.format(parent_group, child_key_list[i+2])
                    elif child_key_list[i-1:i] and child_key_list[i-2:i-1]:
                        p_child_group = '{}/{}'.format(parent_group, child_key_list[i-1])
                        n_child_group = '{}/{}'.format(parent_group, child_key_list[i-2])
                    else:
                        continue
                    src.append(f[child_group][()])
                    p_src.append(f[p_child_group][()])
                    n_src.append(f[n_child_group][()])
                    target1.append(f[child_group].attrs['part'])
                    target2.append(f[child_group].attrs['mayo'])
                    if 'colon' in data_dpath:
                        if target2[-1] > 1:
                            target2[-1] = 1
                        elif target2[-1] <= 1:
                            target2[-1] = 0
        
    src = np.asarray(src)
    p_src = np.asarray(p_src)
    n_src = np.asarray(n_src)
    if src.max() > 1:
        src = src / 255
        p_src = p_src / 255
        n_src = n_src / 255
        # srcs = srcs / srcs.max()
    src = np.transpose(src, (0, 3, 1, 2))
    p_src = np.transpose(p_src, (0, 3, 1, 2))
    n_src = np.transpose(n_src, (0, 3, 1, 2))
    target1 = np.asarray(target1)
    target2 = np.asarray(target2)
    src = torch.from_numpy(src).float()
    p_src = torch.from_numpy(p_src).float()
    n_src = torch.from_numpy(n_src).float()
    target1 = torch.from_numpy(target1).long()
    target2 = torch.from_numpy(target2).long()
    return (src, p_src, n_src), target1, target2


def get_triplet_flatted_data_with_idx(data_dpath, label_decomp=True):
    src = []
    idx, p_idx, n_idx = [], [], []
    target1, target2 = [], []
    inc = 0
    with h5py.File(data_dpath, 'r') as f:
        for group_key in f.keys():
            for parent_key in f[group_key].keys():
                parent_group = '{}/{}'.format(group_key, parent_key)
                child_key_list = list(f[parent_group].keys())
                for i, child_key in enumerate(child_key_list):
                    child_group = '{}/{}'.format(parent_group, child_key)
                    if child_key_list[i+1:i+2] and child_key_list[i+2:i+3]:
                        p_child_group = '{}/{}'.format(parent_group, child_key_list[i+1])
                        n_child_group = '{}/{}'.format(parent_group, child_key_list[i+2])
                        p_inc = inc + 1
                        n_inc = inc + 2
                    elif child_key_list[i-1:i] and child_key_list[i-2:i-1]:
                        p_child_group = '{}/{}'.format(parent_group, child_key_list[i-1])
                        n_child_group = '{}/{}'.format(parent_group, child_key_list[i-2])
                        p_inc = inc - 1
                        n_inc = inc - 2
                    else:
                        continue
                    src.append(f[child_group][()])
                    idx.append(inc)
                    p_idx.append(p_inc)
                    n_idx.append(n_inc)
                    target1.append(f[child_group].attrs['part'])
                    target2.append(f[child_group].attrs['mayo'])
                    inc += 1

    if 'colon' in data_dpath:
        if label_decomp:
            target2 = [1 if cat_target2 > 1 else 0 for cat_target2 in target2]
    src = np.asarray(src)
    idx = np.array(idx)
    p_idx = np.array(p_idx)
    n_idx = np.array(n_idx)
    if src.max() > 1:
        src = src / 255
    src = np.transpose(src, (0, 3, 1, 2))
    target1 = np.asarray(target1)
    target2 = np.asarray(target2)
    src = torch.from_numpy(src).float()
    target1 = torch.from_numpy(target1).long()
    target2 = torch.from_numpy(target2).long()
    idx = torch.from_numpy(idx).long()
    p_idx = torch.from_numpy(p_idx).long()
    n_idx = torch.from_numpy(n_idx).long()
    return src, target1, target2, (idx, p_idx, n_idx)


def random_label_replace(src, ratio=0.1, value=-1, seed=1):
    torch.manual_seed(seed)
    rn.seed(seed)
    np.random.seed(seed)
    
    dst = copy.deepcopy(src[:])
    for uniq_src in np.unique(src):
        idx = np.where(src==uniq_src)[0]
        pick_idx = np.random.choice(idx, size=int(len(idx)*ratio))
        dst[pick_idx] = value
        
    return dst