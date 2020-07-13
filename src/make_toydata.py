import sys
import random as rn
import cv2
import numpy as np
import h5py

n_part = 3
n_color = 2
n_subcolor = 3
reccrent_prob_part = .8
reccrent_prob_color = .8
reccrent_prob_subcolor = 1/n_subcolor
part_list = ('ellipse', 'rectangle', 'triangle')
color_list = (((0, 0, 255),), (((0, 255, 0), (255, 0, 0), (255, 255, 255))))
tran_prob_part = np.ones((n_part, n_part)) * (1 - reccrent_prob_part)/ (n_part - 1)
tran_prob_color = np.ones((n_color, n_color)) * (1 - reccrent_prob_color)/ (n_color - 1)
tran_prob_subcolor = np.ones((n_subcolor, n_subcolor)) * (1 - reccrent_prob_subcolor)/ (n_subcolor - 1)
flatten = tran_prob_part.flatten()
flatten[::n_part+1] = reccrent_prob_part
tran_prob_part = np.resize(flatten, (n_part, n_part))
flatten = tran_prob_color.flatten()
flatten[::n_color+1] = reccrent_prob_color
tran_prob_color = np.resize(flatten, (n_color, n_color))
flatten = tran_prob_subcolor.flatten()
flatten[::n_subcolor+1] = reccrent_prob_subcolor
tran_prob_subcolor = np.resize(flatten, (n_subcolor, n_subcolor))
print(tran_prob_part)
print(tran_prob_color)
print(tran_prob_subcolor)
sys.exit()
n_seq = 10
min_len = 10
out_img = True
out_dpath = './data/toy_data.hdf5'
with h5py.File(out_dpath, 'w') as f:
    for seq_id in range(n_seq):
        s0 = np.random.choice(n_part, size=1)[0]
        s1 = np.random.choice(n_color, size=1)[0]
        s2 = 0
        if s1 > 0:
            s2 = np.random.choice(n_subcolor, size=1, p=tran_prob_subcolor[0])[0]
        continue_prob = [1.0, 0]
        seq_continue = True
        seq_len = 0
        x0 = np.full((256, 256, 3), 0, dtype=np.uint8)
        while seq_continue:
            seq_len += 1
            img = np.full((256, 256, 3), 0, dtype=np.uint8)
            part = part_list[s0]
            color = color_list[s1][s2]
            cx, cy = np.random.randint(64, 256-64, size=2) 
            dx, dy = np.random.randint(32, 64, size=2)
            dx = dy
            angle = np.random.randint(360, size=1)[0]
            if part == 'ellipse':
                box = ((cx, cy), (dx, dy), angle)
                cv2.ellipse(img, box, color, thickness=-1)
            elif part == 'rectangle':
                (x1, y1), (x2, y2) = (dx/2, dy/2), (-dx/2, -dy/2)
                rot_mat = np.array([[np.cos(np.pi*angle/360), -np.sin(np.pi*angle/360)],
                            [np.sin(np.pi*angle/360), np.cos(np.pi*angle/360)]])
                rx11, ry11 = np.dot(rot_mat, np.array([x1, y1]))
                rx12, ry12 = np.dot(rot_mat, np.array([x1, y2]))
                rx21, ry21 = np.dot(rot_mat, np.array([x2, y1]))
                rx22, ry22 = np.dot(rot_mat, np.array([x2, y2]))
                pts = np.array(((cx+rx11, cy+ry11), (cx+rx21, cy+ry21), (cx+rx22, cy+ry22), (cx+rx12, cy+ry12))).astype(np.int)
                cv2.fillConvexPoly(img, pts, color)
            if part == 'triangle':
                rot_mat = np.array([[np.cos(np.pi*angle/360), -np.sin(np.pi*angle/360)],
                            [np.sin(np.pi*angle/360), np.cos(np.pi*angle/360)]])
                (x1, y1), (x2, y2), (x3, y3)= (0, -dy*2/3), (-dx/2, dy/3), (dx/2, dy/3)
                rx1, ry1 = np.dot(rot_mat, np.array([x1, y1]))
                rx2, ry2 = np.dot(rot_mat, np.array([x2, y2]))
                rx3, ry3 = np.dot(rot_mat, np.array([x3, y3]))
                pts = np.array(((cx+rx1, cy+ry1), (cx+rx2, cy+ry2), (cx+rx3, cy+ry3))).astype(np.int)
                cv2.fillConvexPoly(img, pts, color)
            if out_img:
               cv2.imwrite('./data/test{:03d}_{:03d}.png'.format(seq_id, seq_len), img)
            else:
                f.create_dataset(name='img/{:04d}/{:04d}'.format(seq_id, seq_len), data=img)
                f['img/{:04d}/{:04d}'.format(seq_id, seq_len)].attrs['part'] = s0
                f['img/{:04d}/{:04d}'.format(seq_id, seq_len)].attrs['color'] = s1
            s0 = np.random.choice(n_part, size=1, p=tran_prob_part[s0])[0]
            s1 = np.random.choice(n_color, size=1, p=tran_prob_color[s1])[0]
            s2 = 0
            if s1 > 0:
                s2 = np.random.choice(n_subcolor, size=1, p=tran_prob_subcolor[0])[0]
            if seq_len >= min_len:
                seq_continue = False
                continue
                # continue_prob[0] = 1/np.sqrt(2*np.pi*1)*np.exp(-(seq_len-20)**2/2)
                continue_prob[0] = continue_prob[0]*.9
                continue_prob[1] = 1 - continue_prob[0]
                seq_continue = np.random.choice([True, False], size=1, p=continue_prob)[0]