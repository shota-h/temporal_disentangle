import random as rn
import cv2
import numpy as np
import h5py

n_part = 3
n_color = 3
reccrent_prob_part = .8
reccrent_prob__color = 1/n_color
part_list = ('ellipse', 'rectangle', 'triangle')
color_list = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
tran_prob_part = np.ones((n_part, n_part)) * (1 - reccrent_prob_part)/ (n_part - 1)
tran_prob_color = np.ones((n_color, n_color)) * (1 - reccrent_prob__color)/ (n_color - 1)
flatten = tran_prob_color.flatten()
flatten[::n_color+1] = reccrent_prob__color
tran_prob_color = np.resize(flatten, (n_color, n_color))
flatten = tran_prob_part.flatten()
flatten[::n_part+1] = reccrent_prob_part
tran_prob_part = np.resize(flatten, (n_part, n_part))

n_seq = 10000
out_dpath = './img/'
for seq_id in range(n_seq):
    s0 = np.random.choice(n_part, size=1)[0]
    s1 = np.random.choice(n_color, size=1)[0]
    continue_prob = [1.0, 0]
    seq_continue = True
    seq_len = 0
    with h5py.File('')
    while seq_continue:
        seq_len += 1
        img = np.full((256, 256, 3), 0, dtype=np.uint8)
        part = part_list[s0]
        color = color_list[s1]
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
            
        cv2.imwrite('./img/test{:03d}_{:03d}.png'.format(seq_id, seq_len), img)
        s0 = np.random.choice(n_part, size=1, p=tran_prob_part[s0])[0]
        s1 = np.random.choice(n_color, size=1, p=tran_prob_color[s1])[0]
        # continue_prob[1] = 1/np.sqrt(2*np.pi*1)*np.exp(-(seq_len-20)**2/2)
        if seq_len >= 20:
            continue_prob[0] = continue_prob[0]*.9
            continue_prob[1] = 1 - continue_prob[0]
            flag = np.random.choice(2, size=1, p=continue_prob)[0]
            seq_continue = [True, False][flag]
            # seq_continue = [True, False][1]