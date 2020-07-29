import sys
import cv2
import itertools
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
from data_handling import get_flatted_data
from losses import Fourier_mse
# A = np.arange(224)
# X = np.zeros((224, 224))
# X[:] = A
# U, V = np.arange(112), np.arange(112)
# Y = X.T
# Ms = {}
# for u, v in itertools.product(U, V):
#     M = np.exp(-2j*np.pi*(u*X/224 + v*Y/224))
#     Ms['u{}:v{}'.format(u, v)] = M
# cv2.imwrite('./reports/test.png', Ms['u0:v0'].real*255)

#grayscaleで読み込み
src, target1, target2 = get_flatted_data('./data/colon_renew.hdf5')
n = 0
for i in range(src.size(0)):
    # if target2[i] == 0:
    #     continue

    cri = Fourier_mse(img_w=224, img_h=224)
    # l = cri(src[:10], src[:10])
    l, _ = cri.predict(src[:2])
    ll = l[0].numpy()
    cv2.imwrite('./reports/test1.png', ll[0])
    ll = l[1].numpy()
    cv2.imwrite('./reports/test2.png', ll[0])
    ll = l[2].numpy()
    cv2.imwrite('./reports/test3.png', ll[0])
    # cv2.imwrite('./reports/test2.png', l[1])
    # cv2.imwrite('./reports/test3.png', l[2])

    img = src[i].numpy()
    img = np.transpose(img, (1,2,0))
    # cv2.imwrite('./reports/test.png', img[:, :, ::-1]*255)
    f = np.fft.fft2(img[:,:, 0])
    cv2.imwrite('./reports/test.png', f[:, :].real)
    sys.exit()
    fimg = []
    fill_img = []
    if n == 0:
        rows, cols, color = img.shape
        crow,ccol = int(rows/2) , int(cols/2)
        reg = 1
        fil2 = 30
        mask = np.zeros((rows,cols),np.uint8)
        for ii, jj in itertools.product(range(-fil2, fil2), range(-fil2, fil2)):
            if np.abs(ii) + np.abs(jj) <= fil2:
                mask[crow+ii, ccol+jj] = 1
        # mask[crow-fil2:crow+fil2, ccol-fil2:ccol+fil2] = 1
        mask = -mask + 1
    #フーリエ変換
    for c in range(3):

        f = np.fft.fft2(img[:,:, c])
        F_real = []
        F_imag = []
        F_abs = []
        for v, u in itertools.product(V, U):
            f_real = np.dot(img[:, :, c].flatten(), Ms['u{}:v{}'.format(u, v)].real.flatten())
            f_imag = np.dot(img[:, :, c].flatten(), Ms['u{}:v{}'.format(u, v)].imag.flatten())
            F_real.append(f_real)
            F_imag.append(f_imag)
            F_abs.append(np.sqrt(f_imag**2+f_real**2))
        F_abs = np.asarray(F_abs)
        F_real = np.asarray(F_real)
        F_imag = np.asarray(F_imag)
        F_abs = np.reshape(F_abs, (112, 112))
        cv2.imwrite('./reports/test.png', F_abs)
        # fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(f)
        cat_spec = magnitude_spectrum[:112, :112]
        print(F_abs.max())
        print(cat_spec.max())
        print(F_real.max())
        print(f.real.max())
        print(F_imag.max())
        print(f.imag.max())
        f_src = torch.fft(src[i][c])
        print(f_src.imag.max())

        cv2.imwrite('./reports/test1.png', cat_spec)
        cv2.imwrite('./reports/test2.png', cat_spec-F_abs)
        

        sys.exit()

    #画像中心を原点に変更
        fshift = np.fft.fftshift(f)
        fshift_copy = copy.copy(fshift)
        # fshift[crow-reg:crow+reg, ccol-reg:ccol+reg] = 0
        fshift = fshift*mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        fill_img.append(img_back)
    #フーリエ変換結果は複素数なので絶対値にし、logにしている
        # magnitude_spectrum = np.abs(fshift)
        # print(magnitude_spectrum.max())
        magnitude_spectrum = 20*np.log(np.abs(fshift_copy))
        fimg.append(magnitude_spectrum)
    
    fig = plt.figure(figsize=(16*5, 9))
    ax = fig.add_subplot(1, 5, 1)
    ax.imshow(img)
    for ii, fi in zip(range(2, 5), fimg):
        ax = fig.add_subplot(1, 5, ii)
        ax.imshow(fi, cmap = 'gist_gray')
    ax = fig.add_subplot(1, 5, 5)
    ax.imshow(np.transpose(np.asarray(fill_img), (1,2,0)))
    fig.savefig('./reports/fft_analysis/test{}.png'.format(n))
    plt.close(fig)
    if n >= 10: break
    n += 1

    # cv2.imwrite('./reports/test{}.png'.format(c), magnitude_spectrum)
    
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()