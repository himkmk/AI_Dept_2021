from PIL import Image
from torchvision.transforms.functional import to_pil_image,to_tensor
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import os

import skimage.restoration as skimage
import torch.nn.functional as F
import cv2
import numpy as np
import torch

def normalize(img, type):
    if type=="phase":
        # img = torch.fft.ifftshift(img)
        img = (img)/(img).max()
        img = torch.abs(img)
        img = torch.clamp(img[0,:,:],0,1)
    if type=="mag":
        img = torch.log(img)/torch.log(img).max()
        img = torch.clamp(img[:,:,:],0,1)
    if type=="HSV":
        img = img - img.min()
        img = img/img.max()  # img should be 0~1
        img = torch.clamp(img,0,1)  # assurance
        img = img * 179

    return img

def unwrap_npbase(phase, dim):
    phase = phase.detach().to(torch.device("cpu")).numpy()
    phase = skimage.unwrap_phase(phase)
    return torch.Tensor(phase)

def getMagnPhase(img):
    img = to_tensor(img).cuda()

    fft_img = torch.fft.fft2(img)
    fft_img = torch.fft.fftshift(fft_img)
    mag_img = torch.abs(fft_img)
    phase_img = torch.atan2(fft_img.imag, fft_img.real)
    phase_img = unwrap_torchbase(phase_img, dim=-1)

    # visualize
    img = to_pil_image(img)
    mag_img = to_pil_image(normalize(mag_img, "mag"))
    phase_img = to_pil_image(normalize(phase_img, "phase"))

    return img,mag_img,phase_img

def phase2rgb(phase, dir):

    phase = to_tensor(phase)
    phase = phase[0,:,:]
    H = normalize(phase, type="HSV")
    S = torch.ones_like(phase)*255
    V = torch.ones_like(phase)*255


    HSV = np.zeros((list(phase.shape)[0], list(phase.shape)[1], 3), dtype=np.uint8)
    HSV[:,:,0] = H
    HSV[:,:,1] = S
    HSV[:,:,2] = V

    HSV = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    # cv2.imshow(dir, HSV)

    return HSV


def unwrap_torchbase(phi, dim=-1):

    def diff(x, dim=-1):
        # assert dim==-1 or dim==-2, f"dim should be -1 or -2, but given {dim}"
        if dim==-1:
            return F.pad(x[..., 1:]-x[..., :-1], (1, 0))
        elif dim==-2:
            return F.pad(x[...,1:,:]-x[...,:-1,:], (0,0,1, 0))



    dphi = diff(phi, dim=dim)
    dphi_m = ((dphi + np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m == -np.pi) & (dphi > 0)] = np.pi
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < np.pi] = 0

    return phi + phi_adj.cumsum(dim)

def PNGnum(dir):
    return int(os.path.basename(dir)[0:4])

def preprocess(GT_dir, LR_dir, post_dir, startNum, endNum):

    GT_dir_ = glob.glob(GT_dir + post_dir)
    LR_dir_ = glob.glob(LR_dir + post_dir)
    GT_dir = []
    LR_dir = []

    for dir in GT_dir_:
        if (startNum <= PNGnum(dir) <= endNum):
            GT_dir.append(dir)

    for dir in LR_dir_:
        if (startNum <= PNGnum(dir) <= endNum):
            LR_dir.append(dir)

    return GT_dir, LR_dir


# main
indir1 = "./DIV2K/DIV2K_train_HR"
indir2 = "./DIV2K/DIV2K_train_HR"
VISUALIZE = True
SAVE = False

dir1,dir2 = preprocess(indir1,indir2,"/*.png", 801, 900)
for img_dir1,img_dir2 in tqdm(zip(dir1, dir2)):

    # IO
    print("")
    print(img_dir1)
    print(img_dir2)
    img1 = Image.open(img_dir1)
    img2 = Image.open(img_dir2)

    # FFT, MAG, PHASE

    img1, mag_img1, phase_img1 = getMagnPhase(img1)
    img2, mag_img2, phase_img2 = getMagnPhase(img2)

    phase2rgb_img1 = phase2rgb(phase_img1,"dir1")
    phase2rgb_img2 = phase2rgb(phase_img2,"dir2")

    if VISUALIZE:
        mag_img1.show()
        phase_img1.show()


    if SAVE:
        img1.SAVE(os.path.join("./inspection_result", os.path.basename(img_dir1)))
        phase_img1.SAVE(os.path.join("./inspection_result", os.path.basename(img_dir1).split(".")[0] + "_phase1.png"))
        phase_img2.SAVE(os.path.join("./inspection_result", os.path.basename(img_dir2).split(".")[0] + "_phase2.png"))
        phase2rgb_img1.SAVE(os.path.join("./inspection_result", os.path.basename(img_dir1).split(".")[0] + "_rgb.png"))
        phase2rgb_img2.SAVE(os.path.join("./inspection_result", os.path.basename(img_dir2).split(".")[0] + "_rgb.png"))

