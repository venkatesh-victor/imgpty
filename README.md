import numpy as np
from skimage import color, io, transform
from sklearn import preprocessing
import os


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, rho

def cart2polar(region_size):
    """
    return: 2 np.array objects
            first one is radius, second one is angle
    """
    radius = np.zeros(region_size)
    angle = np.zeros(region_size)
    center = [region_size[0]//2, region_size[1]//2]

    for row in range(0, region_size[0]):
        for col in range(0, region_size[1]):
            theta, rho = cart2pol(row - center[0], col - center[1])
            radius[row, col] = np.log(rho + 1e-1)
            angle[row, col] = np.degrees(theta) + 180

    return radius, angle


def get_bin(radius, angle, region_size):
    max_radius = np.max(radius)
    bin = [[[] for _ in range(3)] for _ in range(15)]

    for m in range(15):
        theta_low = m * 24
        theta_up = (m + 1) * 24

        for n in range(3):
            rho_low = max_radius * n / 3
            rho_up = max_radius * (n + 1) / 3

            temp = []
            num = 0
            for row in range(region_size[0]):
                for col in range(region_size[1]):
                    if (rho_low <= radius[row,col] <= rho_up and
                        theta_low <= angle[row, col] <= theta_up):
                        num += 1
                        temp.append([row, col])
            bin[m][n] = temp

    return bin

def cal_ssd(patch, region, alpha, center_patch):
    patch_size = patch.shape
    region_size = region.shape
    ssd_region = np.zeros((region_size[0], region_size[1]))

    for row in range(center_patch[0], region_size[0] - center_patch[0]):
        for col in range(center_patch[1], region_size[1] - center_patch[1]):
            temp = region[row - center_patch[0] : row + center_patch[0] + 1,
                          col - center_patch[1] : col + center_patch[1] + 1] - patch
            ssd_region[row, col] = np.sum(temp ** 2)
            ssd_region[row, col] = np.exp(-alpha * ssd_region[row, col])

    return ssd_region

def get_self_sim_vec(ssd_region, bin, vec_size):
    self_similarities_vec = np.zeros((1, vec_size))
    
    num = 0
    for m in range(15):  
        for n in range(3):  
            temp = bin[m][n]
            max_value = 0
            
            for loc in range(len(temp)):
                row, col = temp[loc]
                max_value = max(max_value, np.max(ssd_region[row, col]))
            
            num += 1
            self_similarities_vec[num-1] = max_value

    return self_similarities_vec

def com_self_similarites(src_img, region_size, patch_size, bin):
    img_lab = color.rgb2lab(src_img)
    lab_size = img_lab.shape
    vec_size = 45
    alpha = 1 / (85**2)

    self_similarites = np.zeros((lab_size[0], lab_size[1], vec_size))
    center_region = [region_size[0]//2, region_size[1]//2]
    center_patch = [patch_size[0]//2, patch_size[1]//2]

    for row in range(center_region[0], lab_size[0] - center_region[1]):
        for col in range(center_region[1], lab_size[1] - center_region[0]):
            patch = img_lab[row - center_patch[0]:row + center_patch[0] + 1,
                            col - center_patch[1]:col + center_patch[1] + 1]
            region = img_lab[row - center_patch[0]:row + center_patch[0] + 1,
                            col - center_patch[1]:col + center_patch[1] + 1]
            
            ssd_region = cal_ssd(patch, region, alpha, center_patch)
            vec = get_self_sim_vec(ssd_region, bin, vec_size) ## finished?
            lssd = preprocessing.minmax_scale(vec, (0, 1))
            self_similarites[row, col, :] = lssd  ## without transposing

    return self_similarites
