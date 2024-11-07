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
            radius[row, col] = np.log(rho)
            angle[row, col] = np.degrees(theta) + 180

    return radius, angle

def get_bin(radius, angle, region_size):
    """ Returns: bins - 3d array """

    max_radius = np.max(radius)
    bins = [[[] for _ in range(3)] for _ in range(15)]
    
    for m in range(15):
        theta_low = m * 24
        theta_up = (m + 1) * 24

        for n in range(3):
            rho_low = max_radius * n / 3
            rho_up = max_radius * (n + 1) / 3

            for row in range(region_size[0]):
                for col in range(region_size[1]):
                    if (rho_low <= radius[row, col] <= rho_up and 
                        theta_low <= angle[row, col] <= theta_up):
                            bins.append(row, col)
    
    return bins

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
            temp = bin[m, n]
            max_value = 0

            temp_size = temp.shape
            for loc in range(1, temp_size[1] + 1):
                row = temp[0, loc]
                col = temp[1, loc]
                max_value = max(max_value, ssd_region[row, col])

            num += 1
            self_similarities_vec[num] = max_value

    return self_similarities_vec

def com_self_similarites(src_img, region_size, patch_size, bin):
    img_lab = color.rgb2lab(src_img)
    lab_size = img_lab.shape
    vec_size = 45
    alpha = 1 / (85**2)

    self_similarites = np.zeros((lab_size[0], lab_size[1], vec_size))
    center_region = [region_size[0]//2, region_size[1]//2]
    center_patch = [patch_size[0]//2, patch_size(1)//2]

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


n_img = 5;
n_test = 2
region_size = [45, 37]
patch_size = [5, 5]
radius, angle = cart2polar(region_size)
bins = get_bin(radius, angle, region_size)

similarity_list = []

for file in os.listdir('Input/'):
    img = io.imread(os.path.join('Input/', file))
    imgRgb = transform.rescale(img, 1/3)
    self_similarites = com_self_similarites(imgRgb, region_size, patch_size, bin)
    similarity_list.append(self_similarites)

width = height = 1
center_sub = [width//2, height//2]
p = 1
## self_similarites = cell(1, n_img)

sig_score_list = []

for m in range(len(similarity_list)):
    self_similarites1 = similarity_list[m]
    img_size1 = similarity_list.shape
    src_img = io.imread('Input/{m}.jpg')
    imgRgb = transform.rescale(src_img, 1/3)
    sig_score_img = np.zeros((img_size1[0], img_size1[1]))

    for row1 in range(center_sub[0] + 1, img_size1[0] - center_sub[1]):
        for col1 in range(center_sub[1] + 1, img_size1[1] - center_sub[1] - 1):
            sub1 = self_similarites1[row1-center_sub[0] : row1+center_sub[0],
                                     col1-center_sub[1] : col1+center_sub[1], :]
            max_match = np.zeros(1, len(similarity_list) - 1)
            num_img = 1
            match_score = np.zeros((n_img, n_img))

            for n in range(n_img):
                if n != m:
                    self_similarities2 = similarity_list[n]
                    temp1 = np.full(self_similarities2.shape[0], 
                                    self_similarities2.shape[1], fill_value=sub1)
                    temp2 = -1 * np.sum((self_similarities2 - temp1)**2, axis=3)
                    max_match[0, num_img] = np.max(temp2)
                    match_score[n_img] = temp2.reshape(-1)
                    num_img += 1;

            temp3 = np.array([match_score[0], match_score[1], match_score[2], match_score[3], match_score[4]])
            avgMatch = np.mean(temp3)
            stdMatch = np.std(temp3)
            sig_score_img[row1, col1] = np.sum(max_match - avgMatch) / stdMatch

    sig_score_list.append(sig_score_img)
