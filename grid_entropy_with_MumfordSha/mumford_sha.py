import cv2
from AmbrosioTortorelliMinimizer import *
from tqdm import tqdm
import os


def apply_mumford_sha(img, alpha=10000, beta=0.1, epsilon=0.01, kernel_size=5):
    gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    sharped = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0, gaussian)

    result, edges = [], []
    for c in cv2.split(sharped):
        solver = AmbrosioTortorelliMinimizer(c, alpha=alpha, beta=beta, epsilon=epsilon)

        f, v = solver.minimize()
        result.append(f)
        edges.append(v)

    result = cv2.merge(result)
    edges = np.maximum(*edges)

    # shaprned_result = cv2.filter2D(result, -1, sharp_kernel)
    gaussian_result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    sharped_result = cv2.addWeighted(result, 1.5, gaussian_result, -0.5, 0, gaussian_result)
    return sharped_result


filenames = [f for f in os.listdir('Images/Images_from_Liu_Bolin_s_site/')
                 if f.endswith('PNG')]

filename_pbar = tqdm(filenames)
for fn in filename_pbar:
    filename_pbar.set_description("Processing %s" % fn)
    img = cv2.imread('Images/Images_from_Liu_Bolin_s_site/' + fn)

    # sharp_kernel = np.array(([0, -1, 0],
    #                      [-1, 5, -1],
    #                      [0, -1, 0])).astype(dtype=np.float)
    # shaprned = cv2.filter2D(img, -1, sharp_kernel)

    first_iter = apply_mumford_sha(img)
    second_iter = apply_mumford_sha(first_iter, alpha=100000)

    # cv2.imwrite('mumfordshah/'+fn, np.hstack((first_iter, second_iter)))
    cv2.imwrite('mumfordshah/'+fn, first_iter)
    break
