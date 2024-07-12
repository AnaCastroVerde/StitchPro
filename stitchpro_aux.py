## --------------------------------------####### STITCHING SCRIPT #######-------------------------------------------- ##

## ------------------------------------------------ Import packages ------------------------------------------------- ##

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import color, data
import scipy.ndimage as ndi
from skimage import morphology
from skimage.feature import canny
from itertools import combinations
from scipy.spatial.distance import cdist
from skimage.transform import rescale, resize
import argparse
import tifffile
from tiatoolbox.wsicore import WSIReader
import time
from pathlib import Path

## ------------------------------------------------ Parse arguments ------------------------------------------------ ##

parser = argparse.ArgumentParser(
    description='Stitching algorithm for reconstruction of tissue quadrants into \
         a complete pseudo-whole-mount histopathology prostate section.')

parser.add_argument('--input_path', dest = 'input_path', required = True,
                    help = 'Path to folder containing the images.')
parser.add_argument('--output_path', dest = 'output_path', required = True,
                    help = 'Path to store the registration results.')
parser.add_argument('--output_name', dest = 'output_name', required = True,
                    help = 'Name of the stitching output file.')
parser.add_argument('--lr', dest = 'lr', required = True,
                    help = 'Lower-right histopathology fragment.')
parser.add_argument('--ll', dest = 'll', required = True,
                    help = 'Lower-left histopathology fragment.')
parser.add_argument('--ur', dest = 'ur', required = True,
                    help = 'Upper-right histopathology fragment.')
parser.add_argument('--ul', dest = 'ul', required = True,
                    help = 'Upper-left histopathology fragment.')
parser.add_argument('--lr_aux', dest = 'lr_aux', required = True,
                    help = 'Lower-right histopathology fragment (auxiliary).')
parser.add_argument('--ll_aux', dest = 'll_aux', required = True,
                    help = 'Lower-left histopathology fragmen (auxiliary).')
parser.add_argument('--ur_aux', dest = 'ur_aux', required = True,
                    help = 'Upper-right histopathology fragment (auxiliary).')
parser.add_argument('--ul_aux', dest = 'ul_aux', required = True,
                    help = 'Upper-left histopathology fragment (auxiliary).')
parser.add_argument('--median_filter_ur', dest = 'median_filter_ur', default = 20,
                    help = 'Size of median filter to reduce noise on thresholded upper-right fragment (int number).')
parser.add_argument('--median_filter_lr', dest = 'median_filter_lr', default = 20,
                    help = 'Size of median filter to reduce noise on thresholded lower-right fragment (int number).')
parser.add_argument('--median_filter_ll', dest = 'median_filter_ll', default = 20,
                    help = 'Size of median filter to reduce noise on thresholded lower-left fragment (int number).')
parser.add_argument('--median_filter_ul', dest = 'median_filter_ul', default = 20,
                    help = 'Size of median filter to reduce noise on thresholded upper-left fragment (int number).')
parser.add_argument('--closing_ur', dest = 'closing_ur', default = 15,
                    help = 'Binary closing factor to eliminate holes on thresholded upper-right fragment (int number).')
parser.add_argument('--closing_lr', dest = 'closing_lr', default = 15,
                    help = 'Binary closing factor to eliminate holes on thresholded lower-right fragment (int number).')
parser.add_argument('--closing_ll', dest = 'closing_ll', default = 15,
                    help = 'Binary closing factor to eliminate holes on thresholded lower-left fragment (int number).')
parser.add_argument('--closing_ul', dest = 'closing_ul', default = 15,
                    help = 'Binary closing factor to eliminate holes on thresholded upper-left fragment (int number).')
parser.add_argument('--sub_bound_x', dest = 'sub_bound_x', default = 0,
                    help = 'Value to subtract from x boundary of output image, to remove whitespace (int number).')
parser.add_argument('--sub_bound_y', dest = 'sub_bound_y', default = 0,
                    help = 'Value to subtract from y boundary of output image, to remove whitespace (int number).')
parser.add_argument('--original_spacing', dest = 'original_spacing', default = 0.5,
                    help = 'Image spacing of the original image (in micrometers).')
parser.add_argument('--level', dest = 'level', default = 5,
                    help = 'Downsampling level of the original image.')
parser.add_argument('--seed', dest = 'seed', default = 42, help = 'Random seed.')
parser.add_argument('--show_image', dest = 'show_image', action = 'store_true',
                    help = 'Show intermediate images and stitching result.')

args = parser.parse_args()

print("Stitching algorithm in progress..")
start_time = time.time()

## ------------------------------------------------ Read fragments ------------------------------------------------ ##

histo_fragment_lr = imageio.imread(args.input_path + args.lr)
histo_fragment_ll = imageio.imread(args.input_path + args.ll)
histo_fragment_ur = imageio.imread(args.input_path + args.ur)
histo_fragment_ul = imageio.imread(args.input_path + args.ul)

original_height_ur, original_width_ur = histo_fragment_ur.shape[:2]
original_height_lr, original_width_lr = histo_fragment_lr.shape[:2]
original_height_ll, original_width_ll = histo_fragment_ll.shape[:2]
original_height_ul, original_width_ul = histo_fragment_ul.shape[:2]

histo_fragment_lr_aux = imageio.imread(args.input_path + args.lr_aux)
histo_fragment_ll_aux = imageio.imread(args.input_path + args.ll_aux)
histo_fragment_ur_aux = imageio.imread(args.input_path + args.ur_aux)
histo_fragment_ul_aux = imageio.imread(args.input_path + args.ul_aux)

original_height_ur_aux, original_width_ur_aux = histo_fragment_ur_aux.shape[:2]
original_height_lr_aux, original_width_lr_aux = histo_fragment_lr_aux.shape[:2]
original_height_ll_aux, original_width_ll_aux = histo_fragment_ll_aux.shape[:2]
original_height_ul_aux, original_width_ul_aux = histo_fragment_ul_aux.shape[:2]

## ------------------------------------------- Downsample the fragments ------------------------------------------- ##

DOWNSAMPLE_LEVEL = 4  # avoid running the optimization for the entire image
add = 0
histo_fragment_lr = rescale(histo_fragment_lr, 1 / DOWNSAMPLE_LEVEL, channel_axis=2,
                            preserve_range=True).astype(np.uint8)
histo_fragment_ll = rescale(histo_fragment_ll, 1 / DOWNSAMPLE_LEVEL, channel_axis=2,
                            preserve_range=True).astype(np.uint8)
histo_fragment_ur = rescale(histo_fragment_ur, 1 / DOWNSAMPLE_LEVEL, channel_axis=2,
                            preserve_range=True).astype(np.uint8)
histo_fragment_ul = rescale(histo_fragment_ul, 1 / DOWNSAMPLE_LEVEL, channel_axis=2,
                            preserve_range=True).astype(np.uint8)

## --------------------------------------------- Obtain image contours -------------------------------------------- ##

## Convert from RGB image to grayscale
histo_fragment_gray_binary_ll = color.rgb2gray(histo_fragment_ll)
histo_fragment_gray_ll = (histo_fragment_gray_binary_ll * 256).astype('uint8')
histo_fragment_gray_binary_lr = color.rgb2gray(histo_fragment_lr)
histo_fragment_gray_lr = (histo_fragment_gray_binary_lr * 256).astype('uint8')
histo_fragment_gray_binary_ul = color.rgb2gray(histo_fragment_ul)
histo_fragment_gray_ul = (histo_fragment_gray_binary_ul * 256).astype('uint8')
histo_fragment_gray_binary_ur = color.rgb2gray(histo_fragment_ur)
histo_fragment_gray_ur = (histo_fragment_gray_binary_ur * 256).astype('uint8')

if args.show_image == True:
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].imshow(histo_fragment_gray_ul, cmap="gray")
    axs[0, 1].imshow(histo_fragment_gray_ur, cmap="gray")
    axs[1, 0].imshow(histo_fragment_gray_ll, cmap="gray")
    axs[1, 1].imshow(histo_fragment_gray_lr, cmap="gray")

    ## Plot the intensity histogram
    hist_ul = ndi.histogram(histo_fragment_gray_ul, min=0, max=255, bins=256)
    hist_ur = ndi.histogram(histo_fragment_gray_ur, min=0, max=255, bins=256)
    hist_lr = ndi.histogram(histo_fragment_gray_lr, min=0, max=255, bins=256)
    hist_ll = ndi.histogram(histo_fragment_gray_ll, min=0, max=255, bins=256)
    fig, axs = plt.subplots(nrows=4, ncols=1)
    axs[0].plot(hist_ul)
    axs[1].plot(hist_ur)
    axs[2].plot(hist_ll)
    axs[3].plot(hist_lr)

## Image segmentation based on threshold
thresh = 220

image_thresholded_ul = histo_fragment_gray_ul < thresh
image_thresholded_ur = histo_fragment_gray_ur < thresh
image_thresholded_lr = histo_fragment_gray_lr < thresh
image_thresholded_ll = histo_fragment_gray_ll < thresh

if args.show_image == True:
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].imshow(image_thresholded_ul, cmap="gray")
    axs[0, 1].imshow(image_thresholded_ur, cmap="gray")
    axs[1, 0].imshow(image_thresholded_ll, cmap="gray")
    axs[1, 1].imshow(image_thresholded_lr, cmap="gray")

## Apply median filter to reduce the noise
image_thresholded_filtered_ul = ndi.median_filter(image_thresholded_ul, size=int(args.median_filter_ul))
image_thresholded_filtered_ur = ndi.median_filter(image_thresholded_ur, size=int(args.median_filter_ur))
image_thresholded_filtered_ll = ndi.median_filter(image_thresholded_ll, size=int(args.median_filter_ll))
image_thresholded_filtered_lr = ndi.median_filter(image_thresholded_lr, size=int(args.median_filter_lr))

if args.show_image == True:
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].imshow(image_thresholded_filtered_ul, cmap="gray")
    axs[0, 1].imshow(image_thresholded_filtered_ur, cmap="gray")
    axs[1, 0].imshow(image_thresholded_filtered_ll, cmap="gray")
    axs[1, 1].imshow(image_thresholded_filtered_lr, cmap="gray")

## Erode the image to eliminate holes
image_thresholded_filtered_closed_ul = morphology.binary_closing(image_thresholded_filtered_ul,
                                                                 footprint=morphology.square(
                                                                     int(args.closing_ul)))  # morphology.square(30) 28 disk
image_thresholded_filtered_closed_ur = morphology.binary_closing(image_thresholded_filtered_ur,
                                                                 footprint=morphology.square(
                                                                     int(args.closing_ur)))  # morphology.square(84) 26
image_thresholded_filtered_closed_ll = morphology.binary_closing(image_thresholded_filtered_ll,
                                                                 footprint=morphology.square(
                                                                     int(args.closing_ll)))  # morphology.square(100) 30
image_thresholded_filtered_closed_lr = morphology.binary_closing(image_thresholded_filtered_lr,
                                                                 footprint=morphology.square(
                                                                     int(args.closing_lr)))  # morphology.square(150) 26

if args.show_image == True:
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].imshow(image_thresholded_filtered_closed_ul, cmap="gray")
    axs[0, 1].imshow(image_thresholded_filtered_closed_ur, cmap="gray")
    axs[1, 0].imshow(image_thresholded_filtered_closed_ll, cmap="gray")
    axs[1, 1].imshow(image_thresholded_filtered_closed_lr, cmap="gray")

## Detect corner with minimumn intensity

dim_x_ur, dim_y_ur = image_thresholded_filtered_closed_ur.shape[:2]

factor = 1/6

corner_intensity_ur_0 = image_thresholded_filtered_closed_ur[int((factor)*dim_x_ur), int(dim_y_ur-(factor)*dim_y_ur)]
corner_intensity_ur_1 = image_thresholded_filtered_closed_ur[int(dim_x_ur-(factor)*dim_x_ur), int(dim_y_ur-(factor)*dim_y_ur)]
corner_intensity_ur_2 = image_thresholded_filtered_closed_ur[int(dim_x_ur-(factor)*dim_x_ur), int((factor)*dim_y_ur)]
corner_intensity_ur_3 = image_thresholded_filtered_closed_ur[int((factor)*dim_x_ur), int((factor)*dim_y_ur)]
corners = [int(corner_intensity_ur_0), int(corner_intensity_ur_1), int(corner_intensity_ur_2), int(corner_intensity_ur_3)]
background_value_ur = min(corners)
min_index_ur = corners.index(background_value_ur)
image_thresholded_filtered_closed_ur = np.array(image_thresholded_filtered_closed_ur).astype(int)

dim_x_lr, dim_y_lr = image_thresholded_filtered_closed_lr.shape[:2]

corner_intensity_lr_0 = image_thresholded_filtered_closed_lr[int((factor)*dim_x_lr), int(dim_y_lr-(factor)*dim_y_lr)]
corner_intensity_lr_1 = image_thresholded_filtered_closed_lr[int(dim_x_lr-(factor)*dim_x_lr), int(dim_y_lr-(factor)*dim_y_lr)]
corner_intensity_lr_2 = image_thresholded_filtered_closed_lr[int(dim_x_lr-(factor)*dim_x_lr), int((factor)*dim_y_lr)]
corner_intensity_lr_3 = image_thresholded_filtered_closed_lr[int((factor)*dim_x_lr), int((factor)*dim_y_lr)]
corners = [int(corner_intensity_lr_0), int(corner_intensity_lr_1), int(corner_intensity_lr_2), int(corner_intensity_lr_3)]
background_value_lr = min(corners)
min_index_lr = corners.index(background_value_lr)
image_thresholded_filtered_closed_lr = np.array(image_thresholded_filtered_closed_lr).astype(int)

dim_x_ll, dim_y_ll = image_thresholded_filtered_closed_ll.shape[:2]

corner_intensity_ll_0 = image_thresholded_filtered_closed_ll[int((factor)*dim_x_ll), int(dim_y_ll-(factor)*dim_y_ll)]
corner_intensity_ll_1 = image_thresholded_filtered_closed_ll[int(dim_x_ll-(factor)*dim_x_ll), int(dim_y_ll-(factor)*dim_y_ll)]
corner_intensity_ll_2 = image_thresholded_filtered_closed_ll[int(dim_x_ll-(factor)*dim_x_ll), int((factor)*dim_y_ll)]
corner_intensity_ll_3 = image_thresholded_filtered_closed_ll[int((factor)*dim_x_ll), int((factor)*dim_y_ll)]
corners = [int(corner_intensity_ll_0), int(corner_intensity_ll_1), int(corner_intensity_ll_2), int(corner_intensity_ll_3)]
background_value_ll = min(corners)
min_index_ll = corners.index(background_value_ll)
image_thresholded_filtered_closed_ll = np.array(image_thresholded_filtered_closed_ll).astype(int)

dim_x_ul, dim_y_ul = image_thresholded_filtered_closed_ul.shape[:2]

corner_intensity_ul_0 = image_thresholded_filtered_closed_ul[int((factor)*dim_x_ul), int(dim_y_ul-(factor)*dim_y_ul)]
corner_intensity_ul_1 = image_thresholded_filtered_closed_ul[int(dim_x_ul-(factor)*dim_x_ul), int(dim_y_ul-(factor)*dim_y_ul)]
corner_intensity_ul_2 = image_thresholded_filtered_closed_ul[int(dim_x_ul-(factor)*dim_x_ul), int((factor)*dim_y_ul)]
corner_intensity_ul_3 = image_thresholded_filtered_closed_ul[int((factor)*dim_x_ul), int((factor)*dim_y_ul)]
corners = [int(corner_intensity_ul_0), int(corner_intensity_ul_1), int(corner_intensity_ul_2), int(corner_intensity_ul_3)]
background_value_ul = min(corners)
min_index_ul = corners.index(background_value_ul)
image_thresholded_filtered_closed_ul = np.array(image_thresholded_filtered_closed_ul).astype(int)

## Rotation for UR fragment
if min_index_ur == 1:
    image_thresholded_filtered_closed_ur = cv2.rotate(image_thresholded_filtered_closed_ur, cv2.ROTATE_90_COUNTERCLOCKWISE)
    histo_fragment_ur = cv2.rotate(histo_fragment_ur, cv2.ROTATE_90_COUNTERCLOCKWISE)
    histo_fragment_ur_aux = cv2.rotate(histo_fragment_ur_aux, cv2.ROTATE_90_COUNTERCLOCKWISE)
if min_index_ur == 2:
    image_thresholded_filtered_closed_ur = cv2.rotate(image_thresholded_filtered_closed_ur, cv2.ROTATE_180)
    histo_fragment_ur = cv2.rotate(histo_fragment_ur, cv2.ROTATE_180)
    histo_fragment_ur_aux = cv2.rotate(histo_fragment_ur_aux, cv2.ROTATE_180)
if min_index_ur == 3:
    image_thresholded_filtered_closed_ur = cv2.rotate(image_thresholded_filtered_closed_ur, cv2.ROTATE_90_CLOCKWISE)
    histo_fragment_ur = cv2.rotate(histo_fragment_ur, cv2.ROTATE_90_CLOCKWISE)
    histo_fragment_ur_aux = cv2.rotate(histo_fragment_ur_aux, cv2.ROTATE_90_CLOCKWISE)
else:
    image_thresholded_filtered_closed_ur = image_thresholded_filtered_closed_ur
    histo_fragment_ur= histo_fragment_ur
    histo_fragment_ur_aux= histo_fragment_ur_aux
image_thresholded_filtered_closed_ur = np.array(image_thresholded_filtered_closed_ur).astype(bool)

## Rotation for LR fragment
if min_index_lr == 0:
    image_thresholded_filtered_closed_lr = cv2.rotate(image_thresholded_filtered_closed_lr, cv2.ROTATE_90_CLOCKWISE)
    histo_fragment_lr = cv2.rotate(histo_fragment_lr, cv2.ROTATE_90_CLOCKWISE)
    histo_fragment_lr_aux = cv2.rotate(histo_fragment_lr_aux, cv2.ROTATE_90_CLOCKWISE)
if min_index_lr == 2:
    image_thresholded_filtered_closed_lr = cv2.rotate(image_thresholded_filtered_closed_lr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    histo_fragment_lr = cv2.rotate(histo_fragment_lr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    histo_fragment_lr_aux = cv2.rotate(histo_fragment_lr_aux, cv2.ROTATE_90_COUNTERCLOCKWISE)
if min_index_lr == 3:
    image_thresholded_filtered_closed_lr = cv2.rotate(image_thresholded_filtered_closed_lr, cv2.ROTATE_180)
    histo_fragment_lr = cv2.rotate(histo_fragment_lr, cv2.ROTATE_180)
    histo_fragment_lr_aux = cv2.rotate(histo_fragment_lr_aux, cv2.ROTATE_180)
else:
    image_thresholded_filtered_closed_lr = image_thresholded_filtered_closed_lr
    histo_fragment_lr= histo_fragment_lr
    histo_fragment_lr_aux= histo_fragment_lr_aux
image_thresholded_filtered_closed_lr = np.array(image_thresholded_filtered_closed_lr).astype(bool)

## Rotation for ll fragment
if min_index_ll == 0:
    image_thresholded_filtered_closed_ll = cv2.rotate(image_thresholded_filtered_closed_ll, cv2.ROTATE_180)
    histo_fragment_ll = cv2.rotate(histo_fragment_ll, cv2.ROTATE_180)
    histo_fragment_ll_aux = cv2.rotate(histo_fragment_ll_aux, cv2.ROTATE_180)
if min_index_ll == 1:
    image_thresholded_filtered_closed_ll = cv2.rotate(image_thresholded_filtered_closed_ll, cv2.ROTATE_90_CLOCKWISE)
    histo_fragment_ll = cv2.rotate(histo_fragment_ll, cv2.ROTATE_90_CLOCKWISE)
    histo_fragment_ll_aux = cv2.rotate(histo_fragment_ll_aux, cv2.ROTATE_90_CLOCKWISE)
if min_index_ll == 3:
    image_thresholded_filtered_closed_ll = cv2.rotate(image_thresholded_filtered_closed_ll, cv2.ROTATE_90_COUNTERCLOCKWISE)
    histo_fragment_ll = cv2.rotate(histo_fragment_ll, cv2.ROTATE_90_COUNTERCLOCKWISE)
    histo_fragment_ll_aux = cv2.rotate(histo_fragment_ll_aux, cv2.ROTATE_90_COUNTERCLOCKWISE)
else:
    image_thresholded_filtered_closed_ll = image_thresholded_filtered_closed_ll
    histo_fragment_ll= histo_fragment_ll
    histo_fragment_ll_aux= histo_fragment_ll_aux
image_thresholded_filtered_closed_ll = np.array(image_thresholded_filtered_closed_ll).astype(bool)

## Rotation for ul fragment
if min_index_ul == 0:
    image_thresholded_filtered_closed_ul = cv2.rotate(image_thresholded_filtered_closed_ul, cv2.ROTATE_90_COUNTERCLOCKWISE)
    histo_fragment_ul = cv2.rotate(histo_fragment_ul, cv2.ROTATE_90_COUNTERCLOCKWISE)
    histo_fragment_ul_aux = cv2.rotate(histo_fragment_ul_aux, cv2.ROTATE_90_COUNTERCLOCKWISE)
if min_index_ul == 1:
    image_thresholded_filtered_closed_ul = cv2.rotate(image_thresholded_filtered_closed_ul, cv2.ROTATE_180)
    histo_fragment_ul = cv2.rotate(histo_fragment_ul, cv2.ROTATE_180)
    histo_fragment_ul_aux = cv2.rotate(histo_fragment_ul_aux, cv2.ROTATE_180)
if min_index_ul == 2:
    image_thresholded_filtered_closed_ul = cv2.rotate(image_thresholded_filtered_closed_ul, cv2.ROTATE_90_CLOCKWISE)
    histo_fragment_ul = cv2.rotate(histo_fragment_ul, cv2.ROTATE_90_CLOCKWISE)
    histo_fragment_ul_aux = cv2.rotate(histo_fragment_ul_aux, cv2.ROTATE_90_CLOCKWISE)
else:
    image_thresholded_filtered_closed_ul = image_thresholded_filtered_closed_ul
    histo_fragment_ul= histo_fragment_ul
    histo_fragment_ul_aux= histo_fragment_ul_aux
image_thresholded_filtered_closed_ul = np.array(image_thresholded_filtered_closed_ul).astype(bool)

## Identify the image boundary
canny_edges_ul = canny(image_thresholded_filtered_closed_ul, sigma=5)
canny_edges_ur = canny(image_thresholded_filtered_closed_ur, sigma=5)
canny_edges_ll = canny(image_thresholded_filtered_closed_ll, sigma=5)
canny_edges_lr = canny(image_thresholded_filtered_closed_lr, sigma=5)

if args.show_image == True:
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0,0].imshow(histo_fragment_ul)
    axs[0,0].contour(canny_edges_ul, colors='r')
    axs[0,1].imshow(histo_fragment_ur)
    axs[0,1].contour(canny_edges_ur, colors='r')
    axs[1,0].imshow(histo_fragment_ll)
    axs[1,0].contour(canny_edges_ll, colors='r')
    axs[1,1].imshow(histo_fragment_lr)
    axs[1,1].contour(canny_edges_lr, colors='r')
    plt.show()

stitching_time = time.time()

## --------------------------------------- To fit a circle arc to each segment ------------------------------------ ##

def circle_arc_loss_cv(par, mask, pad=20):
    mask = cv2.copyMakeBorder(
        mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    cx, cy, r1, r2, theta_1, theta_plus = par
    cx, cy = cx + pad, cy + pad
    theta_2 = theta_1 + theta_plus
    theta_1, theta_2 = np.rad2deg([theta_1, theta_2])
    O = np.zeros_like(mask)
    cv2.ellipse(O, (int(cy), int(cx)), (int(r1), int(r2)), angle=0,
                startAngle=theta_1, endAngle=theta_2, color=1,
                thickness=-1)
    fn = np.sum((mask == 1) & (O == 0))
    I = np.sum(O * mask)
    U = np.sum(O + mask) - I
    return 1 - I / U + fn / mask.sum()


def calculate_intersection(m1, b1, m2, b2):
    cx = (b2 - b1) / (m1 - m2)
    cy = m1 * cx + b1
    return cx, cy


def pol2cart(rho, phi):
    # from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def radius_at_angle(theta, a, b):
    n = a * b
    d = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    return n / d


from scipy.optimize import NonlinearConstraint
from scipy import optimize

POP_SIZE = 10

data_dict = []

tissue_masks = [image_thresholded_filtered_closed_ur, image_thresholded_filtered_closed_lr,
                image_thresholded_filtered_closed_ll, image_thresholded_filtered_closed_ul]
images = [histo_fragment_ur, histo_fragment_lr, histo_fragment_ll, histo_fragment_ul]
tissue_masks_closed = [canny_edges_ur, canny_edges_lr, canny_edges_ll, canny_edges_ul]

N = len(tissue_masks)

curr_theta = -np.pi
extra_theta = 2 * np.pi

N_segments = len(images)
segment_angle = 2 * np.pi / N_segments

for i in range(len(tissue_masks)):
    x = tissue_masks[i]
    x_out = tissue_masks_closed[i]
    x = x.copy().astype(np.uint8)
    c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    x = np.zeros_like(x)
    Mx, My = x.shape
    M = np.maximum(Mx, My)
    hull = cv2.convexHull(c[0])
    cv2.drawContours(x, [hull], -1, 1, -1)
    points = np.stack(np.where(x > 0), axis=1)
    points_out = np.stack(np.where(x_out > 0), axis=1)
    pad = np.maximum(int(M * 0.1), 100)
    initializations = []

    for init_x in [0, Mx / 2, Mx]:
        for init_y in [0, My / 2, My]:
            initializations.append((init_y, init_x))

    solutions = []
    for init in initializations:
        print("Trying initialization for segment {}: {}".format(i, init))

        bounds = [(0, M), (0, M), (0, M), (0, M),
                  (curr_theta - np.pi / 8, curr_theta + extra_theta),
                  (segment_angle * 0.8, segment_angle * 1.2)]
        x0 = [init[0], init[1], M / 2, M / 2,
              curr_theta, segment_angle]
        solution = optimize.differential_evolution(
            circle_arc_loss_cv, x0=x0, bounds=bounds,
            args=[x, pad], popsize=POP_SIZE, maxiter=250, workers=1)  # updating='immediate'
        solutions.append(solution)

    print([s.fun for s in solutions])
    solution_idx = np.argmin([s.fun for s in solutions])
    solution = solutions[solution_idx]

    cy, cx, r1, r2, theta_1, theta_plus = solution.x
    theta_2 = theta_1 + theta_plus
    curr_theta = theta_1
    extra_theta = theta_plus
    theta = theta_plus
    theta_1_, theta_2_ = np.rad2deg([theta_1, theta_2])
    O = images[i].copy()

    # retrieve points from extremal landmarks
    r_ant = radius_at_angle(theta_1, r1, r2)
    points_cart_ant = np.array([pol2cart(r_ant, theta_1)]) + [cx, cy]
    pca_sorted = np.array([points_cart_ant[0][::-1]])
    point_ant = points_out[
        np.argmin(cdist(points_out, pca_sorted), axis=0)]

    r_pos = radius_at_angle(theta_2, r1, r2)
    points_cart_pos = np.array([pol2cart(r_pos, theta_2)]) + [cx, cy]
    pcp_sorted = np.array([points_cart_pos[0][::-1]])
    point_pos = points_out[
        np.argmin(cdist(points_out, pcp_sorted[::-1]), axis=0)]

    # retrieve points for histogram
    ant_axis_line_mask = np.zeros_like(x, dtype=np.uint8)
    cv2.line(ant_axis_line_mask, [int(cx), int(cy)],
             np.int32(point_ant[0][::-1]), 1, 80)
    pos_points = np.roll(
        np.array(np.where(ant_axis_line_mask * x_out)).T, 1, 1)

    pos_axis_line_mask = np.zeros_like(x, dtype=np.uint8)
    cv2.line(pos_axis_line_mask, [int(cx), int(cy)],
             np.int32(point_pos[0][::-1]), 1, 80)
    ant_points = np.roll(
        np.array(np.where(pos_axis_line_mask * x_out)).T, 1, 1)

    # draw examples
    cv2.ellipse(O, (int(cx), int(cy)), (int(r1), int(r2)), angle=0,
                startAngle=theta_1_, endAngle=theta_2_, color=1,
                thickness=5)
    cv2.drawMarker(O, [int(cx), int(cy)], [0, 255, 0],
                   markerType=cv2.MARKER_CROSS, thickness=10)

    cv2.drawMarker(O, np.int32(points_cart_ant[0]), [0, 0, 255],
                   markerType=cv2.MARKER_CROSS, thickness=10)
    cv2.drawMarker(O, np.int32(points_cart_pos[0]), [0, 0, 255],
                   markerType=cv2.MARKER_TRIANGLE_UP, thickness=10)

    cv2.line(O, [int(cx), int(cy)], np.int32(point_ant[0][::-1]), [255, 128, 128], 1)
    cv2.line(O, [int(cx), int(cy)], np.int32(point_pos[0][::-1]), [128, 255, 128], 1)

    cv2.drawMarker(O, point_ant[0][::-1], [255, 0, 0],
                   markerType=cv2.MARKER_CROSS, thickness=10)
    cv2.drawMarker(O, point_pos[0][::-1], [255, 0, 0],
                   markerType=cv2.MARKER_TRIANGLE_UP, thickness=10)
    if args.show_image == True:
        plt.imshow(O)
        plt.show()

    data_dict.append({
        "image": images[i],
        "tissue_mask": tissue_masks[i],
        "tissue_mask_closed": tissue_masks_closed[i],
        "quadrant": i,
        "ant_line": np.array([[int(cx), int(cy)], np.int32(point_ant[0][::-1])]),
        "pos_line": np.array([[int(cx), int(cy)], np.int32(point_pos[0][::-1])]),
        "ant_points": ant_points,
        "pos_points": pos_points})

print(cx, cy)

## ----------------------------------- Calculate histograms and distances between histograms ------------------------------- ##

# set up functions to calculate colour histograms
square_size = 64
n_bins = 32

def calculate_histogram(image, mask, center, n_bins, size):
    x, y = center
    Mx, My = mask.shape
    x1, x2 = np.maximum(x - size // 2, 0), np.minimum(x + size // 2, Mx)
    y1, y2 = np.maximum(y - size // 2, 0), np.minimum(y + size // 2, My)
    mask = mask[x1:x2, y1:y2]
    sub_image = image[x1:x2, y1:y2]
    sub_image = sub_image.reshape([-1, 3])[mask.reshape([-1]) == 1]

    r_hist = np.histogram(sub_image[:, 0], n_bins, range=[0, 256], density=True)[0]
    g_hist = np.histogram(sub_image[:, 1], n_bins, range=[0, 256], density=True)[0]
    b_hist = np.histogram(sub_image[:, 2], n_bins, range=[0, 256], density=True)[0]

    out = np.concatenate([r_hist, g_hist, b_hist])
    return out

histograms = []
print("Calculating histograms for all sections along the x and y edges...")
for i in range(len(data_dict)):
    data = data_dict[i]
    hx, hy = [], []
    for y, x in data["ant_points"]:
        H = calculate_histogram(
            data['image'], data['tissue_mask'], [x, y], n_bins, square_size)
        hx.append(H)
    for y, x in data["pos_points"]:
        H = calculate_histogram(
            data['image'], data['tissue_mask'], [x, y], n_bins, square_size)
        hy.append(H)
    hx = np.array(hx)
    hy = np.array(hy)
    data_dict[i]['histograms_ant'] = hx
    data_dict[i]['histograms_pos'] = hy

print("Calculating correlation distances between histograms for every tissue section pair...")
histogram_dists = {}
for i, j in combinations(range(len(data_dict) + 1), 2):
    i = i % len(data_dict)
    j = j % len(data_dict)
    histogram_dists[i, j] = {
        "ant_pos": cdist(
            data_dict[i]['histograms_ant'], data_dict[j]['histograms_pos'],
            metric="correlation"),
        "pos_ant": cdist(
            data_dict[i]['histograms_pos'], data_dict[j]['histograms_ant'],
            metric="correlation")}

# ensuring that max distance == 1
for i, j in histogram_dists:
    histogram_dists[i, j]['ant_pos'] = np.divide(
        histogram_dists[i, j]['ant_pos'], histogram_dists[i, j]['ant_pos'].max())
    histogram_dists[i, j]['pos_ant'] = np.divide(
        histogram_dists[i, j]['pos_ant'], histogram_dists[i, j]['pos_ant'].max())

## ------------------------ Differential evolution for optimization of tissue segment translations ------------------------ ##

POPSIZE = 25  # parameter for the evolutionary optimization (population size)
MAXITER = 200  # parameter for the evolutionary optimization (population size)

from skimage.transform import AffineTransform

from scipy.optimize import NonlinearConstraint
from scipy import optimize

np.random.seed(args.seed)

output_size = max([max(x['image'].shape) for x in data_dict]) * 2
output_size = [output_size, output_size]
print('output_size', output_size)


def par_to_H(theta, tx, ty):
    # converts a set of three parameters to
    # a homography matrix
    H = AffineTransform(
        scale=1, rotation=theta, shear=None, translation=[tx, ty])
    return H.params


def M_to_quadrant_dict(M, quadrants, anchor):
    H_dict = {}
    Q = [q for q in quadrants if q != anchor]
    for i, q in enumerate(Q):
        H_dict[q] = par_to_H(*[M[i] for i in range(i * 3, i * 3 + 3)])
    return H_dict


def warp(coords, H):
    out = cv2.perspectiveTransform(
        np.float32(coords[:, np.newaxis, :]), H)[:, 0, :]
    return out


print("Optimizing tissue mosaic...")


def loss_fn(M, quadrants, anchor, data_dict, histogram_dists, max_size, alpha=0.1, d=32):
    # M is a list of parameters for homography matrices (every three parameters is
    # converted into a homography matrix). For convenience, I maintain the upper-left
    # quadrant as the fixed image
    # alpha is a scaling factor for the misalignment loss
    hist_loss = 0.
    mis_loss = 0.
    H_dict = M_to_quadrant_dict(M, quadrants, anchor)
    for idx in range(len(data_dict)):
        i, j = idx, (idx + 1) % len(data_dict)
        data1 = data_dict[i]
        data2 = data_dict[j]
        q1 = data1['quadrant']
        q2 = data2['quadrant']

        axis1 = data1['pos_line']
        axis2 = data2['ant_line']
        points1 = data1['pos_points']
        points2 = data2['ant_points']
        hist_dist = histogram_dists[i, j]["pos_ant"]

        if q1 in H_dict and q1 != anchor:
            axis1 = warp(axis1, H_dict[q1])
            points1 = warp(points1, H_dict[q1])
        if q2 in H_dict and q2 != anchor:
            axis2 = warp(axis2, H_dict[q2])
            points2 = warp(points2, H_dict[q2])
        mis_loss += np.mean((axis1 / max_size - axis2 / max_size) ** 2)
        ss1 = np.random.choice(len(points1),
                               size=len(points1) // 4,
                               replace=False)
        ss2 = np.random.choice(len(points2),
                               size=len(points2) // 4,
                               replace=False)
        point_distance = cdist(points1[ss1], points2[ss2])
        nx, ny = np.where(point_distance < d)
        nx, ny = ss1[nx], ss2[ny]
        hnb = hist_dist[nx, ny]
        if hnb.size > 0:
            h = np.mean(hnb)
        else:
            h = 0.
        # penalty if points have no nearby pixels
        max_dist = hist_dist.max()
        no_nearby_x = point_distance.shape[0] - np.unique(nx).size
        no_nearby_x = no_nearby_x / point_distance.shape[0]
        no_nearby_y = point_distance.shape[1] - np.unique(ny).size
        no_nearby_y = no_nearby_y / point_distance.shape[1]
        hist_loss += h + (no_nearby_x + no_nearby_y) * max_dist
    loss = (mis_loss) * alpha + (1 - alpha) * hist_loss
    return loss

quadrant_list = list(range(len(data_dict)))
anchor = len(data_dict) - 1
bounds = []
# initialize image positions to the edges where they *should* be
x0 = []
a, b, _ = [x['image'] for x in data_dict if x['quadrant'] == anchor][0].shape
for q in quadrant_list:
    if q != anchor:
        bounds.extend(
            [(-np.pi / 6, np.pi / 6),
             (-output_size[0], output_size[0]),
             (-output_size[0], output_size[0])])

        x0.extend([0, a, b])

de_result = optimize.differential_evolution(
    loss_fn, bounds, popsize=POPSIZE, maxiter=MAXITER, disp=True, x0=x0,
    mutation=[0.2, 1.0], seed = args.seed,
    args=[quadrant_list, anchor, data_dict, histogram_dists, output_size[0] / 100, 0.1, square_size // 2])

print(de_result)

if min_index_ur == 1 or min_index_ur == 3 :
    original_height_ur_n = original_width_ur
    original_width_ur_n = original_height_ur
    original_height_ur_aux_n = original_width_ur_aux
    original_width_ur_aux_n = original_height_ur_aux
else:
    original_height_ur_n = original_height_ur
    original_width_ur_n = original_width_ur
    original_height_ur_aux_n = original_height_ur_aux
    original_width_ur_aux_n = original_width_ur_aux

if min_index_lr == 0 or min_index_lr == 2:
    original_height_lr_n = original_width_lr
    original_width_lr_n = original_height_lr
    original_height_lr_aux_n = original_width_lr_aux
    original_width_lr_aux_n = original_height_lr_aux
else:
    original_height_lr_n = original_height_lr
    original_width_lr_n = original_width_lr
    original_height_lr_aux_n = original_height_lr_aux
    original_width_lr_aux_n = original_width_lr_aux

if min_index_ll == 1 or min_index_ll == 3:
    original_height_ll_n = original_width_ll
    original_width_ll_n = original_height_ll
    original_height_ll_aux_n = original_width_ll_aux
    original_width_ll_aux_n = original_height_ll_aux
else:
    original_height_ll_n = original_height_ll
    original_width_ll_n = original_width_ll
    original_height_ll_aux_n = original_height_ll_aux
    original_width_ll_aux_n = original_width_ll_aux

if min_index_ul == 0 or min_index_ul == 2:
    original_height_ul_n = original_width_ul
    original_width_ul_n = original_height_ul
    original_height_ul_aux_n = original_width_ul_aux
    original_width_ul_aux_n = original_height_ul_aux
else:
    original_height_ul_n = original_height_ul
    original_width_ul_n = original_width_ul
    original_height_ul_aux_n = original_height_ul_aux
    original_width_ul_aux_n = original_width_ul_aux

## Rescale images back to original
histo_fragment_ur = resize(histo_fragment_ur, (original_height_ur_n, original_width_ur_n),
                            preserve_range=True).astype(np.uint8)
histo_fragment_ur_aux = resize(histo_fragment_ur_aux, (original_height_ur_aux_n, original_width_ur_aux_n),
                            preserve_range=True).astype(np.uint8)                          
histo_fragment_lr = resize(histo_fragment_lr, (original_height_lr_n, original_width_lr_n),
                            preserve_range=True).astype(np.uint8)
histo_fragment_lr_aux = resize(histo_fragment_lr_aux, (original_height_lr_aux_n, original_width_lr_aux_n),
                            preserve_range=True).astype(np.uint8)
histo_fragment_ll = resize(histo_fragment_ll, (original_height_ll_n, original_width_ll_n),
                            preserve_range=True).astype(np.uint8)
histo_fragment_ll_aux = resize(histo_fragment_ll_aux, (original_height_ll_aux_n, original_width_ll_aux_n),
                            preserve_range=True).astype(np.uint8)
histo_fragment_ul = resize(histo_fragment_ul, (original_height_ul_n, original_width_ul_n),
                            preserve_range=True).astype(np.uint8)
histo_fragment_ul_aux = resize(histo_fragment_ul_aux, (original_height_ul_aux_n, original_width_ul_aux_n),
                            preserve_range=True).astype(np.uint8)


images_original = [histo_fragment_ur, histo_fragment_lr, histo_fragment_ll, histo_fragment_ul]
images_original_aux = [histo_fragment_ur_aux, histo_fragment_lr_aux, histo_fragment_ll_aux, histo_fragment_ul_aux]

output_size = max([max(x.shape) for x in images_original]) * 2
output_size = [output_size, output_size]
output = np.zeros((*output_size, 3), dtype=np.int32)
output_aux = np.zeros((*output_size, 3), dtype=np.int32)
output_d = np.zeros((*output_size, 3), dtype=np.int32)
output_d_aux = np.zeros((*output_size, 3), dtype=np.int32)

sc = np.array(
    [[1, 1, DOWNSAMPLE_LEVEL], [1, 1, DOWNSAMPLE_LEVEL], [0, 0, 1]])

axis_1_final = []
axis_2_final = []
H_dict = M_to_quadrant_dict(
    de_result.x, quadrant_list, anchor)
for i, data in enumerate(data_dict):
    image = images_original[i][:, :, :3]
    image_aux = images_original_aux[i][:, :, :3]
    sh = image.shape
    mask = resize(data['tissue_mask'], [sh[0], sh[1]], order=0)
    q = data['quadrant']
    if q in H_dict:
        print(q)
        im = cv2.warpPerspective(
            np.uint8(image * mask[:, :, np.newaxis]),
            sc * H_dict[q], output_size)
        im_aux = cv2.warpPerspective(
            np.uint8(image_aux * mask[:, :, np.newaxis]),
            sc * H_dict[q], output_size)
        axis_1_q = data['pos_line']
        axis_1_q = warp(axis_1_q, H_dict[q])
        print("AXIS1", axis_1_q)
        axis_2_q = data['ant_line']
        axis_2_q = warp(axis_2_q, H_dict[q])
        print("AXIS2", axis_2_q)
        axis_1_final.append(axis_1_q)
        axis_2_final.append(axis_2_q)
    else:
        print(q)
        im = cv2.warpPerspective(
            np.uint8(image * mask[:, :, np.newaxis]),
            par_to_H(0, 0, 0), output_size)
        im_aux = cv2.warpPerspective(
            np.uint8(image_aux * mask[:, :, np.newaxis]),
            par_to_H(0, 0, 0), output_size)
        axis_1_anchor = data['pos_line']
        axis_1_anchor = warp(axis_1_anchor, par_to_H(0,0,0))
        print("AXIS1_anchor", axis_1_anchor)
        axis_2_anchor = data['ant_line']
        axis_2_anchor = warp(axis_2_anchor, par_to_H(0,0,0))
        print("AXIS2_anchor", axis_2_anchor)
        axis_1_final.append(axis_1_anchor)
        axis_2_final.append(axis_2_anchor)

    if q in quadrant_list:
        output[im[:, :, :] > 0] += im[im[:, :, :] > 0]
        output_aux[im_aux[:, :, :] > 0] += im_aux[im_aux[:, :, :] > 0]
        output_d[im[:, :, :] > 0] += 1
        output_d_aux[im_aux[:, :, :] > 0] += 1

## -------------------------------------------------- Compute Euclidean distance and time ------------------------------------------ ##

euclidean_distance_0_1_center = np.sqrt((axis_2_final[1][0][0] - axis_1_final[0][0][0]) ** 2 + (axis_2_final[1][0][1] - axis_1_final[0][0][1]) ** 2)
euclidean_distance_0_1_out = np.sqrt((axis_2_final[1][1][0] - axis_1_final[0][1][0]) ** 2 + (axis_2_final[1][1][1] - axis_1_final[0][1][1]) ** 2)
euclidean_distance_1_2_center = np.sqrt((axis_2_final[2][0][0] - axis_1_final[1][0][0]) ** 2 + (axis_2_final[2][0][1] - axis_1_final[1][0][1]) ** 2)
euclidean_distance_1_2_out = np.sqrt((axis_2_final[2][1][0] - axis_1_final[1][1][0]) ** 2 + (axis_2_final[2][1][1] - axis_1_final[1][1][1]) ** 2)
euclidean_distance_2_3_center = np.sqrt((axis_2_final[3][0][0] - axis_1_final[2][0][0]) ** 2 + (axis_2_final[3][0][1] - axis_1_final[2][0][1]) ** 2)
euclidean_distance_2_3_out = np.sqrt((axis_2_final[3][1][0] - axis_1_final[2][1][0]) ** 2 + (axis_2_final[3][1][1] - axis_1_final[2][1][1]) ** 2)
euclidean_distance_3_0_center = np.sqrt((axis_2_final[0][0][0] - axis_1_final[3][0][0]) ** 2 + (axis_2_final[0][0][1] - axis_1_final[3][0][1]) ** 2)
euclidean_distance_3_0_out = np.sqrt((axis_2_final[0][1][0] - axis_1_final[3][1][0]) ** 2 + (axis_2_final[0][1][1] - axis_1_final[3][1][1]) ** 2)

average_euclidean_distance_units = (euclidean_distance_0_1_center + euclidean_distance_0_1_out + euclidean_distance_1_2_center
                              + euclidean_distance_1_2_out + euclidean_distance_2_3_center + euclidean_distance_2_3_out
                              + euclidean_distance_3_0_center + euclidean_distance_3_0_out)/8

output[output == 0] = 255
output = np.where(output_d > 1, output / output_d, output)
output[np.sum(output, axis = -1) > 650] = 0
output = output.astype(np.uint8)

#output_aux[output_aux == 0] = 255
#output_aux = np.where(output_d_aux > 1, output_aux / output_d_aux, output_aux)
output_aux = output_aux.astype(np.uint8)

if args.show_image == True:
    fig, ax = plt.subplots(1,2)
    ax[0] = plt.imshow(output)
    ax[1] = plt.imshow(output_aux)
    plt.show()

reader = WSIReader.open(output)
info_dict = reader.info.as_dict()
## Remove the excessive white space around the output image
bounds = [0, 0, info_dict['level_dimensions'][0][0]-int(args.sub_bound_x), info_dict['level_dimensions'][0][1]-int(args.sub_bound_y)]
region = reader.read_bounds(bounds, resolution=0, units="level", coord_space = "resolution")

reader_aux = WSIReader.open(output_aux)
info_dict_aux = reader_aux.info.as_dict()
## Remove the excessive white space around the output image
bounds = [0, 0, info_dict_aux['level_dimensions'][0][0]-int(args.sub_bound_x), info_dict['level_dimensions'][0][1]-int(args.sub_bound_y)]
region_aux = reader_aux.read_bounds(bounds, resolution=0, units="level", coord_space = "resolution")

original_spacing = (float(args.original_spacing), float(args.original_spacing))
new_spacing = (2**int(args.level))*original_spacing[0]

tifffile.imwrite(args.output_path+str(args.output_name)+".tif", np.array(region), photometric='rgb', imagej=True, resolution = (1/new_spacing,1/new_spacing), metadata={'spacing': new_spacing, 'unit': 'um'})
tifffile.imwrite(args.output_path+str(args.output_name)+"_aux.tif", np.array(region_aux), photometric='rgb', imagej=True, resolution = (1/new_spacing,1/new_spacing), metadata={'spacing': new_spacing, 'unit': 'um'})

average_euclidean_distance_mm = average_euclidean_distance_units * new_spacing * (10**(-3))
print('Average Euclidean Distance between corner points:', round(average_euclidean_distance_mm,2), 'millimeters')

end_time = time.time()
stitching_time_calc = end_time - stitching_time
elapsed_time = end_time - start_time
print('Execution time of stitching:', round(stitching_time_calc,2), 'seconds')
print('Total execution time of algorithm:', round(elapsed_time,2), 'seconds')

## Open a file and append Dice score
file_object = open(str(Path(str(args.input_path)).parent)+"/Stitching_quantitative.txt", "a")
file_object.write(str(args.input_path)+";")
file_object.write(str(args.output_name)+";")
file_object.write(str(round(average_euclidean_distance_mm,2))+";")
file_object.write(str(round(elapsed_time,2))+"\n")
file_object.close()