import os

import matplotlib.image as mpimg
import numpy as np
from scipy.spatial import Delaunay

import cv2
import dlib

dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(
    'data/shape_predictor_68_face_landmarks.dat')


def bgr2rgb(img):
    # OpenCV's BGR to RGB
    rgb = np.copy(img)
    rgb[..., 0], rgb[..., 2] = img[..., 2], img[..., 0]
    return rgb


def list_imgpaths(imgfolder):
    for fname in os.listdir(imgfolder):
        if (fname.lower().endswith('.jpg') or
            fname.lower().endswith('.png') or
                fname.lower().endswith('.jpeg')):
            yield os.path.join(imgfolder, fname)


def boundary_points(points, width_percent=0.1, height_percent=0.1):
    x, y, w, h = cv2.boundingRect(np.array([points], np.int32))
    spacerw = int(w * width_percent)
    spacerh = int(h * height_percent)

    return [[x + spacerw, y + spacerh],
            [x + w - spacerw, y + spacerh]]


def average_points(point_set):
    return np.mean(point_set, 0).astype(np.int32)


def face_points_dlib(img, add_boundary_points=True):
    try:
        points = []
        rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rects = dlib_detector(rgbimg, 1)

        if rects and len(rects) > 0:
            shapes = dlib_predictor(rgbimg, rects[0])
            points = np.array([(shapes.part(i).x, shapes.part(i).y)
                               for i in range(68)], np.int32)

            if add_boundary_points:
                points = np.vstack([points,
                                    boundary_points(points, 0.1, -0.03),
                                    boundary_points(points, 0.13, -0.05),
                                    boundary_points(points, 0.15, -0.08),
                                    boundary_points(points, 0.33, -0.12)])

        return points

    except Exception as e:
        print(e)

        return[]


def bilinear_interpolate(img, coords):
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    q11 = img[y0, x0]
    q21 = img[y0, x0+1]
    q12 = img[y0+1, x0]
    q22 = img[y0+1, x0+1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T


def grid_coordinates(points):
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1

    return np.asarray([(x, y) for y in range(ymin, ymax) for x in range(xmin, xmax)], np.uint32)


def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    roi_coords = grid_coordinates(dst_points)
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None


def triangular_affine_matrices(vertices, src_points, dest_points):
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, [1, 1, 1]))
        dst_tri = np.vstack((dest_points[tri_indices, :].T, [1, 1, 1]))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]

        yield mat


def warp_image(src_img, src_points, dest_points, dest_shape, dtype=np.uint8):
    src_img = src_img[:, :, :3]

    rows, cols = dest_shape[:2]
    result_img = np.zeros((rows, cols, 3), dtype)

    delaunay = Delaunay(dest_points)
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dest_points)))

    process_warp(src_img, result_img, tri_affines, dest_points, delaunay)

    return result_img


def positive_cap(num):
    if num < 0:
        return 0, abs(num)
    else:
        return num, 0


def roi_coordinates(rect, size, scale):
    rectx, recty, rectw, recth = rect
    new_h, new_w = size
    mid_x = int((rectx + rectw/2) * scale)
    mid_y = int((recty + recth/2) * scale)
    roi_x = mid_x - int(new_w/2)
    roi_y = mid_y - int(new_h/2)

    roi_x, border_x = positive_cap(roi_x)
    roi_y, border_y = positive_cap(roi_y)

    return roi_x, roi_y, border_x, border_y


def scaling_factor(rect, size):
    new_height, new_width = size
    rect_h, rect_w = rect[2:]
    h_ratio = rect_h / new_height
    w_ratio = rect_w / new_width
    scale = 1
    if h_ratio > w_ratio:
        new_recth = 0.8 * new_height
        scale = new_recth / rect_h
    else:
        new_rectw = 0.8 * new_width
        scale = new_rectw / rect_w

    return scale


def resize_image(img, scale):
    im_h, im_w = img.shape[:2]
    new_height = int(scale * im_h)
    new_width = int(scale * im_w)

    return cv2.resize(img, (new_width, new_height))


def mask_from_points(size, points):
    kernel = np.ones((10, 10), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    mask = cv2.erode(mask, kernel)

    return mask


def resize_align(img, points, size):
    new_height, new_width = size

    rect = cv2.boundingRect(np.array([points], np.int32))
    scale = scaling_factor(rect, size)
    img = resize_image(img, scale)

    cur_h, cur_w = img.shape[:2]
    roi_x, roi_y, border_x, border_y = roi_coordinates(rect, size, scale)
    roi_h = np.min([new_height - border_y, cur_h - roi_y])
    roi_w = np.min([new_width - border_x, cur_w - roi_x])

    crop = np.zeros((new_height, new_width, 3), img.dtype)
    crop[border_y:border_y + roi_h, border_x:border_x + roi_w] = (
        img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w])

    points[:, 0] = (points[:, 0] * scale) + (border_x - roi_x)
    points[:, 1] = (points[:, 1] * scale) + (border_y - roi_y)

    return(crop, points)


def load_image_points(path, size):
    img = cv2.imread(path)
    points = face_points_dlib(img)

    if len(points) == 0:
        print('No face found in %s' % path)

        return None, None
    else:

        return resize_align(img, points, size)


def overlay_image(fg_image, mask, bg_image):
    fg_pixels = mask > 0
    bg_image[..., :3][fg_pixels] = fg_image[..., :3][fg_pixels]

    return bg_image


def averager(imgpaths):
    size = (500, 600)
    images = []
    point_set = []
    for path in imgpaths:
        img, points = load_image_points(path, size)
        if img is not None:
            images.append(img)
            point_set.append(points)
    dest_img = np.zeros(images[0].shape, np.uint8)
    dest_points = average_points(point_set)
    num_images = len(images)
    result_images = np.zeros(images[0].shape, np.float32)
    for i in range(num_images):
        result_images += warp_image(images[i],
                                    point_set[i], dest_points, size, np.float32)
    result_image = np.uint8(result_images / num_images)
    face_indexes = np.nonzero(result_image)
    dest_img[face_indexes] = result_image[face_indexes]

    mpimg.imsave('./result.png', cv2.cvtColor(dest_img, cv2.COLOR_BGR2RGB))


averager(list_imgpaths('images'))
