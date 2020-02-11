import cv2
import dlib
import numpy as np
from scipy.spatial import distance

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

def getM(kp1, kp2):

	kp1 = np.float64(kp1)
	kp2 = np.float64(kp2)

	c1 = np.mean(kp1, axis = 0)
	c2 = np.mean(kp2, axis = 0)

	kp1 = kp1 - c1
	kp2 = kp2 - c2

	s1 = np.std(kp1)
	s2 = np.std(kp2)

	kp1 = kp1 / s1
	kp2 = kp2 / s2

	U, S, Vt = np.linalg.svd(kp1.T * kp2)

	R = (U * Vt).T

	M = np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T))])

	return M

def correct_color(img1, img2, landmarks1):

	left_eye_center = np.mean(landmarks1[list(range(42, 48))], axis = 0)
	right_eye_center = np.mean(landmarks1[list(range(36 , 42))], axis = 0)

	blur = int(0.6 * distance.euclidean(left_eye_center, right_eye_center))
	blur += int(blur % 2 == 0)

	img1_blur = cv2.GaussianBlur(img1, (blur, blur), 0)
	img2_blur = cv2.GaussianBlur(img2, (blur, blur), 0)

	img2_blur = img2_blur + np.uint8(128 * (img2_blur <= 1.0))

	corrected_img = np.float64(img2) * np.float64(img1_blur) / np.float64(img2_blur)

	return corrected_img


def mask_img(img1, mask1, img2, mask2):

	combined_mask = np.max([mask1, mask2], axis = 0)
	inv_combined_mask = 1.0 - combined_mask

	output_img = img1 * inv_combined_mask + img2 * combined_mask

	return output_img

def get_landmarks(img):

	bbox = detector(img, 1)

	landmark_list = []
	for i in predictor(img, bbox[0]).parts():
		landmark_list.append([i.x, i.y])

	return np.matrix(landmark_list)

def get_mask(img, landmarks):

	img = np.zeros(img.shape[:2], dtype=np.float64)

	eye_point = cv2.convexHull(landmarks[EYES])
	nm_pont = cv2.convexHull(landmarks[NOSE_MOUTH])
	cv2.fillConvexPoly(img, eye_point, color=1)
	cv2.fillConvexPoly(img, nm_pont, color=1)

	img = np.array([img, img, img]).transpose((1, 2, 0))

	img = cv2.GaussianBlur(img, (11, 11), 0)
	img[img > 0.0] = 1.0

	img = cv2.GaussianBlur(img, (15, 15), 0)

	return img

def warp_img(img, M, dshape):

	warped_img = np.zeros(dshape, dtype=img.dtype)

	cv2.warpAffine(img, M, (dshape[1], dshape[0]),
		dst = warped_img,
		borderMode = cv2.BORDER_TRANSPARENT,
		flags = cv2.WARP_INVERSE_MAP)
	return warped_img

def faceswap(img1, img2):
	landmarks1 = get_landmarks(img1)
	landmarks2 = get_landmarks(img2)

	M = getM(landmarks1[EYES + NOSE_MOUTH], landmarks2[EYES + NOSE_MOUTH])

	mask1 = get_mask(img1, landmarks1)
	mask2 = get_mask(img2, landmarks2)

	warped_img2 = warp_img(img2, M, img1.shape)
	warped_mask2 = warp_img(mask2, M, img1.shape)

	warped_corrected_img2 = correct_color(img1, warped_img2, landmarks1)

	output = mask_img(img1, mask1, warped_corrected_img2, warped_mask2)

	return output

EYES = list(range(17, 27)) + list(range(36, 48))
NOSE_MOUTH = list(range(27, 35)) + list(range(48, 61))

img1 = cv2.imread(input('Filename of the first image:\n'), cv2.IMREAD_COLOR)
img2 = cv2.imread(input('Filename of the second image:\n'), cv2.IMREAD_COLOR)

first_out = cv2.imwrite("first_to_second.png", faceswap(img1, img2))
second_image = cv2.imwrite("second_to_first.png", faceswap(img2, img1))
