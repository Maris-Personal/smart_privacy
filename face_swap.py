import os
import random

import numpy as np
from scipy.spatial import distance

import cv2
import dlib
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['2', '6', '12', '20', '32', '43', '50', '60']
gender_list = ['Male', 'Female']

def initialize_caffe_models():

    age_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_age.prototxt',
        'data/age_net.caffemodel')

    gender_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_gender.prototxt',
        'data/gender_net.caffemodel')

    return(age_net, gender_net)

def predict_age(age_net, gender_net, x, y, w, h, face_img):
    
    face_img = face_img[y:y+h, x:x+w].copy()
    blob = cv2.dnn.blobFromImage(
        np.float32(face_img), 1, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    return gender, age


def getM(kp1, kp2):

    kp1 = np.float64(kp1)
    kp2 = np.float64(kp2)

    c1 = np.mean(kp1, axis=0)
    c2 = np.mean(kp2, axis=0)

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

    left_eye_center = np.mean(landmarks1[list(range(42, 48))], axis=0)
    right_eye_center = np.mean(landmarks1[list(range(36, 42))], axis=0)

    blur = int(0.6 * distance.euclidean(left_eye_center, right_eye_center))
    blur += int(blur % 2 == 0)

    img1_blur = cv2.GaussianBlur(img1, (blur, blur), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur, blur), 0)

    img2_blur = img2_blur + np.uint8(128 * (img2_blur <= 1.0))

    corrected_img = np.float64(
        img2) * np.float64(img1_blur) / np.float64(img2_blur)

    return corrected_img


def mask_img(img1, mask1, img2, mask2):

    combined_mask = np.max([mask1, mask2], axis=0)
    inv_combined_mask = 1.0 - combined_mask

    output_img = img1 * inv_combined_mask + img2 * combined_mask

    return output_img


def get_landmarks(img):

    bbox = detector(img, 1)

    
    bbox_land = []
    for j in range(len(bbox)):
        landmark_list = []
        for i in predictor(img, bbox[j]).parts():
            landmark_list.append([i.x, i.y])
        bbox_land.append(np.matrix(landmark_list))

    return bbox, bbox_land


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
                   dst=warped_img,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return warped_img


def faceswap(base_face):
    base_list = []
    age_net, gender_net = initialize_caffe_models()
    bbox, base_land = get_landmarks(base_face)
    for i in range(len(bbox)):
        base_list.append([bbox[i], base_land[i]])
    for faces in base_list:
        x, y, w, h = face_utils.rect_to_bb(faces[0])
        gender, age = predict_age(age_net, gender_net, x, y, w, h, base_face)
        if gender == 'Male':
            if int(age) <= 20:
                swap_face = cv2.imread(
                    r'./results/m/c/'+random.choice(os.listdir(r'./results/m/c/')), cv2.IMREAD_COLOR)
            elif 20 < int(age) <= 50:
                swap_face = cv2.imread(
                    r'./results/m/a/'+random.choice(os.listdir(r'./results/m/a/')), cv2.IMREAD_COLOR)
            elif int(age) > 50:
                swap_face = cv2.imread(
                    r'./results/m/o/'+random.choice(os.listdir(r'./results/m/o/')), cv2.IMREAD_COLOR)
        elif gender == 'Female':
            if int(age) <= 20:
                swap_face = cv2.imread(
                    r'./results/f/c/'+random.choice(os.listdir(r'./results/f/c/')), cv2.IMREAD_COLOR)
            elif 20 < int(age) <= 50:
                swap_face = cv2.imread(
                    r'./results/f/a/'+random.choice(os.listdir(r'./results/f/a/')), cv2.IMREAD_COLOR)
            elif int(age) > 50:
                swap_face = cv2.imread(
                    r'./results/f/o/'+random.choice(os.listdir(r'./results/f/o/')), cv2.IMREAD_COLOR)
        _, landmarks = get_landmarks(swap_face)
        base_landmarks = np.matrix(faces[1])
        face_landmarks = landmarks[0]
        M = getM(base_landmarks[EYES + NOSE_MOUTH], face_landmarks[EYES + NOSE_MOUTH])

        mask1 = get_mask(base_face, base_landmarks)
        mask2 = get_mask(swap_face, face_landmarks)

        warped_img2 = warp_img(swap_face, M, base_face.shape)
        warped_mask2 = warp_img(mask2, M, base_face.shape)

        warped_corrected_img2 = correct_color(base_face, warped_img2, base_landmarks)

        base_face = mask_img(
            base_face, mask1, warped_corrected_img2, warped_mask2)

    return base_face


EYES = list(range(17, 27)) + list(range(36, 48))
NOSE_MOUTH = list(range(27, 35)) + list(range(48, 61))

base_face = cv2.imread('./group.jpg', cv2.IMREAD_COLOR)
faces_swap = faceswap(base_face)
first_out = cv2.imwrite("anonymous.png", faces_swap)
