import os
import random

import numpy as np
from scipy.spatial import distance

import cv2
import dlib
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

mdl_mean = (78.4263377603, 87.7689143744, 114.895847746)
age = ['2', '6', '12', '20', '32', '43', '50', '60']
gender = ['Male', 'Female']

def model_initialization():

    age_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_age.prototxt',
        'data/age_net.caffemodel')

    gender_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_gender.prototxt',
        'data/gender_net.caffemodel')

    return(age_net, gender_net)

def age_prediction(age_net, gender_net, x, y, w, h, face_img):
    
    face_img = face_img[y:y+h, x:x+w].copy()
    blob = cv2.dnn.blobFromImage(
        np.float32(face_img), 1, (227, 227), mdl_mean, swapRB=True)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender[gender_preds[0].argmax()]

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age[age_preds[0].argmax()]

    return gender, age


def m_calculation(kp1, kp2):

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


def colour_correction(img1, img2, landmarks1):

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


def image_from_mask(img1, first_mask, img2, second_mask):

    combined_mask = np.max([first_mask, second_mask], axis=0)
    inv_combined_mask = 1.0 - combined_mask

    output_img = img1 * inv_combined_mask + img2 * combined_mask

    return output_img


def landmark_detection(img):

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

    eye_point = cv2.convexHull(landmarks[eye_landmarks])
    nm_pont = cv2.convexHull(landmarks[nose_and_mouth_landmarks])
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
    age_net, gender_net = model_initialization()
    bbox, base_land = landmark_detection(base_face)
    for i in range(len(bbox)):
        base_list.append([bbox[i], base_land[i]])
    for faces in base_list:
        x, y, w, h = face_utils.rect_to_bb(faces[0])
        gender, age = age_prediction(age_net, gender_net, x, y, w, h, base_face)
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
        _, landmarks = landmark_detection(swap_face)
        base_landmarks = np.matrix(faces[1])
        face_landmarks = landmarks[0]
        m_value = m_calculation(base_landmarks[eye_landmarks + nose_and_mouth_landmarks], face_landmarks[eye_landmarks + nose_and_mouth_landmarks])

        first_mask = get_mask(base_face, base_landmarks)
        second_mask = get_mask(swap_face, face_landmarks)

        image2_warp = warp_img(swap_face, m_value, base_face.shape)
        mask_warped = warp_img(second_mask, m_value, base_face.shape)

        warped_corrected = colour_correction(base_face, image2_warp, base_landmarks)

        base_face = image_from_mask(
            base_face, first_mask, warped_corrected, mask_warped)

    return base_face


eye_landmarks = list(range(17, 27)) + list(range(36, 48))
nose_and_mouth_landmarks = list(range(27, 35)) + list(range(48, 61))

base_face = cv2.imread('./group.jpg', cv2.IMREAD_COLOR)
faces_swap = faceswap(base_face)
cv2.imwrite("anonymous.png", faces_swap)
