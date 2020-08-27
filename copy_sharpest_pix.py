#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import pprint
import argparse
import datetime

import cv2
import numpy

def printProgressBar (iteration=0, total=0, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def fix_image_size(image, expected_pixels=2E6):
    ratio = float(expected_pixels) / float(image.shape[0] * image.shape[1])
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur(image, threshold=100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = numpy.var(blur_map)
    return blur_map, score, bool(score < threshold)

def pretty_blur_map(blur_map, sigma=5):
    abs_image = numpy.log(numpy.abs(blur_map).astype(numpy.float32))
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)


def get_pix_dict(pix_dir = "", end_index = 4):
    dict_imgs = {}
    if pix_dir and os.path.exists(pix_dir):
        for img in os.listdir(pix_dir):
            filename = os.path.join(pix_dir, img)
            img_name = str(img).split(".")[0]
            key_name = "".join(img_name[:end_index])
            if key_name in dict_imgs:
                dict_imgs[key_name].append(filename)
            else:
                dict_imgs[key_name] = [filename]
                
    return dict_imgs

def main():
    parser = argparse.ArgumentParser(description='Copy Sharpest Pix')
    parser.add_argument('-i', '--input_dir', dest="input_dir", type=str, required=True, help="directory of images")
    parser.add_argument('-o', '--save_dir', dest='save_dir', type=str, required=False, help="path to save output")
    parser.add_argument("-t", "--threshold", dest='threshold', type=float, default=100, help="blurry threshold")


    args = parser.parse_args()
    if args:
        dict_pix = get_pix_dict(args.input_dir)

        dst_dir = args.save_dir
        if not dst_dir:
            dst_dir = os.path.join(args.input_dir, "selected")

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        rbr = 0
        dt_start = datetime.datetime.now()

        for key in dict_pix:
            lst_img = dict_pix.get(key)
            this_img_path = ""
            max_score = 0
            if lst_img:

                for img_path in lst_img:
                    if str(key).startswith("C"):    # image name start with C
                        input_image = cv2.imread(img_path, args.threshold)
                        blur_map, score, blurry = estimate_blur(input_image)

                        if score > max_score:
                            max_score = score
                            this_img_path = img_path
                    else:
                        shutil.move(img_path, dst_dir)

                if this_img_path:
                    shutil.move(this_img_path, dst_dir)

            rbr += 1
            printProgressBar(rbr, len(dict_pix))

        dt_end = datetime.datetime.now() - dt_start
        print (f"Finish time: {dt_end}")

if __name__ == "__main__":
    main()