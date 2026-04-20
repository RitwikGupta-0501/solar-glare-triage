import glob
import os
import random

import cv2
import numpy as np


def add_glare(image):
    h, w = image.shape[:2]
    overlay = image.copy()

    cx = random.randint(int(w * 0.2), int(w * 0.8))
    cy = random.randint(int(h * 0.2), int(h * 0.8))

    axes = (random.randint(w // 4, w // 2), random.randint(h // 10, h // 6))
    angle = random.randint(0, 180)

    cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, (255, 255, 255), -1)

    sigma = random.randint(50, 150)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=sigma, sigmaY=sigma)

    alpha = random.uniform(0.6, 0.85)
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return output


def process_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = glob.glob(os.path.join(input_dir, "*.jpg"))
    print(f"Injecting glare into {len(images)} images...")

    for img_path in images:
        name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        dirty_img = add_glare(img)
        cv2.imwrite(os.path.join(output_dir, name), dirty_img)


if __name__ == "__main__":
    process_dataset("data/raw", "data/dirty")
    print("Done! Check data/dirty/")
