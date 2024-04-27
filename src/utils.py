import cv2
import os
import matplotlib.pyplot as plt

def calculate_sharpness(image, thresh = 500):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    q_score = round(laplacian)
    if q_score >= thresh:
        norm_q_score = 1
    else:
        norm_q_score = q_score/thresh

    # norm score 0 to 0.3 very poor
    # norm score 0.3 to 0.6 poor
    # norm score 0.6 to 0.8 moderate
    # norm score 0.8>  good
    return norm_q_score

def assess_image_quality(image_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return

    # Calculate image sharpness
    sharpness = calculate_sharpness(image)

    # Display image sharpness
    print(f"Image Sharpness (Laplacian): {sharpness}")

    # Compute the variance of Laplacian (VoL)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vol = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Display VoL (a higher value indicates less blurriness)
    print(f"Variance of Laplacian (VoL): {vol}")

    # Optionally, display the image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the path to your image
    bad_img_list = ["VizWiz_val_00000006.jpg", "VizWiz_val_00000007.jpg", "VizWiz_val_00000008.jpg",
     "VizWiz_val_00000010.jpg", "VizWiz_val_00000014.jpg"]
    good_img_list = ["VizWiz_val_00000018.jpg", "VizWiz_val_00000019.jpg", "VizWiz_val_00000023.jpg", "VizWiz_val_00000024.jpg",
     "VizWiz_val_00000025.jpg"]

    imgs_ls = bad_img_list + good_img_list
    print(len(imgs_ls))

    base_path = "./data/train/images"

    for img_name in imgs_ls:
        img_path = os.path.join(base_path, img_name)
        # Assess the image quality
        thresh = 500
        image = cv2.imread(img_path)
        norm_q_score = calculate_sharpness(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.title(f"image quality score: {round(norm_q_score, 3)}")
        plt.axis('off')
        plt.savefig(f'plots/image_quality/{img_name}')
