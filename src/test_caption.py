from infer import create_vision_gpt_model
from infer_vision_gpt2 import create_vision_gpt_hf
import pandas as pd
from datetime import date
import torch
from time import process_time
from PIL import Image
import os
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class CaptionGenerator:
    def __init__(self, model_name, model_path=None):
        if model_name == "VisionGPT":
            if model_path is not None:
                self.model = create_vision_gpt_model(model_path)
            else:
                raise {"model path is None"}
        if model_name == "VisionGPT_hf":
            self.model = create_vision_gpt_hf()

    def get_caption(self, img):
        return self.model.generate_caption(img)
    



def find_files(directory, pattern):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(pattern):  # You can customize this condition
                matched_files.append((os.path.join(root, file), file))
    return matched_files








    

if __name__ == "__main__":
    print(f"Cuda is :{torch.cuda.is_available()}")
    
    # give path of the model that you want use
    MODEL_NAME = "VisionGPT_hf"
    # required for visiongpt
    # MODEL_PATH = "/media/arnav/MEDIA/Abhishek/vision_gpt2/captioner_old/captioner_2_new.pt"
    caption_genrator = CaptionGenerator(
        model_name = MODEL_NAME,
    )
    # test infer for one image
    infer_image = True
    if infer_image:
        img_path = "/media/arnav/MEDIA/Abhishek/vision_gpt2/test_image.jpg"
        image = Image.open(img_path).convert('RGB')
        prediction = caption_genrator.get_caption(image)
        
    # test infer for file
    infer_on_file = False
    if infer_on_file:
        img_dir_pth = "/media/arnav/MEDIA/Abhishek/vision_gpt2/data/val/images"
        file_ls = find_files(directory=img_dir_pth, pattern=".jpg")
        img_df = pd.DataFrame(file_ls, columns=["image_path", "image_name"])
        num_img = img_df.shape[0]

        # annot_img_csv_pth = '/media/arnav/MEDIA/Abhishek/vision_gpt2/data/ImageSpeak/viz_data.csv'
        # annot_df = pd.read_csv(annot_img_csv_pth,sep='|')

        # mg_df = pd.merge(img_df, annot_df, left_on="image_name", right_on="image_name")
        # mg_df.rename({'caption': 'orginal_caption'}, inplace=True, axis = "columns")
        output_ls = []

        for i in range(num_img):
            img_path = img_df.loc[i, "image_path"]
            img_name = img_df.loc[i,"image_name"],
            # orignal_caption = mg_df.loc[i, "orginal_caption"]
            image = Image.open(img_path)
            start_time = process_time()

            predicted_caption = caption_genrator.get_caption(image)
            end_time = process_time()
            diff_time = end_time -start_time
            output_ls.append((img_name, predicted_caption))
            # plt.imshow(image.convert('RGB'))
            # plt.title(f"actual: {orignal_caption}\n predict: {predicted_caption}", fontsize=7)
            # plt.axis('off')
            # plt.savefig(f'plots/pred_img/pred_vs_gen_cap_{i}.jpg')

        output_df = pd.DataFrame(output_ls, columns = ["image_name", "predicted_caption"])
        file_name = f"test_score_{date.today().strftime('%d')}_output.csv"
        file_new_path = os.path.join('./output_csv', file_name)
        output_df.to_csv(file_new_path, index = False, sep = "|")
