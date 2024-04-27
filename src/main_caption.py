from vision_caption_infer import create_vision_gpt_model
import pandas as pd
from datetime import date
import torch
from time import process_time
from PIL import Image
import os
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class CaptionGenrator:
    def __init__(self, model_name, model_path):
        if model_name == "VisionGPT":
            self.model = create_vision_gpt_model(model_path)

    # functionrturn in this way if we need param to give to caption genrato
    def get_caption(self, img):

        return self.model.generate_caption(img)
    



def find_files(directory, pattern):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(pattern):  # You can customize this condition
                matched_files.append((os.path.join(root, file), file))
    return matched_files




# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])





# dataset = ImageCaptionDataset(df = mg_df)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False)



# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)
#     print("code running on multiple gpus")

    

if __name__ == "__main__":
    print(f"Cuda is :{torch.cuda.is_available()}")
    
    # give path of the model that you want use
    MODEL_NAME = "VisionGPT"
    MODEL_PATH = "/media/arnav/MEDIA/Abhishek/vision_gpt2/captioner/captioner_16.pt"
    caption_genrator = CaptionGenrator(
        model_name = MODEL_NAME,
        model_path = MODEL_PATH
    )
    infer_image = False
    if infer_image:
        img_path = "/media/arnav/MEDIA/Abhishek/vision_gpt2/VizWiz_test_00000313.jpg"
        image = Image.open(img_path).convert('RGB')
        prediction = caption_genrator.get_caption(image)

    infer_on_file = True
    if infer_on_file:
        img_dir_pth = "/media/arnav/MEDIA/Abhishek/vision_gpt2/data/ImageSpeak/images"
        file_ls = find_files(directory=img_dir_pth, pattern=".jpg")
        img_df = pd.DataFrame(file_ls, columns=["image_path", "image_name"])
        num_img = img_df.shape[0]

        annot_img_csv_pth = '/media/arnav/MEDIA/Abhishek/vision_gpt2/data/ImageSpeak/viz_data.csv'
        annot_df = pd.read_csv(annot_img_csv_pth,sep='|')

        mg_df = pd.merge(img_df, annot_df, left_on="image_name", right_on="image_name")
        mg_df.rename({'caption': 'orginal_caption'}, inplace=True, axis = "columns")
        output_ls = []

        for i in range(50):
            img_path = mg_df.loc[i, "image_path"]
            img_name = mg_df.loc[i,"image_name"],
            orignal_caption = mg_df.loc[i, "orginal_caption"]
            image = Image.open(img_path)
            start_time = process_time()

            predicted_caption = caption_genrator.get_caption(image)
            end_time = process_time()
            diff_time = end_time -start_time
            output_ls.append((img_name, predicted_caption, orignal_caption, diff_time))
            plt.imshow(image.convert('RGB'))
            plt.title(f"actual: {orignal_caption}\n predict: {predicted_caption}", fontsize=7)
            plt.axis('off')
            plt.savefig(f'plots/pred_img/pred_vs_gen_cap_{i}.jpg')

        output_df = pd.DataFrame(output_ls, columns = ["image_name", "predicted_caption", "orginal_caption", "time_take_in_infer"])
        file_name = f"{MODEL_NAME}_{date.today().strftime('%d')}_output.csv"
        file_new_path = os.path.join('./output_csv', file_name)
        output_df.to_csv(file_new_path, index = False, sep = "|")
