import json
import os
from pathlib import Path
import pandas as pd

json_file = Path(os.getcwd())/"data/ImageSpeak/annotations/annotations.json"

with open(json_file, "rb") as f:
    annot_json = json.load(f)

print(annot_json.keys())


annot_cap_ls = [(item["caption"], item["image_id"]) for item in annot_json["annotations"]]

annot_df = pd.DataFrame(annot_cap_ls, columns = ["caption", "image_id"])
print(f"annot_df_shape: {annot_df.shape}")

image_ls = [(item["file_name"], item["id"]) for item in annot_json["images"]]
image_df = pd.DataFrame(image_ls, columns = ["image_name", "image_id"])
print(f"image_df_shape: {image_df.shape}")
output_df = pd.merge(image_df, annot_df, on="image_id")

output_df = output_df[["image_name", "caption"]]

print(f"output_df shape: {output_df.shape},  output_df_columns: {output_df.columns}")
print(output_df.head())

# json_file = Path(os.getcwd())/"data/ImageSpeak/annotations/annotations.json"
output_df.to_csv("data/ImageSpeak/viz_data.csv", index=False, sep="|")
