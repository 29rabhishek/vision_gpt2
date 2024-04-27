import pandas as pd
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk import word_tokenize

img_dir_pth = "output_csv/VisionGPT_26_output.csv"
df = pd.read_csv(img_dir_pth, sep="|")
df.rename({
    "comment": "orginal_caption"
}, axis="columns", inplace=True)
print(df.head())
print(df.shape)

img_names = list(set(df["image_name"]))

print(img_names)
print(df.columns)
collect_ls = []
for idx, img_name in enumerate(img_names):

    orginal_captions = list(df[(df["image_name"] == img_name)]["orginal_caption"])
    predicted_captions = list(df[(df["image_name"] == img_name)]["predicted_caption"])
    pred_ls = []

    for orginal_caption in orginal_captions:

        beleu_score_list = []
        reference = word_tokenize(orginal_caption.lower())
        for predicted_caption in predicted_captions:
            # Tokenize the reference and candidate captions
            candidate = word_tokenize(predicted_caption.lower())
            # Compute BLEU score (using a list of lists for reference as required by NLTK)
            bleu_score = sentence_bleu([reference], candidate)
            beleu_score_list.append(bleu_score)
            # rouge scores
            # rouge_scores = rouge.get_scores(' '.join(candidate), ' '.join(reference))[0]
            # # score_list.append((orginal_caption, predicted_caption, bleu_score, rouge_scores['rouge-1']['f']))

        score_idx = beleu_score_list.index(max(beleu_score_list))
        collect_ls.append((img_name, orginal_caption, predicted_captions[score_idx], beleu_score_list[score_idx]))

output_df = pd.DataFrame(collect_ls, columns=["image_name", "orginal_caption", "predicted_caption", "best_score_match"])

output_df.to_csv("output_csv/output_best_caption_score.csv", index =False, sep="|")
