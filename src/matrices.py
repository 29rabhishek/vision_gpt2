import pandas as pd
from pathlib import Path
import os
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk import word_tokenize

def compute_scores(data):
    
    # Prepare the ROUGE evaluator data is dataframe
    rouge = Rouge()
    
    # List to store results
    results = []

    # Iterate over each row in the dataframe
    for index, row in data.iterrows():
        # Tokenize the reference and candidate captions
        reference = word_tokenize(row['orginal_caption'].lower())
        candidate = word_tokenize(row['predicted_caption'].lower())

        # Compute BLEU score (using a list of lists for reference as required by NLTK)
        bleu_score = sentence_bleu([reference], candidate)

        # Compute ROUGE scores
        rouge_scores = rouge.get_scores(' '.join(candidate), ' '.join(reference))[0]

        # Store results along with identifiers
        results.append({
            'image_name': row['image_name'],
            'orginal_caption': row['orginal_caption'],
            'predicted_caption': row['predicted_caption'],
            'BLEU': bleu_score,
            'ROUGE-1': rouge_scores['rouge-1']['f'],
            'ROUGE-2': rouge_scores['rouge-2']['f'],
            'ROUGE-L': rouge_scores['rouge-l']['f']
        })

    return pd.DataFrame(results)

# Example usage
# results_df = compute_scores('path_to_your_csv_file.csv')
# print(results_df)

if __name__ == "__main__":
    file_name = "VisionGPT_26_output.csv"
    path_to_file = os.path.join("output_csv", "VisionGPT_26_output.csv")
    output_file = pd.read_csv(path_to_file, sep = "|")
    print(output_file.head())
    result_df = compute_scores(output_file)
    # print(result_df.head())
    # print(result_df.shape)
    ##############################
    # WARNING : CHANGE THE FILE NAME ACCORDING TO YOURSELF
    ###################################
    result_df.to_csv("output_csv/VisionGPT_26_output_score.csv", sep = "|", index =False)
