# Vision GPT2

Vision GPT2 is an image captioning model based on the GPT-2 architecture. This project includes scripts for training the model, testing it on images, running an API server, and evaluating its performance using metrics such as BLEU and ROUGE scores.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Testing the Model](#testing-the-model)
  - [Running the API Server](#running-the-api-server)
  - [Evaluating the Model](#evaluating-the-model)
- [Contributing](#contributing)
- [License](#license)

## Project Structure


vision-gpt2/
│
├── src/
│ ├── train.py - Script for training the model on provided data.
│ ├── test.py - Test the model on single or multiple images.
│ ├── infer_vision_gpt2.py - Perform inference with models from Hugging Face.
│ ├── run_api.py - Code to run the API server for model serving.
│ └── metrics.py - Calculate BLEU and ROUGE scores for model evaluation.
│
└── requirements.txt - Required libraries.