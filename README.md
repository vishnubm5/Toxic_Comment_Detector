# Toxic_Comment_Detector
Hate Speech Detection with Bi‑LSTM

Project Overview

This repository implements a text classification pipeline to detect hate speech, offensive language, and neutral tweets using a balanced Bi‑Directional LSTM (Bi‑LSTM) model with pre‑trained GloVe embeddings.

Key features:

Data preprocessing (tokenization, stopword removal, normalization)

Handling imbalanced classes via upsampling and custom class‑weighted loss

Use of TensorFlow 2.x, Keras API, and GloVe embeddings (100d)

Bi‑directional LSTM network with dropout and early stopping

Performance evaluation with classification report and confusion matrix

Visualization of training history

Repository Structure
```bash
├── Hate_Speech.csv        # Raw dataset file (tweets and class labels)
├── glove.6B.100d.txt      # Pre-trained GloVe embeddings (100d)
├── src/                   # Python scripts
│   └── train.py           # Main training and evaluation script
├── requirements.txt       # Python package dependencies
└── README.md              # Project overview and instructions
```
Requirements

Python 3.8+

pip

Install dependencies:
```bash
pip install -r requirements.txt
```
Contents of requirements.txt:

pandas
numpy
tensorflow
scikit-learn
matplotlib
nltk

Data Preparation

Place your Hate_Speech.csv file at the project root.

Download and place the GloVe embeddings: glove.6B.100d.txt

In a Python shell, download NLTK resources:
```bash
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
Usage

Navigate to the project directory:
```bash
cd path/to/project
```
Run the training script:
```bash
python src/train.py --data Hate_Speech.csv --glove glove.6B.100d.txt
```
Optional arguments:
```bash
--epochs: Number of training epochs (default: 20)

--batch_size: Batch size for training (default: 32)

--max_words: Vocabulary size (default: 5000)

--max_len: Sequence length (default: 100)
```
Results

<pre> ```bash Classification Report: precision recall f1-score support 0 0.83 0.98 0.90 832 1 0.96 0.76 0.85 833 2 0.90 0.93 0.92 833 accuracy 0.89 2498 macro avg 0.90 0.89 0.89 2498 weighted avg 0.90 0.89 0.89 2498 Confusion Matrix: [[814 11 7] [126 631 76] [ 39 18 776]] ``` </pre>



After training, the script will output:

Class distribution before and after balancing

Training/validation loss and accuracy plots

Classification report (precision, recall, F1-score)

Confusion matrix

Customization

Model architecture: Modify layers in train.py to experiment with different LSTM units or additional layers.

Embeddings: Swap GloVe for other embeddings (FastText, Word2Vec) by adjusting the embedding loader.

Preprocessing: Tweak preprocess_text for language-specific rules or additional cleaning.

License

This project is licensed under the MIT License. Feel free to use and modify for research and education.
