
# Sentiment Analysis with BERT on IMDb Dataset 🎬💬

This project focuses on building a robust Sentiment Analysis model using the IMDb dataset and a pre-trained BERT model (bert-base-uncased). The model classifies movie reviews as **positive** or **negative**.

## 📌 Features

- Utilizes the IMDb dataset (balanced for binary sentiment classification)
- Data cleaning and preprocessing pipeline
- Tokenization using Hugging Face's BERT tokenizer
- Fine-tuning BERT for sequence classification
- Evaluation with accuracy, loss, and classification report
- Prediction function for real-world input

## 🛠 Tech Stack

- Python 3.x
- PyTorch
- Hugging Face Transformers
- IMDb Dataset from 🤗 `datasets` library
- Scikit-learn, Pandas, tqdm

## 🧪 Model Architecture

- BERT as base model (`bert-base-uncased`)
- Dropout layers to reduce overfitting
- Dense layers for classification
- CrossEntropyLoss and AdamW optimizer
- Learning rate scheduler (ReduceLROnPlateau)

## 📊 Performance

- Achieved high accuracy on validation and test sets
- Can classify custom review texts

## 🚀 How to Use

1. Install dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas tqdm
```

2. Run the script to train the model:

```bash
python sentiment_analysis_bert.py
```

3. Use the `predict_sentiment(text, model, tokenizer)` function to classify new texts.

## 📁 Files

- `sentiment_analysis_bert.py` - Main training and evaluation script
- `best_sentiment_model.pth` - Saved best model weights
- `README.md` - Project documentation

## 💡 Example

```python
predict_sentiment("I loved the movie. Great acting and direction!", model, tokenizer)
# Output: "Positive"
```

## 📬 Contact

Feel free to reach out if you're interested in collaborating or discussing NLP projects!

---

⭐ Star the repo if you find it helpful!  
📌 #NLP #BERT #SentimentAnalysis #IMDb #MachineLearning #PyTorch #Transformers
