# Disaster Tweets Classification with BERT

## Project Overview

This project is part of the **Natural Language Processing with Disaster Tweets** competition on Kaggle. The goal is to develop a model that can accurately classify tweets as either disaster-related or not. Using BERT (Bidirectional Encoder Representations from Transformers), the project leverages the power of transfer learning to analyze the context of tweets and predict their relevance in disaster situations.

I approached this challenge to further develop my NLP skills, focusing on text preprocessing, fine-tuning transformers, and evaluation metrics.

## Approach

### Key Steps in Workflow:

**1. Data Preprocessing:**
   - Handling missing values in the dataset.
   - Tokenizing text with BERT’s tokenizer.
   - Encoding and padding sequences for compatibility with BERT.
   - Exploratory Data Analysis (EDA) to understand patterns in disaster-related language.

**2. Model Training:**
   - Fine-tuning a pre-trained BERT model for binary classification.
   - Experimenting with various configurations, such as learning rate and batch size, to optimize the model.
   
**3. Evaluation:**
   - Evaluating the model using metrics like accuracy, precision, recall, and F1-score.
   - Visualizing training and validation losses to monitor overfitting or underfitting.
   - Analyzing model performance through confusion matrices and precision-recall curves.

## Challenges & Solutions

- **Handling Imbalanced Data**: Many tweets were not disaster-related, creating an imbalance. I used weighted loss functions to address this imbalance.
- **Optimizing Fine-Tuning**: Fine-tuning BERT on smaller datasets can lead to overfitting. I mitigated this with regularization techniques and by experimenting with dropout rates.

## Results

The model achieved a high accuracy on the test set and ranked in the top 100 on Kaggle, demonstrating its ability to understand disaster-related language patterns. The following visualizations were essential in understanding the model’s performance:

- **Learning Curves**: Monitored to ensure the model was not overfitting.
- **Confusion Matrix**: Showed a clear separation between disaster and non-disaster tweets.
  
## Future Improvements

- **Hyperparameter Tuning**: Further tuning of hyperparameters could improve accuracy and generalization.
- **Exploring Other Models**: Testing other transformer models, like RoBERTa or DistilBERT, might enhance performance.
- **Enhanced Feature Engineering**: Additional feature extraction from tweet metadata, such as user information or hashtags, could provide extra insights.

## Conclusion

This project served as a hands-on application of NLP techniques in a real-world context, highlighting the capabilities of BERT for text classification. It allowed me to deepen my skills in data preprocessing, model fine-tuning, and evaluation while gaining insights into how language models can aid in disaster response by filtering relevant information from social media.
