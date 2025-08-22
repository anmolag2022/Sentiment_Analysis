
---

# Sentiment Analysis of Amazon Musical Instrument Reviews

This project analyzes sentiment from Amazon customer reviews on musical instruments, using both classical NLP techniques and modern machine learning. You can use either the Jupyter notebook (`Sentiment-Analysis.ipynb`) or the Python script version (`Sentiment-Analysis.py`).

---

## Overview

* **Goal:** Classify Amazon musical instrument product reviews into Positive, Neutral, or Negative categories based on review text and ratings.
* **Techniques used:** Text cleaning, feature engineering, data resampling, n-gram analysis, classical ML classifiers (Logistic Regression, SVM, Random Forest, Naive Bayes, KNN, Decision Tree), hyperparameter tuning, and visualization.
* **Dataset:** Amazon Musical Instruments Reviews (CSV file placed locally in the project).

---

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Major Steps](#major-steps)
* [Results](#results)
* [Citation](#citation)
* [References](#references)

---

## Installation

### 1. Clone the Repository & Get the Data

Place the CSV file (e.g., `Musical_instruments_reviews.csv` or `Instruments_Reviews.csv`) in the project folder.

### 2. Install Required Packages

Use pip to install all necessary libraries:

```bash
pip install pandas numpy matplotlib scikit-learn nltk textblob wordcloud imbalanced-learn
```

*Note*: If you’re running the notebook and some packages are missing, you may install them directly in a cell as:

```python
!pip install <package-name>
```

### 3. Download NLTK Data

The script/notebook will attempt to download needed NLTK datasets automatically. If you face errors, pre-download them using:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Usage

### For the Jupyter Notebook (`Sentiment-Analysis.ipynb`):

1. Open the notebook in Jupyter or VS Code.
2. Run the cells in order.
3. Ensure the CSV data file is present in the working directory.

### For the Python Script (`Sentiment-Analysis.py`):

1. Ensure all dependencies and the data file are available in the working directory.
2. Run via terminal:

```bash
python Sentiment-Analysis.py
```

*Note*: Comment out or adapt any notebook-specific code (e.g., plot displays).

---

## Project Structure

```
Sentiment-Analysis.ipynb      # Jupyter notebook (recommended for interactive exploration)
Sentiment-Analysis.py         # Python script version (for batch runs)
<your-data-file>.csv          # Amazon Musical Instruments Reviews dataset (CSV)
README.md                     # Project documentation
```

---

## Major Steps

1. **Data Loading & Inspection** – Load dataset, check shape, missing values, and basic stats.
2. **Data Preprocessing** – Fill missing values, combine review text & summary, drop unused columns.
3. **Exploratory Analysis** – Visualize ratings, sentiment distribution, review lengths, word counts.
4. **Label Engineering** – Categorize reviews into `Positive`, `Neutral`, `Negative` based on ratings.
5. **Text Cleaning** – Lowercasing, punctuation/number/link removal, lemmatization, stopword filtering (keeping “not”).
6. **N-gram Analysis & Word Clouds** – Analyze top unigrams, bigrams, trigrams; visualize with word clouds.
7. **Feature Engineering & Resampling** – Encode labels, vectorize text (TF-IDF), balance data using SMOTE.
8. **Model Training & Evaluation** – Train classifiers, select best model (Logistic Regression) via cross-validation and tuning; evaluate using accuracy, confusion matrix, and classification report.

---

## Results

* **Best Model:** Logistic Regression (with hyperparameter tuning)
* **Accuracy:** >95% on test set
* **Key Insights:**

  * Most reviews are positive.
  * Guitars and accessories are the most discussed.
  * Imbalanced data impacts minority class performance.

---

## Citation

If you use this codebase or analysis, please cite the dataset source and the libraries used.

---

## References

* **Libraries:** pandas, scikit-learn, nltk, matplotlib, wordcloud, textblob, imbalanced-learn

---
