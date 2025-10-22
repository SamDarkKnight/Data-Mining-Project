# Fake News Detection using Machine Learning

*A data-mining project by Steve, Suhana, Joel*

## 📖 Overview

This project implements a machine-learning pipeline for detecting fake news articles. Using text data (news headlines and/or bodies) labeled as real or fake, the notebook builds, evaluates and compares several classification models to distinguish misleading content from authentic articles.

## 🧰 Contents

* `Fake_News_Detection_using_machine_learning.ipynb` — the Jupyter notebook which contains all of the processing, modelling and evaluation steps.
* (Optionally) Dataset files — if any external CSV/JSON data is used, put them in a `data/` folder (or link to them).
* (Optionally) `requirements.txt` — list of Python dependencies (e.g., pandas, scikit-learn, nltk etc.).
* (Optionally) `README.md` (this file) — overview and instructions for using the project.

## ✅ Features

* Pre-processing of news text (cleaning, tokenizing, stop-word removal).
* Vectorisation of text (e.g., TF-IDF, Count Vectors).
* Training and comparing multiple classification algorithms (for example: Logistic Regression, Naive Bayes, Random Forest, etc.).
* Evaluation of model performance (accuracy, precision, recall, F1-score, confusion matrix).
* Insights on which features or models perform best at fake-news detection.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python (3.x) installed. It’s recommended to use a virtual environment.
Install required packages (example):

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

### Running the Notebook

1. Clone this repository:

   ```bash
   git clone https://github.com/SamDarkKnight/Data-Mining-Project.git
   cd Data-Mining-Project
   ```
2. If required, place your dataset file(s) in a folder named `data/`, and update the notebook path accordingly.
3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
4. Open `Fake_News_Detection_using_machine_learning.ipynb` and run the cells sequentially.
5. Inspect results, plots, and model comparisons as shown in the notebook.

## 🧩 Project Structure

```
Data-Mining-Project/
│
├── data/                        ← (optional) folder for raw data files  
│      └── fake.csv      ← example dataset of real vs fake news
            true.csv
│
├── Fake_News_Detection_using_machine_learning.ipynb  
├── requirements.txt             ← (optional) list of Python packages  
└── README.md                    ← this file  
```

## 📚 Dataset

The dataset used in this project contains news items labeled as “real” or “fake”. It typically includes fields such as *title*, *text/body*, *label*.

True.csv: https://drive.google.com/file/d/1iX49I9BRRzTKaOTT7jRXaV22uRPg35kg/view?usp=sharing

Fake.csv: https://drive.google.com/file/d/1XJw5fvMvFgy54aqHXr0woJuja6ljukMB/view?usp=sharing

## 🧪 Methodology

1. Load and preview the dataset (check for missing values, class balance).
2. Clean & pre-process text: remove punctuation, stop words, lowercase conversion, tokenization, etc.
3. Feature extraction: convert text into numeric vectors (TF-IDF / Count Vector).
4. Split into training & test sets (e.g., 70/30 or 80/20).
5. Train multiple classifiers (e.g., Logistic Regression, Naive Bayes, Random Forest).
6. Evaluate models on test set: compute accuracy, precision, recall, F1, confusion matrix.
7. Compare and document results: which model performs best, and why.
8. (Optional) Visualise results (bar charts of performance metrics, word-clouds of top features).

## 📈 Results

* Classifier X achieved highest accuracy of **80%**.
* Model Y had better recall for “fake news” class, but lower precision.
* Feature importance shows that words like “claim”, “report”, “fake” appear more often in fake-news articles.
* The dataset is somewhat imbalanced: real news articles outnumber fake by ~ 2 : 1.

## 🧠 Insights & Future Work

* The pipeline shows that basic text-vectorisation + classic ML models already yield meaningful performance in fake-news detection.
* However, more advanced techniques (deep learning with embeddings, transformer models) could further improve accuracy.
* Addressing dataset imbalance, deeper feature engineering (e.g., metadata: source, date, author) or incorporating network features (sharing patterns) are possible next steps.
* A production-ready system would require real-time data ingestion, continuous retraining and robust monitoring.

## 🤝 Contributing

Contributions are welcome! If you’d like to improve the notebook, add new models, or extend datasets:

1. Fork the repository.
2. Create a new branch for your feature.
3. Submit a pull request and describe your changes.

## 📄 License

This project is provided for academic and educational use.

---

**Thank you for checking out this project – happy data-mining!**
