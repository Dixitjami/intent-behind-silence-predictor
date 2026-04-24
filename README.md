# Intent Behind Silence Predictor

A Streamlit NLP app that predicts the likely intent behind delayed message replies using TF-IDF, Logistic Regression, and probability visualization.

## Files

- `app.py` - Streamlit user interface and prediction logic.
- `model.pkl` - Trained classifier.
- `tfidf.pkl` - Saved TF-IDF vectorizer.
- `label_encoder.pkl` - Saved label encoder.
- `requirements.txt` - Python dependencies for deployment.

## Run Locally

```bash
python -m streamlit run app.py
```

## Deploy

Deploy this repository on Streamlit Community Cloud:

1. Go to `https://share.streamlit.io`.
2. Click `Create app`.
3. Select this GitHub repository.
4. Set the branch to `main`.
5. Set the main file path to `app.py`.
6. In Advanced settings, choose a Python version compatible with the dependencies.
7. Click `Deploy`.
