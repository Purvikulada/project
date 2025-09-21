# Review Analyzer

This project is a sentiment analysis web application built with Flask. It uses a Naive Bayes model trained on review data to predict sentiment.

## Setup Instructions

1. Create a virtual environment (recommended):

```bash
python -m venv venv
```

2. Activate the virtual environment:

- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r review-analyzer/requirements.txt
```

4. Train the model:

```bash
python review-analyzer/train_model.py
```

This will train the sentiment analysis model and save it to `review-analyzer/model/sentiment_model.joblib`.

5. Run the Flask app:

```bash
python review-analyzer/app.py
```

6. Open your browser and go to:

```
http://127.0.0.1:5000/
```

to access the web app.

## Notes

- Make sure `review-analyzer/reviews.csv` is present before training the model.
- The app expects the trained model to be present in `review-analyzer/model/`.
- For production deployment, consider using `gunicorn` or other WSGI servers.
