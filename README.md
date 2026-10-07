# Student Performance Prediction

Machine Learning project jo student ki performance (High / Medium / Low) predict karta hai.
Flask web app ke saath.

## Features
- Data analysis (graphs `graphs/` folder mein save hote hain)
- 5 models ka comparison (Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, KNN)
- Cross-validation aur GridSearchCV se tuning
- Web app: validation, confidence %, personalised suggestions, feature importance, prediction history (SQLite)

## Folder Structure
```
dataset/student_data.csv
analysis.py
train_model.py
app.py
templates/index.html
templates/history.html
model/            (train_model.py se banta hai)
graphs/           (analysis.py aur train_model.py se bante hain)
```

## Run kaise karein
```
pip install -r requirements.txt
python analysis.py
python train_model.py
python app.py
```
Phir browser mein http://127.0.0.1:5000 kholein.

## Dataset Columns
study_hours, attendance, previous_score, assignments_completed, performance
