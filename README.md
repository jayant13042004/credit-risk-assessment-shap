# Credit Risk Assessment & Explainable ML

End-to-end credit default prediction on **32,581 loan records**, with calibrated probabilities, SHAP explanations, and a FastAPI prediction service.

## What it does
- Preprocessing and feature engineering on loan applicant data
- XGBoost (tree-based) classifier for default prediction
- Probability calibration so outputs can be read as default probabilities
- SHAP explainability to show which features drive each prediction
- FastAPI `POST /predict` endpoint with input validation, configurable classification thresholds, default probability scores, and risk categories
- Interactive web interface for running predictions

## Tech stack
Python, Pandas, Scikit-learn, XGBoost, SHAP, FastAPI

## Results
| Metric | Value |
|---|---|
| ROC-AUC | [fill from your notebook] |
| Precision / Recall / F1 | [fill from your notebook] |

(Add a SHAP summary plot here: `![SHAP summary](images/shap_summary.png)`)

## Run locally
```bash
git clone https://github.com/jayant13042004/credit-risk-assessment-shap.git
cd credit-risk-assessment-shap
pip install -r requirements.txt
uvicorn [app_file]:app --reload     # API docs at http://127.0.0.1:8000/docs
```

## Example request
```json
POST /predict
{ "[feature_1]": [value], "[feature_2]": [value] }
```
Response: default probability, predicted class at the chosen threshold, and risk category.

## Project structure
```
data/  notebooks/  src/  app/  requirements.txt
```
[adjust to your actual folders]

## Limitations
- Trained on a single public dataset; not validated on real lending data.
- Not for real credit decisions.

## Author
Jayant Jagtap · [LinkedIn](https://www.linkedin.com/in/jayantjagtap/)
