# SwarSaathi – Machine Learning-Based Aging Risk Assessment

An **educational** machine-learning project (MCA). From demographic, lifestyle and health inputs it
predicts one of three categories from the supplied dataset: **Healthy**, **At-risk** or **Frail**
(target column `aging_outcome`), explains the result, and lets you explore "what if" changes.

> **Disclaimer:** SwarSaathi is NOT a medical device. It gives no diagnosis and has not been
> clinically validated. Always consult a qualified healthcare professional.

---
## 1. How to run it in VS Code (Windows)

1. Install **Python 3.10+** and **VS Code** with the *Python* extension.
2. Unzip the project, then in VS Code: **File → Open Folder →** select the `SwarSaathi` folder.
3. Open a terminal: **Terminal → New Terminal** (PowerShell) and run:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.
4. *(Recommended)* re-train so the saved model matches your scikit-learn version (2–5 minutes):
   ```powershell
   python -m src.train_models
   ```
   (`python -m src.train_models --skip-tuning` is faster.) A trained model is already included.
5. Start the app:
   ```powershell
   streamlit run app.py
   ```
   Your browser opens at http://localhost:8501.

---
## 2. Results 

| Test accuracy | 0.881 | **0.918** |
| Test macro-F1 | 0.751 | **0.859** |
| Out-of-fold accuracy (all unique rows) | – | **0.909** |
| Out-of-fold macro-F1 (all unique rows) | – | **0.834** |

The improvement is real but modest, and the Healthy class (22 unique records) still limits
reliability. Out-of-fold Healthy recall is 0.73 and precision 0.67 on those 22 cases. Numbers can shift slightly
on your machine; always read the `reports/` folder.

### 3. New app features (Streamlit, 6 pages)
## Page 1 – Home

**Purpose:** a welcome page that explains the project in 30 seconds.

**What you see**
- A banner with the project idea.
- Four number cards: unique records (2,439), input features (13), test accuracy (about 92%) and out-of-fold macro-F1 (about 0.83).
- Six small cards describing the other pages.
- Four "How it works" steps: Clean, Compare, Select, Explain.
- A button that takes you to the Assessment page.

**Good to know:** the numbers on the cards are read from the file `reports/evaluation_summary.json`, so they always match the last training run.

## Page 2 – Assessment (main page)

**Purpose:** predict the category for one person.

**Step by step**
1. *(Optional)* Pick an example in **Quick start**. It fills in all inputs for you. Examples are made-up people.
2. Check or change the inputs. All 13 are shown together in one box, in three columns:
   - **Personal:** age, gender, education years, income level.
   - **Lifestyle:** physical activity, smoking, diet quality, alcohol, social engagement.
   - **Health measures:** number of chronic conditions, memory score, BMI, blood pressure.
3. Click **Run assessment**. **Reset form** returns to the starting values.

All 13 inputs are combined into one single prediction. The three columns are only for tidy layout.

**What the result shows**
- **Predicted category** card (green = Healthy, amber = At-risk, red = Frail) with the model's confidence.
- **Risk index gauge (0–100):** a summary number. It equals 0 × P(Healthy) + 50 × P(At-risk) + 100 × P(Frail). It is not a medical score.
- **Class probabilities:** how likely the model thinks each of the three categories is. Very sure answers are shown as ">99.9%" or "<0.1%" because a model is never truly 100% sure.
- Three tabs:
  - **Why this result:** which inputs pushed the estimate up (red) or down (green) compared with a typical person in the dataset.
  - **How to improve:** which lifestyle habit, if improved, would lower the model's risk score the most; a lifestyle radar chart; and general wellness tips marked High / Medium / Good.
  - **Report:** download a one-page PDF or a CSV of the result.
- **Assessment history** (bottom): every assessment you ran in this session, with a download button.

**Why do I often see "Frail"?** About 64% of the training records are Frail, and in this dataset the label depends on combinations such as several chronic conditions, weaker diet and low social engagement. Older or higher-risk profiles therefore often come out Frail. Use the Quick start examples to see all three categories.

**Behind the scenes:** the inputs are checked for valid ranges, sent to the saved model, and then the model is re-run on slightly changed copies of the profile to work out the "Why this result" and "How to improve" charts.

## Page 3 – What-if Simulator

**Purpose:** see how the model's answer changes if some habits change.

**How to use:** first run an assessment. Then change the physical activity, diet, smoking, alcohol, social engagement or BMI. Everything else stays as it was.

**What you see:** the new category, the new risk index, the new chance of Frail, and a before-and-after bar chart.

**Good to know:** if nothing changes, the model was already very sure about that profile. This is a limit of the data, not proof that habits do not matter. It simulates the model, not a real person.

## Page 4 – Model Insights

**Purpose:** show how good the model is and how it was chosen. This page is very useful in your viva.

**Tabs**
- **Comparison:** all 7 algorithms compared (Logistic Regression, KNN, SVM, Random Forest, Decision Tree, Gradient Boosting and 
the Soft Voting Ensemble) before and after tuning. The best one is chosen using cross-validation on training data only.
- **Per-class results:** precision, recall and F1 for Healthy, At-risk and Frail.
- **Confusion matrices:** a table of right and wrong predictions, on the test set and on all records (out-of-fold).
- **Feature importance:** which inputs the model relies on most.
- **Data quality:** how many rows were duplicates, and why only 22 unique Healthy records remain.

**Words explained:** *accuracy* = share of correct predictions. *Macro-F1* = average quality across the three classes, so small classes count equally. *Out-of-fold* = each record was predicted by a model that never saw it.

## Page 5 – Data Explorer

**Purpose:** look at the dataset with interactive charts.

**Controls:** choose the unique records (used for training) or the original file (with many repeated rows), pick which outcome classes to show and set an age range.

**Tabs:** numeric features (histograms by outcome), categorical features (outcome mix per group), correlation heatmap, and the data table with a download button.

## Page 6 – About

**Purpose:** project summary, technology list, and an honest list of limitations.

**Limitations in short**
- The dataset's origin is unverified.
- Only 22 unique Healthy records exist, so Healthy results are unreliable.
- The labels look rule-generated.
- Explanations show patterns in the data, not medical causes.
- It is not a medical device.

### 4. Features 
* Probabilities are shown as `>99.9%` / `<0.1%` instead of "100%".
* Explanations use the log-odds scale so they stay informative even when probabilities saturate, and
  are labelled as *associations in this dataset, not medical causes*.
* The risk index (0×P(Healthy) + 50×P(At-risk) + 100×P(Frail)) is documented as a summary, not a clinical score.


## 5. Dataset (verified from the supplied CSV)
| Fact | Value |
| Rows × columns | 7,488 × 14 |
| Target classes | Healthy, At-risk, Frail (2,496 each in the original file) |
| Exact duplicate rows | 5,049 (67.4%) |
| Unique rows | 2,439 → Frail 1,556 · At-risk 861 · Healthy 22 |
| Truly missing values | 0 |

## 6. Project structure
```
SwarSaathi/
├── app.py                  App entry point (navigation + theme)
├── views/                  One file per page: home, assess, whatif, insights, explorer, about
├── ui/                     theme.py (CSS/cards), charts.py (Plotly), common.py (helpers)
├── src/
│   ├── config.py           paths, columns, valid ranges, class colours
│   ├── data_loader.py      safe CSV reading
│   ├── data_cleaning.py    audit, invalid values, de-duplication, leakage check
│   ├── preprocessing.py    pipeline, 7 models incl. the ensemble, stratified split
│   ├── tune_models.py      GridSearchCV with repeated CV
│   ├── train_models.py     train → compare → tune → select → out-of-fold check → save
│   ├── evaluate_models.py  metrics and plots
│   ├── predict.py          validation, prediction, risk index
│   ├── explain.py          risk drivers, what-if, opportunities, wellness tips           
│   └── report.py           PDF report
├── docs/                   SwarSaathi_Page_Guide.md (what each page does)
├── data/                   original dataset + sample_batch_input.csv
├── models/  reports/  notebooks/  tests/
└── .vscode/                interpreter, pytest and F5 launch settings
```

## 7. How the final model is chosen
* 80/20 stratified split (`random_state=42`) after removing exact duplicates.
* Every model is scored with 3×5-fold stratified CV on the training data (metric: macro-F1); the top 3 are tuned
  with GridSearchCV (same metric, training data only).
* **Selection metric: cross-validated macro-F1 on training data.** Test data only reports results.
* Selected here: **Soft Voting Ensemble (tuned)**, CV macro-F1 0.740.
* Tuned CV scores are the best of several tries, so they are slightly optimistic.


