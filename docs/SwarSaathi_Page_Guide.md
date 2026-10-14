# SwarSaathi Guide

A simple explanation of what every page does, how to use it, and what happens behind the scenes.

SwarSaathi is a machine-learning app. You give it 13 pieces of information about a person (age, habits, health measures). It compares them with patterns it learned from a dataset of 2,439 unique records and answers one question: **does this profile look most like Healthy, At-risk or Frail?**

It is a learning project, not a medical tool. It gives no diagnosis.

The sidebar on the left has 7 pages. Use them in this order the first time: Home → Assessment → What-if → Model Insights → Data Explorer → About.

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


## Where things live in the project

Folder or file with What it holds -

`app.py`  - Starts the app and builds the sidebar menu |
`views/`  - One file for each page above |
`ui/`     - Colours and styling, charts, shared helpers |
`src/`    - The machine-learning code: cleaning, training, prediction, explanations, PDF, batch |
`models/` - The saved trained model |
`reports/`- Charts and tables made during training |
`data/`   - The dataset and the sample batch file |
`tests/`  - 34 automatic checks (run with `python -m pytest`) |

## How to run

1. Open the `SwarSaathi` folder in VS Code.
2. In the terminal: `py -m venv .venv`, then `.\.venv\Scripts\Activate.ps1`, then `python -m pip install -r requirements.txt`.
3. Optional re-train: `python -m src.train_models`.
4. Start: `python -m streamlit run app.py` (opens at http://localhost:8501).
