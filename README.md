# Automated Insight Engine

The **Automated Insight Engine** is a web-based tool built with Streamlit that accelerates the initial phases of a data science project. It allows users to upload a CSV file and guides them through data understanding, cleaning, and building baseline machine learning models without writing code.

![App Screenshot](https://via.placeholder.com/800x450.png?text=App+Screenshot+Here)
*(Suggestion: Replace the placeholder above with a real screenshot of your running application)*

---

## Features

The application is organized into a series of tabs, each handling a specific step in the ML workflow:

1.  **Data Understanding**:
    *   Automatically detects column types (numeric, categorical, datetime, etc.).
    *   Identifies the likely target variable and infers the ML task (classification or regression).
    *   Checks for class imbalance in classification tasks.
    *   Flags skewed numeric features and potential data leakage.

2.  **Diagnosis**:
    *   Visualizes missing data patterns.
    *   Identifies duplicate rows.
    *   Detects potential outliers using the IQR method.
    *   Calculates and visualizes correlation hotspots for both numeric (Pearson) and categorical (Cramer's V) features.

3.  **Cleaning**:
    *   Removes duplicate rows.
    *   Drops columns with high sparsity (configurable threshold).
    *   Imputes missing values (median for numeric, 'Unknown' for categorical).
    *   Caps outliers to reduce their effect.

4.  **Features**:
    *   (Experimental) Basic feature engineering, such as creating interaction features from categorical columns.

5.  **Models (Preview)**:
    *   Lists candidate models suitable for the inferred task and dataset size.

6.  **Leaderboard**:
    *   Runs a cross-validation competition across multiple baseline models (e.g., Logistic Regression, Random Forest, XGBoost).
    *   Ranks models by performance metrics (Accuracy/F1 for classification, R²/RMSE for regression).
    *   Visualizes model performance with bar charts for easy comparison.

7.  **Train & Test**:
    *   Allows you to train a selected model on a train/test split of the data.
    *   Displays detailed performance metrics, a confusion matrix (for classification), and a decision tree plot (if applicable).
    *   Provides a human-readable summary of the model's performance.

## Tech Stack

- **Backend & ML**: Python, Pandas, Scikit-learn, XGBoost, LightGBM
- **Frontend**: Streamlit
- **Plotting**: Plotly, Matplotlib

## Setup and Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

- Python 3.8 or newer
- `pip` for package management

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/singh2894/Automated-Insight-Engine.git
    cd AutoML
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## How to Run

Once the setup is complete, you can run the Streamlit application using the provided script or the direct command.

**Using the script:**
```sh
# For Windows
.\run.bat

# For macOS/Linux
chmod +x run.sh
./run.sh
```

**Or using the direct command:**
```sh
streamlit run ui/app.py
```

The application will open in a new tab in your default web browser.