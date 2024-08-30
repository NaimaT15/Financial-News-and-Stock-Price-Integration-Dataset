<h1 align="center">Welcome to Financial-News-and-Stock-Price-Integration-Dataset 👋</h1>
<p>
Project Structure

.
├── .github
│   └── workflows
│       └── unittests.yml       # CI/CD pipeline configuration for unit tests

├── .notebook
│   └──                         # Jupyter Notebooks directory

├── .vscode
│   └── settings.json           # VS Code specific settings

├── data
│   └──                         # Directory for datasets

├── notebooks
│   ├── eda.ipynb               # Notebook for Exploratory Data Analysis

│   └── financial_analysis_notebook.ipynb  # Notebook for detailed financial analysis

├── scripts
│   ├── financial_analysis.py   # Script for performing financial analysis

│   └── load_data.py            # Script for loading data

├── src
│   └──                         # Main source code for the project

├── venv
│   └──                         # Virtual environment directory

├── .gitignore                  # Specifies files to ignore in git

├── README.md                   # General project documentation

└── requirements.txt            # Lists project dependencies



Installation


    1. Clone the Repository:
    
    git clone  https://github.com/NaimaT15/Financial-News-and-Stock-Price-Integration-Dataset.git
    
    cd FinancialandStock
    
    2. Setup Virtual Environment (Optional but recommended):
    
    python -m venv venv
    
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    
    3. Install Dependencies:
    
    pip install -r requirements.txt


Usage
Exploratory Data Analysis: Open the eda.ipynb notebook within Jupyter Notebook or Lab to explore the dataset:
jupyter notebook notebooks/eda.ipynb

Financial Analysis: Review the financial_analysis_notebook.ipynb for detailed financial analysis:
jupyter notebook notebooks/financial_analysis_notebook.ipynb

Running Scripts: Execute scripts directly from the command line:
python scripts/financial_analysis.py

Features

Sentiment Analysis: Analyzes the sentiment of financial news headlines to determine market sentiment.

Topic Modeling: Utilizes LDA to extract prominent topics from a large corpus of financial texts.

Quantitative Analysis: Employs TA-Lib and PyNance to compute technical indicators and financial metrics.

Contribution
Contributions are welcome! Please create a pull request or issue to suggest improvements or add new features.


</p>

## Author

👤 **Naima Tilahun**

* Github: [@NaimaT15](https://github.com/NaimaT15)

## Show your support

Give a ⭐️ if this project helped you!
