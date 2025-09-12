📊 Streamlit Data Cleaner & AI Analyst

An interactive data cleaning, visualization, and AI-powered analysis tool built with Streamlit.
Upload your dataset, clean it, explore insights visually, and ask questions in natural language using Groq’s LLaMA-3.3-70B model.

✨ Features

✅ Smart Upload & Preview – Upload CSV/Excel and preview before cleaning
✅ Data Cleaning – Remove missing values, duplicates, normalize column names
✅ Data Quality Reports – Missing values, duplicates, correlations, high-correlation warnings
✅ Interactive Visualizations – Histograms, box plots, scatter plots, correlation heatmaps
✅ AI-Powered Queries – Ask questions about your dataset in plain English
✅ Download Processed Data – Export cleaned data as CSV or Excel
✅ Customizable Sidebar – Control cleaning options before applying
✅ Landing Page – Friendly instructions when no dataset is uploaded

🛠️ Tech Stack

Frontend: Streamlit

Data Handling: Pandas, NumPy, Scikit-learn

Visualization: Matplotlib, Seaborn, Plotly

AI Integration: LangChain + Groq API (LLaMA-3.3-70B)

Export: CSV + Excel (XlsxWriter)

📦 Installation

Clone the repo:

git clone https://github.com/your-username/streamlit-data-cleaner.git
cd streamlit-data-cleaner


Create a virtual environment & install dependencies:

pip install -r requirements.txt


Add your Groq API key to a .env file:

GROQ_API_KEY=your_api_key_here

🚀 Run Locally
streamlit run app.py


The app will start at 👉 http://localhost:8501

Live working link:https://ai-data-analyst-agent-nk734wznzewtxzncszkkja.streamlit.app

