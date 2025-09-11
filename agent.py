import os
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------------------
# Streamlit App Configuration
# ----------------------------
st.set_page_config(page_title="AI Data Analyst Agent", layout="wide", initial_sidebar_state="expanded")
st.title("📊 AI Data Analyst Agent")
st.markdown("*Interactive Data Cleaning + EDA + AI-Powered Analysis*")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
cleaning_options = st.sidebar.multiselect(
    "Select cleaning operations:",
    ["Remove Duplicates", "Handle Missing Values", "Fix Data Types", "Cap Outliers", "Normalize Data"],
    default=["Remove Duplicates", "Handle Missing Values"]
)

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read dataset with error handling
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Store original data for comparison
        if 'original_df' not in st.session_state:
            st.session_state.original_df = df.copy()

        st.success("✅ Dataset loaded successfully!")
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "🔍 Data Quality", "📊 Visualizations", "🤖 AI Analysis"])
        
        with tab1:
            st.subheader("📌 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data types summary
            st.subheader("📋 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)

        with tab2:
            st.subheader("🛑 Data Quality Report")
            
            # Missing values visualization
            if df.isnull().sum().sum() > 0:
                st.write("**Missing Values Pattern:**")
                missing_data = df.isnull().sum().sort_values(ascending=False)
                missing_data = missing_data[missing_data > 0]
                
                if not missing_data.empty:
                    fig_missing = px.bar(
                        x=missing_data.values, 
                        y=missing_data.index,
                        orientation='h',
                        title="Missing Values by Column"
                    )
                    st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("🎉 No missing values detected!")

            # Duplicate analysis
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                st.warning(f"⚠️ Found {duplicates} duplicate rows ({duplicates/len(df)*100:.2f}%)")
            else:
                st.success("🎉 No duplicate rows found!")

            # Correlation heatmap for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                st.subheader("📈 Correlation Analysis")
                fig, ax = plt.subplots(figsize=(10, 6))
                correlation_matrix = numeric_df.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap="RdYlBu_r", center=0, ax=ax)
                st.pyplot(fig)
                
                # Find high correlations
                high_corr = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr.append({
                                'Column 1': correlation_matrix.columns[i],
                                'Column 2': correlation_matrix.columns[j],
                                'Correlation': round(corr_val, 3)
                            })
                
                if high_corr:
                    st.warning("🔥 High Correlations Detected (|r| > 0.7):")
                    st.dataframe(pd.DataFrame(high_corr), use_container_width=True)

        with tab3:
            st.subheader("📊 Interactive Visualizations")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_column = st.selectbox("Select column for analysis:", df.columns)
            with col2:
                chart_type = st.selectbox("Chart type:", ["Auto", "Histogram", "Box Plot", "Bar Chart", "Scatter Plot"])

            if selected_column:
                if df[selected_column].dtype in ['int64', 'float64'] and chart_type in ["Auto", "Histogram"]:
                    fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical summary
                    stats = df[selected_column].describe()
                    st.write("**Statistical Summary:**")
                    st.dataframe(stats.to_frame().T, use_container_width=True)
                
                elif df[selected_column].dtype in ['int64', 'float64'] and chart_type == "Box Plot":
                    fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type in ["Auto", "Bar Chart"] or df[selected_column].dtype == 'object':
                    value_counts = df[selected_column].value_counts().head(15)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                title=f"Top 15 Categories in {selected_column}")
                    st.plotly_chart(fig, use_container_width=True)

            # Multi-column analysis
            st.subheader("🔀 Multi-Column Analysis")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_columns)
                with col2:
                    y_col = st.selectbox("Y-axis:", numeric_columns)
                
                if x_col and y_col and x_col != y_col:
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)

        with tab4:
            # ----------------------------
            # Interactive Data Cleaning
            # ----------------------------
            st.subheader("🧹 Data Cleaning")
            
            if st.button("🧹 Apply Selected Cleaning Operations", type="primary"):
                cleaned_df = df.copy()
                cleaning_log = []

                with st.spinner("⚡ Cleaning dataset..."):
                    # Remove duplicates
                    if "Remove Duplicates" in cleaning_options:
                        initial_shape = cleaned_df.shape[0]
                        cleaned_df.drop_duplicates(inplace=True)
                        removed = initial_shape - cleaned_df.shape[0]
                        if removed > 0:
                            cleaning_log.append(f"✅ Removed {removed} duplicate rows")

                    # Fix data types
                    if "Fix Data Types" in cleaning_options:
                        for c in cleaned_df.columns:
                            if cleaned_df[c].dtype == 'object':
                                # Try datetime conversion
                                try:
                                    pd.to_datetime(cleaned_df[c])
                                    cleaned_df[c] = pd.to_datetime(cleaned_df[c])
                                    cleaning_log.append(f"✅ Converted {c} to datetime")
                                    continue
                                except:
                                    pass
                                
                                # Try numeric conversion
                                try:
                                    pd.to_numeric(cleaned_df[c])
                                    cleaned_df[c] = pd.to_numeric(cleaned_df[c])
                                    cleaning_log.append(f"✅ Converted {c} to numeric")
                                except:
                                    pass

                    # Handle missing values
                    if "Handle Missing Values" in cleaning_options:
                        for col in cleaned_df.columns:
                            missing_count = cleaned_df[col].isnull().sum()
                            if missing_count > 0:
                                if cleaned_df[col].dtype in ['int64', 'float64']:
                                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                                    cleaning_log.append(f"✅ Filled {missing_count} missing values in {col} with median")
                                else:
                                    mode_val = cleaned_df[col].mode()
                                    if len(mode_val) > 0:
                                        cleaned_df[col].fillna(mode_val[0], inplace=True)
                                        cleaning_log.append(f"✅ Filled {missing_count} missing values in {col} with mode")

                    # Outlier treatment
                    if "Cap Outliers" in cleaning_options:
                        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            Q1 = cleaned_df[col].quantile(0.25)
                            Q3 = cleaned_df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5 * IQR
                            upper = Q3 + 1.5 * IQR
                            
                            outliers_count = ((cleaned_df[col] < lower) | (cleaned_df[col] > upper)).sum()
                            if outliers_count > 0:
                                cleaned_df[col] = cleaned_df[col].clip(lower, upper)
                                cleaning_log.append(f"✅ Capped {outliers_count} outliers in {col}")

                    # Normalize numeric data
                    if "Normalize Data" in cleaning_options:
                        scaler = StandardScaler()
                        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            cleaned_df[numeric_cols] = scaler.fit_transform(cleaned_df[numeric_cols])
                            cleaning_log.append(f"✅ Normalized {len(numeric_cols)} numeric columns")

                # Update the main dataframe
                df = cleaned_df
                
                st.success("✅ Data cleaning completed!")
                
                # Show cleaning log
                if cleaning_log:
                    st.write("**Cleaning Operations Applied:**")
                    for log in cleaning_log:
                        st.write(log)
                else:
                    st.info("No cleaning operations were needed or selected.")

                # Show before/after comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Before Cleaning:**")
                    st.write(f"Shape: {st.session_state.original_df.shape}")
                    st.write(f"Missing values: {st.session_state.original_df.isnull().sum().sum()}")
                with col2:
                    st.write("**After Cleaning:**")
                    st.write(f"Shape: {df.shape}")
                    st.write(f"Missing values: {df.isnull().sum().sum()}")

                st.dataframe(df.head(), use_container_width=True)

            # ----------------------------
            # Download Options
            # ----------------------------
            st.subheader("📥 Download Cleaned Data")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="CleanedData")
                excel_data = excel_buffer.getvalue()

                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name="cleaned_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            # ----------------------------
            # AI Query System (Groq LLM)
            # ----------------------------
            st.subheader("🤖 Ask AI about your Data")

            # Query examples
            with st.expander("💡 Example Questions"):
                st.write("""
                - "What is the average Age?"
                - "Show me the distribution of [column_name]"
                - "Find correlations between numeric columns"
                - "What are the top 5 categories in [column_name]?"
                - "Calculate summary statistics for all numeric columns"
                - "Identify outliers in the dataset"
                - "What patterns do you see in the data?"
                """)

            query = st.text_input("💬 Type your question (e.g., 'What is the average Age?')")

            if query:
                try:
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile",
                        api_key=GROQ_API_KEY,
                        temperature=0
                    )
                    
                    agent = create_pandas_dataframe_agent(
                        llm,
                        df,
                        verbose=False,
                        allow_dangerous_code=True,
                        handle_parsing_errors=True
                    )
                    
                    with st.spinner("🤖 Analyzing your query..."):
                        response = agent.run(query)
                    
                    st.success("✅ AI Response:")
                    st.write(response)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("💡 Try rephrasing your question or check if the column names are correct.")

    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.info("💡 Please check if your file is properly formatted and try again.")

else:
    # Landing page
    st.markdown("""
    ## 🚀 Welcome to AI Data Analyst Agent!
    
    This powerful tool helps you:
    
    - 📊 **Analyze** your data with interactive visualizations
    - 🧹 **Clean** your data automatically with customizable options
    - 🔍 **Explore** data quality issues and patterns
    - 🤖 **Query** your data using natural language with AI
    
    ### 📋 Instructions:
    1. Upload your CSV or Excel file using the uploader above
    2. Explore your data in the different tabs
    3. Clean your data with customizable cleaning options
    4. Ask questions about your data using natural language
    5. Download your cleaned dataset
    
    ### 📁 Supported File Formats:
    - CSV files (.csv)
    - Excel files (.xlsx)
    
    **Ready to get started? Upload your dataset above!** 📈
    """)

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit, Pandas, Groq AI, and LangChain*")