import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Global Migration Data Visualization",
    page_icon="🌍",
    layout="wide"
)

# Title and description
st.title("Migration Data Visualization Platform")
st.markdown("This application provides interactive visualizations for global migration trends data analysis.")

# Load data
@st.cache_data
def load_data():
    file_path = r"C:\Users\dell\Desktop\country_of_destination_migration.xlsx"
    return pd.read_excel(file_path)

# Handle file loading with error handling
try:
    df = load_data()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading the file: {e}")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Detailed Analysis", "Correlation Analysis", "Geographic Visualization", "Custom Filters"])

# Overview page
if page == "Overview":
    st.header("Data Overview")
    
    # Display data summary
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Sample")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Data Statistics")
        st.dataframe(df.describe())
    
    # Display data info
    st.subheader("Data Information")
    buffer = pd.io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_data[missing_data > 0].plot(kind='bar', ax=ax)
        plt.title("Missing Values by Column")
        plt.xlabel("Columns")
        plt.ylabel("Count")
        st.pyplot(fig)
    else:
        st.write("No missing values found in the dataset!")

# Detailed Analysis page
elif page == "Detailed Analysis":
    st.header("Detailed Data Analysis")
    
    # Select columns for analysis
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Visualization options
    analysis_type = st.selectbox("Select Analysis Type", ["Numerical Analysis", "Categorical Analysis"])
    
    if analysis_type == "Numerical Analysis":
        if not numeric_columns:
            st.warning("No numerical columns found in the dataset")
        else:
            selected_column = st.selectbox("Select a Numerical Column", numeric_columns)
            
            # Distribution analysis
            st.subheader(f"Distribution of {selected_column}")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=selected_column, nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y=selected_column)
                st.plotly_chart(fig, use_container_width=True)
            
            # Time series if date column exists
            date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            if date_columns:
                st.subheader("Time Series Analysis")
                date_col = st.selectbox("Select Date Column", date_columns)
                
                # Aggregate by date
                time_series = df.groupby(df[date_col].dt.date)[selected_column].mean().reset_index()
                fig = px.line(time_series, x=date_col, y=selected_column)
                st.plotly_chart(fig, use_container_width=True)
    
    else:  # Categorical Analysis
        if not categorical_columns:
            st.warning("No categorical columns found in the dataset")
        else:
            selected_column = st.selectbox("Select a Categorical Column", categorical_columns)
            
            # Count of categories
            st.subheader(f"Count of {selected_column}")
            count_df = df[selected_column].value_counts().reset_index()
            count_df.columns = [selected_column, 'Count']
            
            fig = px.bar(count_df, x=selected_column, y='Count')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show percentage distribution
            st.subheader("Factors of Migration(in millions)")
            fig = px.pie(count_df, names=['job opportunity','education','war/refugee','seeking asylum','natural disasters','environmental changes'], values=[169,6.1,32.5,2.2,32.6,22.3])
            st.plotly_chart(fig, use_container_width=True)

# Correlation Analysis page
elif page == "Correlation Analysis":
    st.header("Correlation Analysis")
    
    # Select columns for correlation
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.warning("Need at least two numerical columns for correlation analysis")
    else:
        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr = df[numeric_columns].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Scatter plot for selected columns
        st.subheader("Scatter Plot Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis", ['Country', 'Continent'])
        with col2:
           y_col = st.selectbox("Select Y-axis", [col for col in numeric_columns if col not in ['Country', 'Continent']])
        
    
        # Color by categorical column if available
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        color_col = None
        if categorical_columns:
            color_col = st.selectbox("Color by (optional)", ["None"] + categorical_columns)
            if color_col == "None":
                color_col = None
        
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig, use_container_width=True)

# Geographic Visualization page
elif page == "Geographic Visualization":
    st.header("Geographic Visualization")
    
    # Check if there are columns that might contain geographic information
    potential_geo_columns = [col for col in df.columns if any(geo_term in col.lower() for geo_term in ['country', 'region', 'city', 'state', 'province', 'location', 'geo', 'lat', 'lon', 'lng'])]
    
    if not potential_geo_columns:
        st.info("No geographic columns detected in the dataset. Geographic visualization requires columns with location data.")
    else:
        geo_col = st.selectbox("Select Geographic Column", potential_geo_columns)
        
        # Check if we have count data for choropleth
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_columns:
            value_col = st.selectbox("Select Value Column for Mapping", numeric_columns)
            
            # Aggregate data by geographic column
            geo_data = df.groupby(geo_col)[value_col].sum().reset_index()
            
            # If lat/lon columns exist, create scatter map
            lat_cols = [col for col in df.columns if 'lat' in col.lower()]
            lon_cols = [col for col in df.columns if 'lon' in col.lower() or 'lng' in col.lower()]
            
            if lat_cols and lon_cols:
                st.subheader("Geographic Scatter Plot")
                lat_col = lat_cols[0]
                lon_col = lon_cols[0]
                
                fig = px.scatter_geo(df, 
                                   lat=lat_col,
                                   lon=lon_col,
                                   color=value_col,
                                   hover_name=geo_col,
                                   size=value_col,
                                   title=f"{value_col} by Location")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Create a bar chart instead
                st.subheader(f"{value_col} by {geo_col}")
                fig = px.bar(geo_data.sort_values(value_col, ascending=False).head(20), 
                           x=geo_col, 
                           y=value_col,
                           title=f"Top 20 {geo_col} by {value_col}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numerical columns available for mapping values.")

# Custom Filters page
elif page == "Custom Filters":
    st.header("Custom Data Filters")
    
    # Allow selection of columns to display
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select Columns to Display", all_columns, default=all_columns[:5])
    
    # Add filters for each column
    st.subheader("Apply Filters")
    filters = {}
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Select columns to filter
    filter_columns = st.multiselect("Select Columns to Filter", all_columns)
    
    # Create appropriate filters based on data type
    filtered_df = df.copy()
    for column in filter_columns:
        st.markdown(f"**Filter for {column}:**")
        
        if column in categorical_columns:
            unique_values = df[column].dropna().unique().tolist()
            selected_values = st.multiselect(f"Select values for {column}", unique_values)
            if selected_values:
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
        
        elif column in numeric_columns:
            min_val = float(filtered_df[column].min())
            max_val = float(filtered_df[column].max())
            range_val = st.slider(f"Range for {column}", min_val, max_val, (min_val, max_val))
            filtered_df = filtered_df[(filtered_df[column] >= range_val[0]) & (filtered_df[column] <= range_val[1])]
        
        elif column in date_columns:
            min_date = filtered_df[column].min().date()
            max_date = filtered_df[column].max().date()
            date_range = st.date_input(f"Range for {column}", (min_date, max_date))
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[(filtered_df[column].dt.date >= start_date) & (filtered_df[column].dt.date <= end_date)]
    
    # Display filtered data
    st.subheader("Filtered Data")
    st.dataframe(filtered_df[selected_columns])
    
    # Download button for filtered data
    if not filtered_df.empty:
        st.download_button(
            label="Download Filtered Data as CSV",
            data=filtered_df[selected_columns].to_csv(index=False).encode('utf-8'),
            file_name='filtered_migration_data.csv',
            mime='text/csv',
        )

# Add footer
st.markdown("---")
st.markdown("Migration Data Analysis Platform | Created with Streamlit")