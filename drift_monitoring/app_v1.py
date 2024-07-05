import streamlit as st
import boto3
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset,DataQualityPreset
from datetime import datetime,timedelta

# Initialize the S3 client
s3 = boto3.client('s3')

# S3 bucket name
bucket_name = 'loanprediction'

# Function to list folders in the specified S3 bucket and prefix
def list_folders(bucket_name, prefix):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    return [content.get('Prefix') for content in response.get('CommonPrefixes', [])]

# Function to list CSV files in a specified S3 folder
def list_csv_files(bucket_name, folder):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder)
    return [content['Key'] for content in response.get('Contents', []) if content['Key'].endswith('.csv')]

# Function to download a CSV file from S3 and load it into a DataFrame
def load_csv_from_s3(bucket_name, key):
    response = s3.get_object(Bucket=bucket_name, Key=key)
    return pd.read_csv(response['Body'])

# Function to calculate data drift using Evidently
def calculate_data_drift_evidently(baseline_df, latest_df):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline_df, current_data=latest_df)
    return report

def calculate_data_quality_evidently(baseline_df, latest_df):
    report = Report(metrics=[DataQualityPreset()])
    report.run(reference_data=baseline_df, current_data=latest_df)
    return report

# Function to find the most recent folder
def find_most_recent_folder(bucket_name, prefix, max_days=7):
    for i in range(max_days):
        check_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        check_date_folder = f'{prefix}{check_date}/'
        if check_date_folder in list_folders(bucket_name, prefix):
            return check_date_folder
    return None

# Main Streamlit app
def main():

    # add_custom_css()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ["Data Drift", "Data Quality"])

    #st.title('Data Analysis')
    # st.title('Data Drift Analysis')
    if page == "Data Drift":
        st.header('Data Drift Analysis')
        prefix = 'datadrift/'
        # current_date = datetime.now().strftime('%Y-%m-%d')
        # current_date_folder = f'{prefix}{current_date}/'

        most_recent_folder = find_most_recent_folder(bucket_name, prefix)

        # List all folders in the datadrift directory
        # folders = list_folders(bucket_name, prefix)
        
        # if current_date_folder in folders:
        if most_recent_folder:
            # Load the baseline CSV
            baseline_csv_key = 'datadrift/baseline.csv'
            baseline_df = load_csv_from_s3(bucket_name, baseline_csv_key)

            # Drop 'Loan_ID' and 'Loan_Status' columns from the baseline DataFrame
            baseline_df = baseline_df.drop(columns=['Loan_ID', 'Loan_Status'])

            # List all CSV files in the current date folder
            latest_csv_files = list_csv_files(bucket_name, most_recent_folder)
            
            # Streamlit dropdown for selecting the target dataset
            selected_file = st.selectbox('Select the target dataset', latest_csv_files)

            if selected_file:
                # Load and process the selected target dataset
                latest_df = load_csv_from_s3(bucket_name, selected_file)

                # Calculate data drift
                drift_report = calculate_data_drift_evidently(baseline_df, latest_df)
                
                #Replace colons with underscores in the filename
                safe_filename = selected_file.split("/")[-1].replace(":", "_")
                
                report_filename = f'drift_report_{safe_filename}.html'
                drift_report.save_html(report_filename)
                
                
                # Display the report
                with open(report_filename, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    st.components.v1.html(html_content, height=1000,width=1000,scrolling=True)
        else:
            st.write(f'No folder found ')

    elif page == "Data Quality":
        st.header('Data Quality Analysis')
        prefix = 'datadrift/'
        # current_date = datetime.now().strftime('%Y-%m-%d')
        # current_date_folder = f'{prefix}{current_date}/'

        # List all folders in the datadrift directory
        #folders = list_folders(bucket_name, prefix)
        most_recent_folder = find_most_recent_folder(bucket_name, prefix)
        
        if  most_recent_folder:
            # Load the baseline CSV
            baseline_csv_key = 'datadrift/baseline.csv'
            baseline_df = load_csv_from_s3(bucket_name, baseline_csv_key)

            # Drop 'Loan_ID' and 'Loan_Status' columns from the baseline DataFrame
            baseline_df = baseline_df.drop(columns=['Loan_ID', 'Loan_Status'])

            # List all CSV files in the current date folder
            latest_csv_files = list_csv_files(bucket_name, most_recent_folder)
            
            # Streamlit dropdown for selecting the target dataset
            selected_file = st.selectbox('Select the target dataset', latest_csv_files)

            if selected_file:
                # Load and process the selected target dataset
                latest_df = load_csv_from_s3(bucket_name, selected_file)

                latest_df=latest_df.drop(['Prediction'],axis=1)

                # Calculate data drift
                drift_report = calculate_data_quality_evidently(baseline_df, latest_df)
                
                #Replace colons with underscores in the filename
                safe_filename = selected_file.split("/")[-1].replace(":", "_")
                
                report_filename = f'drift_report_{safe_filename}.html'
                drift_report.save_html(report_filename)
                
                
                # Display the report
                with open(report_filename, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    st.components.v1.html(html_content, height=1000,width=1300,scrolling=True)
      
        else:
            st.write(f'No folder found ')


if __name__ == "__main__":
    main()
