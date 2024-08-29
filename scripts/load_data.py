import pandas as pd
import os

def load_data(file_paths):
    dataframes = []
    for file_path in file_paths:
        print(f"Attempting to load: {file_path}")  # Log file being processed
        if not os.path.isfile(file_path):
            dataframes.append(f"File does not exist: {file_path}")
            continue
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type for {file_path}")
            dataframes.append(df)
            print(f"Successfully loaded: {file_path}")  # Log successful load
        except Exception as e:
            dataframes.append(f"Error loading {file_path}: {str(e)}")
            print(f"Error loading {file_path}: {str(e)}")  # Log the error
    return dataframes
