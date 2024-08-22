import os
import pandas as pd

# Define the base directories
base_dir = '/Users/zw/Desktop/FinanceReport/FinanceReport'
output_dir = '/Users/zw/Desktop/FinanceReport/FinanceReport/QFII'
day_base_dir = '/Users/zw/Desktop/DataBase'
day_liq_free_base_dir = '/Users/zw/Desktop/DataBase-1'

# Define the subfolders
subfolders = ['FSk', 'FRDay']

# Function to extract the date from the filename
def extract_date(filename):
    date_str = filename.split('_')[1].split('.')[0]
    return date_str

# Dictionary to store lists of files by date
files_by_date = {}

# Populate the dictionary with filenames grouped by date
for subfolder in subfolders:
    subfolder_path = os.path.join(base_dir, subfolder)
    if os.path.exists(subfolder_path):
        # Get a list of files and sort them by date
        sorted_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.txt')], key=extract_date)
        for file_name in sorted_files:
            date_str = extract_date(file_name)
            if date_str not in files_by_date:
                files_by_date[date_str] = {}
            files_by_date[date_str][subfolder] = os.path.join(subfolder_path, file_name)

# Sort the dates
sorted_dates = sorted(files_by_date.keys())

# Process and merge files by date
os.makedirs(output_dir, exist_ok=True)

combined_df = pd.DataFrame()

for date_str in sorted_dates:
    print(date_str)
    file_dict = files_by_date[date_str]
    # Read and merge files of the same date
    df_list = []
    for subfolder in subfolders:
        if subfolder in file_dict:
            df = pd.read_csv(file_dict[subfolder], header=None, delimiter='\t')
            # Generate column names based on subfolder
            num_columns = df.shape[1]
            column_names = [f"{subfolder}_{i+1}" for i in range(num_columns)]
            df.columns = column_names
            df['Date'] = date_str  # Add the extracted date as a new column
            df_list.append(df)
        else:
            print(f"Warning: Missing file for {subfolder} on {date_str}")
            continue

    # Concatenate dataframes horizontally
    if df_list:
        temp_df = pd.concat(df_list, axis=1)
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

# Save the combined DataFrame to CSV
output_file = os.path.join(output_dir, 'combined_FRDay.csv')
combined_df.to_csv(output_file, index=False)
print(f"Saved combined file to {output_file}")

print("All files processed and saved.")
