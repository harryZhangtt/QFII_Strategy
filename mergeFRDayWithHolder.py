import pandas as pd
from tqdm import tqdm

# Define file paths
frday_file_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/QFII/combined_FRDay.csv'
topten1_file_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/QFII/topten1.xlsx'
topten2_file_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/QFII/topten2.xlsx'
names_file_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/QFII/QFII_LIST.xlsx'

# Read the CSV and Excel files
frday_df = pd.read_csv(frday_file_path, encoding='utf-8')
topten1_df = pd.read_excel(topten1_file_path, engine='openpyxl')
topten2_df = pd.read_excel(topten2_file_path, engine='openpyxl')

# Read the names file with openpyxl engine
try:
    names_df = pd.read_excel(names_file_path, engine='openpyxl')
except Exception as e:
    print(f"Error reading {names_file_path}: {e}")
    exit()

# Ensure the number of columns is consistent
expected_columns = 6
topten1_df = topten1_df[topten1_df.apply(lambda x: len(x) == expected_columns, axis=1)]
topten2_df = topten2_df[topten2_df.apply(lambda x: len(x) == expected_columns, axis=1)]

# Rename columns in topten1_df and topten2_df to match for merging
topten1_df.columns = ['SECU_NAME', 'Date', 'Holder_Name', 'Holder_Type', 'Share_Percentage', 'SECU_CODE']
topten2_df.columns = ['SECU_NAME', 'Date', 'Holder_Name', 'Holder_Type', 'Share_Percentage', 'SECU_CODE']

# Merge topten1_df and topten2_df
combined_topten_df = pd.concat([topten1_df, topten2_df], ignore_index=True)

# Merge the combined_topten_df with frday_df on 'Date' and 'SECU_CODE'
merged_df = pd.merge(combined_topten_df, frday_df, on=['Date', 'SECU_CODE'], how='left')

# Fill missing values in merged_df with the corresponding FRDay data for the same ticker on the same date
frday_columns = frday_df.columns.difference(['Date', 'SECU_CODE'])

for column in frday_columns:
    merged_df[column] = merged_df.groupby(['Date', 'SECU_CODE'])[column].transform(lambda x: x.ffill().bfill())

# Replace "Date" column values with "FRDay" column values and rename "FRDay" to "Date"
if 'FRDay' in merged_df.columns:
    merged_df['Date'] = merged_df['FRDay']
    merged_df = merged_df.drop(columns=['FRDay'])

# Remove "国际有限公司" characters
chars_to_remove = "国际有限公司"

# Preprocess the Holder_Name column to convert all non-string values to NaN
merged_df['Holder_Name'] = merged_df['Holder_Name'].apply(lambda x: ''.join([char for char in str(x) if char not in chars_to_remove]) if isinstance(x, str) else None)

# Remove characters from CHNAME in names_df
names_df['CHNAME'] = names_df['CHNAME'].apply(lambda x: ''.join([char for char in x if char not in chars_to_remove]))

# Create a dictionary for O(1) lookup
names_df['CHNAME'] = names_df['CHNAME'].astype(str)
names_df['ENNAME'] = names_df['ENNAME'].astype(str)
name_dict = {row['CHNAME']: row['ENNAME'] for _, row in names_df.iterrows()}

# Function to check similarity based on character matching
def is_similar(name, name_dict):
    if pd.isnull(name):
        return False
    for chname, engname in name_dict.items():
        sim_chname = sum(1 for a, b in zip(name, chname) if a == b) / max(len(name), len(chname))
        sim_engname = sum(1 for a, b in zip(name, engname) if a == b) / max(len(name), len(engname))
        if sim_chname > 0.70 or sim_engname > 0.70:
            ##87
            ##60
            ##70
            ##75

            return True
    return False

# Filter out Holder_Name that are similar to any CHNAME or ENNAME with progress bar
tqdm.pandas(desc="Filtering Holder_Name")
merged_df=merged_df.sort_values(by='Date',ascending=True)


filtered_merged_df = merged_df[merged_df['Holder_Name'].progress_apply(lambda x: is_similar(x, name_dict))]
filtered_merged_df['Holder_Type']= filtered_merged_df['Holder_Type'].astype(str)
filtered_merged_df = filtered_merged_df[~filtered_merged_df['Holder_Type'].str.contains('国有|境内')]
filtered_merged_df.to_csv('/Users/zw/Desktop/FinanceReport/FinanceReport/QFII/filtered_names(debug).csv', index=False, encoding='utf-8-sig')

# Group by 'Date' and 'SECU_CODE', summing 'Share_Percentage'
aggregated_df = filtered_merged_df.groupby(['Date', 'SECU_CODE'])['Share_Percentage'].sum().reset_index()
aggregated_df=aggregated_df.sort_values(by='Date',ascending=True)
# Save the aggregated DataFrame to a new CSV file
output_file_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/QFII/aggregated_share_percentage.csv'
aggregated_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

print(f"Saved aggregated file to {output_file_path}")


