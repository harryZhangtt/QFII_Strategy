import gc
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from statsmodels.robust.scale import mad
"""
this is the third part of the project. The first and second step construct the matrix sorted by date and ticker name that contains each stocks's share percentage held by foreign investors. 

on this QFIIProcessor class, we combine this dataframe with price and volume dataframe we constructed through previous project, and then we construct our QFII strategy as taking the top 1/3 of 
stocks invested by foreign investors(Market Cap weighted). The strategy is backtested and examined detailed(shown on the report on QFII strategy)
"""


class QFIIProcessor:
    def __init__(self, qp_folder_path, percentage_file_path, output_path, industry_file_path,CSI800_path):
        self.start_date=pd.to_datetime('2017-03-31')
        self.end_date=pd.to_datetime('2024-06-30')
        self.qp_folder_path = qp_folder_path
        self.percentage_file_path = percentage_file_path
        self.output_path = output_path
        self.industry_file_path = industry_file_path
        self.percentage_df = None
        self.qp_merged_df = None
        self.mapped_df = None
        self.all_stocks = None
        self.all_factors=[]
        self.CSI800_path= CSI800_path
        self.industry_data = pd.read_excel(industry_file_path, engine='openpyxl')

    def get_sorted_files(self, folder_path, prefix):
        files = [f for f in os.listdir(folder_path) if f.startswith(prefix)]
        return sorted(files)

    def extract_date_from_filename(self, filename):
        # Assumes the date is in YYYYMMDD format somewhere in the filename
        date_str = ''.join(filter(str.isdigit, filename))
        return pd.to_datetime(date_str, format='%Y%m%d')
    """
    load price and volume data
    """
    def load_qp_data(self):
        qp_files = self.get_sorted_files(self.qp_folder_path, 'merged')[2912:]
        # qp_files = self.get_sorted_files(self.qp_folder_path, 'merged')[3400:3700]

        qp_merged_df = pd.DataFrame()

        for file_name in qp_files:
            print(file_name)
            date = self.extract_date_from_filename(file_name)
            if self.start_date <= date <= self.end_date:
                qp_file_path = os.path.join(self.qp_folder_path, file_name)
                qp_df = pd.read_csv(qp_file_path)
                qp_df['Date'] = date
                qp_merged_df = pd.concat([qp_merged_df, qp_df], ignore_index=True)

        qp_merged_df.rename(columns={'sk_1': 'SECU_CODE', 'CP_1': 'ADJ_CLOSE_PRICE'}, inplace=True)

        qp_merged_df = qp_merged_df.sort_values(by=['Date', 'SECU_CODE'])

        return qp_merged_df
    """
    load share percentage data(percentage held by foreign investors) 
    """
    def load_data(self):
        self.percentage_df = pd.read_csv(self.percentage_file_path, encoding='utf-8')

        self.percentage_df['Date'] = pd.to_datetime(self.percentage_df['Date'], format='%Y%m%d')

        # Sort by Date to ensure proper forward filling
        self.percentage_df = self.percentage_df.sort_values(by=['Date'])

    def merge_data(self, qp_merged_df):
        final_df = qp_merged_df.copy()
        unique_dates = final_df['Date'].unique()

        # Sort industry data by SECU_CODE
        self.industry_data.sort_values(by='SECU_CODE', inplace=True)

        all_combined = []

        for unique_date in unique_dates:
            print(unique_date)
            qp_rows = final_df[final_df['Date'] == unique_date].copy()

            # Merge industry information
            qp_rows.sort_values(by='SECU_CODE', inplace=True)
            industry_info = self.industry_data[['SECU_CODE', 'industry']].set_index('SECU_CODE')
            qp_rows = qp_rows.join(industry_info, on='SECU_CODE')

            all_combined.append(qp_rows)

        final_df = pd.concat(all_combined)

        # Only filter stocks that are not ST and with price greater than 1
        final_df = final_df[(final_df['sk_2'] == 1) & (final_df['ADJ_CLOSE_PRICE'] >= 1)]

        return final_df

    """
    this function here is tricky as well because share percentage df and qp data don't share the same time series( due to the infrequent nature of financial report),
    therefore, after merge, ffill is important
    """
    def map_and_fill(self, final_df):
        # Set multi-index for mapping
        self.percentage_df.set_index(['Date', 'SECU_CODE'], inplace=True)
        final_df.set_index(['Date', 'SECU_CODE'], inplace=True)

        # Sort the indices to ensure proper join operations
        self.percentage_df.sort_index(inplace=True)
        final_df.sort_index(inplace=True)

        # Map the QFII percentage to the corresponding ticker and date
        self.mapped_df = final_df.join(self.percentage_df, how='left')

        # Debugging: Ensure the mapping is successful
        sample_date = pd.to_datetime('2017-04-14')
        sample_secu_code = 2415
        print(f"Sample mapping check for Date: {sample_date} and SECU_CODE: {sample_secu_code}")
        print("Mapped DataFrame row:", self.mapped_df.loc[(sample_date, sample_secu_code)])

        # Check if the 'Share_Percentage' column exists after the join
        if 'Share_Percentage' not in self.mapped_df.columns:
            raise KeyError("Column 'Share_Percentage' not found after join operation")

        # Forward fill the percentage values within each SECU_CODE group
        self.mapped_df=self.mapped_df.sort_values(by='Date')
        self.mapped_df['Share_Percentage'] = self.mapped_df.groupby(level='SECU_CODE')['Share_Percentage'].ffill()


        # Reindex to include all combinations of dates and SECU_CODEs from final_df
        self.mapped_df = self.mapped_df.reindex(final_df.index)

        # Reset the index to preserve 'Date' as a column again
        self.mapped_df.reset_index(inplace=True)
        # Shift Share_Percentage by one to avoid future data leakage
        self.mapped_df['Share_Percentage'] = self.mapped_df.groupby('SECU_CODE')['Share_Percentage'].shift(1)
        # Fill missing Share_Percentage values with 0
        self.mapped_df['Share_Percentage'] = self.mapped_df['Share_Percentage'].fillna(0)

        print(self.mapped_df.columns)
        print(self.mapped_df)
    """
    in this strategy, the stocks pool is securites from Chinese CSI800 index, this function filter out stocks within the index
    """
    def filter_CSI800(self):
        df = self.mapped_df.copy()
        df = df.sort_values(by=['Date', 'SECU_CODE'])

        # Get the list of files from the directories
        csi300_files = sorted(
            [os.path.join(self.CSI800_path, 'CSI300', f) for f in os.listdir(os.path.join(self.CSI800_path, 'CSI300'))
             if f.endswith('.txt')])
        csi500_files = sorted(
            [os.path.join(self.CSI800_path, 'CSI500', f) for f in os.listdir(os.path.join(self.CSI800_path, 'CSI500'))
             if f.endswith('.txt')])

        # Convert self.start and self.end to datetime for comparison
        start_date = self.start_date
        end_date = self.end_date

        # Function to extract date from filename
        def get_date_from_filename(filename):
            base = os.path.basename(filename)
            date_str = base.split('_')[-1].split('.')[0]
            return pd.to_datetime(date_str, format='%Y%m%d')

        # Filter files based on the date range
        csi300_files = [f for f in csi300_files if start_date <= get_date_from_filename(f) <= end_date]
        csi500_files = [f for f in csi500_files if start_date <= get_date_from_filename(f) <= end_date]

        # Create an empty DataFrame to store the filtered results
        filtered_df = pd.DataFrame()

        # Iterate through the files and merge them based on dates
        for i, (csi300_file, csi500_file) in enumerate(zip(csi300_files, csi500_files)):
            # Ensure the files are for the same date
            date_300 = get_date_from_filename(csi300_file)
            date_500 = get_date_from_filename(csi500_file)

            if date_300 != date_500:
                continue

            # Read the files
            df_300 = pd.read_csv(csi300_file, header=None)
            df_500 = pd.read_csv(csi500_file, header=None)

            # Merge the dataframes
            merged_df = pd.concat([df_300, df_500], ignore_index=True)

            # Rename the first column to 'SECU_CODE' and drop the second column by keeping only the first column and the remaining columns
            merged_df = merged_df.iloc[:, [0] + list(range(2, merged_df.shape[1]))].rename(columns={0: 'SECU_CODE'})

            # Clean and convert SECU_CODE to integer, handle potential conversion issues
            merged_df['SECU_CODE'] = merged_df['SECU_CODE'].apply(lambda x: str(x).strip().split()[0]).astype(int)

            # Filter the main DataFrame by SECU_CODE and Date
            daily_filter = merged_df['SECU_CODE']
            daily_filtered_df = df[df['SECU_CODE'].isin(daily_filter) & (df['Date'] == date_300)]

            # Append the filtered daily data to the overall filtered DataFrame
            filtered_df = pd.concat([filtered_df, daily_filtered_df])

            # Show progress
            print(f"Processed {i + 1}/{len(csi300_files)} files")

        # Set the filtered DataFrame as the new mapped_df
        self.mapped_df = filtered_df.sort_values(by=['Date', 'SECU_CODE'])
        print(self.mapped_df)
    """
    the alpha value of QFII strategy is calculated as the ratio of amount of money foreign investors put on particular stocks 
    and the total amount of money invested in Chinese A share stocks market(within CSI800)
    """
    def calculate_weighted_price(self):
        df = self.mapped_df.copy()
        self.all_stocks=df
        df=df.sort_values(by=['Date','SECU_CODE'])

        """
        when processing stocks, only consider stocks picked by QFII
        """
        df['Market_Cap'] =df['MvTotal_1'
        df['QFII_SHARE'] = 0.01 * df['Share_Percentage'] * df['Market_Cap']
        df['QFII_ALPHA'] = df['QFII_SHARE'] / df.groupby('Date')['QFII_SHARE'].transform('sum')
        df['next_day_return'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].diff()
        df['next_day_return'] = df['next_day_return'].fillna(0)
        """
        this filter can reduce amount of work done significantly because it reduce the size of the matrix, however, this should be done 
        after the calculation of next_day_return to avoid missing data
        """
        df = df[df['QFII_ALPHA'] > 0]


        self.all_factors.append('QFII_ALPHA')
        self.mapped_df=df




    def backtest_alpha(self, initial_capital=1e8):
        df = self.mapped_df.copy()
        df = df.sort_values(by=['Date', 'SECU_CODE'])
        df['pct_return']=df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].pct_change()

        def weight_assignment(df, vector):
            df['vector'] = df[vector].astype(float)
            df = df.sort_values(by='Date', ascending=True)

            # Define masks for long and short investments based on the normalized factor


            # Define masks for long investments based on the top 1/3 quantile
            def normalize_weights(group):
                """
                pick top 1/3 of stocks invested by foreign investors(market cap weighted), normalized their weight 
                to make sure they sum up to one
                """
                # Divide the data into 3 quantiles, dropping duplicates
                group['quantile'] = pd.qcut(group['vector'], 3, labels=False, duplicates='drop')

                # Select the top quantile group (highest quantile)
                top_quantile = group['quantile'].max()
                mask = group['quantile'] == top_quantile

                # Initialize 'weight' column with zeros, ensuring the same dtype as 'vector'
                group['weight'] = pd.Series([0] * len(group), dtype=group['vector'].dtype)

                # Assign 'vector' values to 'weight' for the top quantile group
                group.loc[mask, 'weight'] = group.loc[mask, 'vector']

                # Normalize the weights so they sum up to 1 for the top quantile
                total_weight = group.loc[mask, 'weight'].sum()
                if total_weight > 0:
                    group.loc[mask, 'weight'] = group.loc[mask, 'weight'] / total_weight

                # Drop the quantile column as it's no longer needed
                group = group.drop(columns=['quantile'])
                return group

            # Apply normalization per date
            df = df.groupby('Date').apply(normalize_weights).reset_index(drop=True)

            # Calculate excess weight
            # Cap the weights at 0.1 and calculate excess weight within transform
            """
            for risk control, we want to ensure no tickers receive weight greater than 0.1, 
            this is hard to achieve while maintaining weight sum of one, 
            so the portfolio get rebalanced by distributing excess weight(those greater than 0.1) evenly to stocks 
            that have weight less than 0.1
            """
            for date, group in df.groupby('Date'):
                if len(group) > 30:
                    # Copy the weight column to be adjusted
                    adjusted_weights = group['weight'].copy()

                    # Cap the weights at 0.1 and calculate the excess weight as a single value
                    capped_weights = np.where(group['weight'] > 0.1, 0.1, group['weight'])
                    excess_weight = (group['weight'] - capped_weights).sum()

                    # Apply the capped weights to the adjusted weights
                    adjusted_weights.loc[group.index] = capped_weights

                    # Create a mask for weights less than or equal to 0.1
                    #make sure to ignore those outside the 1/3
                    mask = (group['weight'] <= 0.1) & (group['weight'] > 0)

                    # Calculate the total number of eligible weights
                    total_eligible = mask.sum()

                    # Calculate the excess distribution
                    excess_distribution = excess_weight / total_eligible if total_eligible != 0 else 0

                    # Distribute the excess weight evenly among eligible weights
                    adjusted_weights.loc[group.index[mask]] += excess_distribution

                    # Set the adjusted weights back to the original weight column
                    df.loc[group.index, 'weight'] = adjusted_weights.loc[group.index]

            return df

        for value_factor in self.all_factors:
            df['vector']=df[value_factor]
            df = weight_assignment(df, 'vector')
            df['sum_weight']= df.groupby('Date')['weight'].transform('sum')


            df['long_capital_allocation'] = initial_capital * df['weight']
            df['long_investments'] = ((df['long_capital_allocation'] / df['ADJ_CLOSE_PRICE']) // 100) * 100

            df['investment'] = df['long_investments']

            df['previous_investment'] = df.groupby('SECU_CODE')['investment'].shift(1)
            df['investment_change'] = (df['investment'] - df['previous_investment']).fillna(0)
            df['abs_investment_change'] = abs(df['investment_change'])

            df['pnl'] = df['investment'] * df['next_day_return']
            df['pnl'] = df['pnl'].fillna(0)

            df['Date'] = pd.to_datetime(df['Date'])

            df['tvr_shares'] = df['abs_investment_change']
            df['tvr_values'] = df['abs_investment_change'] * df['ADJ_CLOSE_PRICE']
            df['tvr_shares'] = df['tvr_shares'].fillna(0)
            df['tvr_values'] = df['tvr_values'].fillna(0)
            # Calculate monthly average vector values
            df['year_month'] = df['Date'].dt.to_period('M')

            # df = self.calculate_factor_exposure(df, value_factor)
            # df = self.calculate_factor_stability_coefficient(df)
            # df.rename(columns={'factor_stability_coefficient_y': 'factor_stability_coefficient'}, inplace=True)
            # print(df['factor_stability_coefficient'])
            aggregated = df.groupby('Date').agg(
                year_month=('year_month', 'last'),
                pnl=('pnl', 'sum'),
                tvrshares=('tvr_shares', 'sum'),
                tvrvalues=('tvr_values', 'sum'),
                # factor_exposure=('factor_exposure', 'mean'),
                # factor_stability_coefficient=('factor_stability_coefficient', 'last')
            ).reset_index()


            aggregated['cum_pnl'] = aggregated['pnl'].cumsum() / (initial_capital)
            # Extract year from Date
            aggregated['year'] = aggregated['Date'].dt.year
            # Calculate annualized return for each year
            annual_returns = aggregated.groupby('year')['pnl'].sum().reset_index()
            annual_returns.columns = ['year', 'annualized_return']
            annual_returns['annualized_return'] = annual_returns['annualized_return'] / (initial_capital)
            df['TOTALVALUE'] = df['MvTotal_1']

            # Merge annualized return back to aggregated DataFrame
            aggregated = pd.merge(aggregated, annual_returns, on='year', how='left')
            # Calculate Sharpe Ratio
            daily_returns = (aggregated['pnl'] / (2 * initial_capital)).fillna(0)
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            aggregated['sharpe_ratio'] = sharpe_ratio

            # Calculate the rolling mean of 21 days for 'vector' grouped by 'SECU_CODE'
            df['year_month'] = df['Date'].dt.to_period('M')  # Assuming you have a 'Date' column
            df['vector_rolling_mean'] = df.groupby('SECU_CODE')['vector'].transform(
                lambda x: x.rolling(window=21, min_periods=1).mean())

            # Merge the rolling mean vector back to the original DataFrame
            df = pd.merge(df, df[['Date', 'SECU_CODE', 'vector_rolling_mean']], on=['Date', 'SECU_CODE'],
                          suffixes=('', '_monthly'))
            print(df['vector_rolling_mean_monthly'])

            # Shift the returns by one month to align with the previous month's vector values
            df['next_month_return'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].pct_change(
                periods=21)  # Assuming 21 trading days in a month

            # Calculate Information Coefficient (IC)
            monthly_ic = df.groupby('year_month').apply(
                lambda x: x[['vector_rolling_mean_monthly', 'next_month_return']].corr().iloc[0, 1]
            ).reset_index()
            monthly_ic.columns = ['year_month', 'IC']

            aggregated = pd.merge(aggregated, monthly_ic, on='year_month', how='left')
            aggregated['cum_IC'] = aggregated['IC'].cumsum()
            aggregated['IC_YAvg'] = aggregated.groupby('year')['IC'].transform('mean')

            def calculate_max_drawdown(aggregated):
                max_drawdown = 0
                for i in range(1, len(aggregated)):
                    drawdown = aggregated.loc[i, 'pnl'] / initial_capital
                    if drawdown < max_drawdown:
                        max_drawdown = drawdown
                    aggregated.loc[i, 'mdd'] = max_drawdown
                return aggregated

            aggregated = calculate_max_drawdown(aggregated)

            def plot_combined_graphs(aggregated, df, initial_principal, vector):
                # Ensure Date is treated as datetime
                aggregated['Date'] = pd.to_datetime(aggregated['Date'], format='%Y%m%d')
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
                df['pct_return'] = np.log(df['ADJ_CLOSE_PRICE'] / df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(1))
                self.all_stocks['pct_return']=np.log(self.all_stocks['ADJ_CLOSE_PRICE'] / self.all_stocks.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(1))
                cumulative_avg_return = self.all_stocks.groupby('Date')['pct_return'].mean().cumsum()
                # Calculate cumulative return for vector > 0
                cumulative_return_vector_positive = df[df['vector'] > 0].groupby('Date')['pct_return'].mean().cumsum()

                # Calculate TVR ratio
                aggregated['tvr_ratio'] = aggregated['tvrvalues'] / initial_principal

                # Calculate excess returns
                aggregated[f'{vector}_excess_pnl'] = aggregated['cum_pnl'] - cumulative_avg_return.reindex(
                    aggregated['Date']).values

                fig, axs = plt.subplots(3, 1, figsize=(10, 8))

                # Plot cumulative PnL
                axs[0].plot(aggregated['Date'], aggregated['cum_pnl'], label='Cumulative PnL')
                axs[0].plot(cumulative_avg_return.index, cumulative_avg_return.values,
                            label='Cumulative Average Return')
                axs[0].plot(cumulative_return_vector_positive.index, cumulative_return_vector_positive.values,
                            label='Cumulative Return (vector > 0)')
                axs[0].xaxis.set_major_locator(mdates.YearLocator())
                axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                axs[0].set_title('Cumulative PnL and Cumulative Average Return', fontsize='small')
                axs[0].set_xlabel('Trading Day', fontsize='small')
                axs[0].set_ylabel('Cumulative Return', fontsize='small')
                axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                axs[0].grid(True)

                # Plot histogram of TVR ratio
                axs[1].hist(aggregated['tvr_ratio'], bins=30, color='blue', edgecolor='black', alpha=0.7)
                axs[1].set_title('Distribution of TVR Ratio', fontsize='small')
                axs[1].set_xlabel('TVR Ratio', fontsize='small')
                axs[1].set_ylabel('Frequency', fontsize='small')
                axs[1].grid(True)

                # Plot annualized return
                axs[2].plot(aggregated['Date'], aggregated['annualized_return'], label='Annualized Return')
                axs[2].xaxis.set_major_locator(mdates.YearLocator())
                axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                axs[2].set_title('Annualized Return Over Time', fontsize='small')
                axs[2].set_xlabel('Trading Day', fontsize='small')
                axs[2].set_ylabel('Annualized Return', fontsize='small')
                axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                axs[2].grid(True)

                # Save the first plot
                plt.tight_layout()
                plt.savefig(f'{output_path}/{value_factor}_pnl.png')
                plt.close(fig)

                fig, axs = plt.subplots(2, 1, figsize=(14, 8))

                # Plot excess returns
                axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_pnl'], label='Excess PnL')
                axs[0].xaxis.set_major_locator(mdates.YearLocator())
                axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                axs[0].set_title('Excess Returns', fontsize='small')
                axs[0].set_xlabel('Trading Day', fontsize='small')
                axs[0].set_ylabel('Excess Return', fontsize='small')
                axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                axs[0].grid(True)

                # Plot cumulative IC
                axs[1].plot(aggregated['Date'], aggregated['cum_IC'], label='Cumulative IC')
                axs[1].xaxis.set_major_locator(mdates.YearLocator())
                axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                axs[1].set_title('Cumulative IC', fontsize='small')
                axs[1].set_xlabel('Trading Day', fontsize='small')
                axs[1].set_ylabel('Cumulative IC', fontsize='small')
                axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                axs[1].grid(True)

                # Save the second plot
                plt.tight_layout()
                plt.savefig(f'{output_path}/{value_factor}_excess.png')
                plt.close(fig)

            plot_combined_graphs(aggregated, df, initial_capital, value_factor)

           
            def grouping_analysis(df, output_path, value_factor):
                # Sort the dataframe by vector and trading day for proper quantile grouping
                df = df.sort_values(by=['Date', 'vector'])
                # Only need to analyze the values that are not zero
                df = df[df['vector']> 0]

                # Calculate size exposure by dividing the data into 10 size groups and calculating average capital allocation
                df['size_group'] = pd.qcut(df['TOTALVALUE'], q=10, labels=False, duplicates='drop')
                size_exposure = df.groupby('size_group').agg(
                    long_capital_allocation=('long_capital_allocation', 'sum'),
                    avg_size=('TOTALVALUE', 'mean')
                ).reset_index()
                total_allocation = df['long_capital_allocation'].sum()
                size_exposure['allocation_percentage'] = size_exposure['long_capital_allocation'] / total_allocation
                size_exposure = size_exposure.sort_values(by='size_group', ascending=True)

                # Calculate industry exposure as percentage of the total allocation within each industry
                industry_exposure = df.groupby('industry')['long_capital_allocation'].sum().reset_index()
                industry_exposure['allocation_percentage'] = industry_exposure[
                                                                 'long_capital_allocation'] / total_allocation
                industry_exposure.columns = ['industry', 'long_capital_allocation', 'allocation_percentage']
                industry_exposure = industry_exposure.sort_values(by='allocation_percentage', ascending=False)

                # Create subplots
                fig, axs = plt.subplots(2, 1, figsize=(14, 10))

                # Plotting size allocation percentages as a pie chart with average size noted
                size_labels = [f'Group {int(g)}: Avg Size {s:.2f}' for g, s in
                               zip(size_exposure['size_group'], size_exposure['avg_size'])]
                axs[0].pie(size_exposure['allocation_percentage'], labels=size_labels,
                           autopct='%1.1f%%', colors=plt.cm.tab20.colors)
                axs[0].set_title('Capital Allocation by Size Group', fontsize='small')

                # Plotting industry allocation percentages as a pie chart
                axs[1].pie(industry_exposure['allocation_percentage'], labels=industry_exposure['industry'],
                           autopct='%1.1f%%', colors=plt.cm.tab20.colors)
                axs[1].set_title('Capital Allocation by Industry', fontsize='small')

                # Save the plots
                plt.tight_layout()
                plt.savefig(f'{output_path}/{value_factor}_grouping.png')
                plt.close(fig)

            grouping_analysis(df, self.output_path, value_factor)

            output_file = os.path.join(output_path, f'{value_factor}_results_2.0.csv')
            print('latest_date up to', aggregated['Date'].max())
            aggregated.to_csv(output_file, index=False)

            overall_pnl = df['pnl'].sum()

            print(f"{value_factor} PnL: {overall_pnl}")
            self.mapped_df=df
            gc.collect()


    def save_results(self):
        # Save the monthly aggregated DataFrame to a new CSV file
        output_file_path = os.path.join(self.output_path, 'monthly_aggregated_file.csv')
        self.mapped_df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"Saved monthly aggregated file to {output_file_path}")

    def process(self):
        qp_merged_df = self.load_qp_data()
        self.load_data()
        final_df = self.merge_data(qp_merged_df)
        self.map_and_fill(final_df)
        self.filter_CSI800()
        self.calculate_weighted_price()
        self.backtest_alpha()
        self.save_results()


# Example usage
qp_folder_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/MergedCSVs/'
percentage_file_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/QFII/aggregated_share_percentage.csv'
output_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/QFII'
industry_file_path = '/Users/zw/Desktop/IndustryCitic_with_industry.xlsx'
CSI800_path='/Users/zw/Desktop/Index_Weight'

processor = QFIIProcessor(qp_folder_path, percentage_file_path, output_path, industry_file_path,CSI800_path)
processor.process()
