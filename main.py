#!/usr/bin/env python3
"""
===============================================================================
COVID-19 CRISIS FINANCIAL ANALYSIS: PFIZER (PFE) 
===============================================================================

Analyzes Pfizer (PFE) stock performance across three COVID-19 crisis periods:
1. BEFORE CRISIS: February 1, 2019 – January 31, 2020
2. DURING CRISIS: February 1, 2020 – March 31, 2020  
3. RECOVERY PHASE: April 1, 2020 – March 31, 2021

===============================================================================
"""

# =============================================================================
# IMPORT REQUIRED LIBRARIES
# =============================================================================
import pandas as pd                    # Data manipulation and analysis
import numpy as np                     # Numerical computations
from datetime import datetime, timedelta
import os                             # Operating system interface
import yfinance as yf                 # Yahoo Finance data download
import statsmodels.formula.api as sm  # Statistical modeling (OLS regression)
import matplotlib.pyplot as plt       # Data visualization
from statsmodels.graphics.regressionplots import abline_plot
from scipy import stats               # Statistical functions (Q-Q plots)

# =============================================================================
# SETUP AND CONFIGURATION
# =============================================================================
# Set working directory to current project folder
os.chdir('/Users/maritzamancillas/Desktop/fin_650')


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================
# Load historical market data from CSV files
snp500 = pd.read_csv('SP500_201209014_20220901.csv')  # S&P 500 index data
rf = pd.read_csv('DGS1MO_20120901_20220901.csv')      # 1-month Treasury rate (risk-free rate)

# Display data structure for verification
snp500.head(10)
rf.head(5)

# Download Pfizer (PFE) stock data from Yahoo Finance
# This gets OHLCV data for the entire analysis period
stock_df = yf.download("PFE", start="2018-01-01", end="2021-12-31")
print("Stock data columns:", stock_df.columns.tolist())
print("Stock data shape:", stock_df.shape)
print("First few rows of stock data:")
print(stock_df.head())

# =============================================================================
# DATA CLEANING AND PREPARATION
# =============================================================================
# Rename columns for consistency
rf = rf.rename(columns={'DGS1MO': 'rf_rate'})           # Risk-free rate
snp500 = snp500.rename(columns={'SP500': 'sp500_close'}) # S&P 500 close prices

# Extract only the closing price from the multi-level column structure
# yfinance returns data with (Price, Ticker) column structure
stock_df = stock_df[('Close', 'PFE')].copy()
stock_df = stock_df.rename('stock_close')
stock_df = stock_df.reset_index(level=0)  # Convert index to column
stock_df = stock_df.rename({'Date': 'DATE'}, axis=1)

# Convert date columns to datetime format for proper merging
rf["DATE"] = pd.to_datetime(rf["DATE"])
snp500["DATE"] = pd.to_datetime(snp500["DATE"])
stock_df["DATE"] = pd.to_datetime(stock_df["DATE"])

# Check data types and convert to numeric
print("Risk-free rate data types:")
print(rf['rf_rate'].dtypes)

# Convert string data to numeric, handling any non-numeric values
rf['rf_rate'] = pd.to_numeric(rf['rf_rate'], errors='coerce')
snp500['sp500_close'] = pd.to_numeric(snp500['sp500_close'], errors='coerce')

# =============================================================================
# DEFINE ANALYSIS PERIODS
# =============================================================================
# Define the three COVID-19 crisis periods for analysis
periods = {
    'Before Crisis': ('2019-02-01', '2020-01-31'),    # Pre-crisis period
    'During Crisis': ('2020-02-01', '2020-03-31'),    # Peak crisis period
    'Recovery Phase': ('2020-04-01', '2021-03-31')    # Post-crisis recovery
}

# =============================================================================
# ANALYSIS FUNCTION DEFINITION
# =============================================================================
# Function to analyze a specific period
# This function filters data for a given time period and prepares it for regression analysis
def analyze_period(period_name, start_date, end_date, rf_data, snp500_data, stock_data):
    print(f"\n{'='*80}")
    print(f"ANALYZING PERIOD: {period_name.upper()}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"{'='*80}")
    
    # Filter data for the specific period
    period_rf = rf_data[(rf_data['DATE'] >= pd.to_datetime(start_date)) & 
                       (rf_data['DATE'] <= pd.to_datetime(end_date))].copy()
    period_snp500 = snp500_data[(snp500_data['DATE'] >= pd.to_datetime(start_date)) & 
                               (snp500_data['DATE'] <= pd.to_datetime(end_date))].copy()
    period_stock = stock_data[(stock_data['DATE'] >= pd.to_datetime(start_date)) & 
                             (stock_data['DATE'] <= pd.to_datetime(end_date))].copy()
    
    # Reset indices
    period_rf = period_rf.reset_index(drop=True)
    period_snp500 = period_snp500.reset_index(drop=True)
    period_stock = period_stock.reset_index(drop=True)
    
    print(f"Data points: RF={len(period_rf)}, S&P500={len(period_snp500)}, Stock={len(period_stock)}")
    
    # Merge the data
    merge_rf_snp500 = pd.merge(period_rf, period_snp500, how='left', on='DATE')
    merge_df = pd.merge(merge_rf_snp500, period_stock, how='left', on='DATE')
    
    # Keep only the required columns
    merge_df = merge_df[['DATE', 'rf_rate', 'sp500_close', 'stock_close']].copy()
    
    # Calculate daily returns using percentage change formula
    # Daily Return = (Price_today - Price_yesterday) / Price_yesterday
    row_num = merge_df.shape[0]
    for i in range(1, row_num):
        merge_df.loc[i, 'stock_daily_rtn'] = (merge_df.loc[i, 'stock_close'] - merge_df.loc[i-1, 'stock_close']) / merge_df.loc[i-1, 'stock_close']
        merge_df.loc[i, 'sp500_daily_rtn'] = (merge_df.loc[i, 'sp500_close'] - merge_df.loc[i-1, 'sp500_close']) / merge_df.loc[i-1, 'sp500_close']
    
    # Calculate risk-free daily rate and excess returns for CAPM analysis
    # Convert annual risk-free rate to daily rate: (annual_rate / 100) / 365
    merge_df['rf_daily'] = (merge_df['rf_rate'] * 0.01) / 365
    # Market premium = Market return - Risk-free rate
    merge_df['mrkt_premium'] = merge_df['sp500_daily_rtn'] - merge_df['rf_daily']
    # Stock excess return = Stock return - Risk-free rate
    merge_df['stock_exc_rtn'] = merge_df['stock_daily_rtn'] - merge_df['rf_daily']
    
    # Remove first row (no returns calculated)
    merge_df = merge_df[1:].copy()
    
    print(f"Final dataset: {len(merge_df)} observations")
    print(f"Date range: {merge_df['DATE'].min().strftime('%Y-%m-%d')} to {merge_df['DATE'].max().strftime('%Y-%m-%d')}")
    
    return merge_df

# =============================================================================
# MAIN ANALYSIS EXECUTION
# =============================================================================
# Store results for all periods in a dictionary
all_results = {}

# Loop through each COVID-19 crisis period and perform analysis
for period_name, (start_date, end_date) in periods.items():
    # Get data for this period
    period_data = analyze_period(period_name, start_date, end_date, rf, snp500, stock_df)
    
    if len(period_data) < 10:  # Skip if insufficient data
        print(f"⚠️  Insufficient data for {period_name}. Skipping...")
        continue
    
    # =============================================================================
    # CAPM REGRESSION ANALYSIS
    # =============================================================================
    # Run CAPM regression for this period
    # CAPM equation: R_stock - R_f = α + β(R_market - R_f) + ε
    # Where: R_stock = stock return, R_f = risk-free rate, R_market = market return
    #        α = Jensen's Alpha (excess return), β = Beta (systematic risk)
    print(f"\n📊 Running CAPM regression for {period_name}...")
    try:
        # OLS regression with HAC (Heteroscedasticity and Autocorrelation Consistent) standard errors
        # This accounts for potential heteroscedasticity and autocorrelation in financial time series
        model = sm.ols(formula='stock_exc_rtn ~ mrkt_premium', data=period_data).fit(cov_type='HAC', cov_kwds={'maxlags': 3}, use_t=True)
        
        # Store results
        all_results[period_name] = {
            'data': period_data,
            'model': model,
            'beta': model.params['mrkt_premium'],
            'alpha': model.params['Intercept'],
            'r_squared': model.rsquared,
            'p_value_beta': model.pvalues['mrkt_premium'],
            'p_value_alpha': model.pvalues['Intercept'],
            'observations': len(period_data)
        }
        
        print(f"\n📈 {period_name} CAPM Results:")
        print(f"   • Beta (β): {model.params['mrkt_premium']:.4f}")
        print(f"   • Jensen's Alpha (α): {model.params['Intercept']:.4f}")
        print(f"   • R-squared: {model.rsquared:.4f}")
        print(f"   • P-value (Beta): {model.pvalues['mrkt_premium']:.4f}")
        print(f"   • P-value (Alpha): {model.pvalues['Intercept']:.4f}")
        print(f"   • Observations: {len(period_data)}")
        
        # Print full regression table
        print(f"\n{'-'*60}")
        print(f"CAPM REGRESSION TABLE - {period_name.upper()}")
        print(f"{'-'*60}")
        print(model.summary())
        
    except Exception as e:
        print(f"Error running regression for {period_name}: {e}")
        continue

# =============================================================================
# CREATE COMPARATIVE VISUALIZATIONS
# =============================================================================
# This section creates comprehensive visualizations comparing all three periods
# Includes scatter plots, bar charts, time series, and summary tables
print("\n" + "="*80)
print("CREATING COVID-19 CRISIS PERIOD COMPARISON VISUALIZATIONS")
print("="*80)

if len(all_results) > 0:
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Pfizer (PFE) Beta and Alpha Analysis: COVID-19 Crisis Periods', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green']
    period_names = list(all_results.keys())
    
    # Plot 1: CAPM Scatter Plots for Each Period
    for i, (period_name, results) in enumerate(all_results.items()):
        data = results['data']
        model = results['model']
        color = colors[i % len(colors)]
        
        # Ensure data alignment
        clean_data = data.dropna(subset=['mrkt_premium', 'stock_exc_rtn'])
        if len(clean_data) > 0 and len(clean_data) == len(model.fittedvalues):
            axes[0, 0].scatter(clean_data['mrkt_premium'], clean_data['stock_exc_rtn'], 
                              alpha=0.6, color=color, label=f'{period_name} (β={results["beta"]:.3f})')
            axes[0, 0].plot(clean_data['mrkt_premium'], model.fittedvalues, 
                           color=color, linewidth=2, alpha=0.8)
    
    axes[0, 0].set_xlabel('Market Premium')
    axes[0, 0].set_ylabel('Stock Excess Return')
    axes[0, 0].set_title('CAPM: Stock Excess Return vs Market Premium by Period')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Beta Comparison Bar Chart
    periods = list(all_results.keys())
    betas = [all_results[p]['beta'] for p in periods]
    alphas = [all_results[p]['alpha'] for p in periods]
    
    x_pos = np.arange(len(periods))
    bars = axes[0, 1].bar(x_pos, betas, color=colors[:len(periods)], alpha=0.7)
    axes[0, 1].set_xlabel('Period')
    axes[0, 1].set_ylabel('Beta (β)')
    axes[0, 1].set_title('Beta Comparison Across Crisis Periods')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(periods, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, beta) in enumerate(zip(bars, betas)):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{beta:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Alpha Comparison Bar Chart
    bars = axes[0, 2].bar(x_pos, alphas, color=colors[:len(periods)], alpha=0.7)
    axes[0, 2].set_xlabel('Period')
    axes[0, 2].set_ylabel("Jensen's Alpha (α)")
    axes[0, 2].set_title("Jensen's Alpha Comparison Across Crisis Periods")
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(periods, rotation=45)
    axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, alpha) in enumerate(zip(bars, alphas)):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.0001 if alpha >= 0 else -0.0001),
                       f'{alpha:.4f}', ha='center', va='bottom' if alpha >= 0 else 'top', fontweight='bold')
    
    # Plot 4: R-squared Comparison
    r_squareds = [all_results[p]['r_squared'] for p in periods]
    bars = axes[1, 0].bar(x_pos, r_squareds, color=colors[:len(periods)], alpha=0.7)
    axes[1, 0].set_xlabel('Period')
    axes[1, 0].set_ylabel('R-squared')
    axes[1, 0].set_title('Model Fit (R-squared) Across Crisis Periods')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(periods, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, r2) in enumerate(zip(bars, r_squareds)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Time Series of Returns (Combined)
    for i, (period_name, results) in enumerate(all_results.items()):
        data = results['data']
        color = colors[i % len(colors)]
        axes[1, 1].plot(data['DATE'], data['stock_daily_rtn'], 
                       label=f'{period_name} Stock', alpha=0.7, color=color, linewidth=2)
    
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Daily Stock Returns')
    axes[1, 1].set_title('Pfizer Daily Returns Across Crisis Periods')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Summary Statistics Table
    axes[1, 2].axis('off')
    
    # Create summary table
    summary_data = []
    for period_name, results in all_results.items():
        summary_data.append([
            period_name,
            f"{results['beta']:.4f}",
            f"{results['alpha']:.4f}",
            f"{results['r_squared']:.4f}",
            f"{results['observations']}"
        ])
    
    table = axes[1, 2].table(cellText=summary_data,
                            colLabels=['Period', 'Beta (β)', "Alpha (α)", 'R²', 'Obs'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 2].set_title('Summary Statistics by Period', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('covid_crisis_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ COVID-19 crisis analysis visualizations created and saved as 'covid_crisis_analysis.png'")
    
    # Show the plot
    plt.show()
else:
    print(" No data available for visualization")

# =============================================================================
# COMPREHENSIVE SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*80)
print("COVID-19 CRISIS PERIOD ANALYSIS SUMMARY")
print("="*80)

if len(all_results) > 0:
    print(f"\n📊 PFE (Pfizer) Beta and Jensen's Alpha Analysis:")
    print(f"{'Period':<20} {'Beta (β)':<12} {'Alpha (α)':<12} {'R²':<8} {'Obs':<6} {'P-value (β)':<12}")
    print("-" * 80)
    
    for period_name, results in all_results.items():
        print(f"{period_name:<20} {results['beta']:<12.4f} {results['alpha']:<12.4f} "
              f"{results['r_squared']:<8.4f} {results['observations']:<6} {results['p_value_beta']:<12.4f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    # Find periods with highest/lowest beta and alpha
    betas = [(name, results['beta']) for name, results in all_results.items()]
    alphas = [(name, results['alpha']) for name, results in all_results.items()]
    
    max_beta_period = max(betas, key=lambda x: x[1])
    min_beta_period = min(betas, key=lambda x: x[1])
    max_alpha_period = max(alphas, key=lambda x: x[1])
    min_alpha_period = min(alphas, key=lambda x: x[1])
    
    print(f"🔍 BETA ANALYSIS:")
    print(f"   • Highest Beta: {max_beta_period[0]} (β = {max_beta_period[1]:.4f})")
    print(f"   • Lowest Beta:  {min_beta_period[0]} (β = {min_beta_period[1]:.4f})")
    print(f"   • Beta Range:   {max_beta_period[1] - min_beta_period[1]:.4f}")
    
    print(f"\n🔍 JENSEN'S ALPHA ANALYSIS:")
    print(f"   • Highest Alpha: {max_alpha_period[0]} (α = {max_alpha_period[1]:.4f})")
    print(f"   • Lowest Alpha:  {min_alpha_period[0]} (α = {min_alpha_period[1]:.4f})")
    print(f"   • Alpha Range:   {max_alpha_period[1] - min_alpha_period[1]:.4f}")
    
    # Statistical significance analysis
    significant_betas = [(name, results) for name, results in all_results.items() 
                        if results['p_value_beta'] < 0.05]
    significant_alphas = [(name, results) for name, results in all_results.items() 
                         if results['p_value_alpha'] < 0.05]
    
    print(f"\n📈 STATISTICAL SIGNIFICANCE:")
    print(f"   • Periods with significant Beta (p < 0.05): {len(significant_betas)}")
    for name, results in significant_betas:
        print(f"     - {name}: β = {results['beta']:.4f} (p = {results['p_value_beta']:.4f})")
    
    print(f"   • Periods with significant Alpha (p < 0.05): {len(significant_alphas)}")
    for name, results in significant_alphas:
        print(f"     - {name}: α = {results['alpha']:.4f} (p = {results['p_value_alpha']:.4f})")
    
    print(f"\n📊 MODEL FIT ANALYSIS:")
    r_squareds = [(name, results['r_squared']) for name, results in all_results.items()]
    max_r2_period = max(r_squareds, key=lambda x: x[1])
    min_r2_period = min(r_squareds, key=lambda x: x[1])
    
    print(f"   • Best Model Fit: {max_r2_period[0]} (R² = {max_r2_period[1]:.4f})")
    print(f"   • Worst Model Fit: {min_r2_period[0]} (R² = {min_r2_period[1]:.4f})")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"   • Beta measures the stock's sensitivity to market movements")
    print(f"   • Alpha measures the stock's excess return relative to CAPM expectations")
    print(f"   • Positive Alpha: Stock outperformed CAPM expectations")
    print(f"   • Negative Alpha: Stock underperformed CAPM expectations")
    print(f"   • Beta > 1: More volatile than market; Beta < 1: Less volatile than market")
    
    # =============================================================================
    # CREATE INDIVIDUAL REGRESSION TABLES AND GRAPHS FOR EACH PERIOD
    # =============================================================================

    print("\n" + "="*80)
    print("CREATING INDIVIDUAL REGRESSION TABLES AND GRAPHS")
    print("="*80)
    
    period_colors = {'Before Crisis': 'blue', 'During Crisis': 'red', 'Recovery Phase': 'green'}
    
    for period_name, results in all_results.items():
        print(f"\n{'='*60}")
        print(f"INDIVIDUAL ANALYSIS: {period_name.upper()}")
        print(f"{'='*60}")
        
        data = results['data']
        model = results['model']
        color = period_colors[period_name]
        
        # Print individual regression table
        print(f"\n📊 REGRESSION TABLE - {period_name.upper()}")
        print(f"Date Range: {data['DATE'].min().strftime('%Y-%m-%d')} to {data['DATE'].max().strftime('%Y-%m-%d')}")
        print(f"Observations: {results['observations']}")
        print(f"{'-'*60}")
        print(model.summary())
        
        # Create individual graph for this period
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Pfizer (PFE) CAPM Analysis: {period_name}', fontsize=16, fontweight='bold')
        
        # Clean data for plotting
        clean_data = data.dropna(subset=['mrkt_premium', 'stock_exc_rtn'])
        
        # Plot 1: CAPM Scatter Plot
        axes[0, 0].scatter(clean_data['mrkt_premium'], clean_data['stock_exc_rtn'], 
                          alpha=0.6, color=color, s=50)
        axes[0, 0].plot(clean_data['mrkt_premium'], model.fittedvalues, 
                       color='red', linewidth=3, label=f'CAPM Line')
        axes[0, 0].set_xlabel('Market Premium', fontsize=12)
        axes[0, 0].set_ylabel('Stock Excess Return', fontsize=12)
        axes[0, 0].set_title(f'CAPM: Stock Excess Return vs Market Premium\n{period_name}', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add regression equation
        equation = f'y = {results["alpha"]:.4f} + {results["beta"]:.4f}x\nR² = {results["r_squared"]:.4f}'
        axes[0, 0].text(0.05, 0.95, equation, transform=axes[0, 0].transAxes, 
                       verticalalignment='top', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # Plot 2: Residuals vs Fitted Values
        axes[0, 1].scatter(model.fittedvalues, model.resid, alpha=0.6, color=color, s=50)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[0, 1].set_xlabel('Fitted Values', fontsize=12)
        axes[0, 1].set_ylabel('Residuals', fontsize=12)
        axes[0, 1].set_title(f'Residuals vs Fitted Values\n{period_name}', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Observations: {results["observations"]}\nP-value (β): {results["p_value_beta"]:.4f}\nP-value (α): {results["p_value_alpha"]:.4f}'
        axes[0, 1].text(0.05, 0.95, stats_text, transform=axes[0, 1].transAxes, 
                       verticalalignment='top', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        # Plot 3: Time Series of Returns
        axes[1, 0].plot(data['DATE'], data['stock_daily_rtn'], 
                       label='Pfizer Returns', color=color, linewidth=2, alpha=0.8)
        axes[1, 0].plot(data['DATE'], data['sp500_daily_rtn'], 
                       label='S&P 500 Returns', color='orange', linewidth=2, alpha=0.8, linestyle='--')
        axes[1, 0].set_xlabel('Date', fontsize=12)
        axes[1, 0].set_ylabel('Daily Returns', fontsize=12)
        axes[1, 0].set_title(f'Daily Returns Over Time\n{period_name}', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Q-Q Plot for residuals
        from scipy import stats
        stats.probplot(model.resid, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'Q-Q Plot of Residuals\n{period_name}', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual graph
        filename = f'{period_name.lower().replace(" ", "_")}_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Individual analysis saved as '{filename}'")
        
        # Show the plot
        plt.show()
        
        # Print summary statistics for this period
        print(f"\n📈 SUMMARY STATISTICS - {period_name.upper()}")
        print(f"{'='*50}")
        print(f"Beta (β):           {results['beta']:.4f}")
        print(f"Jensen's Alpha (α): {results['alpha']:.4f}")
        print(f"R-squared:          {results['r_squared']:.4f}")
        print(f"Adjusted R²:        {model.rsquared_adj:.4f}")
        print(f"F-statistic:        {model.fvalue:.4f}")
        print(f"P-value (F-test):   {model.f_pvalue:.4f}")
        print(f"Observations:       {results['observations']}")
        print(f"Date Range:         {data['DATE'].min().strftime('%Y-%m-%d')} to {data['DATE'].max().strftime('%Y-%m-%d')}")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION - {period_name.upper()}")
        print(f"{'='*50}")
        if results['beta'] > 1:
            print(f"• Beta > 1: Pfizer is MORE volatile than the market")
        elif results['beta'] < 1:
            print(f"• Beta < 1: Pfizer is LESS volatile than the market")
        else:
            print(f"• Beta = 1: Pfizer has SAME volatility as the market")
            
        if results['alpha'] > 0:
            print(f"• Positive Alpha: Pfizer OUTPERFORMED CAPM expectations")
        elif results['alpha'] < 0:
            print(f"• Negative Alpha: Pfizer UNDERPERFORMED CAPM expectations")
        else:
            print(f"• Zero Alpha: Pfizer performed EXACTLY as expected by CAPM")
            
        print(f"• R² = {results['r_squared']:.1%}: Market movements explain {results['r_squared']:.1%} of Pfizer's returns")
        
        if results['p_value_beta'] < 0.05:
            print(f"• Beta is statistically significant (p < 0.05)")
        else:
            print(f"• Beta is NOT statistically significant (p ≥ 0.05)")
            
        if results['p_value_alpha'] < 0.05:
            print(f"• Alpha is statistically significant (p < 0.05)")
        else:
            print(f"• Alpha is NOT statistically significant (p ≥ 0.05)")
    
    print(f"\n📁 INDIVIDUAL FILES CREATED:")
    for period_name in all_results.keys():
        filename = f'{period_name.lower().replace(" ", "_")}_analysis.png'
        print(f"   • {filename} - Complete analysis for {period_name}")
    print(f"   • Terminal output - Individual regression tables for each period")
    
else:
    print(" No analysis results available. Please check data availability for the specified periods.")

