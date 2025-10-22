import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import argparse
from scipy.stats import spearmanr, ttest_ind
import numpy as np

from feature_datasets.common import *
from probe_experiment import load_probe_results
from analysis.generalization import*
from analysis.probe_plots import *

# Constants from the notebook
NS_PER_YEAR = 1e9 * 60 * 60 * 24 * 365.25

parser = argparse.ArgumentParser()
parser.add_argument('--output_type', type=str, default='when_w_period')
parser.add_argument('--exp', type=str, default='art/president')
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--president_input', type=str, default='real')
args = parser.parse_args()

def create_headline_time_plot(experiment_name, model, layer, output_dir="figures"):
    # Set seaborn style
    sns.set_context("talk", font_scale=2.5)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load headline data
    headline_df = load_entity_data('headline')
    
    # Load probe results
    probe_result = load_probe_results(
        experiment_name=experiment_name,
        model=model,
        entity_type='headline',
        feature_name='pub_date',
        prompt='when_w_period',
        output_type='when_w_period'
    )
    
    # Convert projection to datetime
    headline_pred_datetime = pd.to_datetime(probe_result['projections'][layer].projection * NS_PER_YEAR)
    headline_true_datetime = pd.to_datetime(headline_df.pub_date.values)
    headline_is_test = headline_df.is_test.values
    
    # Calculate Spearman correlation
    headline_spearman = spearmanr(
        headline_true_datetime[headline_is_test], 
        headline_pred_datetime[headline_is_test]
    ).correlation
    
    # Calculate MSE (in years squared)
    headline_rmse = np.sqrt(np.mean((
        (headline_true_datetime[headline_is_test] - headline_pred_datetime[headline_is_test]).dt.total_seconds() / (24*3600*365.25)
    ) ** 2))
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of test predictions - darker dots
    plt.scatter(
        headline_true_datetime[headline_is_test], 
        headline_pred_datetime[headline_is_test], 
        s=10, alpha=0.8
    )
    
    # Add ideal prediction line
    plt.plot(
        [headline_true_datetime.min(), headline_true_datetime.max()], 
        [headline_true_datetime.min(), headline_true_datetime.max()], 
        color='red', lw=1, alpha=0.75
    )
    
    # Add labels and details - no title
    plt.xlabel('True publication date')
    plt.ylabel('Predicted publication date')
    plt.annotate(
        f'Spearman $r$={headline_spearman:.3f}\nRMSE={headline_rmse:.1f} years', 
        xy=(0.97, 0.03), 
        xycoords='axes fraction', 
        ha='right', 
        va='bottom'
    )
    
    # Clean up plot design
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)  # You can adjust thickness as needed
        spine.set_color('black')   # You can change the color if desired
    
    # Set x and y ticks to 5-year intervals starting from 1945
    start_year = 1945
    ax = plt.gca()
    
    # Set major ticks every 5 years
    years_major_locator = mdates.YearLocator(5, month=1, day=1)
    years_minor_locator = mdates.YearLocator(1, month=1, day=1)  # Add minor ticks at each year
    years_format = mdates.DateFormatter('%Y')
    
    ax.xaxis.set_major_locator(years_major_locator)
    ax.xaxis.set_minor_locator(years_minor_locator)  # Add minor ticks
    ax.xaxis.set_major_formatter(years_format)
    
    ax.yaxis.set_major_locator(years_major_locator)
    ax.yaxis.set_minor_locator(years_minor_locator)  # Add minor ticks
    ax.yaxis.set_major_formatter(years_format)
    
    # Make minor ticks visible
    ax.tick_params(which='minor', length=2, color='gray')
    ax.tick_params(which='major', length=5)
    
    # Sanitize model name for filename by replacing slashes with underscores
    safe_model_name = model.replace('/', '_')
    
    # Save the figure with explicit error handling
    output_path = os.path.join(output_dir, f'character_headline_time_plot_{safe_model_name}_layer{layer}.pdf')
    print(f"Attempting to save plot to: {output_path}")
    
    try:
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', format='pdf')
        print(f"Plot successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")
        # Try with a simpler directory and filename as fallback
        try:
            plt.savefig("headline_plot.pdf", format='pdf')
            print("Saved to headline_plot.pdf in current directory as fallback")
        except Exception as e2:
            print(f"Fallback save also failed: {e2}")
    
    # Show plot
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")
    
    return headline_spearman, headline_rmse

def create_art_time_plot(experiment_name, model, layer, output_dir="final_figures/llama/appendix"):
    # Set seaborn style
    sns.set_style("white")
    sns.set_context("talk", font_scale=2.5)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load art data
    art_df = load_entity_data('art')
    # Load probe results
    probe_result = load_probe_results(
        experiment_name=experiment_name,
        model=model,
        entity_type='art',
        feature_name='release_date',
        prompt='release',
        output_type=args.output_type
    )
    # Convert projection to datetime
    art_pred_datetime = pd.to_datetime(probe_result['projections'][layer].projection * NS_PER_YEAR)
    art_true_datetime = pd.to_datetime(art_df.release_date.values)
    art_is_test = art_df.is_test.values
    
    # Calculate Spearman correlation and p-value
    art_spearman_result = spearmanr(
        art_true_datetime[art_is_test], 
        art_pred_datetime[art_is_test]
    )
    art_spearman = art_spearman_result.correlation
    art_spearman_pvalue = art_spearman_result.pvalue
    
    # Calculate RMSE and perform t-test for significance
    true_seconds = art_true_datetime[art_is_test].view('int64') / 1e9
    pred_seconds = art_pred_datetime[art_is_test].view('int64') / 1e9
    residuals = (true_seconds - pred_seconds) / (24*3600*365.25)  # Convert to years
    art_rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Perform t-test on residuals (compared to zero)
    rmse_ttest = ttest_ind(residuals**2, np.zeros_like(residuals))
    art_rmse_pvalue = rmse_ttest.pvalue
    
    # Sanitize model name for filename by replacing slashes with underscores
    safe_model_name = model.replace('/', '_')
    
    # Define the date ranges
    early_start = pd.Timestamp('1950-01-01')
    early_end = pd.Timestamp('1985-01-01')
    late_start = pd.Timestamp('1985-01-01')
    late_end = pd.Timestamp('2020-01-01')
    
    # Create two separate plots
    for period, start_date, end_date in [
        ('early', early_start, early_end),
        ('late', late_start, late_end)
    ]:
        # Create the plot
        plt.figure(figsize=(18,16))
        
        # Scatter plot of test predictions (filter by date range) - darker dots
        mask = (art_true_datetime >= start_date) & (art_true_datetime <= end_date) & art_is_test
        
        plt.scatter(
            art_true_datetime[mask], 
            art_pred_datetime[mask], 
            s=15, alpha=0.95
        )
        
        # Add ideal prediction line
        plt.plot(
            [start_date, end_date], 
            [start_date, end_date], 
            color='red', lw=5, alpha=0.75
        )
        
        # Add labels and details - no title
        plt.xlabel('True release date')
        plt.ylabel('Predicted release date')
        
        # Calculate period-specific correlation if there's enough data
        if sum(mask) > 5:  # Only calculate if we have sufficient data points
            period_spearman = spearmanr(art_true_datetime[mask], art_pred_datetime[mask]).correlation
            period_rmse = np.sqrt(np.mean(((art_true_datetime[mask] - art_pred_datetime[mask]).dt.total_seconds() / (24*3600*365.25)) ** 2))
            period_spearman_pvalue = spearmanr(art_true_datetime[mask], art_pred_datetime[mask]).pvalue
            period_rmse_pvalue = spearmanr(art_true_datetime[mask], art_pred_datetime[mask]).pvalue
            corr_text = f'Period Spearman $r$={period_spearman:.3f} (p={period_spearman_pvalue:.3e}), RMSE={period_rmse:.1f} years (p={period_rmse_pvalue:.3e})\nOverall $r$={art_spearman:.3f} (p={art_spearman_pvalue:.3e}), RMSE={art_rmse:.1f} years (p={art_rmse_pvalue:.3e})'
        else:
            corr_text = f'Overall Spearman $r$={art_spearman:.3f} (p={art_spearman_pvalue:.3e})\nRMSE={art_rmse:.1f} years (p={art_rmse_pvalue:.3e})'
        
        plt.annotate(
            corr_text, 
            xy=(0.97, 0.03), 
            xycoords='axes fraction', 
            ha='right', 
            va='bottom'
        )
        
        # Set specific date range
        if period == 'early':
            plt.xlim(pd.Timestamp('1948-01-01'), pd.Timestamp('1987-01-01'))
            plt.ylim(pd.Timestamp('1948-01-01'), pd.Timestamp('1987-01-01'))
        else:
            plt.xlim(pd.Timestamp('1983-01-01'), pd.Timestamp('2022-01-01'))
            plt.ylim(pd.Timestamp('1983-01-01'), pd.Timestamp('2022-01-01'))
        
        # Clean up plot design
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)  # You can adjust thickness as needed
            spine.set_color('black')   # You can change the color if desired
        
        # Set x and y ticks to 5-year intervals starting from 1945
        ax = plt.gca()
        
        # Set major ticks every 5 years
        years_major_locator = mdates.YearLocator(5, month=1, day=1)
        years_minor_locator = mdates.YearLocator(1, month=1, day=1)  # Add minor ticks at each year
        years_format = mdates.DateFormatter('%Y')
        
        ax.xaxis.set_major_locator(years_major_locator)
        ax.xaxis.set_minor_locator(years_minor_locator)  # Add minor ticks
        ax.xaxis.set_major_formatter(years_format)
        
        ax.yaxis.set_major_locator(years_major_locator)
        ax.yaxis.set_minor_locator(years_minor_locator)  # Add minor ticks
        ax.yaxis.set_major_formatter(years_format)
        
        # Make minor ticks visible
        ax.tick_params(which='minor', length=2, color='gray')
        ax.tick_params(which='major', length=5)
        
        # Save the figure with explicit error handling
        output_path = os.path.join(output_dir, f'{args.output_type}_art_time_plot_{period}_{safe_model_name}_layer{layer}.pdf')
        print(f"Attempting to save {period} plot to: {output_path}")
        
        try:
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', format='pdf')
            print(f"Plot successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
            # Try with a simpler directory and filename as fallback
            try:
                plt.savefig(f"char_art_plot_{period}.pdf", format='pdf')
                print(f"Saved to char_art_plot_{period}.pdf in current directory as fallback")
            except Exception as e2:
                print(f"Fallback save also failed: {e2}")
        
        # Show plot
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
    
    return art_spearman, art_rmse

def create_art_time_plot_full(experiment_name, model, layer, output_dir="final_figures/llama/appendix"):
    # Set seaborn style
    sns.set_style("white")
    sns.set_context("talk", font_scale=2.5)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load art data
    art_df = load_entity_data('art')
    # Load probe results
    probe_result = load_probe_results(
        experiment_name=experiment_name,
        model=model,
        entity_type='art',
        feature_name='release_date',
        prompt='release',
        output_type=args.output_type
    )
    # Convert projection to datetime
    art_pred_datetime = pd.to_datetime(probe_result['projections'][layer].projection * NS_PER_YEAR)
    art_true_datetime = pd.to_datetime(art_df.release_date.values)
    art_is_test = art_df.is_test.values
    
    # Calculate Spearman correlation and p-value
    art_spearman_result = spearmanr(
        art_true_datetime[art_is_test], 
        art_pred_datetime[art_is_test]
    )
    art_spearman = art_spearman_result.correlation
    art_spearman_pvalue = art_spearman_result.pvalue
    
    # Calculate RMSE and perform t-test for significance
    true_seconds = art_true_datetime[art_is_test].view('int64') / 1e9
    pred_seconds = art_pred_datetime[art_is_test].view('int64') / 1e9
    residuals = (true_seconds - pred_seconds) / (24*3600*365.25)  # Convert to years
    art_rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Perform t-test on residuals (compared to zero)
    rmse_ttest = ttest_ind(residuals**2, np.zeros_like(residuals))
    art_rmse_pvalue = rmse_ttest.pvalue
    
    # Sanitize model name for filename by replacing slashes with underscores
    safe_model_name = model.replace('/', '_')
    
    # Create a single plot for full date range
    plt.figure(figsize=(18, 16))
    
    # Scatter plot of test predictions
    plt.scatter(
        art_true_datetime[art_is_test], 
        art_pred_datetime[art_is_test], 
        s=15, alpha=0.95
    )
    
    # Add ideal prediction line
    plt.plot(
        [art_true_datetime.min(), art_true_datetime.max()], 
        [art_true_datetime.min(), art_true_datetime.max()], 
        color='red', lw=5, alpha=0.75
    )
    
    # Add labels and details - no title
    plt.xlabel('True release date')
    plt.ylabel('Predicted release date')
    
    plt.annotate(
        f'Spearman $r$={art_spearman:.3f}\nRMSE={art_rmse:.1f} years', 
        xy=(0.97, 0.03), 
        xycoords='axes fraction', 
        ha='right', 
        va='bottom'
    )
    print(art_spearman, art_spearman_pvalue, art_rmse, art_rmse_pvalue)
    # Set specific date range for full timeline
    min_year = pd.Timestamp('1948-01-01')
    max_year = pd.Timestamp('2022-01-01')
    plt.xlim(min_year, max_year)
    plt.ylim(min_year, max_year)
    
    # Clean up plot design
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)  # You can adjust thickness as needed
        spine.set_color('black')   # You can change the color if desired
    
    # Set x and y ticks
    ax = plt.gca()
    
    # Set major ticks every 10 years
    years_major_locator = mdates.YearLocator(10, month=1, day=1)
    years_minor_locator = mdates.YearLocator(2, month=1, day=1)  # Add minor ticks every 2 years
    years_format = mdates.DateFormatter('%Y')
    
    ax.xaxis.set_major_locator(years_major_locator)
    ax.xaxis.set_minor_locator(years_minor_locator)
    ax.xaxis.set_major_formatter(years_format)
    
    ax.yaxis.set_major_locator(years_major_locator)
    ax.yaxis.set_minor_locator(years_minor_locator)
    ax.yaxis.set_major_formatter(years_format)
    
    # Make minor ticks visible
    ax.tick_params(which='minor', length=2, color='gray')
    ax.tick_params(which='major', length=5)
    
    # Save the figure with explicit error handling
    output_path = os.path.join(output_dir, f'{args.output_type}_art_time_plot_full_{safe_model_name}_layer{layer}.pdf')
    print(f"Attempting to save full timeline plot to: {output_path}")
    
    try:
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', format='pdf')
        print(f"Plot successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")
        # Try with a simpler directory and filename as fallback
        try:
            plt.savefig(f"art_plot_full.pdf", format='pdf')
            print(f"Saved to art_plot_full.pdf in current directory as fallback")
        except Exception as e2:
            print(f"Fallback save also failed: {e2}")
    
    # Show plot
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")
    
    return art_spearman, art_rmse

def create_historical_figure_time_plot(experiment_name, model, layer, output_dir="figures", data_dir=None):
    # Set seaborn style
    sns.set_context("talk", font_scale=2.5)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set DATA_DIR environment variable if provided
    if data_dir:
        os.environ['DATA_DIR'] = data_dir
    
    # Load historical figure data
    figure_df = load_entity_data('historical_figure')
    
    # Load probe results
    probe_result = load_probe_results(
        experiment_name=experiment_name,
        model=model,
        entity_type='historical_figure',
        feature_name='death_year',
        prompt='when'
    )
    
    # Get projections directly (not datetime for historical figures)
    figure_pred = probe_result['projections'][layer].projection
    figure_true = figure_df.death_year.values
    figure_is_test = figure_df.is_test.values
    
    # Calculate Spearman correlation
    figure_spearman = spearmanr(
        figure_true[figure_is_test], 
        figure_pred[figure_is_test]
    ).correlation
    
    # Calculate MSE (in years squared)
    figure_rmse = np.sqrt(np.mean((figure_true[figure_is_test] - figure_pred[figure_is_test]) ** 2))
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of test predictions - darker dots
    plt.scatter(
        figure_true[figure_is_test], 
        figure_pred[figure_is_test], 
        s=10, alpha=0.8
    )
    
    # Add ideal prediction line
    plt.plot(
        [figure_true.min(), figure_true.max()], 
        [figure_true.min(), figure_true.max()], 
        color='red', lw=1, alpha=0.75
    )
    
    # Add labels and details - no title
    plt.xlabel('True death year')
    plt.ylabel('Predicted death year')
    plt.annotate(
        f'Spearman $r$={figure_spearman:.3f}\nRMSE={figure_rmse:.1f} years', 
        xy=(0.97, 0.03), 
        xycoords='axes fraction', 
        ha='right', 
        va='bottom'
    )
    
    # Set specific year range like in the notebook
    plt.ylim(450, 2020)
    plt.xlim(450, 2020)
    
    # Clean up plot design
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)  # You can adjust thickness as needed
        spine.set_color('black')   # You can change the color if desired
    
    # Set x and y ticks
    ax = plt.gca()
    
    # Set major and minor ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))  # Add minor ticks at each year
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))  # Add minor ticks at each year
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
    
    # Make minor ticks visible
    ax.tick_params(which='minor', length=2, color='gray')
    ax.tick_params(which='major', length=5)
    
    # Sanitize model name for filename by replacing slashes with underscores
    safe_model_name = model.replace('/', '_')
    
    # Save the figure with explicit error handling
    output_path = os.path.join(output_dir, f'historical_figure_time_plot_{safe_model_name}_layer{layer}.pdf')
    print(f"Attempting to save plot to: {output_path}")
    
    try:
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', format='pdf')
        print(f"Plot successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")
        # Try with a simpler directory and filename as fallback
        try:
            plt.savefig("historical_figure_plot.pdf", format='pdf')
            print("Saved to char_historical_figure_plot.pdf in current directory as fallback")
        except Exception as e2:
            print(f"Fallback save also failed: {e2}")
    
    # Show plot
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")
    
    return figure_spearman, figure_rmse

def create_president_time_plot(experiment_name, model, layer, president_input, output_dir="final_figures/llama/presidential", data_dir=None):
    # Set seaborn style
    sns.set_style("white")
    sns.set_context("talk", font_scale=2.5)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set DATA_DIR environment variable if provided
    if data_dir:
        os.environ['DATA_DIR'] = data_dir
    
    # Load president data
    #FIXME: change type
    president_df = load_entity_data("presidential_baseline")
    
    # Load probe results
    probe_result = load_probe_results(
        experiment_name=president_input,
        model=model,
        entity_type=president_input,
        feature_name='presidential',
        prompt='president_normal',
        output_type=args.output_type
    )
    
    # Get projections directly (not datetime for president years)
    president_pred = probe_result['projections'][layer].projection
    president_true = president_df.year.values
    president_is_test = president_df.is_test.values
    
    # Calculate overall Spearman correlation and p-value
    president_spearman_result = spearmanr(
        president_true[president_is_test], 
        president_pred[president_is_test]
    )
    president_spearman = president_spearman_result.correlation
    president_spearman_pvalue = president_spearman_result.pvalue
    
    # Calculate RMSE and perform t-test for significance
    residuals = president_true[president_is_test] - president_pred[president_is_test]
    president_rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Perform t-test on residuals (compared to zero)
    rmse_ttest = ttest_ind(residuals**2, np.zeros_like(residuals))
    president_rmse_pvalue = rmse_ttest.pvalue
    
    # Sanitize model name for filename by replacing slashes with underscores
    safe_model_name = model.replace('/', '_')
    
    # Define the year ranges
    early_start = 1780
    early_end = 1900
    late_start = 1900
    late_end = 2025
    
    # Create two separate plots
    for period, start_year, end_year in [
        ('early', early_start, early_end),
        ('late', late_start, late_end)
    ]:
        # Create the plot
        plt.figure(figsize=(18, 16))
        
        # Scatter plot of test predictions (filter by date range)
        mask = (president_true >= start_year) & (president_true <= end_year) & president_is_test
        
        plt.scatter(
            president_true[mask], 
            president_pred[mask], 
            s=3000, alpha=0.95
        )
        
        # Add ideal prediction line
        plt.plot(
            [start_year, end_year], 
            [start_year, end_year], 
            color='red', lw=5, alpha=0.75
        )
        
        # Add labels and details - no title
        plt.xlabel('True presidential year')
        plt.ylabel('Predicted presidential year')
        
        # Calculate period-specific correlation if there's enough data
        if sum(mask) > 5:  # Only calculate if we have sufficient data points
            period_spearman = spearmanr(president_true[mask], president_pred[mask]).correlation
            period_rmse = np.sqrt(np.mean((president_true[mask] - president_pred[mask]) ** 2))
            period_spearman_pvalue = spearmanr(president_true[mask], president_pred[mask]).pvalue
            period_rmse_pvalue = spearmanr(president_true[mask], president_pred[mask]).pvalue
            corr_text = f'Period Spearman $r$={period_spearman:.3f} (p={period_spearman_pvalue:.3e}), RMSE={period_rmse:.1f} years (p={period_rmse_pvalue:.3e})\nOverall $r$={president_spearman:.3f} (p={president_spearman_pvalue:.3e}), RMSE={president_rmse:.1f} years (p={president_rmse_pvalue:.3e})'
        else:
            corr_text = f'Overall Spearman $r$={president_spearman:.3f} (p={president_spearman_pvalue:.3e})\nRMSE={president_rmse:.1f} years (p={president_rmse_pvalue:.3e})'
        
        plt.annotate(
            corr_text, 
            xy=(0.97, 0.03), 
            xycoords='axes fraction', 
            ha='right', 
            va='bottom'
        )
        
        # Set specific year ranges for each period
        if period == 'early':
            plt.xlim(1780, 1900)
            plt.ylim(1780, 1900)
        else:
            plt.xlim(1900, 2021)
            plt.ylim(1900, 2021)
        
        # Clean up plot design
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)  # You can adjust thickness as needed
            spine.set_color('black')   # You can change the color if desired
        
        # Set x and y ticks
        ax = plt.gca()
        
        # Adjust tick spacing based on period
        if period == 'early':
            major_tick_interval = 20
            minor_tick_interval = 4
        else:
            major_tick_interval = 20
            minor_tick_interval = 4
        
        # Set major and minor ticks
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        
        ax.yaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
        
        # Make minor ticks visible
        ax.tick_params(which='minor', length=2, color='gray')
        ax.tick_params(which='major', length=5)
        
        # Save the figure with explicit error handling
        output_path = os.path.join(output_dir, f'{args.output_type}_plot_{period}_{safe_model_name}_layer{layer}.pdf')
        print(f"Attempting to save {period} plot to: {output_path}")
        
        try:
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', format='pdf')
            print(f"Plot successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
            # Try with a simpler directory and filename as fallback
            try:
                plt.savefig(f"president_plot_{period}.pdf", format='pdf')
                print(f"Saved to president_plot_{period}.pdf in current directory as fallback")
            except Exception as e2:
                print(f"Fallback save also failed: {e2}")
        
        # Show plot
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
    
    return president_spearman, president_rmse

def create_president_time_plot_full(experiment_name, model, layer, president_input, output_dir="final_figures/llama/appendix", data_dir=None):
    # Set seaborn style
    sns.set_style("white")
    sns.set_context("talk", font_scale=2.5)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set DATA_DIR environment variable if provided
    if data_dir:
        os.environ['DATA_DIR'] = data_dir
    
    # Load president data
    president_df = load_entity_data("presidential_baseline")
    
    # Load probe results
    probe_result = load_probe_results(
        experiment_name=president_input,
        model=model,
        entity_type=president_input,
        feature_name='presidential',
        prompt='president_normal',
        output_type=args.output_type
    )
    
    # Get projections directly (not datetime for president years)
    president_pred = probe_result['projections'][layer].projection
    president_true = president_df.year.values
    president_is_test = president_df.is_test.values
    
    # Calculate overall Spearman correlation and p-value
    president_spearman_result = spearmanr(
        president_true[president_is_test], 
        president_pred[president_is_test]
    )
    president_spearman = president_spearman_result.correlation
    president_spearman_pvalue = president_spearman_result.pvalue
    
    # Calculate RMSE and perform t-test for significance
    residuals = president_true[president_is_test] - president_pred[president_is_test]
    president_rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Perform t-test on residuals (compared to zero)
    rmse_ttest = ttest_ind(residuals**2, np.zeros_like(residuals))
    president_rmse_pvalue = rmse_ttest.pvalue
    
    # Sanitize model name for filename by replacing slashes with underscores
    safe_model_name = model.replace('/', '_')
    
    # Create a single plot for full range
    plt.figure(figsize=(18, 16))
    
    # Scatter plot of test predictions
    plt.scatter(
        president_true[president_is_test], 
        president_pred[president_is_test], 
        s=1800, alpha=0.7, linewidth=1
    )
    
    # Add ideal prediction line
    plt.plot(
        [president_true.min(), president_true.max()], 
        [president_true.min(), president_true.max()], 
        color='red', lw=5, alpha=0.75
    )
    
    # Add labels and details - no title
    plt.xlabel('True presidential year')
    plt.ylabel('Predicted presidential year')
    
    plt.annotate(
        f'Spearman $r$={president_spearman:.3f}\nRMSE={president_rmse:.1f} years', 
        xy=(0.97, 0.03), 
        xycoords='axes fraction', 
        ha='right', 
        va='bottom'
    )
    print(president_spearman, president_spearman_pvalue, president_rmse, president_rmse_pvalue)
    # Set specific year range for the full timeline
    plt.xlim(1780, 2021)
    plt.ylim(1780, 2021)
    
    # Clean up plot design
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)  # You can adjust thickness as needed
        spine.set_color('black')   # You can change the color if desired
    
    # Set x and y ticks
    ax = plt.gca()
    
    # Set major and minor ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(40))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    ax.yaxis.set_major_locator(plt.MultipleLocator(40))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
    
    # Make minor ticks visible
    ax.tick_params(which='minor', length=2, color='gray')
    ax.tick_params(which='major', length=5)
    
    # Save the figure with explicit error handling
    output_path = os.path.join(output_dir, f'{args.output_type}_plot_full_{safe_model_name}_layer{layer}.pdf')
    print(f"Attempting to save full timeline plot to: {output_path}")
    
    try:
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', format='pdf')
        print(f"Plot successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")
        # Try with a simpler directory and filename as fallback
        try:
            plt.savefig(f"president_plot_full.pdf", format='pdf')
            print(f"Saved to president_plot_full.pdf in current directory as fallback")
        except Exception as e2:
            print(f"Fallback save also failed: {e2}")
    
    # Show plot
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")
    
    return president_spearman, president_rmse

if __name__ == "__main__":
    # create_headline_time_plot(experiment_name="headline_experiment_whenw", model="meta-llama/Llama-3.1-8B-Instruct", layer=20)
    
    # You can uncomment the below line to also run the art time plot
    # create_art_time_plot(experiment_name="art", model="meta-llama/Llama-3.1-8B-Instruct", layer=31)
    # create_art_time_plot(experiment_name="art", model="google/gemma-2-9b-it", layer=31)
    # create_historical_figure_time_plot(experiment_name="historical_death_year", model="meta-llama/Llama-3.1-8B-Instruct", layer=31)
    if args.exp == 'president':
        # create_president_time_plot(experiment_name="presidential", model=args.model, layer=31, president_input=args.president_input)
        create_president_time_plot_full(experiment_name="presidential", model=args.model, layer=31, president_input=args.president_input)
    elif args.exp == 'art':
        # create_art_time_plot(experiment_name="art", model=args.model, layer=31)
        create_art_time_plot_full(experiment_name="art", model=args.model, layer=31)