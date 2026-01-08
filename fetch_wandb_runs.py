"""
Script to fetch all runs from Weights & Biases and save to CSV file.
"""
import pandas as pd
import wandb
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
import json
import matplotlib.pyplot as plt
import os
import ast


def check_filter_condition(config: Dict[str, Any], key: str, filter_value: Any) -> bool:
    """
    Check if a config value matches the filter condition.

    Args:
        config: Run configuration dictionary
        key: Configuration key to check
        filter_value: Expected value or condition

    Returns:
        True if condition matches, False otherwise
    """
    # Handle tuple conditions for "not in" (e.g., ("not_in", [0.2, 0.5]))
    # If the key is missing and this is a "not_in" condition, pass the filter
    # (missing key means it's not in the excluded values)
    if isinstance(filter_value, tuple) and filter_value[0] == "not_in":
        if key not in config:
            return True  # Missing key passes "not_in" filter
        return config[key] not in filter_value[1]

    # For all other filter types, missing key means filter fails
    if key not in config:
        return False

    config_value = config[key]

    # Handle list conditions (e.g., model in ["lstm", "FCN3"])
    if isinstance(filter_value, list):
        return config_value in filter_value

    # Handle simple equality
    return config_value == filter_value


def apply_filters(config: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Apply all filter conditions to a run configuration.

    Args:
        config: Run configuration dictionary
        filters: Dictionary of filter conditions

    Returns:
        True if all conditions match, False otherwise
    """
    if not filters:
        return True

    for key, filter_value in filters.items():
        if not check_filter_condition(config, key, filter_value):
            return False

    return True


def fetch_wandb_runs(
        entity: str = "rausch-technology",
        project: str = "ma-wahl-forecast",
        output_file: str = "runs.csv",
        filters: Optional[Dict[str, Any]] = None
):
    """
    Fetch all runs from a Weights & Biases project and save to CSV.

    Args:
        entity: W&B entity/organization name
        project: W&B project name
        output_file: Output CSV filename
        filters: Dictionary of filter conditions. Examples:
            - Simple equality: {"energy": "all", "res": "hourly"}
            - List membership: {"model": ["lstm", "FCN3"]}
            - Exclusion: {"remove_per": ("not_in", [0.2, 0.5])}

    Example:
        filters = {
            "energy": "all",
            "res": "hourly",
            "model": ["lstm", "FCN3"],
            "dataset": "building",
            "interpolate": 1,
            "remove_per": ("not_in", [0.2, 0.5])
        }
    """
    print(f"Connecting to W&B project: {entity}/{project}")
    api = wandb.Api()

    # Fetch all runs from the project
    runs = api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} runs")

    if filters:
        print(f"Applying filters: {filters}")

    summary_list, config_list, name_list, metadata_list, system_metrics_list = [], [], [], [], []
    filtered_count = 0

    print("Processing runs...")
    for i, run in enumerate(runs, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(runs)} runs...")

        # Get config (remove special values that start with _)
        config = {k: v for k, v in run.config.items()
                  if not k.startswith('_')}

        # Apply filters
        if filters and not apply_filters(config, filters):
            filtered_count += 1
            continue

        # .summary contains the output keys/values for metrics like accuracy.
        # We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # Store config
        config_list.append(config)

        # Store metadata (contains system/hardware info)
        metadata_list.append(run.metadata if hasattr(run, 'metadata') else {})

        # Store system_metrics (contains GPU power usage, etc.)
        system_metrics_list.append(run.system_metrics if hasattr(run, 'system_metrics') else {})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    # Create DataFrame
    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "metadata": metadata_list,
        "system_metrics": system_metrics_list,
        "name": name_list
    })

    # Save to CSV
    runs_df.to_csv(output_file, index=False)

    print(f"\nFiltered out: {filtered_count} runs")
    print(f"Successfully saved {len(runs_df)} runs to {output_file}")

    return runs_df


def analyze_top_models(runs_df: pd.DataFrame, output_dir: str = "."):
    """
    Analyze top 5 models per experiment and generate LaTeX tables.

    Args:
        runs_df: DataFrame with runs data (from fetch_wandb_runs)
        output_dir: Directory to save LaTeX table files

    Returns:
        DataFrame with all model statistics for visualization
    """
    print("\n=== Analyzing Top Models ===")

    # Parse config and summary from string/dict format
    if isinstance(runs_df['config'].iloc[0], str):
        runs_df['config'] = runs_df['config'].apply(ast.literal_eval)
    if isinstance(runs_df['summary'].iloc[0], str):
        runs_df['summary'] = runs_df['summary'].apply(ast.literal_eval)

    # Extract relevant fields
    records = []

    for idx, row in runs_df.iterrows():
        config = row['config']
        summary = row['summary']

        # Extract key fields (use lowercase keys as they appear in wandb)
        record = {
            'model': config.get('model'),
            'res': config.get('res'),
            'n_out': config.get('n_out'),
            'feature_code': config.get('feature_code'),
            'Test_RMSE': summary.get('test_rmse'),  # lowercase in wandb
            'Test_MAE': summary.get('test_mae'),  # lowercase in wandb
        }

        # Only include if we have the essential fields
        if record['model'] and record['res'] and record['n_out'] is not None and record['Test_RMSE'] is not None and \
                record['Test_MAE'] is not None and record["feature_code"] is not None:
            records.append(record)

    df = pd.DataFrame(records)
    print(f"Found {len(df)} runs with complete data")

    # Group by experiment (res, n_out) and model
    experiments = df.groupby(['res', 'n_out'])

    all_stats = []

    for (res, n_out), exp_data in experiments:
        print(f"\nProcessing experiment: {res}, n_out={n_out}")

        # Store results for this experiment
        model_stats = []

        # Process each model type
        for model_type in df['model'].unique():
            model_data = exp_data[exp_data['model'] == model_type]

            if len(model_data) == 0:
                continue

            # Select top 5 by Test_RMSE
            top_5 = model_data.nsmallest(5, 'Test_RMSE')

            if len(top_5) == 0:
                continue

            # Calculate statistics
            rmse_values = top_5['Test_RMSE'].values
            mae_values = top_5['Test_MAE'].values
            n_runs = len(top_5)  # Number of runs used for average

            # Calculate mean, std, and 95% confidence interval
            rmse_mean = np.mean(rmse_values)
            mae_mean = np.mean(mae_values)
            rmse_std = np.std(rmse_values, ddof=1) if len(rmse_values) > 1 else 0.0
            mae_std = np.std(mae_values, ddof=1) if len(mae_values) > 1 else 0.0

            # Calculate confidence intervals
            if len(rmse_values) > 1:
                rmse_ci = stats.t.interval(0.95, len(rmse_values) - 1,
                                           loc=rmse_mean,
                                           scale=stats.sem(rmse_values))
                mae_ci = stats.t.interval(0.95, len(mae_values) - 1,
                                          loc=mae_mean,
                                          scale=stats.sem(mae_values))
            else:
                rmse_ci = (rmse_mean, rmse_mean)
                mae_ci = (mae_mean, mae_mean)

            model_stats.append({
                'model': model_type,
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std,
                'rmse_ci_lower': rmse_ci[0],
                'rmse_ci_upper': rmse_ci[1],
                'mae_mean': mae_mean,
                'mae_std': mae_std,
                'mae_ci_lower': mae_ci[0],
                'mae_ci_upper': mae_ci[1],
                'n_runs': n_runs,
                'res': res,
                'n_out': n_out
            })

        if not model_stats:
            print(f"  No models found for {res}, n_out={n_out}")
            continue

        # Create LaTeX table
        stats_df = pd.DataFrame(model_stats)
        stats_df = stats_df.sort_values('rmse_mean')

        # Add to all stats for visualization
        all_stats.extend(model_stats)

        latex_table = generate_latex_table(stats_df, res, n_out)

        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/{res}_{n_out}.tex"
        with open(filename, 'w') as f:
            f.write(latex_table)

        print(f"  Saved LaTeX table to {filename}")
        print(f"  Models: {len(stats_df)}")

    return pd.DataFrame(all_stats)


def generate_latex_table(stats_df: pd.DataFrame, res: str, n_out: int) -> str:
    """
    Generate LaTeX table from statistics DataFrame.

    Args:
        stats_df: DataFrame with model statistics
        res: Resolution (daily/hourly)
        n_out: Number of prediction steps

    Returns:
        LaTeX table as string
    """
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Top 5 Model Performance for {res.capitalize()} Forecasting with n\\_out={n_out}}}")
    latex.append(f"\\label{{tab:{res}_{n_out}}}")
    latex.append("\\begin{tabular}{lccccc}")
    latex.append("\\toprule")
    latex.append("Model & RMSE & Std & MAE & Std & \\# Runs \\\\")
    latex.append("\\midrule")

    for _, row in stats_df.iterrows():
        model = row['model']
        rmse = f"{row['rmse_mean']:.4f}"
        rmse_std = f"{row['rmse_std']:.4f}"
        rmse_ci = f"[{row['rmse_ci_lower']:.4f}, {row['rmse_ci_upper']:.4f}]"
        mae = f"{row['mae_mean']:.4f}"
        mae_std = f"{row['mae_std']:.4f}"
        mae_ci = f"[{row['mae_ci_lower']:.4f}, {row['mae_ci_upper']:.4f}]"
        n_runs = int(row['n_runs'])

        latex.append(f"{model} & {rmse} {rmse_ci} & {rmse_std} & {mae} {mae_ci} & {mae_std} & {n_runs} \\\\")

    latex.append("\\midrule")
    latex.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Note: Values shown as mean [95\\% CI]}}}} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def visualize_model_performance(stats_df: pd.DataFrame, output_dir: str = "."):
    """
    Create visualizations of RMSE and MAE for each experiment with confidence intervals.

    Args:
        stats_df: DataFrame with model statistics from analyze_top_models
        output_dir: Directory to save visualization files
    """
    print("\n=== Creating Visualizations ===")
    os.makedirs(output_dir, exist_ok=True)

    # Group by experiment (res, n_out)
    experiments = stats_df.groupby(['res', 'n_out'])

    for (res, n_out), exp_data in experiments:
        print(f"Creating plots for {res}, n_out={n_out}")

        # Sort by RMSE for consistent ordering
        exp_data = exp_data.sort_values('rmse_mean')

        models = exp_data['model'].values
        x_pos = np.arange(len(models))

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot RMSE
        rmse_means = exp_data['rmse_mean'].values
        rmse_ci_lower = exp_data['rmse_ci_lower'].values
        rmse_ci_upper = exp_data['rmse_ci_upper'].values
        rmse_errors = np.array([rmse_means - rmse_ci_lower, rmse_ci_upper - rmse_means])

        ax1.bar(x_pos, rmse_means, color='steelblue', alpha=0.7, label='RMSE')
        ax1.errorbar(x_pos, rmse_means, yerr=rmse_errors, fmt='none',
                     ecolor='black', capsize=5, capthick=2, label='95% CI')
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('RMSE', fontsize=12)
        ax1.set_title(f'RMSE for {res.capitalize()} Forecasting (n_out={n_out})', fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Plot MAE
        mae_means = exp_data['mae_mean'].values
        mae_ci_lower = exp_data['mae_ci_lower'].values
        mae_ci_upper = exp_data['mae_ci_upper'].values
        mae_errors = np.array([mae_means - mae_ci_lower, mae_ci_upper - mae_means])

        ax2.bar(x_pos, mae_means, color='coral', alpha=0.7, label='MAE')
        ax2.errorbar(x_pos, mae_means, yerr=mae_errors, fmt='none',
                     ecolor='black', capsize=5, capthick=2, label='95% CI')
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.set_title(f'MAE for {res.capitalize()} Forecasting (n_out={n_out})', fontsize=14)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save figure
        filename = f"{output_dir}/{res}_{n_out}_metrics.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved visualization to {filename}")


def analyze_best_model_runtimes(runs_df: pd.DataFrame, output_dir: str = "./reports/experiment_runtimes", average_power_draw_cpu: float = 50.0, common_grid_factor: float = 187.08):
    """
    Analyze runtime and hardware information for the best performing model in each experiment.

    Args:
        runs_df: DataFrame with runs data (from fetch_wandb_runs)
        output_dir: Directory to save runtime CSV files
        average_power_draw_cpu: Average power draw for CPU training in watts (default: 50W)
        common_grid_factor: Grid carbon intensity in gCO2e/kWh (default: 187.08)
    """
    print("\n=== Analyzing Best Model Runtimes ===")

    # Parse config, summary, metadata, and system_metrics from string/dict format
    if isinstance(runs_df['config'].iloc[0], str):
        runs_df['config'] = runs_df['config'].apply(ast.literal_eval)
    if isinstance(runs_df['summary'].iloc[0], str):
        runs_df['summary'] = runs_df['summary'].apply(ast.literal_eval)
    if 'metadata' in runs_df.columns and isinstance(runs_df['metadata'].iloc[0], str):
        runs_df['metadata'] = runs_df['metadata'].apply(ast.literal_eval)
    if 'system_metrics' in runs_df.columns and isinstance(runs_df['system_metrics'].iloc[0], str):
        runs_df['system_metrics'] = runs_df['system_metrics'].apply(ast.literal_eval)

    # Extract relevant fields
    records = []

    for idx, row in runs_df.iterrows():
        config = row['config']
        summary = row['summary']
        name = row['name']
        metadata = row.get('metadata', {}) if 'metadata' in row else {}
        system_metrics = row.get('system_metrics', {}) if 'system_metrics' in row else {}

        # Extract runtime - check various possible locations
        runtime = None
        if '_runtime' in summary:
            runtime = summary['_runtime']
        elif '_wandb' in summary and isinstance(summary['_wandb'], dict):
            if 'runtime' in summary['_wandb']:
                runtime = summary['_wandb']['runtime']

        # Extract hardware info from metadata (preferred) or fallback to config/summary
        if isinstance(metadata, dict) and metadata:
            # Metadata has the most accurate hardware info
            gpu_type = metadata.get('gpu', 'Unknown')
            gpu_count = metadata.get('gpu_count', 0)
            cpu_count = metadata.get('cpu_count', metadata.get('cpu_count_logical', 'Unknown'))

            # Get device info from config if available
            device = config.get('device', 'cuda' if gpu_count > 0 else 'cpu')
        else:
            # Fallback to config/summary if metadata not available
            gpu_type = config.get('gpu_type') or config.get('device_name') or summary.get('gpu_type', 'Unknown')
            gpu_count = config.get('gpu_count', 0)
            cpu_count = config.get('cpu_count') or config.get('num_cpus', 'Unknown')
            device = config.get('device', 'Unknown')

        # Determine if CPU or GPU
        hardware_type = 'GPU' if (device and 'cuda' in str(device).lower()) or gpu_count > 0 else 'CPU'

        # Extract GPU power usage from system_metrics
        gpu_power_watts = None
        if isinstance(system_metrics, dict) and system_metrics:
            # Get power for all GPUs and calculate average
            gpu_powers = []
            for i in range(gpu_count):
                power_key = f'system.gpu.{i}.powerWatts'
                if power_key in system_metrics:
                    gpu_powers.append(system_metrics[power_key])

            if gpu_powers:
                gpu_power_watts = sum(gpu_powers) / len(gpu_powers)

        # Calculate average power draw
        if hardware_type == 'GPU' and gpu_power_watts is not None:
            average_power_draw = gpu_power_watts
        elif hardware_type == 'GPU' and gpu_power_watts is None:
            # GPU training but no power data available
            average_power_draw = None
        else:
            # CPU training - check if logged value exists in summary
            if 'average_power_consumption_cpu' in summary:
                average_power_draw = summary['average_power_consumption_cpu']
            else:
                average_power_draw = average_power_draw_cpu

        # Calculate total energy consumption in kWh
        total_power_draw_kwh = None

        # For CPU training, check if logged value exists in summary
        if hardware_type == 'CPU' and 'total_power_consumption_kwh' in summary:
            total_power_draw_kwh = summary['total_power_consumption_kwh']
        elif runtime is not None and average_power_draw is not None:
            runtime_hours = runtime / 3600  # Convert seconds to hours
            total_power_draw_kwh = (runtime_hours * average_power_draw) / 1000  # Convert Wh to kWh

        # Calculate CO2 emissions in gCO2e
        co2_emission = None
        if total_power_draw_kwh is not None:
            co2_emission = total_power_draw_kwh * common_grid_factor  # gCO2e

        # Extract other fields
        record = {
            'name': name,
            'model': config.get('model'),
            'res': config.get('res'),
            'n_out': config.get('n_out'),
            'test_rmse': summary.get('test_rmse'),
            'test_mae': summary.get('test_mae'),
            'runtime_seconds': runtime,
            'hardware_type': hardware_type,
            'cpu_count': cpu_count,
            'gpu_count': gpu_count,
            'gpu_type': gpu_type,
            'device': device,
            'epochs': config.get('epochs') or config.get('n_epochs') or config.get('max_epochs', 'Unknown'),
            'average_power_draw': average_power_draw,
            'total_power_draw_kwh': total_power_draw_kwh,
            'co2_emission_gco2e': co2_emission
        }

        # Only include if we have essential fields
        if record['model'] and record['res'] and record['n_out'] is not None and record['test_rmse'] is not None:
            records.append(record)

    df = pd.DataFrame(records)
    print(f"Found {len(df)} runs with complete data")

    if len(df) == 0:
        print("No runs found with complete data. Cannot proceed.")
        return

    # Group by experiment (res, n_out) and find best model per model type
    experiments = df.groupby(['res', 'n_out'])

    os.makedirs(output_dir, exist_ok=True)

    for (res, n_out), exp_data in experiments:
        print(f"\nProcessing experiment: {res}, n_out={n_out}")

        # Find best model within each model type
        best_models_list = []

        for model_type in exp_data['model'].unique():
            model_data = exp_data[exp_data['model'] == model_type]

            if len(model_data) == 0:
                continue

            # Get the best model (lowest RMSE) for this model type
            best_model = model_data.nsmallest(1, 'test_rmse')
            best_models_list.append(best_model)

            print(f"  Best {model_type}: RMSE = {best_model['test_rmse'].iloc[0]:.4f}")

        if len(best_models_list) == 0:
            print(f"  No best models found for {res}, n_out={n_out}")
            continue

        # Combine all best models
        best_models = pd.concat(best_models_list, ignore_index=True)

        # Select columns for output
        output_cols = [
            'name', 'model', 'test_rmse', 'test_mae', 'runtime_seconds',
            'hardware_type', 'cpu_count', 'gpu_count', 'gpu_type', 'device', 'epochs',
            'average_power_draw', 'total_power_draw_kwh', 'co2_emission_gco2e'
        ]

        best_models_output = best_models[output_cols].copy()

        # Convert runtime to hours:minutes:seconds format if available
        runtime_formatted = []
        for idx, row in best_models_output.iterrows():
            if row['runtime_seconds'] is not None:
                runtime_sec = row['runtime_seconds']
                hours = int(runtime_sec // 3600)
                minutes = int((runtime_sec % 3600) // 60)
                seconds = int(runtime_sec % 60)
                runtime_formatted.append(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            else:
                runtime_formatted.append(None)

        best_models_output['runtime_formatted'] = runtime_formatted

        # Sort by RMSE for better readability
        best_models_output = best_models_output.sort_values('test_rmse')

        # Save to CSV
        filename = f"{output_dir}/runtimes_{res}_{n_out}.csv"
        best_models_output.to_csv(filename, index=False)

        print(f"  Saved runtime info to {filename}")
        print(f"  Total best models saved: {len(best_models_output)}")


if __name__ == "__main__":
    # Example 1: Fetch all runs without filters
    # df = fetch_wandb_runs(
    #     entity="rausch-technology",
    #     project="ma-wahl-forecast",
    #     output_file="runs.csv"
    # )

    feature_code = "all"

    # Example 2: Fetch runs with filters
    filters = {
        "energy": "all",
        # Include both daily and hourly for analysis
        "model": ["lstm", "FCN3", "transformer", "xlstm"],  # model must be lstm OR FCN3 or transformer or xlstm
        "dataset": "building",
        "feature_code": [12, 15, 16],  # all possible feature combinations
        "interpolate": 1,
        "remove_per": ("not_in", [0.2, 0.5])  # remove_per must NOT be 0.2 or 0.5
    }

    df = fetch_wandb_runs(
        entity="rausch-technology",
        project="ma-wahl-forecast",
        output_file=f"runs_feature_code_{feature_code}.csv",
        filters=filters
    )

    df = pd.read_csv(f"runs_feature_code_{feature_code}.csv")

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # # Analyze top models and generate LaTeX tables
    # stats_df = analyze_top_models(df, output_dir=f"./reports/top_k_model_feature_code_{feature_code}_stats")
    #
    # # Create visualizations
    # visualize_model_performance(stats_df, output_dir=f"./reports/top_k_model_feature_code_{feature_code}_stats")

    # Analyze runtime information for best models
    analyze_best_model_runtimes(df, output_dir="./reports/experiment_runtimes")