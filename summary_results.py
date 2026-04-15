from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_resize_images(results_folder,folders, image_names, image_size, fold_num):
    """Load and resize images or insert a placeholder if missing."""
    all_images = []
    for folder in folders:
        row_images = []
        for img_name in image_names:
            img_path = os.path.join(results_folder,folder, fold_num, img_name)
            if os.path.exists(img_path):
                img = Image.open(img_path).resize(image_size)
            else:
                img = Image.new('RGB', image_size, color=(255, 255, 255))  # white placeholder
            row_images.append(img)
        all_images.append(row_images)
    return all_images


def create_image_grid(all_images, image_size):
    """Create a single grid image from rows of images."""
    rows = len(all_images)
    cols = len(all_images[0]) if rows > 0 else 0
    grid_width = cols * image_size[0]
    grid_height = rows * image_size[1]

    grid_img = Image.new('RGB', (grid_width, grid_height))
    for row_idx, row_images in enumerate(all_images):
        for col_idx, img in enumerate(row_images):
            position = (col_idx * image_size[0], row_idx * image_size[1])
            grid_img.paste(img, position)

    return grid_img


def save_grid_image(grid_img, results_folder, filename, use_summary_folder = True):
    """Save the final grid image to disk."""
    save_dir = os.path.join(results_folder, 'summary_results') if use_summary_folder else results_folder
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    grid_img.save(save_path)
   # print(f"Grid image saved as '{save_path}'")


def collect_metrics_from_folds(results_folder,base_dir, filename="result_train.csv", num_folds=5):
    all_metrics = []

    for i in range(0, num_folds):
        file_path = os.path.join(results_folder, base_dir, f"fold_{i}", filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                line = f.readline().strip()
                metrics = list(map(float, line.split(',')))
                all_metrics.append(metrics)
        else:
            print(f"File not found: {file_path}")

    return all_metrics


def summarize_metrics(base_dir, file_label_pairs):
    shown_metric_names = [ 'mse','rmse', 'pearson']  # only show these

    for filename, label in file_label_pairs:
        all_metrics = collect_metrics_from_folds(base_dir, filename=filename)
        data = np.array(all_metrics)

        if data.size == 0:
            print(f"No data for {label}.")
            continue

        means = np.mean(data, axis=0)[:3]  # take only rmse, mse, pearson
        stds = np.std(data, axis=0, ddof=1)[:3]

        output = [f"{mean:.4f} ({std:.4f})" for mean, std in zip(means, stds)]

        print(f"\n{label} results:")
        print("  " + " &  ".join(shown_metric_names))
        print("  " + " &  ".join(output))


def summarize_all_metrics(results_folder,folders, file_label_pairs, num_folds=5, round_decimals=4):
    shown_metric_names = ['mse', 'rmse', 'pearson']

    summary_data = []  # for Table 1f
    raw_data = []      # for Table 2

    for folder in folders:
      #  model_num = folder.split('_')[1]
        model_name = f'model_{folder}'

        for filename, label in file_label_pairs:
            metrics = collect_metrics_from_folds(results_folder,folder, filename, num_folds=num_folds)
            if not metrics:
                continue

            metrics_arr = np.array(metrics)
            means = np.mean(metrics_arr, axis=0)[:3]
            stds = np.std(metrics_arr, axis=0, ddof=1)[:3]

            # Table 1: Summary per model and data split
            summary_data.append({
                "Model": folder,
                "Split": label,
                **{f"{metric}_mean": round(mean, round_decimals)for metric, mean in zip(shown_metric_names, means)},
                **{f"{metric}_std": round(std, round_decimals) for metric, std in zip(shown_metric_names, stds)},
            })

            # Table 2: Raw metrics per fold
            for fold_idx, row in enumerate(metrics_arr):
                raw_data.append({
                    "Model": folder,
                    "Fold": fold_idx,
                    "Split": label,
                    **{metric: round(row[i], round_decimals) for i, metric in enumerate(shown_metric_names)},
                })

    summary_df = pd.DataFrame(summary_data)
    raw_df = pd.DataFrame(raw_data)

    return summary_df, raw_df


def plot_summary_table(summary_df):
    print("\nSummary Table (Mean ± Std):\n")
    print(summary_df.to_string(index=False))


def plot_raw_table(raw_df):
    print("\nRaw Metrics Table (Per Fold):\n")
    print(raw_df.to_string(index=False))


def plot_bar_chart_summary(summary_df, output_dir="summary_results"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['mse', 'rmse', 'pearson']
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        for model_name, group in summary_df.groupby("Model"):
            x = group["Split"]
            y = group[f"{metric}_mean"]
            yerr = group[f"{metric}_std"]
            ax.errorbar(x, y, yerr=yerr, label=model_name, capsize=5, marker='o', linestyle='-')

        ax.set_title(f"{metric.upper()} by Split")
        ax.set_ylabel(metric.upper())
        ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{metric}_summary_barplot.png")
        plt.savefig(plot_path)
        plt.close()
       # print(f"Saved: {plot_path}")


def plot_raw_boxplots(raw_df, output_dir="summary_results"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['mse', 'rmse', 'pearson']
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        raw_df.boxplot(column=metric, by=["Model", "Split"], ax=ax, grid=False)
        plt.title(f"{metric.upper()} Distribution by Model and Split")
        plt.suptitle("")
        plt.xlabel("Model - Split")
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{metric}_raw_boxplot.png")
        plt.savefig(plot_path)
        plt.close()
       # print(f"Saved: {plot_path}")


def save_df_as_table_image(df, output_dir, output_path, title=None, font_size=10, use_summary_folder = True ):
    """
    Save a pandas DataFrame as a styled table image.

    Args:
        df (pd.DataFrame): The DataFrame to render.
        output_path (str): Full path to save the PNG image.
        title (str): Optional title for the table.
        font_size (int): Font size used in the table.
    """
    # Setup figure size dynamically
    n_rows, n_cols = df.shape
    fig_height = max(1.2, 0.4 * n_rows)
    fig, ax = plt.subplots(figsize=(min(18, 0.8 * n_cols + 2), fig_height))

    ax.axis('off')
    if title:
        plt.title(title, fontsize=font_size + 2, weight='bold')

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colLoc='center'
    )

    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.2, 1.2)

    # Alternate row background colors
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            cell = table[(row_idx + 1, col_idx)]  # +1 because row 0 is the header
            if row_idx % 2 == 1:
                cell.set_facecolor('#f0f0f0')  # light gray
            else:
                cell.set_facecolor('white')  # white
    # Save as image

    save_dir = os.path.join(output_dir, 'summary_results') if use_summary_folder else output_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,output_path)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
   # print(f"Saved table image to: {output_path}")


def save_df_per_model_as_table_images(df, output_dir, output_path, title_prefix=None, font_size=10, use_summary_folder=True):
    """Save one PNG table image per model from a DataFrame that contains a 'Model' column."""
    if 'Model' not in df.columns:
        raise ValueError("The DataFrame must contain a 'Model' column.")

    save_dir = os.path.join(output_dir, 'summary_results') if use_summary_folder else output_dir
    os.makedirs(save_dir, exist_ok=True)

    for model_name, group_df in df.groupby('Model'):
        # Drop the Model column for clarity in the image
        display_df = group_df.drop(columns=['Model'])

        # Setup figure size dynamically
        n_rows, n_cols = display_df.shape
        fig_height = max(1.2, 0.4 * n_rows)
        fig, ax = plt.subplots(figsize=(min(18, 0.8 * n_cols + 2), fig_height))
        ax.axis('off')

        title = f"{title_prefix} - {model_name}" if title_prefix else model_name
        plt.title(title, fontsize=font_size + 2, weight='bold')

        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            loc='center',
            cellLoc='center',
            colLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1.2, 1.2)

        # Alternate row background colors
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                cell = table[(row_idx + 1, col_idx)]  # +1 because row 0 is the header
                if row_idx % 2 == 1:
                    cell.set_facecolor('#f0f0f0')  # light gray
                else:
                    cell.set_facecolor('white')  # white


        # Define image filename
        safe_model_name = model_name.replace(" ", "_")
        save_path = os.path.join(save_dir, f"{output_path}_{safe_model_name}.png")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

       # print(f"Saved table image for model '{model_name}' to: {output_path}")


def summary_results_main(results_folder):

    #folders = [ "ligand","ligand_3D", "ligand_BD","ligand_3D_BD", "ligand_prot_seq","ligand_prot_seq_3D",
    #            "ligand_prot_seq_BD","ligand_prot_seq_3D_BD"   ]
    #folders = ["ligand", "ligand_3D", "ligand_BD", "ligand_3D_BD", "ligand_prot_seq", "ligand_prot_seq_3D",
    #           "ligand_prot_seq_BD", "ligand_prot_seq_3D_BD", "pocket","pocket_3D","pocket_BD","pocket_3D_BD","pocket_prot_seq",
    #           "pocket_prot_seq_3D","pocket_prot_seq_BD","pocket_prot_seq_3D_BD"]
    folders = ["ligand","ligand_3D", "ligand_BD","ligand_3D_BD","pocket", "pocket_3D", "pocket_BD", "pocket_3D_BD"]
    true_vs_pred = ['true_vs_pred_train.png', 'true_vs_pred_validation.png', 'true_vs_pred_test.png']
    performance = ['pearson.png', 'loss_training.png', 'loss_validation.png']
    image_names = [true_vs_pred, performance]
    output_names = ['true_vs_pred', 'performance']
    size = [(256, 256), (512, 256)]

    file_label_pairs = [
        ("result_train.csv", "Train"),
        ("result_validation.csv", "Validation"),
        ("result_test.csv", "Test")
    ]

    for f in range(5):
        for image_name, output_name, image_size in zip(image_names, output_names, size):
            fold_num = f'fold_{f}'
            all_images = load_and_resize_images(results_folder,folders, image_name, image_size,fold_num)
            grid_img = create_image_grid(all_images, image_size)
            filename = f'{output_name}_{fold_num}.png'
            save_grid_image(grid_img,results_folder, filename)

    summary_df, raw_df = summarize_all_metrics(results_folder,folders, file_label_pairs)
   # plot_summary_table(summary_df)
   # plot_raw_table(raw_df)
    cols = list(summary_df.columns)
    col_to_move = cols.pop(5)
    cols.insert(3, col_to_move)
    col_to_move = cols.pop(6)
    cols.insert(5, col_to_move)
    summary_df_reordered = summary_df[cols]
    print(summary_df_reordered[['Model', 'Split', 'rmse_mean', 'rmse_std','pearson_mean', 'pearson_std']])
    save_df_as_table_image(summary_df_reordered, results_folder, "summary_table.png", title="Summary Metrics")
    save_df_per_model_as_table_images(raw_df, results_folder, "raw_metrics_model","Raw Metrics (Per Fold) Model")
 #   plot_bar_chart_summary(summary_df)
 #   plot_raw_boxplots(raw_df)


if __name__ == "__main__":
    results_folder = r"C:\Users\natal\OneDrive\code_binding\results_main\363_Mpro_Juliol_2025"
    summary_results_main(results_folder)
    results_folder = r"C:\Users\natal\OneDrive\code_binding\results_main\363_Mpro_Juliol_2025\results_22_07_2025\results_22_07_2025"
    summary_results_main(results_folder)
