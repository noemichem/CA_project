import os
import sys
import pandas as pd

def average_execution_time(csv_path):
    """
    Compute the average execution time for CPU or GPU results and save to CSV.
    CPU: group by Num Threads, Input File, Executable
    GPU: group by Threads per Block, Input File, Executable
    Missing thread counts are treated as a separate group (-1).
    """
    df = pd.read_csv(csv_path)

    # Strip whitespace from string columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Detect table type
    if "Num Threads" in df.columns:
        mode = "CPU"
        x_col = "Num Threads"
        output_file = "CPU_means.csv"
    elif "Threads per Block" in df.columns:
        mode = "GPU"
        x_col = "Threads per Block"
        output_file = "GPU_means.csv"
    else:
        raise ValueError("CSV must contain either 'Num Threads' (CPU) or 'Threads per Block' (GPU).")

    # Fill missing thread counts with -1 and convert to integer
    df[x_col] = df[x_col].fillna(-1).astype(int).astype(str)

    # Group by relevant columns and compute mean
    group_cols = [x_col, "Input File", "Executable"]
    grouped = df.groupby(group_cols, as_index=False)["Execution Time (ms)"].mean()

    # Rename column
    grouped = grouped.rename(columns={"Execution Time (ms)": "Avg Execution Time (ms)"})  # type: ignore

    # Ensure thread column is integer (for safety)
    grouped[x_col] = grouped[x_col].astype(int)

    # Save to CSV
    output_path = os.path.join(os.path.dirname(csv_path), output_file)
    grouped.to_csv(output_path, index=False)
    print(f"{mode} averages saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python means.py <input_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    if not os.path.isfile(input_csv):
        print(f"Error: file '{input_csv}' not found.")
        sys.exit(1)

    average_execution_time(input_csv)
