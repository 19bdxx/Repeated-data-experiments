import csv
from pathlib import Path

def analyze_csv(file_path: str | Path) -> None:
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Get the header row
        rows = list(reader)  # Read the rest of the rows
        num_rows = len(rows)
        num_cols = len(header)
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_cols}")
        print(f"Column names: {header}")

if __name__ == "__main__":
    file_path = r"场站数据\LGXRFD\step6-风机宽表修正结果-仅正功率\LGXRFD_202309-202407_fan_wide_correction_threshold_0p15.csv"  # Replace with your CSV file path
    file_path = r"场站数据\LGXRFD\LGXRFD_202309-202407.csv"  # Replace with your CSV file path

    analyze_csv(file_path)