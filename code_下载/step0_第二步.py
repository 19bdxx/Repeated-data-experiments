import os
import pandas as pd


def find_main_csv_files(output_root):
    """
    在“批量处理.py”的输出目录下，找到每个场站的主 CSV。
    排除各种检查结果文件和检测输出文件。
    """
    main_files = []

    skip_keywords = [
        "_duplicate_timestamps",
        "_missing_timestamps",
        "_process_summary",
        "_nulls",
        "_seconds_not_zero",
        "_invalid_timestamps",
        "_check_summary",
        "_timestamp_fixed",
        "_timestamp_changes"
    ]

    for station_folder in os.listdir(output_root):
        station_path = os.path.join(output_root, station_folder)
        if not os.path.isdir(station_path):
            continue

        for file_name in os.listdir(station_path):
            if not file_name.endswith(".csv"):
                continue

            if any(k in file_name for k in skip_keywords):
                continue

            main_files.append(os.path.join(station_path, file_name))

    return sorted(main_files)


def detect_column_repeats(df, columns_to_extract, save_dir, min_repeat=5, timestamp_col='timestamp'):
    """
    检测指定列是否存在连续相同值段。
    输出：
    - 连续相同检测/每列连续重复检测结果.csv
    - 连续相同检测/每列连续重复汇总.csv
    """
    os.makedirs(save_dir, exist_ok=True)

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)

    if timestamp_col not in df.columns:
        raise ValueError(f"未找到 {timestamp_col} 列，无法进行连续重复检查。")

    missing_cols = [c for c in columns_to_extract if c not in df.columns]
    if missing_cols:
        print(f"[提示] 以下列不存在，将跳过: {missing_cols}")

    cols_exist = [c for c in columns_to_extract if c in df.columns]
    if timestamp_col not in cols_exist:
        cols_exist = [timestamp_col] + cols_exist

    if len(cols_exist) <= 1:
        print("📭 没有可用于“每列连续重复检测”的目标列。")
        return

    df_extract = df[cols_exist].copy()
    df_extract[timestamp_col] = pd.to_datetime(df_extract[timestamp_col], errors='coerce')
    df_extract = df_extract.dropna(subset=[timestamp_col]).sort_values(by=timestamp_col).reset_index(drop=True)

    repeat_results = []
    target_cols = [c for c in cols_exist if c != timestamp_col]

    for col in target_cols:
        series = df_extract[col].copy()
        series = series.astype(object).where(pd.notna(series), "__MISSING__")

        same_as_prev = series.eq(series.shift(1))
        group_id = (~same_as_prev).cumsum()

        tmp = pd.DataFrame({
            timestamp_col: df_extract[timestamp_col],
            col: df_extract[col],
            "_group_id": group_id
        })

        for _, group in tmp.groupby("_group_id", sort=False):
            group_len = len(group)

            if group_len >= min_repeat:
                repeat_results.append({
                    "字段名": col,
                    "重复值": group[col].iloc[0],
                    "开始时间": group[timestamp_col].iloc[0],
                    "结束时间": group[timestamp_col].iloc[-1],
                    "持续长度": group_len
                })

    if repeat_results:
        df_repeat = pd.DataFrame(repeat_results)

        detail_path = os.path.join(save_dir, "每列连续重复检测结果.csv")
        df_repeat.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"✅ 每列连续重复明细已保存：{detail_path}")

        df_summary = df_repeat.groupby("字段名", as_index=False).agg(
            重复段数量=("字段名", "count"),
            连续重复总长度=("持续长度", "sum")
        )
        summary_path = os.path.join(save_dir, "每列连续重复汇总.csv")
        df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"✅ 每列连续重复汇总已保存：{summary_path}")
    else:
        print(f"📭 未发现持续长度 >= {min_repeat} 的连续重复段。")


def extract_fan_numbers_sorted_from_df(df, field_prefixes):
    """
    从 DataFrame 列名中自动提取风机编号。
    列名格式示例：STATUS_#1, ACTIVE_POWER_#12
    """
    fan_numbers = set()

    for prefix in field_prefixes:
        fan_columns = [col for col in df.columns if str(col).startswith(prefix)]
        for col in fan_columns:
            parts = str(col).split('#')
            if len(parts) > 1:
                try:
                    number = int(parts[1])
                    fan_numbers.add(number)
                except ValueError:
                    continue

    return sorted(fan_numbers)


def detect_joint_repeats(df, fan_numbers, field_prefixes, timestamp_col='timestamp', min_repeat=5):
    """
    检测每台风机多个字段联合连续相同的区段。
    """
    results = []

    for fan_num in fan_numbers:
        col_names = [f'{prefix}#{fan_num}' for prefix in field_prefixes]
        if not all(col in df.columns for col in col_names):
            continue

        current_joint_val = None
        count = 0
        start_idx = None

        for idx, row in df[col_names].iterrows():
            joint_val = tuple(row)

            if joint_val == current_joint_val:
                count += 1
            else:
                if count >= min_repeat:
                    results.append({
                        '风机编号': fan_num,
                        '重复值组合': current_joint_val,
                        '开始时间': df[timestamp_col].iloc[start_idx],
                        '结束时间': df[timestamp_col].iloc[idx - 1],
                        '持续长度': count
                    })
                current_joint_val = joint_val
                count = 1
                start_idx = idx

        if count >= min_repeat:
            results.append({
                '风机编号': fan_num,
                '重复值组合': current_joint_val,
                '开始时间': df[timestamp_col].iloc[start_idx],
                '结束时间': df[timestamp_col].iloc[len(df) - 1],
                '持续长度': count
            })

    return results


def save_joint_repeat_results(results, save_dir):
    """
    保存风机联合重复检测结果。
    输出：
    - 风机联合连续相同检测/联合重复值检测结果.xlsx
    - 风机联合连续相同检测/联合重复值总时长汇总.csv
    """
    os.makedirs(save_dir, exist_ok=True)

    if not results:
        print("📭 未检测到任何联合重复值段。")
        return

    df_results = pd.DataFrame(results)

    detail_excel = os.path.join(save_dir, "联合重复值检测结果.xlsx")
    with pd.ExcelWriter(detail_excel, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='联合重复检测结果', index=False)
    print(f"✅ 联合重复检测结果已保存：{detail_excel}")

    summary = df_results.groupby('风机编号')['持续长度'].sum().reset_index()
    summary = summary.rename(columns={'持续长度': '联合重复总长度'})

    summary_csv = os.path.join(save_dir, "联合重复值总时长汇总.csv")
    summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 联合重复时长汇总已保存：{summary_csv}")


def run_joint_repeat_check(df, save_dir, field_prefixes, timestamp_col='timestamp', min_repeat=5):
    """
    基于场站主 CSV 运行风机联合连续相同检测。
    """
    df = df.copy()
    if timestamp_col not in df.columns:
        raise ValueError(f"未找到 {timestamp_col} 列，无法进行风机联合重复检测。")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=[timestamp_col]).sort_values(by=timestamp_col).reset_index(drop=True)

    fan_numbers = extract_fan_numbers_sorted_from_df(df, field_prefixes)
    print(f"提取的风机编号列表：{fan_numbers}")

    if not fan_numbers:
        print("📭 未识别到任何风机编号，跳过风机联合连续相同检测。")
        return

    results = detect_joint_repeats(
        df=df,
        fan_numbers=fan_numbers,
        field_prefixes=field_prefixes,
        timestamp_col=timestamp_col,
        min_repeat=min_repeat
    )

    save_joint_repeat_results(results, save_dir)


def process_one_station_csv(csv_path, min_repeat=5):
    """
    对单个场站主 CSV 执行：
    1. 每列连续重复检测
    2. 风机联合连续重复检测
    """
    print("=" * 100)
    print(f"开始处理: {csv_path}")

    df = pd.read_csv(csv_path)
    folder = os.path.dirname(csv_path)

    # 1) 每列连续重复检测
    line_repeat_dir = os.path.join(folder, "连续相同检测")

    columns_to_extract = [
        "timestamp",
        "ACTIVE_POWER_STATION",
        "LIMIT_POWER"
    ]

    detect_column_repeats(
        df=df,
        columns_to_extract=columns_to_extract,
        save_dir=line_repeat_dir,
        min_repeat=min_repeat,
        timestamp_col='timestamp'
    )

    # 2) 风机联合连续重复检测
    joint_repeat_dir = os.path.join(folder, "风机联合连续相同检测")

    field_prefixes = [
        'STATUS_',
        'ACTIVE_POWER_',
        'REACTIVE_POWER_',
        'WINDSPEED_',
        'WINDDIRECTION_'
    ]

    run_joint_repeat_check(
        df=df,
        save_dir=joint_repeat_dir,
        field_prefixes=field_prefixes,
        timestamp_col='timestamp',
        min_repeat=min_repeat
    )

    print("🎉 检测完成。")


if __name__ == "__main__":
    # “批量处理.py”的输出根目录
    output_root = r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\江苏\场站数据"

    # 连续相同最少条数，达到这个值才记录
    min_repeat = 2

    main_csv_files = find_main_csv_files(output_root)

    if not main_csv_files:
        print("没有找到可处理的主 CSV 文件。")
    else:
        print(f"共找到 {len(main_csv_files)} 个主 CSV 文件，开始检测...")
        for csv_path in main_csv_files:
            process_one_station_csv(csv_path, min_repeat=min_repeat)