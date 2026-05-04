import os
import pandas as pd


def load_station_mapping(point_file):
    """
    读取场站点位表，返回：
    - station_code: 点位表文件名里的场站代号，如 GRJHFD
    - station_name: 场站列里的名称
    - point_ids: 该场站所有点位
    - rename_dict: 点位 -> 英文列名
    """
    point_df = pd.read_excel(point_file)

    required_cols = {'点位', '场站', '遥测点-英文'}
    missing_cols = required_cols - set(point_df.columns)
    if missing_cols:
        raise ValueError(f"点位表缺少必要列: {missing_cols}")

    point_df = point_df[['点位', '场站', '遥测点-英文']].drop_duplicates()

    station_code = os.path.basename(point_file).replace('_点位.xlsx', '').strip()
    station_name = str(point_df['场站'].iloc[0]).strip()

    # 点位统一转字符串，避免和 CSV 列名类型不一致
    point_df['点位'] = point_df['点位'].astype(str)

    point_ids = point_df['点位'].tolist()
    rename_dict = dict(zip(point_df['点位'], point_df['遥测点-英文']))

    return station_code, station_name, point_ids, rename_dict


def extract_station_from_monthly_csv(monthly_csv, point_ids, rename_dict):
    """
    从单个月度总表中提取指定场站的数据。
    """
    df = pd.read_csv(monthly_csv)

    if 'timestamp' not in df.columns:
        raise ValueError(f"{monthly_csv} 中没有 timestamp 列")

    # 统一列名为字符串，避免点位列 int/str 不一致
    df.columns = df.columns.map(str)

    available_point_ids = [pid for pid in point_ids if pid in df.columns]
    missing_point_ids = [pid for pid in point_ids if pid not in df.columns]

    if missing_point_ids:
        print(
            f"[提示] {os.path.basename(monthly_csv)} 中缺少 "
            f"{len(missing_point_ids)} 个点位列，示例: {missing_point_ids[:10]}"
        )

    if not available_point_ids:
        return pd.DataFrame(columns=['timestamp'])

    selected_cols = ['timestamp'] + available_point_ids
    sub_df = df[selected_cols].copy()

    # 重命名点位列为英文名
    sub_df.rename(columns=rename_dict, inplace=True)

    return sub_df


def adjust_timestamp(ts):
    """
    时间戳处理规则：
    - 秒数 = 0：不变
    - 0 < 秒数 <= 30：秒数置0，分钟不变
    - 秒数 > 30：分钟 +1，秒数置0
    """
    if pd.isna(ts):
        return pd.NaT

    if ts.second == 0:
        return ts.replace(second=0, microsecond=0)

    if ts.second > 30:
        return (ts + pd.Timedelta(minutes=1)).replace(second=0, microsecond=0)
    else:
        return ts.replace(second=0, microsecond=0)


def clean_timestamp_and_nulls(df):
    """
    统一处理时间戳并删除空值行：
    1. timestamp 转 datetime，无法解析的行删除
    2. 按秒数规则修正 timestamp
    3. 删除包含任意空值的整行
    """
    cleaned_df = df.copy()

    # 1) 时间戳解析
    cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'], errors='coerce')
    invalid_timestamp_rows = int(cleaned_df['timestamp'].isna().sum())

    if invalid_timestamp_rows > 0:
        print(f"⚠️ 删除 {invalid_timestamp_rows} 行无法解析的时间戳。")

    cleaned_df = cleaned_df.dropna(subset=['timestamp']).copy()

    # 2) 修正秒数
    nonzero_seconds_rows = int((cleaned_df['timestamp'].dt.second != 0).sum())
    if nonzero_seconds_rows > 0:
        print(f"⚠️ 有 {nonzero_seconds_rows} 行时间戳秒数不为0，已按规则修正。")

    cleaned_df['timestamp'] = cleaned_df['timestamp'].apply(adjust_timestamp)

    # 3) 删除包含空值的整行
    null_row_count = int(cleaned_df.isnull().any(axis=1).sum())
    if null_row_count > 0:
        print(f"⚠️ 删除 {null_row_count} 行包含空值的记录。")

    cleaned_df = cleaned_df.dropna(axis=0, how='any').copy()

    # 排序
    cleaned_df = cleaned_df.sort_values('timestamp').reset_index(drop=True)

    return cleaned_df, invalid_timestamp_rows, nonzero_seconds_rows, null_row_count


def check_duplicate_timestamps(df, timestamp_column='timestamp'):
    """
    检查重复时间戳
    """
    temp_df = df.copy()
    temp_df[timestamp_column] = pd.to_datetime(temp_df[timestamp_column], errors='coerce')
    duplicated_df = temp_df[temp_df[timestamp_column].duplicated(keep=False)].copy()
    return duplicated_df


def find_missing_timestamps(df, timestamp_column='timestamp'):
    """
    检查按分钟粒度缺失的时间戳
    """
    if df.empty:
        return pd.DataFrame(columns=['Missing_Timestamps'])

    ts = pd.to_datetime(df[timestamp_column], errors='coerce').dropna()
    if ts.empty:
        return pd.DataFrame(columns=['Missing_Timestamps'])

    start_time = ts.min()
    end_time = ts.max()

    complete_range = pd.date_range(start=start_time, end=end_time, freq='min')
    existing_set = set(ts)
    missing_list = sorted(set(complete_range) - existing_set)

    return pd.DataFrame({'Missing_Timestamps': missing_list})


def merge_station_monthly_data(monthly_folder, point_file, output_root):
    """
    将某个场站在所有月份中的数据提取出来，清洗后保存为一个 CSV，
    并输出重复时间戳、缺失时间戳检查结果。
    """
    station_code, station_name, point_ids, rename_dict = load_station_mapping(point_file)

    csv_files = sorted([
        f for f in os.listdir(monthly_folder)
        if f.endswith('.csv')
    ])

    if not csv_files:
        raise ValueError("月度 CSV 文件夹中没有找到 CSV 文件")

    all_data = []
    month_tags = []

    for file_name in csv_files:
        file_path = os.path.join(monthly_folder, file_name)
        print(f"读取月度文件: {file_name}")

        try:
            sub_df = extract_station_from_monthly_csv(file_path, point_ids, rename_dict)
            if not sub_df.empty:
                all_data.append(sub_df)

                base = os.path.splitext(file_name)[0]
                month_str = base.replace('点位数据_', '').replace('-', '')
                month_tags.append(month_str)
        except Exception as e:
            print(f"[跳过] {file_name} 处理失败: {e}")

    if not all_data:
        print(f"{station_code} / {station_name} 没有提取到任何数据。")
        return

    merged_df = pd.concat(all_data, ignore_index=True)

    print(f"原始合并后行数: {len(merged_df)}")

    # 先做时间戳处理和空值删除
    cleaned_df, invalid_timestamp_rows, nonzero_seconds_rows, null_row_count = clean_timestamp_and_nulls(merged_df)

    print(f"清洗后行数: {len(cleaned_df)}")

    # 输出目录
    station_output_folder = os.path.join(output_root, station_code)
    os.makedirs(station_output_folder, exist_ok=True)

    start_month = min(month_tags) if month_tags else "unknown"
    end_month = max(month_tags) if month_tags else "unknown"

    # 保存主文件（保存清洗后的）
    main_csv = os.path.join(
        station_output_folder,
        f"{station_code}_{start_month}-{end_month}.csv"
    )
    cleaned_df.to_csv(main_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 主文件已保存: {main_csv}")

    # 重复时间戳检查（基于清洗后数据）
    duplicate_df = check_duplicate_timestamps(cleaned_df)
    duplicate_count = len(duplicate_df)

    if not duplicate_df.empty:
        duplicate_file = os.path.join(
            station_output_folder,
            f"{station_code}_{start_month}-{end_month}_duplicate_timestamps.csv"
        )
        duplicate_df.to_csv(duplicate_file, index=False, encoding='utf-8-sig')
        print(f"⚠️ 重复时间戳文件已保存: {duplicate_file}")
        print(f"⚠️ 共检测到 {duplicate_count} 行重复时间戳。")
    else:
        print("没有重复时间戳。")

    # 缺失时间戳检查（基于清洗后数据）
    missing_df = find_missing_timestamps(cleaned_df)
    missing_count = len(missing_df)

    if not missing_df.empty:
        missing_file = os.path.join(
            station_output_folder,
            f"{station_code}_{start_month}-{end_month}_missing_timestamps.csv"
        )
        missing_df.to_csv(missing_file, index=False, encoding='utf-8-sig')
        print(f"⚠️ 缺失时间戳文件已保存: {missing_file}")
        print(f"⚠️ 共检测到 {missing_count} 个缺失时间戳。")
    else:
        print("没有缺失时间戳。")

    # 汇总信息
    summary_df = pd.DataFrame([{
        'station_code': station_code,
        'station_name': station_name,
        'start_month': start_month,
        'end_month': end_month,
        'raw_rows': len(merged_df),
        'cleaned_rows': len(cleaned_df),
        'invalid_timestamp_rows_removed': invalid_timestamp_rows,
        'nonzero_seconds_rows_adjusted': nonzero_seconds_rows,
        'null_rows_removed': null_row_count,
        'duplicate_timestamp_rows': duplicate_count,
        'missing_timestamp_rows': missing_count
    }])

    summary_file = os.path.join(
        station_output_folder,
        f"{station_code}_{start_month}-{end_month}_process_summary.csv"
    )
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"📄 处理汇总已保存: {summary_file}")

    print(f"🎉 {station_code} / {station_name} 处理完成。")


if __name__ == "__main__":
    # 1) 月度总表所在文件夹
    monthly_folder = r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\江苏\CODE_整体提取点位数据\分月结果"

    # 2) 输出根目录
    output_root = r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\江苏\场站数据"

    # 3) 要批量处理的场站
    sites = ["LGXRFD", "LGXHFD", "GRJHFD", "JMZSFD", "SXFHFD"]

    for site in sites:
        point_file = rf"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\江苏\分场站点位表\{site}_点位.xlsx"

        if not os.path.exists(point_file):
            print(f"[跳过] 点位表不存在: {point_file}")
            continue

        print("=" * 80)
        print(f"开始处理场站: {site}")
        merge_station_monthly_data(monthly_folder, point_file, output_root)