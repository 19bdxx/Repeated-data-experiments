#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step0_第二步_仅正功率版_简化.py

功能：
1. 保留每列连续重复检测：
   - 连续相同检测/每列连续重复检测结果.csv
   - 连续相同检测/每列连续重复汇总.csv

2. 风机联合连续重复检测只输出“仅正功率重复”结果：
   - 风机联合连续相同检测_仅正功率/联合重复值检测结果.xlsx
   - 风机联合连续相同检测_仅正功率/联合重复值总时长汇总.csv
   - 风机联合连续相同检测_仅正功率/仅正功率筛选统计.csv
   - 风机联合连续相同检测_仅正功率/仅正功率筛选统计_按风机.csv

不再输出：
   - 风机联合连续相同检测/联合重复值检测结果.xlsx
   - 风机联合连续相同检测/联合重复值总时长汇总.csv

筛选规则：
联合重复值组合格式预期为：
    (STATUS, ACTIVE_POWER, REACTIVE_POWER, WINDSPEED, WINDDIRECTION)
仅保留 ACTIVE_POWER > 0 的联合重复段。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_ROOT = r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\江苏\场站数据"
DEFAULT_MIN_REPEAT = 4

COLUMN_REPEAT_TARGETS = [
    "timestamp",
    "ACTIVE_POWER_STATION",
    "LIMIT_POWER",
]

JOINT_FIELD_PREFIXES = [
    "STATUS_",
    "ACTIVE_POWER_",
    "REACTIVE_POWER_",
    "WINDSPEED_",
    "WINDDIRECTION_",
]

COLUMN_REPEAT_DIR_NAME = "连续相同检测"
POSITIVE_JOINT_REPEAT_DIR_NAME = "风机联合连续相同检测_仅正功率"


def read_csv_auto(path: str | Path, **kwargs) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"文件不存在：{path_obj}")

    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path_obj, encoding=enc, **kwargs)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"读取失败：{path_obj}\n最后一次报错：{last_err}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    return out


def find_main_csv_files(output_root: str | Path) -> List[Path]:
    output_root = Path(output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"输出根目录不存在：{output_root}")

    skip_keywords = [
        "_duplicate_timestamps",
        "_missing_timestamps",
        "_process_summary",
        "_nulls",
        "_seconds_not_zero",
        "_invalid_timestamps",
        "_check_summary",
        "_timestamp_fixed",
        "_timestamp_changes",
        "threshold_scan_",
        "freeze_count_",
        "station_anomaly_",
        "repeat_power_",
        "hard_anomaly_",
        "fan_timestamp_",
        "fan_wide_",
    ]

    main_files: List[Path] = []
    for station_folder in sorted(output_root.iterdir()):
        if not station_folder.is_dir():
            continue

        for file_path in sorted(station_folder.glob("*.csv")):
            file_name = file_path.name
            if any(k in file_name for k in skip_keywords):
                continue
            main_files.append(file_path)

    return sorted(main_files)


def detect_column_repeats(
    df: pd.DataFrame,
    columns_to_extract: Iterable[str],
    save_dir: str | Path,
    min_repeat: int,
    timestamp_col: str = "timestamp",
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_columns(df)

    if timestamp_col not in df.columns:
        raise ValueError(f"未找到 {timestamp_col} 列，无法进行连续重复检查。")

    columns_to_extract = list(columns_to_extract)
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
    df_extract[timestamp_col] = pd.to_datetime(df_extract[timestamp_col], errors="coerce")
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
            "_group_id": group_id,
        })

        for _, group in tmp.groupby("_group_id", sort=False):
            group_len = len(group)
            if group_len >= min_repeat:
                repeat_results.append({
                    "字段名": col,
                    "重复值": group[col].iloc[0],
                    "开始时间": group[timestamp_col].iloc[0],
                    "结束时间": group[timestamp_col].iloc[-1],
                    "持续长度": int(group_len),
                })

    if repeat_results:
        df_repeat = pd.DataFrame(repeat_results)

        detail_path = save_dir / "每列连续重复检测结果.csv"
        df_repeat.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"✅ 每列连续重复明细已保存：{detail_path}")

        df_summary = df_repeat.groupby("字段名", as_index=False).agg(
            重复段数量=("字段名", "count"),
            连续重复总长度=("持续长度", "sum"),
        )
        summary_path = save_dir / "每列连续重复汇总.csv"
        df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"✅ 每列连续重复汇总已保存：{summary_path}")
    else:
        print(f"📭 未发现持续长度 >= {min_repeat} 的连续重复段。")


def extract_fan_numbers_sorted_from_df(df: pd.DataFrame, field_prefixes: Iterable[str]) -> List[int]:
    fan_numbers = set()
    columns = [str(c).strip() for c in df.columns]

    for prefix in field_prefixes:
        fan_columns = [col for col in columns if col.startswith(prefix)]
        for col in fan_columns:
            parts = str(col).split("#")
            if len(parts) > 1:
                try:
                    fan_numbers.add(int(parts[1]))
                except ValueError:
                    continue

    return sorted(fan_numbers)


def parse_joint_combo(combo) -> Tuple[float, float, float, float, float]:
    try:
        if isinstance(combo, tuple):
            vals = list(combo)
        else:
            vals = str(combo).strip().strip("()").split(",")
            vals = [v.strip() for v in vals]

        def to_float(x):
            if x is None or str(x).strip() == "":
                return np.nan
            return float(x)

        status = to_float(vals[0]) if len(vals) > 0 else np.nan
        active_power = to_float(vals[1]) if len(vals) > 1 else np.nan
        reactive_power = to_float(vals[2]) if len(vals) > 2 else np.nan
        windspeed = to_float(vals[3]) if len(vals) > 3 else np.nan
        winddirection = to_float(vals[4]) if len(vals) > 4 else np.nan
        return status, active_power, reactive_power, windspeed, winddirection
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def detect_positive_joint_repeats(
    df: pd.DataFrame,
    fan_numbers: Iterable[int],
    field_prefixes: Iterable[str],
    timestamp_col: str,
    min_repeat: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    检测所有联合重复段，但最终只返回 ACTIVE_POWER > 0 的结果。
    同时返回全量结果用于统计，不保存全量明细。
    """
    results = []
    field_prefixes = list(field_prefixes)

    for fan_num in fan_numbers:
        col_names = [f"{prefix}#{fan_num}" for prefix in field_prefixes]
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
                        "风机编号": fan_num,
                        "重复值组合": current_joint_val,
                        "开始时间": df[timestamp_col].iloc[start_idx],
                        "结束时间": df[timestamp_col].iloc[idx - 1],
                        "持续长度": int(count),
                    })
                current_joint_val = joint_val
                count = 1
                start_idx = idx

        if count >= min_repeat:
            results.append({
                "风机编号": fan_num,
                "重复值组合": current_joint_val,
                "开始时间": df[timestamp_col].iloc[start_idx],
                "结束时间": df[timestamp_col].iloc[len(df) - 1],
                "持续长度": int(count),
            })

    full_df = pd.DataFrame(results)
    full_df = add_parsed_combo_columns(full_df)

    if len(full_df) > 0:
        positive_df = full_df[pd.to_numeric(full_df["重复有功功率kW"], errors="coerce") > 0].copy()
    else:
        positive_df = pd.DataFrame(columns=full_df.columns)

    return full_df, positive_df


def add_parsed_combo_columns(df_results: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "风机编号", "重复值组合", "开始时间", "结束时间", "持续长度",
        "重复状态码", "重复有功功率kW", "重复无功功率", "重复风速m/s", "重复风向",
    ]

    if df_results is None or len(df_results) == 0:
        return pd.DataFrame(columns=cols)

    out = df_results.copy()
    parsed = out["重复值组合"].apply(parse_joint_combo)
    parsed_df = pd.DataFrame(
        parsed.tolist(),
        columns=["重复状态码", "重复有功功率kW", "重复无功功率", "重复风速m/s", "重复风向"],
        index=out.index,
    )
    out = pd.concat([out, parsed_df], axis=1)
    out["重复有功功率kW"] = pd.to_numeric(out["重复有功功率kW"], errors="coerce")
    return out


def save_positive_joint_results(positive_df: pd.DataFrame, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if positive_df is None or len(positive_df) == 0:
        print("📭 未检测到任何仅正功率联合重复结果。")
        positive_df = pd.DataFrame(columns=[
            "风机编号", "重复值组合", "开始时间", "结束时间", "持续长度",
            "重复状态码", "重复有功功率kW", "重复无功功率", "重复风速m/s", "重复风向",
        ])

    detail_excel = save_dir / "联合重复值检测结果.xlsx"
    with pd.ExcelWriter(detail_excel, engine="openpyxl") as writer:
        positive_df.to_excel(writer, sheet_name="联合重复检测结果", index=False)
    print(f"✅ 仅正功率联合重复结果已保存：{detail_excel}")

    if len(positive_df) > 0:
        summary = positive_df.groupby("风机编号", as_index=False)["持续长度"].sum()
        summary = summary.rename(columns={"持续长度": "联合重复总长度"})
    else:
        summary = pd.DataFrame(columns=["风机编号", "联合重复总长度"])

    summary_csv = save_dir / "联合重复值总时长汇总.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 仅正功率联合重复时长汇总已保存：{summary_csv}")


def save_positive_filter_stats(full_df: pd.DataFrame, positive_df: pd.DataFrame, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    full_count = int(len(full_df))
    pos_count = int(len(positive_df))
    removed_count = full_count - pos_count

    full_duration = float(pd.to_numeric(full_df.get("持续长度", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if full_count else 0.0
    pos_duration = float(pd.to_numeric(positive_df.get("持续长度", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if pos_count else 0.0
    removed_duration = full_duration - pos_duration

    stats_df = pd.DataFrame([{
        "原始联合重复段数_仅用于统计": full_count,
        "仅正功率重复段数": pos_count,
        "被剔除的零负或无效功率重复段数": removed_count,
        "仅正功率重复段占比": pos_count / full_count if full_count > 0 else np.nan,
        "原始联合重复总持续长度_仅用于统计": full_duration,
        "仅正功率重复总持续长度": pos_duration,
        "被剔除重复总持续长度": removed_duration,
        "仅正功率重复持续长度占比": pos_duration / full_duration if full_duration > 0 else np.nan,
        "筛选规则": "重复值组合中的ACTIVE_POWER > 0",
        "说明": "本脚本不保存全量联合重复明细，只保存仅正功率结果；原始联合重复指标仅用于筛选统计。",
    }])
    stats_path = save_dir / "仅正功率筛选统计.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"✅ 仅正功率筛选统计已保存：{stats_path}")

    if full_count > 0:
        full_by_fan = full_df.groupby("风机编号", as_index=False).agg(
            原始联合重复段数_仅用于统计=("风机编号", "count"),
            原始联合重复总持续长度_仅用于统计=("持续长度", "sum"),
        )
    else:
        full_by_fan = pd.DataFrame(columns=["风机编号", "原始联合重复段数_仅用于统计", "原始联合重复总持续长度_仅用于统计"])

    if pos_count > 0:
        pos_by_fan = positive_df.groupby("风机编号", as_index=False).agg(
            仅正功率重复段数=("风机编号", "count"),
            仅正功率重复总持续长度=("持续长度", "sum"),
        )
    else:
        pos_by_fan = pd.DataFrame(columns=["风机编号", "仅正功率重复段数", "仅正功率重复总持续长度"])

    fan_stats = full_by_fan.merge(pos_by_fan, on="风机编号", how="outer")
    for col in ["原始联合重复段数_仅用于统计", "原始联合重复总持续长度_仅用于统计", "仅正功率重复段数", "仅正功率重复总持续长度"]:
        if col in fan_stats.columns:
            fan_stats[col] = pd.to_numeric(fan_stats[col], errors="coerce").fillna(0)

    if len(fan_stats) > 0:
        fan_stats["被剔除零负或无效功率重复段数"] = fan_stats["原始联合重复段数_仅用于统计"] - fan_stats["仅正功率重复段数"]
        fan_stats["被剔除重复总持续长度"] = fan_stats["原始联合重复总持续长度_仅用于统计"] - fan_stats["仅正功率重复总持续长度"]
        fan_stats["仅正功率重复段占比"] = np.where(
            fan_stats["原始联合重复段数_仅用于统计"] > 0,
            fan_stats["仅正功率重复段数"] / fan_stats["原始联合重复段数_仅用于统计"],
            np.nan,
        )
        fan_stats["仅正功率重复持续长度占比"] = np.where(
            fan_stats["原始联合重复总持续长度_仅用于统计"] > 0,
            fan_stats["仅正功率重复总持续长度"] / fan_stats["原始联合重复总持续长度_仅用于统计"],
            np.nan,
        )
        fan_stats = fan_stats.sort_values(["仅正功率重复总持续长度", "仅正功率重复段数"], ascending=[False, False])

    fan_stats_path = save_dir / "仅正功率筛选统计_按风机.csv"
    fan_stats.to_csv(fan_stats_path, index=False, encoding="utf-8-sig")
    print(f"✅ 仅正功率按风机筛选统计已保存：{fan_stats_path}")


def run_positive_joint_repeat_check(
    df: pd.DataFrame,
    save_dir_positive: str | Path,
    field_prefixes: Iterable[str],
    timestamp_col: str,
    min_repeat: int,
) -> None:
    df = normalize_columns(df)

    if timestamp_col not in df.columns:
        raise ValueError(f"未找到 {timestamp_col} 列，无法进行风机联合重复检测。")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(by=timestamp_col).reset_index(drop=True)

    fan_numbers = extract_fan_numbers_sorted_from_df(df, field_prefixes)
    print(f"提取的风机编号列表：{fan_numbers}")

    if not fan_numbers:
        print("📭 未识别到任何风机编号，跳过风机联合连续相同检测。")
        return

    full_df, positive_df = detect_positive_joint_repeats(
        df=df,
        fan_numbers=fan_numbers,
        field_prefixes=field_prefixes,
        timestamp_col=timestamp_col,
        min_repeat=min_repeat,
    )

    save_positive_joint_results(positive_df, save_dir_positive)
    save_positive_filter_stats(full_df, positive_df, save_dir_positive)


def process_one_station_csv(csv_path: str | Path, min_repeat: int) -> None:
    csv_path = Path(csv_path)
    print("=" * 100)
    print(f"开始处理: {csv_path}")

    df = read_csv_auto(csv_path)
    df = normalize_columns(df)
    folder = csv_path.parent

    line_repeat_dir = folder / COLUMN_REPEAT_DIR_NAME
    detect_column_repeats(
        df=df,
        columns_to_extract=COLUMN_REPEAT_TARGETS,
        save_dir=line_repeat_dir,
        min_repeat=min_repeat,
        timestamp_col="timestamp",
    )

    positive_joint_dir = folder / POSITIVE_JOINT_REPEAT_DIR_NAME
    run_positive_joint_repeat_check(
        df=df,
        save_dir_positive=positive_joint_dir,
        field_prefixes=JOINT_FIELD_PREFIXES,
        timestamp_col="timestamp",
        min_repeat=min_repeat,
    )

    print("🎉 检测完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="STEP0 第二步简化版：只输出仅正功率联合重复结果")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="step0_第一步.py 输出的场站数据根目录")
    parser.add_argument("--station-csv", default=None, help="只处理指定场站主CSV；不填则处理 output-root 下所有场站主CSV")
    parser.add_argument("--min-repeat", type=int, default=DEFAULT_MIN_REPEAT, help="连续相同最少条数，达到该值才记录")
    args = parser.parse_args()

    if args.station_csv:
        main_csv_files = [Path(args.station_csv)]
    else:
        main_csv_files = find_main_csv_files(args.output_root)

    if not main_csv_files:
        print("没有找到可处理的主 CSV 文件。")
        return

    print(f"共找到 {len(main_csv_files)} 个主 CSV 文件，开始检测...")
    print(f"min_repeat = {args.min_repeat}")

    for csv_path in main_csv_files:
        process_one_station_csv(csv_path, min_repeat=args.min_repeat)


if __name__ == "__main__":
    main()
