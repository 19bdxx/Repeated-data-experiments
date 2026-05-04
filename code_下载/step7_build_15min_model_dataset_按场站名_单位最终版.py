#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP7：将 STEP6 分钟级清洗宽表聚合为 15分钟级建模数据集。

默认路径采用“只修改场站名称”的形式：
    输入：场站数据/<场站名称>/step6-风机宽表修正结果-仅正功率/*fan_wide_correction_threshold_*.csv
    输出：场站数据/<场站名称>/step7-15min建模数据/<场站名称>_15min_model_wide.csv


核心规则：
1. 15分钟时间戳采用右端点：
   - 00:15 表示 00:01 ~ 00:15
   - 00:30 表示 00:16 ~ 00:30
   - 00:00 表示前一天 23:46 ~ 当天 00:00

2. 只使用 DATA_USE_RESULT == “可用” 的分钟参与均值计算。

3. 如果某个 15min 窗口内可用分钟数 >= min-valid-minutes，默认 14，
   则该 15min 窗口可用；否则不可用。

4. 均值分母为实际可用分钟数：
   - 15个可用分钟就除以15
   - 14个可用分钟就除以14

5. 风机级只聚合 FINAL_ACTIVE_POWER_#n，不聚合状态码、风速。
   状态码取均值没有实际意义；风速未经过冻结修正，暂不纳入第一版 15min 建模表。

6. 单机风机功率列在分钟级通常是 kW，而全站功率是 MW。
   STEP7 在计算15分钟均值时：
       - 1分钟风机功率 < 0 时按 0 参与均值；
       - 风机功率不做上限裁剪；
       - 输出的 FINAL_ACTIVE_POWER_#n_15MIN 直接转换为 MW。
   因此：
       FINAL_FAN_POWER_SUM_MW_15MIN = sum(FINAL_ACTIVE_POWER_#n_15MIN)
       FINAL_LOSS_MW_15MIN = FINAL_FAN_POWER_SUM_MW_15MIN - ACTIVE_POWER_STATION_15MIN
   后续 STEP8 可以全程使用 MW，避免单位混用。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# 默认路径
# ============================================================
# 只需要修改场站名称即可自动拼接输入输出路径。
DEFAULT_BASE_DIR = r"场站数据"
DEFAULT_STATION_NAME = "LGXRFD"
DEFAULT_STEP6_DIR_NAME = "step6-风机宽表修正结果-仅正功率"
DEFAULT_STEP7_DIR_NAME = "step7-15min建模数据"

DEFAULT_INPUT_FILE = None
DEFAULT_OUTPUT_DIR = None

TIMESTAMP_COL = "timestamp"
DATA_USE_COL = "DATA_USE_RESULT"
DATA_USE_AVAILABLE = "可用"

FINAL_POWER_PREFIX = "FINAL_ACTIVE_POWER_#"

# 直接从 STEP6 做15min均值的场站级原始列。
# 注意：分钟级风机功率通常是 kW，而全站级功率是 MW。
# STEP7 会将 FINAL_ACTIVE_POWER_#n_15MIN 直接输出为 MW。
# 因此 FINAL_FAN_POWER_SUM_MW_15MIN / FINAL_LOSS_MW_15MIN
# 在 STEP7 内基于 MW 级风机15min功率重新计算。
STATION_MEAN_COLS = [
    "ACTIVE_POWER_STATION",
    "LIMIT_POWER",
]

DERIVED_STATION_COLS = [
    "FINAL_FAN_POWER_SUM_MW_15MIN",
    "FINAL_LOSS_MW_15MIN",
    "FINAL_STATION_POWER_MW_15MIN",
]


# ============================================================
# 通用工具
# ============================================================
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


def build_station_dir(base_dir: str | Path, station_name: str) -> Path:
    return Path(base_dir) / str(station_name)


def build_default_step6_dir(base_dir: str | Path, station_name: str, step6_dir_name: str = DEFAULT_STEP6_DIR_NAME) -> Path:
    return build_station_dir(base_dir, station_name) / step6_dir_name


def build_default_step7_dir(base_dir: str | Path, station_name: str, step7_dir_name: str = DEFAULT_STEP7_DIR_NAME) -> Path:
    return build_station_dir(base_dir, station_name) / step7_dir_name


def find_latest_step6_file(search_dir: str | Path = ".") -> Path:
    """
    在当前目录及子目录中尝试寻找 STEP6 输出宽表。
    若你不想自动查找，建议直接用 --input-file 显式指定。
    """
    search_dir = Path(search_dir)
    patterns = [
        "*fan_wide_correction_threshold_*.csv",
        "*_fan_wide_correction_*.csv",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(search_dir.rglob(pat))

    # 排除 summary
    candidates = [p for p in candidates if "summary" not in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(
            f"未自动找到 STEP6 输出宽表。请使用 --input-file 显式指定文件。搜索目录：{search_dir}"
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    print("⚠️ 未指定 --input-file，自动选择最近的 STEP6 输出宽表：")
    print(f"  {candidates[0]}")
    return candidates[0]


def extract_fan_numbers(columns: Iterable[str]) -> List[int]:
    nums = []
    pattern = re.compile(r"^FINAL_ACTIVE_POWER_#(\d+)$")
    for col in columns:
        m = pattern.match(str(col).strip())
        if m:
            nums.append(int(m.group(1)))
    return sorted(set(nums))


def make_window_end(ts: pd.Series, freq_minutes: int = 15) -> pd.Series:
    """
    计算右端点窗口标签。

    规则：
    - 若 timestamp 已经落在 15min 边界，如 00:15，则窗口结束为 00:15
    - 若 timestamp 是 00:01~00:14，则窗口结束为 00:15
    - 若 timestamp 是 00:00，则窗口结束为 00:00，对应前一窗口 23:46~00:00
    """
    ts = pd.to_datetime(ts, errors="coerce")
    day_start = ts.dt.floor("D")
    minutes = ts.dt.hour * 60 + ts.dt.minute

    # ceil 到 freq_minutes 的倍数；0 仍然为 0。
    window_minute = np.ceil(minutes / freq_minutes) * freq_minutes
    window_minute = pd.Series(window_minute, index=ts.index).astype("Int64")

    # 24:00 进位到次日 00:00
    add_days = (window_minute >= 24 * 60).astype(int)
    minute_in_day = (window_minute % (24 * 60)).astype("Int64")

    return day_start + pd.to_timedelta(add_days, unit="D") + pd.to_timedelta(minute_in_day.astype(float), unit="m")


def safe_mean_valid(g: pd.DataFrame, col: str, valid_col: str = "_is_valid") -> float:
    s = pd.to_numeric(g.loc[g[valid_col], col], errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.mean())


def fan_unit_to_mw_scale(unit: str) -> float:
    """
    返回：单机功率原始单位 × scale = MW。
    江苏场站单机功率通常是 kW，因此默认 kW -> MW 的系数是 0.001。
    """
    u = str(unit).strip().lower()
    if u == "kw":
        return 0.001
    if u == "mw":
        return 1.0
    raise ValueError(f"未知风机功率单位：{unit}，仅支持 kW 或 MW")


def safe_mean_valid_fan_power_mw(
    g: pd.DataFrame,
    col: str,
    valid_col: str = "_is_valid",
    source_unit: str = "kW",
    negative_to_zero: bool = True,
) -> float:
    """
    风机功率15min均值，输出统一为 MW。

    规则：
    - 只使用分钟级 DATA_USE_RESULT=可用 的分钟；
    - 若1分钟风机功率为负数，则按0参与均值；
    - 不做上限裁剪，保留实际高于额定的功率；
    - 若源单位为 kW，则最终除以1000输出 MW。
    """
    s = pd.to_numeric(g.loc[g[valid_col], col], errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    if negative_to_zero:
        s = s.clip(lower=0.0)
    return float(s.mean() * fan_unit_to_mw_scale(source_unit))

def concat_reason(values: pd.Series, max_items: int = 5) -> str:
    vals = [str(x).strip() for x in values.dropna().astype(str).tolist() if str(x).strip()]
    if not vals:
        return ""
    counts = pd.Series(vals).value_counts()
    parts = [f"{k}:{int(v)}" for k, v in counts.head(max_items).items()]
    return "；".join(parts)


# ============================================================
# 主聚合逻辑
# ============================================================
def build_15min_dataset(
    minute_df: pd.DataFrame,
    min_valid_minutes: int = 14,
    freq_minutes: int = 15,
    fan_power_source_unit: str = "kW",
    negative_fan_power_to_zero: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = minute_df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"输入文件缺少 {TIMESTAMP_COL} 列")
    if DATA_USE_COL not in df.columns:
        raise ValueError(f"输入文件缺少 {DATA_USE_COL} 列，请确认输入为 STEP6 数据可用标签版输出")

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL]).copy()
    df = df.sort_values(TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL], keep="last").reset_index(drop=True)

    df["_window_end"] = make_window_end(df[TIMESTAMP_COL], freq_minutes=freq_minutes)
    df["_window_start"] = df["_window_end"] - pd.to_timedelta(freq_minutes - 1, unit="m")
    df["_is_valid"] = df[DATA_USE_COL].astype(str).str.strip().eq(DATA_USE_AVAILABLE)

    fan_nums = extract_fan_numbers(df.columns)
    fan_power_cols = [f"{FINAL_POWER_PREFIX}{n}" for n in fan_nums]

    station_cols = [c for c in STATION_MEAN_COLS if c in df.columns]
    missing_station_cols = [c for c in STATION_MEAN_COLS if c not in df.columns]
    if missing_station_cols:
        print("⚠️ 以下场站级列不存在，将不会输出其15min均值：")
        for c in missing_station_cols:
            print(f"  - {c}")

    print(f"识别到 FINAL_ACTIVE_POWER 风机列数量: {len(fan_power_cols)}")
    print(f"场站级均值列: {station_cols}")
    print(f"分钟级风机功率源单位: {fan_power_source_unit}; 转MW系数: {fan_unit_to_mw_scale(fan_power_source_unit)}")
    print(f"风机负功率是否按0参与均值: {negative_fan_power_to_zero}; 风机功率不做上限裁剪")
    print("输出的 FINAL_ACTIVE_POWER_#n_15MIN 单位: MW")
    print(f"最小可用分钟数: {min_valid_minutes}/{freq_minutes}")

    rows = []
    grouped = df.groupby("_window_end", sort=True)

    for window_end, g in grouped:
        total_minutes = int(len(g))
        valid_minutes = int(g["_is_valid"].sum())
        window_start = pd.to_datetime(window_end) - pd.Timedelta(minutes=freq_minutes - 1)

        row = {
            "timestamp": pd.to_datetime(window_end),
            "WINDOW_START": window_start,
            "WINDOW_END": pd.to_datetime(window_end),
            "总分钟数": total_minutes,
            "可用分钟数": valid_minutes,
            "不可用分钟数": total_minutes - valid_minutes,
            "可用分钟占比": valid_minutes / total_minutes if total_minutes > 0 else np.nan,
            "DATA_15MIN_USE_RESULT": "可用" if valid_minutes >= min_valid_minutes else "不可用",
            "DATA_15MIN_USE_REASON": (
                f"可用分钟数{valid_minutes}>={min_valid_minutes}"
                if valid_minutes >= min_valid_minutes
                else f"可用分钟数{valid_minutes}<{min_valid_minutes}"
            ),
        }

        # 不可用原因分布，方便排查为何某个15min窗口不可用。
        if "DATA_USE_REASON" in g.columns:
            row["分钟级不可用/可用原因分布"] = concat_reason(g.loc[~g["_is_valid"], "DATA_USE_REASON"])
        if "DATA_USE_SCENE" in g.columns:
            row["分钟级场景分布"] = concat_reason(g["DATA_USE_SCENE"])
        if "POINT_OUTLIER_RESULT" in g.columns:
            row["点级离群分钟数"] = int(g["POINT_OUTLIER_RESULT"].astype(str).eq("点级明显离群").sum())

        # 场站级字段：只对可用分钟取均值。
        for col in station_cols:
            row[f"{col}_15MIN"] = safe_mean_valid(g, col)

        # 风机级字段：只输出 FINAL_ACTIVE_POWER_#n 的15min均值。
        # 注意：输出单位直接转换为 MW，避免后续 STEP8 再混用 kW/MW。
        for col in fan_power_cols:
            row[f"{col}_15MIN"] = safe_mean_valid_fan_power_mw(
                g,
                col,
                source_unit=fan_power_source_unit,
                negative_to_zero=negative_fan_power_to_zero,
            )

        rows.append(row)

    out = pd.DataFrame(rows)

    # ========================================================
    # 单位修正后的派生场站级字段
    # ========================================================
    fan_out_cols = [f"{c}_15MIN" for c in fan_power_cols]

    if fan_out_cols:
        # 风机功率15min列已经在上方统一转换为 MW，因此这里直接求和。
        out["FINAL_FAN_POWER_SUM_MW_15MIN"] = out[fan_out_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
    else:
        out["FINAL_FAN_POWER_SUM_MW_15MIN"] = np.nan

    if "ACTIVE_POWER_STATION_15MIN" in out.columns:
        out["FINAL_STATION_POWER_MW_15MIN"] = pd.to_numeric(out["ACTIVE_POWER_STATION_15MIN"], errors="coerce")
        out["FINAL_LOSS_MW_15MIN"] = out["FINAL_FAN_POWER_SUM_MW_15MIN"] - out["FINAL_STATION_POWER_MW_15MIN"]
    else:
        out["FINAL_STATION_POWER_MW_15MIN"] = np.nan
        out["FINAL_LOSS_MW_15MIN"] = np.nan

    # 输出列顺序：质量字段 -> 场站级字段 -> 风机功率字段
    quality_cols = [
        "timestamp", "WINDOW_START", "WINDOW_END",
        "DATA_15MIN_USE_RESULT", "DATA_15MIN_USE_REASON",
        "总分钟数", "可用分钟数", "不可用分钟数", "可用分钟占比",
        "点级离群分钟数", "分钟级不可用/可用原因分布", "分钟级场景分布",
    ]
    station_out_cols = [f"{c}_15MIN" for c in station_cols]
    derived_station_cols = [c for c in DERIVED_STATION_COLS if c in out.columns]
    fan_out_cols = [f"{c}_15MIN" for c in fan_power_cols]
    ordered_cols = [c for c in quality_cols + station_out_cols + derived_station_cols + fan_out_cols if c in out.columns]
    out = out[ordered_cols]

    summary_rows = [
        {"统计项": "分钟级原始行数", "数值": len(df)},
        {"统计项": "15min窗口数", "数值": len(out)},
        {"统计项": "可用15min窗口数", "数值": int((out["DATA_15MIN_USE_RESULT"] == "可用").sum())},
        {"统计项": "不可用15min窗口数", "数值": int((out["DATA_15MIN_USE_RESULT"] == "不可用").sum())},
        {"统计项": f"可用分钟数>={min_valid_minutes}窗口数", "数值": int((out["可用分钟数"] >= min_valid_minutes).sum())},
        {"统计项": "可用分钟数=15窗口数", "数值": int((out["可用分钟数"] == freq_minutes).sum())},
        {"统计项": "可用分钟数=14窗口数", "数值": int((out["可用分钟数"] == 14).sum())},
        {"统计项": "可用分钟数<14窗口数", "数值": int((out["可用分钟数"] < 14).sum())},
        {"统计项": "平均可用分钟数", "数值": float(out["可用分钟数"].mean()) if len(out) else np.nan},
        {"统计项": "FINAL_ACTIVE_POWER风机列数量", "数值": len(fan_power_cols)},
        {"统计项": "分钟级风机功率源单位", "数值": fan_power_source_unit},
        {"统计项": "输出风机15min功率单位", "数值": "MW"},
        {"统计项": "风机功率转MW系数", "数值": fan_unit_to_mw_scale(fan_power_source_unit)},
        {"统计项": "风机负功率是否按0参与均值", "数值": negative_fan_power_to_zero},
        {"统计项": "风机功率是否上限裁剪", "数值": False},
        {"统计项": "FINAL_FAN_POWER_SUM_MW_15MIN最大值", "数值": float(out["FINAL_FAN_POWER_SUM_MW_15MIN"].max()) if len(out) else np.nan},
        {"统计项": "FINAL_LOSS_MW_15MIN最大值", "数值": float(out["FINAL_LOSS_MW_15MIN"].max()) if len(out) else np.nan},
        {"统计项": "FINAL_LOSS_MW_15MIN最小值", "数值": float(out["FINAL_LOSS_MW_15MIN"].min()) if len(out) else np.nan},
    ]

    if "点级离群分钟数" in out.columns:
        summary_rows.append({"统计项": "含点级离群的15min窗口数", "数值": int((out["点级离群分钟数"] > 0).sum())})
        summary_rows.append({"统计项": "点级离群分钟数总和", "数值": int(out["点级离群分钟数"].sum())})

    summary = pd.DataFrame(summary_rows)
    return out, summary


# ============================================================
# main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="STEP7：将 STEP6 分钟级清洗宽表聚合为15分钟级建模数据集")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="场站数据根目录；默认：场站数据")
    parser.add_argument("--station-name", default=DEFAULT_STATION_NAME, help="场站名称；只需要修改这个参数即可切换场站")
    parser.add_argument("--step6-dir-name", default=DEFAULT_STEP6_DIR_NAME, help="STEP6输出目录名，位于 base-dir/station-name 下")
    parser.add_argument("--step7-dir-name", default=DEFAULT_STEP7_DIR_NAME, help="STEP7输出目录名，位于 base-dir/station-name 下")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="STEP6 输出的分钟级宽表路径；不填则自动从该场站 STEP6 目录寻找最新宽表")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录；不填则自动使用 base-dir/station-name/step7-dir-name")
    parser.add_argument("--output-name", default=None, help="15min宽表输出文件名；不填则使用 <场站名>_15min_model_wide.csv")
    parser.add_argument("--summary-name", default=None, help="质量统计输出文件名；不填则使用 <场站名>_15min_quality_summary.csv")
    parser.add_argument("--min-valid-minutes", type=int, default=14, help="15min窗口至少需要多少个可用分钟才标记为可用")
    parser.add_argument("--freq-minutes", type=int, default=15, help="聚合频率，默认15分钟")
    parser.add_argument("--fan-power-source-unit", default="kW", choices=["kW", "MW"], help="分钟级风机功率源单位，江苏场站通常为kW；STEP7输出会统一转为MW")
    parser.add_argument("--no-negative-fan-power-to-zero", action="store_true", help="若指定，则风机负功率不按0处理；默认负值按0参与均值")
    parser.add_argument("--only-available-output", action="store_true", help="若指定，则只输出 DATA_15MIN_USE_RESULT=可用 的15min窗口")
    args = parser.parse_args()

    station_dir = build_station_dir(args.base_dir, args.station_name)
    step6_dir = build_default_step6_dir(args.base_dir, args.station_name, args.step6_dir_name)

    input_file = Path(args.input_file) if args.input_file else find_latest_step6_file(step6_dir)
    output_dir = Path(args.output_dir) if args.output_dir else build_default_step7_dir(args.base_dir, args.station_name, args.step7_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = args.output_name if args.output_name else f"{args.station_name}_15min_model_wide.csv"
    summary_name = args.summary_name if args.summary_name else f"{args.station_name}_15min_quality_summary.csv"

    print("=" * 88)
    print("STEP7：分钟级清洗宽表 -> 15分钟级建模数据集")
    print("=" * 88)
    print(f"场站名称: {args.station_name}")
    print(f"场站目录: {station_dir}")
    print(f"STEP6目录: {step6_dir}")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"聚合频率: {args.freq_minutes} min")
    print(f"最小可用分钟数: {args.min_valid_minutes}")
    print(f"分钟级风机功率源单位: {args.fan_power_source_unit}")
    print("风机功率不做上限裁剪")
    print(f"风机负功率是否按0参与均值: {not args.no_negative_fan_power_to_zero}")
    print("输出风机15min功率单位: MW")

    minute_df = read_csv_auto(input_file)
    out, summary = build_15min_dataset(
        minute_df=minute_df,
        min_valid_minutes=args.min_valid_minutes,
        freq_minutes=args.freq_minutes,
        fan_power_source_unit=args.fan_power_source_unit,
        negative_fan_power_to_zero=(not args.no_negative_fan_power_to_zero),
    )

    if args.only_available_output:
        out_to_write = out[out["DATA_15MIN_USE_RESULT"] == "可用"].copy()
    else:
        out_to_write = out

    out_path = output_dir / output_name
    summary_path = output_dir / summary_name

    out_to_write.to_csv(out_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n完成：")
    print(f"- 15min建模宽表: {out_path}")
    print(f"- 质量统计表: {summary_path}")
    print(f"- 15min窗口数: {len(out):,}")
    print(f"- 实际输出窗口数: {len(out_to_write):,}")
    print("\n质量统计：")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
