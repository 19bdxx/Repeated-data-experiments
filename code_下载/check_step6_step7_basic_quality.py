#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 STEP6/STEP7 输出数据的基本情况。

目的：
1. 不修改 STEP6/STEP7，只检查它们生成的数据是否符合后续预测实验要求；
2. 检查分钟级 STEP6：
   - DATA_USE_RESULT 分布
   - 可用分钟中是否存在风机 FINAL_ACTIVE_POWER 缺失
   - 可用分钟中是否存在风机 FINAL_ACTIVE_POWER 负值
   - 可用分钟中是否存在风机 FINAL_ACTIVE_POWER 超额定/超上限
3. 检查 15min STEP7：
   - DATA_15MIN_USE_RESULT 分布
   - 可用15min窗口中是否存在风机 FINAL_ACTIVE_POWER 缺失
   - 可用15min窗口中是否存在风机 FINAL_ACTIVE_POWER 负值
   - 可用15min窗口中是否存在风机 FINAL_ACTIVE_POWER 超额定/超上限
4. 对 STEP6 -> STEP7 聚合关系做抽查：
   - 每个15min窗口中 STEP6 可用分钟数
   - 每台风机在该窗口内有效分钟数的最小值
   - 检查 STEP7 标记可用但个别风机有效分钟不足的窗口

默认单位：
- 风机功率：kW
- 单机额定：4000 kW
- 上限：4000 * 1.05 = 4200 kW
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = r"场站数据"
DEFAULT_STATION_NAME = "LGXRFD"
DEFAULT_STEP6_DIR_NAME = "step6-风机宽表修正结果-仅正功率"
DEFAULT_STEP7_DIR_NAME = "step7-15min建模数据"
DEFAULT_OUTPUT_DIR_NAME = "step8-预测实验"

STEP6_FINAL_PATTERN = re.compile(r"^FINAL_ACTIVE_POWER_#(\d+)$")
STEP7_FINAL_PATTERN = re.compile(r"^FINAL_ACTIVE_POWER_#(\d+)_15MIN$")


def read_csv_auto(path: str | Path, **kwargs) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"文件不存在：{path_obj}")

    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path_obj, encoding=enc, low_memory=False, **kwargs)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"读取失败：{path_obj}\n最后一次报错：{last_err}")


def csv_has_cols(path: Path, required: list[str]) -> bool:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            header = pd.read_csv(path, encoding=enc, nrows=0)
            cols = set(header.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False))
            return all(c in cols for c in required)
        except Exception:
            continue
    return False


def find_latest_step6_file(step6_dir: Path) -> Path:
    patterns = [
        "*fan_wide_correction_data_use_threshold_*.csv",
        "*fan_wide_correction_threshold_*.csv",
        "*_fan_wide_correction_*.csv",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(step6_dir.rglob(pat))
    candidates = [
        p for p in candidates
        if p.is_file()
        and "summary" not in p.name.lower()
        and "quality" not in p.name.lower()
    ]
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)

    # 优先选择包含 DATA_USE_RESULT 的新版 STEP6 输出
    valid = [p for p in candidates if csv_has_cols(p, ["timestamp", "DATA_USE_RESULT"])]
    if valid:
        return valid[0]
    if candidates:
        print("⚠️ 未找到包含 DATA_USE_RESULT 的 STEP6 文件，将使用最近的候选文件，但部分检查会跳过。")
        return candidates[0]
    raise FileNotFoundError(f"未找到 STEP6 输出宽表：{step6_dir}")


def default_step7_file(base_dir: str | Path, station_name: str, step7_dir_name: str) -> Path:
    return Path(base_dir) / station_name / step7_dir_name / f"{station_name}_15min_model_wide.csv"


def default_output_dir(base_dir: str | Path, station_name: str) -> Path:
    return Path(base_dir) / station_name / DEFAULT_OUTPUT_DIR_NAME


def extract_cols(columns: Iterable[str], pattern: re.Pattern) -> List[Tuple[int, str]]:
    pairs = []
    for c in columns:
        m = pattern.match(str(c).strip())
        if m:
            pairs.append((int(m.group(1)), str(c).strip()))
    return sorted(pairs, key=lambda x: x[0])


def make_window_end(ts: pd.Series, freq_minutes: int = 15) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    day_start = ts.dt.floor("D")
    minutes = ts.dt.hour * 60 + ts.dt.minute
    window_minute = np.ceil(minutes / freq_minutes) * freq_minutes
    window_minute = pd.Series(window_minute, index=ts.index).astype("Int64")
    add_days = (window_minute >= 24 * 60).astype(int)
    minute_in_day = (window_minute % (24 * 60)).astype("Int64")
    return day_start + pd.to_timedelta(add_days, unit="D") + pd.to_timedelta(minute_in_day.astype(float), unit="m")


def summarize_basic(df: pd.DataFrame, name: str, use_col: Optional[str]) -> list[dict]:
    rows = [
        {"数据表": name, "统计项": "总行数", "数值": len(df)},
        {"数据表": name, "统计项": "开始时间", "数值": str(df["timestamp"].min()) if "timestamp" in df.columns else ""},
        {"数据表": name, "统计项": "结束时间", "数值": str(df["timestamp"].max()) if "timestamp" in df.columns else ""},
    ]
    if use_col and use_col in df.columns:
        vc = df[use_col].astype(str).value_counts(dropna=False)
        for k, v in vc.items():
            rows.append({"数据表": name, "统计项": f"{use_col}={k}", "数值": int(v)})
    return rows


def analyze_power_matrix(
    df: pd.DataFrame,
    fan_cols: List[str],
    use_col: Optional[str],
    use_value: str,
    rated_kw: float,
    upper_kw: float,
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    返回：
    - row_detail：每个时刻的风机功率完整性统计
    - fan_detail：每台风机功率范围统计
    - summary：汇总统计
    """
    work = df.copy()
    if "timestamp" not in work.columns:
        raise ValueError(f"{name} 缺少 timestamp 列")

    if use_col and use_col in work.columns:
        mask_use = work[use_col].astype(str).str.strip().eq(use_value)
    else:
        mask_use = pd.Series(True, index=work.index)

    power = work[fan_cols].apply(pd.to_numeric, errors="coerce") if fan_cols else pd.DataFrame(index=work.index)

    missing_count = power.isna().sum(axis=1)
    valid_count = power.notna().sum(axis=1)
    neg_count = (power < 0).sum(axis=1)
    below_zero_or_nan_count = power.isna().sum(axis=1) + (power < 0).sum(axis=1)
    over_rated_count = (power > rated_kw).sum(axis=1)
    over_upper_count = (power > upper_kw).sum(axis=1)

    row_detail = pd.DataFrame({
        "timestamp": work["timestamp"],
        "是否表级可用": mask_use,
        "风机总数": len(fan_cols),
        "风机有效数": valid_count,
        "风机缺失数": missing_count,
        "风机负值数": neg_count,
        "风机缺失或负值数": below_zero_or_nan_count,
        "风机超过额定数": over_rated_count,
        "风机超过上限数": over_upper_count,
    })

    if use_col and use_col in work.columns:
        row_detail[use_col] = work[use_col].astype(str)

    fan_rows = []
    for col in fan_cols:
        fan_no = int(re.search(r"#(\d+)", col).group(1))
        s_all = pd.to_numeric(work[col], errors="coerce")
        s_use = s_all[mask_use]
        valid = s_use.dropna()
        fan_rows.append({
            "数据表": name,
            "风机编号": fan_no,
            "列名": col,
            "可用行数": int(mask_use.sum()),
            "有效点数": int(s_use.notna().sum()),
            "缺失点数": int(s_use.isna().sum()),
            "负值点数": int((s_use < 0).fillna(False).sum()),
            "超过额定点数": int((s_use > rated_kw).fillna(False).sum()),
            "超过上限点数": int((s_use > upper_kw).fillna(False).sum()),
            "最小值kW": float(valid.min()) if len(valid) else np.nan,
            "P01kW": float(valid.quantile(0.01)) if len(valid) else np.nan,
            "P50kW": float(valid.quantile(0.50)) if len(valid) else np.nan,
            "P95kW": float(valid.quantile(0.95)) if len(valid) else np.nan,
            "P99kW": float(valid.quantile(0.99)) if len(valid) else np.nan,
            "P99_9kW": float(valid.quantile(0.999)) if len(valid) else np.nan,
            "最大值kW": float(valid.max()) if len(valid) else np.nan,
            "均值kW": float(valid.mean()) if len(valid) else np.nan,
        })
    fan_detail = pd.DataFrame(fan_rows)

    use_rows = row_detail[row_detail["是否表级可用"]].copy()

    summary_rows = [
        {"数据表": name, "统计项": "风机列数量", "数值": len(fan_cols)},
        {"数据表": name, "统计项": "表级可用行数", "数值": int(mask_use.sum())},
        {"数据表": name, "统计项": "表级不可用行数", "数值": int((~mask_use).sum())},
        {"数据表": name, "统计项": "可用行中存在任意风机缺失的行数", "数值": int((use_rows["风机缺失数"] > 0).sum())},
        {"数据表": name, "统计项": "可用行中存在任意风机负值的行数", "数值": int((use_rows["风机负值数"] > 0).sum())},
        {"数据表": name, "统计项": "可用行中存在任意风机缺失或负值的行数", "数值": int((use_rows["风机缺失或负值数"] > 0).sum())},
        {"数据表": name, "统计项": "可用行中存在任意风机超过额定的行数", "数值": int((use_rows["风机超过额定数"] > 0).sum())},
        {"数据表": name, "统计项": "可用行中存在任意风机超过上限的行数", "数值": int((use_rows["风机超过上限数"] > 0).sum())},
        {"数据表": name, "统计项": "可用行中最小风机有效数", "数值": int(use_rows["风机有效数"].min()) if len(use_rows) else np.nan},
        {"数据表": name, "统计项": "可用行中平均风机有效数", "数值": float(use_rows["风机有效数"].mean()) if len(use_rows) else np.nan},
        {"数据表": name, "统计项": "可用行中最小风机缺失数", "数值": int(use_rows["风机缺失数"].min()) if len(use_rows) else np.nan},
        {"数据表": name, "统计项": "可用行中最大风机缺失数", "数值": int(use_rows["风机缺失数"].max()) if len(use_rows) else np.nan},
        {"数据表": name, "统计项": "可用行中最大风机负值数", "数值": int(use_rows["风机负值数"].max()) if len(use_rows) else np.nan},
        {"数据表": name, "统计项": "可用行中最大风机超过上限数", "数值": int(use_rows["风机超过上限数"].max()) if len(use_rows) else np.nan},
        {"数据表": name, "统计项": "可用样本中所有风机最小值kW", "数值": float(fan_detail["最小值kW"].min()) if len(fan_detail) else np.nan},
        {"数据表": name, "统计项": "可用样本中所有风机最大值kW", "数值": float(fan_detail["最大值kW"].max()) if len(fan_detail) else np.nan},
        {"数据表": name, "统计项": "可用样本中所有风机P99最大值kW", "数值": float(fan_detail["P99kW"].max()) if len(fan_detail) else np.nan},
        {"数据表": name, "统计项": "可用样本中所有风机P99_9最大值kW", "数值": float(fan_detail["P99_9kW"].max()) if len(fan_detail) else np.nan},
    ]

    summary = pd.DataFrame(summary_rows)
    return row_detail, fan_detail, summary


def analyze_step6_to_step7_window_consistency(
    step6: pd.DataFrame,
    step7: pd.DataFrame,
    step6_fan_cols: List[str],
    step7_fan_cols: List[str],
    min_valid_minutes: int,
    freq_minutes: int,
) -> pd.DataFrame:
    """
    检查 STEP7 可用窗口是否对应 STEP6 中足够的可用分钟和风机有效分钟。
    """
    if "DATA_USE_RESULT" not in step6.columns or "DATA_15MIN_USE_RESULT" not in step7.columns:
        return pd.DataFrame()

    s6 = step6.copy()
    s6["timestamp"] = pd.to_datetime(s6["timestamp"], errors="coerce")
    s6 = s6.dropna(subset=["timestamp"]).copy()
    s6["_window_end"] = make_window_end(s6["timestamp"], freq_minutes=freq_minutes)
    s6["_minute_usable"] = s6["DATA_USE_RESULT"].astype(str).str.strip().eq("可用")

    fan_power = s6[step6_fan_cols].apply(pd.to_numeric, errors="coerce")
    fan_valid = fan_power.notna()

    rows = []
    for window_end, g_idx in s6.groupby("_window_end").groups.items():
        idx = list(g_idx)
        g = s6.loc[idx]
        usable_idx = g.index[g["_minute_usable"]].tolist()
        usable_minutes = len(usable_idx)

        if usable_minutes > 0 and step6_fan_cols:
            fan_valid_counts = fan_valid.loc[usable_idx, step6_fan_cols].sum(axis=0)
            min_fan_valid_minutes = int(fan_valid_counts.min())
            max_fan_valid_minutes = int(fan_valid_counts.max())
            bad_fan_count = int((fan_valid_counts < min_valid_minutes).sum())
            all_fan_enough = bad_fan_count == 0
        else:
            min_fan_valid_minutes = 0
            max_fan_valid_minutes = 0
            bad_fan_count = len(step6_fan_cols)
            all_fan_enough = False

        rows.append({
            "timestamp": pd.to_datetime(window_end),
            "STEP6窗口总分钟数": int(len(g)),
            "STEP6可用分钟数": int(usable_minutes),
            "STEP6最小风机有效分钟数": min_fan_valid_minutes,
            "STEP6最大风机有效分钟数": max_fan_valid_minutes,
            "STEP6有效分钟不足风机数": bad_fan_count,
            "STEP6是否所有风机有效分钟足够": all_fan_enough,
        })

    win = pd.DataFrame(rows)
    s7 = step7[["timestamp", "DATA_15MIN_USE_RESULT"]].copy()
    s7["timestamp"] = pd.to_datetime(s7["timestamp"], errors="coerce")
    out = s7.merge(win, on="timestamp", how="left")
    out["STEP7是否可用"] = out["DATA_15MIN_USE_RESULT"].astype(str).str.strip().eq("可用")
    out["STEP7可用但STEP6可用分钟不足"] = out["STEP7是否可用"] & (out["STEP6可用分钟数"] < min_valid_minutes)
    out["STEP7可用但存在风机有效分钟不足"] = out["STEP7是否可用"] & (~out["STEP6是否所有风机有效分钟足够"].fillna(False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 STEP6/STEP7 输出数据基本情况")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="场站数据根目录")
    parser.add_argument("--station-name", default=DEFAULT_STATION_NAME, help="场站名称")
    parser.add_argument("--step6-dir-name", default=DEFAULT_STEP6_DIR_NAME, help="STEP6输出目录名")
    parser.add_argument("--step7-dir-name", default=DEFAULT_STEP7_DIR_NAME, help="STEP7输出目录名")
    parser.add_argument("--step6-file", default=None, help="STEP6分钟级宽表；不填则自动查找最新")
    parser.add_argument("--step7-file", default=None, help="STEP7 15min宽表；不填则按场站名自动拼接")
    parser.add_argument("--output-dir", default=None, help="输出目录；默认 场站数据/<场站>/step8-预测实验")
    parser.add_argument("--fan-rated-kw", type=float, default=4000.0, help="单台风机额定功率kW")
    parser.add_argument("--upper-factor", type=float, default=1.05, help="单机允许上限倍率")
    parser.add_argument("--min-valid-minutes", type=int, default=14, help="STEP7可用窗口最小可用分钟数")
    parser.add_argument("--freq-minutes", type=int, default=15, help="聚合频率")
    args = parser.parse_args()

    station_dir = Path(args.base_dir) / args.station_name
    step6_file = Path(args.step6_file) if args.step6_file else find_latest_step6_file(station_dir / args.step6_dir_name)
    step7_file = Path(args.step7_file) if args.step7_file else default_step7_file(args.base_dir, args.station_name, args.step7_dir_name)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.base_dir, args.station_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    upper_kw = args.fan_rated_kw * args.upper_factor

    print("=" * 88)
    print("检查 STEP6/STEP7 输出数据基本情况")
    print("=" * 88)
    print(f"场站: {args.station_name}")
    print(f"STEP6文件: {step6_file}")
    print(f"STEP7文件: {step7_file}")
    print(f"输出目录: {output_dir}")
    print(f"单机额定功率: {args.fan_rated_kw:g} kW")
    print(f"单机上限: {upper_kw:g} kW")

    step6 = read_csv_auto(step6_file)
    step7 = read_csv_auto(step7_file)
    step6.columns = step6.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    step7.columns = step7.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    step6["timestamp"] = pd.to_datetime(step6["timestamp"], errors="coerce")
    step7["timestamp"] = pd.to_datetime(step7["timestamp"], errors="coerce")
    step6 = step6.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    step7 = step7.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    step6_fan_pairs = extract_cols(step6.columns, STEP6_FINAL_PATTERN)
    step7_fan_pairs = extract_cols(step7.columns, STEP7_FINAL_PATTERN)
    step6_fan_cols = [c for _, c in step6_fan_pairs]
    step7_fan_cols = [c for _, c in step7_fan_pairs]

    print(f"STEP6风机列数: {len(step6_fan_cols)}")
    print(f"STEP7风机列数: {len(step7_fan_cols)}")

    all_summary_rows = []
    all_summary_rows.extend(summarize_basic(step6, "STEP6分钟级", "DATA_USE_RESULT"))
    all_summary_rows.extend(summarize_basic(step7, "STEP7_15min", "DATA_15MIN_USE_RESULT"))

    step6_row_detail, step6_fan_detail, step6_power_summary = analyze_power_matrix(
        step6,
        step6_fan_cols,
        use_col="DATA_USE_RESULT",
        use_value="可用",
        rated_kw=args.fan_rated_kw,
        upper_kw=upper_kw,
        name="STEP6分钟级",
    )
    step7_row_detail, step7_fan_detail, step7_power_summary = analyze_power_matrix(
        step7,
        step7_fan_cols,
        use_col="DATA_15MIN_USE_RESULT",
        use_value="可用",
        rated_kw=args.fan_rated_kw,
        upper_kw=upper_kw,
        name="STEP7_15min",
    )

    consistency = analyze_step6_to_step7_window_consistency(
        step6,
        step7,
        step6_fan_cols,
        step7_fan_cols,
        min_valid_minutes=args.min_valid_minutes,
        freq_minutes=args.freq_minutes,
    )

    if len(consistency) > 0:
        all_summary_rows.extend([
            {
                "数据表": "STEP6->STEP7一致性",
                "统计项": "STEP7可用但STEP6可用分钟不足窗口数",
                "数值": int(consistency["STEP7可用但STEP6可用分钟不足"].sum()),
            },
            {
                "数据表": "STEP6->STEP7一致性",
                "统计项": "STEP7可用但存在风机有效分钟不足窗口数",
                "数值": int(consistency["STEP7可用但存在风机有效分钟不足"].sum()),
            },
            {
                "数据表": "STEP6->STEP7一致性",
                "统计项": "STEP7可用窗口中STEP6最小风机有效分钟数最小值",
                "数值": float(consistency.loc[consistency["STEP7是否可用"], "STEP6最小风机有效分钟数"].min()) if consistency["STEP7是否可用"].any() else np.nan,
            },
        ])

    summary = pd.concat(
        [
            pd.DataFrame(all_summary_rows),
            step6_power_summary,
            step7_power_summary,
        ],
        ignore_index=True,
    )

    # 输出
    summary_file = output_dir / "step6_step7_basic_quality_summary.csv"
    step6_row_file = output_dir / "step6_minute_row_quality.csv"
    step7_row_file = output_dir / "step7_15min_row_quality.csv"
    step6_fan_file = output_dir / "step6_fan_power_quality.csv"
    step7_fan_file = output_dir / "step7_fan_power_quality.csv"
    consistency_file = output_dir / "step6_to_step7_window_consistency.csv"

    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    step6_row_detail.to_csv(step6_row_file, index=False, encoding="utf-8-sig")
    step7_row_detail.to_csv(step7_row_file, index=False, encoding="utf-8-sig")
    step6_fan_detail.to_csv(step6_fan_file, index=False, encoding="utf-8-sig")
    step7_fan_detail.to_csv(step7_fan_file, index=False, encoding="utf-8-sig")
    if len(consistency) > 0:
        consistency.to_csv(consistency_file, index=False, encoding="utf-8-sig")

    print("\n完成：")
    print(f"- 汇总统计: {summary_file}")
    print(f"- STEP6逐分钟行质量: {step6_row_file}")
    print(f"- STEP7逐15min行质量: {step7_row_file}")
    print(f"- STEP6风机功率质量: {step6_fan_file}")
    print(f"- STEP7风机功率质量: {step7_fan_file}")
    if len(consistency) > 0:
        print(f"- STEP6到STEP7窗口一致性: {consistency_file}")

    print("\n关键汇总：")
    key_summary = summary[
        summary["统计项"].astype(str).str.contains(
            "可用行中|可用15min|DATA|一致性|风机列数量|所有风机最大值|所有风机P99|缺失|负值|超过上限",
            regex=True,
            na=False,
        )
    ].copy()
    print(key_summary.to_string(index=False))


if __name__ == "__main__":
    main()
