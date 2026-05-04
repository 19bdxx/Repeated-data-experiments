#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 STEP7 15min 建模宽表中每台风机 FINAL_ACTIVE_POWER_#n_15MIN 的功率范围。

用途：
1. 查看每台风机最大值、P99、P99.9；
2. 统计超过额定功率、超过允许上限、负值、缺失值的点数；
3. 为 STEP8 中单机功率物理过滤/裁剪阈值提供依据。

默认：
- 场站：LGXRFD
- 单机额定：4000 kW
- 允许上限：4000 * 1.05 = 4200 kW
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = r"场站数据"
DEFAULT_STATION_NAME = "LGXRFD"
DEFAULT_STEP7_DIR_NAME = "step7-15min建模数据"
DEFAULT_OUTPUT_DIR_NAME = "step8-预测实验"

FINAL_POWER_PATTERN = re.compile(r"^FINAL_ACTIVE_POWER_#(\d+)_15MIN$")


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


def build_default_input_file(base_dir: str | Path, station_name: str, step7_dir_name: str) -> Path:
    return Path(base_dir) / station_name / step7_dir_name / f"{station_name}_15min_model_wide.csv"


def build_default_output_dir(base_dir: str | Path, station_name: str, output_dir_name: str) -> Path:
    return Path(base_dir) / station_name / output_dir_name


def extract_fan_cols(columns: Iterable[str]) -> List[tuple[int, str]]:
    pairs = []
    for col in columns:
        m = FINAL_POWER_PATTERN.match(str(col).strip())
        if m:
            pairs.append((int(m.group(1)), str(col).strip()))
    return sorted(pairs, key=lambda x: x[0])


def calc_quantile(s: pd.Series, q: float) -> float:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if len(s2) == 0:
        return np.nan
    return float(s2.quantile(q))


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 STEP7 每台风机 15min 功率范围")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="场站数据根目录")
    parser.add_argument("--station-name", default=DEFAULT_STATION_NAME, help="场站名称")
    parser.add_argument("--step7-dir-name", default=DEFAULT_STEP7_DIR_NAME, help="STEP7输出目录名")
    parser.add_argument("--input-file", default=None, help="STEP7 15min宽表路径；不填则按场站名自动拼接")
    parser.add_argument("--output-dir", default=None, help="输出目录；不填则输出到 场站数据/<场站>/step8-预测实验")
    parser.add_argument("--fan-rated-kw", type=float, default=4000.0, help="单台风机额定功率kW")
    parser.add_argument("--upper-factor", type=float, default=1.05, help="允许上限倍率，默认1.05")
    parser.add_argument("--only-usable", action="store_true", help="只统计 DATA_15MIN_USE_RESULT=可用 的行")
    args = parser.parse_args()

    input_file = Path(args.input_file) if args.input_file else build_default_input_file(args.base_dir, args.station_name, args.step7_dir_name)
    output_dir = Path(args.output_dir) if args.output_dir else build_default_output_dir(args.base_dir, args.station_name, DEFAULT_OUTPUT_DIR_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    upper_kw = args.fan_rated_kw * args.upper_factor

    print("=" * 88)
    print("检查 STEP7 每台风机功率范围")
    print("=" * 88)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"单机额定功率: {args.fan_rated_kw:g} kW")
    print(f"允许上限: {upper_kw:g} kW")

    df = read_csv_auto(input_file)
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    if args.only_usable:
        if "DATA_15MIN_USE_RESULT" not in df.columns:
            raise ValueError("指定了 --only-usable，但输入文件缺少 DATA_15MIN_USE_RESULT 列")
        before = len(df)
        df = df[df["DATA_15MIN_USE_RESULT"].astype(str).str.strip().eq("可用")].copy()
        print(f"只统计可用行: {before:,} -> {len(df):,}")

    fan_cols = extract_fan_cols(df.columns)
    if not fan_cols:
        raise ValueError("未识别到 FINAL_ACTIVE_POWER_#n_15MIN 风机功率列")

    rows = []
    for fan_no, col in fan_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()

        rows.append({
            "风机编号": fan_no,
            "列名": col,
            "总行数": len(s),
            "有效点数": int(s.notna().sum()),
            "缺失点数": int(s.isna().sum()),
            "缺失占比": float(s.isna().mean()) if len(s) else np.nan,
            "负值点数": int((s < 0).fillna(False).sum()),
            "零值点数": int((s == 0).fillna(False).sum()),
            "超过额定点数": int((s > args.fan_rated_kw).fillna(False).sum()),
            "超过额定占比": float((s > args.fan_rated_kw).fillna(False).mean()) if len(s) else np.nan,
            "超过上限点数": int((s > upper_kw).fillna(False).sum()),
            "超过上限占比": float((s > upper_kw).fillna(False).mean()) if len(s) else np.nan,
            "最小值kW": float(valid.min()) if len(valid) else np.nan,
            "P01kW": calc_quantile(s, 0.01),
            "P50kW": calc_quantile(s, 0.50),
            "P95kW": calc_quantile(s, 0.95),
            "P99kW": calc_quantile(s, 0.99),
            "P99_9kW": calc_quantile(s, 0.999),
            "最大值kW": float(valid.max()) if len(valid) else np.nan,
            "均值kW": float(valid.mean()) if len(valid) else np.nan,
            "标准差kW": float(valid.std()) if len(valid) else np.nan,
        })

    result = pd.DataFrame(rows)

    summary_rows = [
        {"统计项": "风机数量", "数值": len(fan_cols)},
        {"统计项": "统计行数", "数值": len(df)},
        {"统计项": "单机额定功率kW", "数值": args.fan_rated_kw},
        {"统计项": "允许上限倍率", "数值": args.upper_factor},
        {"统计项": "允许上限kW", "数值": upper_kw},
        {"统计项": "存在超过额定点的风机数", "数值": int((result["超过额定点数"] > 0).sum())},
        {"统计项": "存在超过上限点的风机数", "数值": int((result["超过上限点数"] > 0).sum())},
        {"统计项": "超过额定点数总和", "数值": int(result["超过额定点数"].sum())},
        {"统计项": "超过上限点数总和", "数值": int(result["超过上限点数"].sum())},
        {"统计项": "负值点数总和", "数值": int(result["负值点数"].sum())},
        {"统计项": "缺失点数总和", "数值": int(result["缺失点数"].sum())},
        {"统计项": "所有风机最大值kW", "数值": float(result["最大值kW"].max())},
        {"统计项": "所有风机P99最大值kW", "数值": float(result["P99kW"].max())},
        {"统计项": "所有风机P99_9最大值kW", "数值": float(result["P99_9kW"].max())},
    ]
    summary = pd.DataFrame(summary_rows)

    detail_file = output_dir / "fan_power_range_check.csv"
    summary_file = output_dir / "fan_power_range_summary.csv"

    result.to_csv(detail_file, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print("\n完成：")
    print(f"- 风机功率范围明细: {detail_file}")
    print(f"- 汇总统计: {summary_file}")

    print("\n超过上限最多的前20台风机：")
    display_cols = ["风机编号", "有效点数", "超过上限点数", "超过上限占比", "P99kW", "P99_9kW", "最大值kW"]
    print(result.sort_values("超过上限点数", ascending=False)[display_cols].head(20).to_string(index=False))

    print("\n汇总：")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
