#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 版本说明：
# - 已彻底移除二次拟合 / polyfit / polyval 逻辑。
# - 模型中心线全部来自“低功率分箱中位数 + 高功率分箱中位数”的一维线性插值。
# - 保留经验百分位带逻辑；Dash 根据 EMPIRICAL_QUANTILES 自动生成并显示分位带。
# - FAN_SEG 的 0~3MW 超低功率区间继续使用物理规则，不绘制普通模型带。
# - 硬异常同时合并“风机联合连续相同检测”和“站端 ACTIVE_POWER_STATION 连续重复检测”。
#
from __future__ import annotations

import os
import re
import argparse
import warnings
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go


# ============================================================
# 默认参数
# ============================================================
STATION_CT_COL = "ACTIVE_POWER_STATION"
FAN_POWER_COL_PATTERN = r"^ACTIVE_POWER_#\d+$"

LOW_BIN_WIDTH = 0.1
HIGH_BIN_WIDTH = 2.0
MIN_BIN_SAMPLES = 20
MODEL_LOSS_MIN = 0.0
MAX_SCATTER_POINTS = 12000
FAN_THRESHOLD_B = 10.0

# 负损耗是否直接作为异常
NEGATIVE_LOSS_AS_ANOMALY = False

# 残差粗筛
ENABLE_COARSE_FILTER = True
COARSE_FILTER_K = 5.0
COARSE_FILTER_MIN_SAMPLES = 15

# 全站额定功率硬筛选
STATION_RATED_POWER_MW = 300.0
RATED_POWER_MARGIN = 1.05

# 方案显示名（替代 AP / B）
SCHEME_STATION_SEG = "站端功率分段模型"
SCHEME_FAN_SEG = "风机总功率分段模型"

# Dash 展示配置
EMPIRICAL_QUANTILES = tuple([i / 100 for i in range(1, 11)] + [0.50] + [i / 100 for i in range(90, 100)])
# 经验分位带：仅剔除硬异常后计算。
# 默认输出 Q01~Q10、Q50、Q90~Q99，Dash 可自由选择下限/上限组合。
# 如只想输出少量列，也可以改成例如：(0.10, 0.50, 0.90)
SIGMA_BAND_DEFAULT = 2.0                   # 局部 sigma 正常区间：硬异常 + 粗筛后计算
ULTRA_LOW_POWER_MAX_MW = 3.0               # 风机总功率超低功率区间上限
ULTRA_LOW_POWER_LOSS_RATIO_MIN = 0.99      # 超低功率区间：L 下界 = 该比例 × FAN_SUM，上界 = FAN_SUM

# 站端功率连续重复硬异常规则
# 来源：连续相同检测/每列连续重复检测结果.csv
# 规则：字段名 == ACTIVE_POWER_STATION，重复值 > STATION_REPEAT_VALUE_MIN，持续长度 > STATION_REPEAT_MIN_LENGTH
ENABLE_STATION_REPEAT_HARD_ANOMALY = True
STATION_REPEAT_MIN_LENGTH = 3
STATION_REPEAT_VALUE_MIN = 0.0


# ============================================================
# 工具函数
# ============================================================
def sample_df(df: pd.DataFrame, n: int = MAX_SCATTER_POINTS, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).sort_values("timestamp")


def quantile_col_name(q: float) -> str:
    """把 0.1 / 0.05 这类分位数转换为输出表列名 Q10 / Q05。"""
    return f"Q{int(round(float(q) * 100)):02d}"


def quantile_label(q: float) -> str:
    """把 0.1 / 0.05 这类分位数转换为图例标签 P10 / P05。"""
    return f"P{int(round(float(q) * 100)):02d}"


def get_empirical_lower_options(quantiles: Tuple[float, ...] = EMPIRICAL_QUANTILES) -> List[dict]:
    """Dash 经验分位带下限选项：默认取小于 50% 的分位数，例如 P01~P10。"""
    qs = sorted({float(q) for q in quantiles if np.isfinite(q) and 0.0 < float(q) < 0.5})
    return [
        {
            "label": f"{quantile_label(q)} 下限",
            "value": quantile_col_name(q),
        }
        for q in qs
    ]


def get_empirical_upper_options(quantiles: Tuple[float, ...] = EMPIRICAL_QUANTILES) -> List[dict]:
    """Dash 经验分位带上限选项：默认取大于 50% 的分位数，例如 P90~P99。"""
    qs = sorted({float(q) for q in quantiles if np.isfinite(q) and 0.5 < float(q) < 1.0})
    return [
        {
            "label": f"{quantile_label(q)} 上限",
            "value": quantile_col_name(q),
        }
        for q in qs
    ]


def default_empirical_lower_values() -> List[str]:
    """默认经验分位带下限：优先 P10；没有 P10 时取最靠近 50% 的下限。"""
    opts = get_empirical_lower_options(EMPIRICAL_QUANTILES)
    values = [opt["value"] for opt in opts]
    if "Q10" in values:
        return ["Q10"]
    return [values[-1]] if values else []


def default_empirical_upper_values() -> List[str]:
    """默认经验分位带上限：优先 P90；没有 P90 时取最靠近 50% 的上限。"""
    opts = get_empirical_upper_options(EMPIRICAL_QUANTILES)
    values = [opt["value"] for opt in opts]
    if "Q90" in values:
        return ["Q90"]
    return [values[0]] if values else []


def normalize_dropdown_values(values) -> List[str]:
    """兼容 Dash multi=True 返回 None / str / list 的情况。"""
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(v) for v in values if v]


def empirical_band_label(low_col: str, high_col: str) -> str:
    """把 Q10 + Q90 转成 P10-P90。"""
    return f"{low_col.replace('Q', 'P')}-{high_col.replace('Q', 'P')}"


def build_exclude_mask(ts_np: np.ndarray, segs: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(ts_np), dtype=bool)
    if segs is None or len(segs) == 0:
        return mask

    t_int = ts_np.astype("datetime64[ns]").astype("int64")
    for _, row in segs.iterrows():
        s = pd.to_datetime(row["开始时间"]).to_datetime64().astype("datetime64[ns]").astype("int64")
        e = pd.to_datetime(row["结束时间"]).to_datetime64().astype("datetime64[ns]").astype("int64")
        mask |= (t_int >= s) & (t_int <= e)
    return mask


def _empty_bin_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "P_left", "P_right", "P_med", "L_med", "n",
        "base_bin_count", "bin_width", "min_bin_samples",
    ])


def _concat_parts(parts: List[pd.DataFrame]) -> pd.DataFrame:
    valid_parts = [x for x in parts if x is not None and len(x) > 0]
    if not valid_parts:
        return pd.DataFrame(columns=["_P", "_L"])
    return pd.concat(valid_parts, ignore_index=True)


def filter_model_xy(x_vals: np.ndarray, y_vals: np.ndarray, loss_min: float = MODEL_LOSS_MIN):
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.isfinite(loss_min):
        keep &= (y >= float(loss_min))
    return x[keep], y[keep]


def build_fixed_width_median_bins(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    bin_width: float,
    min_bin_samples: int,
    range_start: Optional[float] = None,
    range_end: Optional[float] = None,
    loss_min: float = MODEL_LOSS_MIN,
    max_bins: int = 20000,
) -> pd.DataFrame:
    x, y = filter_model_xy(x_vals, y_vals, loss_min=loss_min)
    if len(x) == 0:
        return _empty_bin_df()

    tmp = pd.DataFrame({"_P": x, "_L": y}).sort_values("_P").reset_index(drop=True)

    if range_start is None:
        range_start = float(np.floor(tmp["_P"].min() / bin_width) * bin_width)
    if range_end is None:
        range_end = float(np.ceil(tmp["_P"].max() / bin_width) * bin_width)

    range_start = float(range_start)
    range_end = float(range_end)
    if not np.isfinite(range_start) or not np.isfinite(range_end) or range_end <= range_start:
        return _empty_bin_df()

    span = range_end - range_start
    n_steps = int(np.ceil(span / bin_width))
    if n_steps <= 0:
        return _empty_bin_df()

    if n_steps > max_bins:
        print(f"⚠️ 分箱范围过大，原始分箱数 {n_steps:,} 超过上限 {max_bins:,}，将按分位数裁剪后分箱。")
        q_low = float(np.nanpercentile(tmp["_P"].values, 0.1))
        q_high = float(np.nanpercentile(tmp["_P"].values, 99.9))
        range_start = float(np.floor(q_low / bin_width) * bin_width)
        range_end = float(np.ceil(q_high / bin_width) * bin_width)
        span = range_end - range_start
        n_steps = int(np.ceil(span / bin_width))
        if n_steps > max_bins:
            range_end = range_start + max_bins * bin_width
            n_steps = max_bins
        tmp = tmp[(tmp["_P"] >= range_start) & (tmp["_P"] <= range_end)].copy()
        if len(tmp) == 0:
            return _empty_bin_df()

    edges = range_start + np.arange(n_steps + 1, dtype=float) * bin_width
    if edges[-1] < range_end:
        edges = np.append(edges, range_end)
    if len(edges) < 2:
        return _empty_bin_df()

    bin_idx = np.searchsorted(edges, tmp["_P"].values, side="left") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    tmp["_bin_idx"] = bin_idx
    grouped = {int(i): g[["_P", "_L"]].copy() for i, g in tmp.groupby("_bin_idx", observed=True)}

    merged_rows = []
    acc_left = None
    acc_right = None
    acc_parts: List[pd.DataFrame] = []
    acc_n = 0
    acc_base_bin_count = 0

    def flush_current() -> None:
        nonlocal acc_left, acc_right, acc_parts, acc_n, acc_base_bin_count, merged_rows
        if acc_left is None or acc_right is None:
            return
        merged_df = _concat_parts(acc_parts)
        if len(merged_df) == 0:
            acc_left = None
            acc_right = None
            acc_parts = []
            acc_n = 0
            acc_base_bin_count = 0
            return
        merged_rows.append({
            "P_left": float(acc_left),
            "P_right": float(acc_right),
            "P_med": float(merged_df["_P"].median()),
            "L_med": float(merged_df["_L"].median()),
            "n": int(len(merged_df)),
            "base_bin_count": int(acc_base_bin_count),
            "bin_width": float(bin_width),
            "min_bin_samples": int(min_bin_samples),
            "_samples": merged_df,
        })
        acc_left = None
        acc_right = None
        acc_parts = []
        acc_n = 0
        acc_base_bin_count = 0

    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        g = grouped.get(i, pd.DataFrame(columns=["_P", "_L"]))
        if acc_left is None:
            acc_left = left
        acc_right = right
        acc_base_bin_count += 1
        if len(g) > 0:
            acc_parts.append(g)
            acc_n += len(g)
        if acc_n >= min_bin_samples:
            flush_current()

    if acc_left is not None and acc_right is not None:
        tail_df = _concat_parts(acc_parts)
        if len(tail_df) > 0:
            if merged_rows and len(tail_df) < min_bin_samples:
                prev = merged_rows[-1]
                merged_df = pd.concat([prev["_samples"], tail_df], ignore_index=True)
                prev["P_right"] = float(acc_right)
                prev["P_med"] = float(merged_df["_P"].median())
                prev["L_med"] = float(merged_df["_L"].median())
                prev["n"] = int(len(merged_df))
                prev["base_bin_count"] = int(prev["base_bin_count"] + acc_base_bin_count)
                prev["_samples"] = merged_df
            else:
                flush_current()

    if not merged_rows:
        return _empty_bin_df()

    out = pd.DataFrame(merged_rows)
    if "_samples" in out.columns:
        out = out.drop(columns=["_samples"])
    return out.sort_values("P_left").reset_index(drop=True)



def build_fixed_width_quantile_bins(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    bin_width: float,
    min_bin_samples: int,
    quantiles: Tuple[float, ...] = EMPIRICAL_QUANTILES,
    range_start: Optional[float] = None,
    range_end: Optional[float] = None,
) -> pd.DataFrame:
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    if len(x) == 0:
        cols = ["P_left", "P_right", "P_med", "n", "base_bin_count", "bin_width", "min_bin_samples"]
        cols += [f"Q{int(round(q * 100)):02d}" for q in quantiles]
        return pd.DataFrame(columns=cols)

    tmp = pd.DataFrame({"_P": x, "_L": y}).sort_values("_P").reset_index(drop=True)

    if range_start is None:
        range_start = float(np.floor(tmp["_P"].min() / bin_width) * bin_width)
    if range_end is None:
        range_end = float(np.ceil(tmp["_P"].max() / bin_width) * bin_width)

    range_start = float(range_start)
    range_end = float(range_end)
    if not np.isfinite(range_start) or not np.isfinite(range_end) or range_end <= range_start:
        return pd.DataFrame()

    n_steps = int(np.ceil((range_end - range_start) / bin_width))
    if n_steps <= 0:
        return pd.DataFrame()

    edges = range_start + np.arange(n_steps + 1, dtype=float) * bin_width
    if edges[-1] < range_end:
        edges = np.append(edges, range_end)

    bin_idx = np.searchsorted(edges, tmp["_P"].values, side="left") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    tmp["_bin_idx"] = bin_idx
    grouped = {int(i): g[["_P", "_L"]].copy() for i, g in tmp.groupby("_bin_idx", observed=True)}

    merged_rows = []
    acc_left = None
    acc_right = None
    acc_parts: List[pd.DataFrame] = []
    acc_n = 0
    acc_base_bin_count = 0

    def flush_current() -> None:
        nonlocal acc_left, acc_right, acc_parts, acc_n, acc_base_bin_count, merged_rows
        if acc_left is None or acc_right is None:
            return
        merged_df = _concat_parts(acc_parts)
        if len(merged_df) == 0:
            acc_left = None
            acc_right = None
            acc_parts = []
            acc_n = 0
            acc_base_bin_count = 0
            return

        row = {
            "P_left": float(acc_left),
            "P_right": float(acc_right),
            "P_med": float(merged_df["_P"].median()),
            "n": int(len(merged_df)),
            "base_bin_count": int(acc_base_bin_count),
            "bin_width": float(bin_width),
            "min_bin_samples": int(min_bin_samples),
            "_samples": merged_df,
        }
        for q in quantiles:
            row[f"Q{int(round(q * 100)):02d}"] = float(np.nanquantile(merged_df["_L"].values, q))
        merged_rows.append(row)

        acc_left = None
        acc_right = None
        acc_parts = []
        acc_n = 0
        acc_base_bin_count = 0

    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        g = grouped.get(i, pd.DataFrame(columns=["_P", "_L"]))

        if acc_left is None:
            acc_left = left
        acc_right = right
        acc_base_bin_count += 1

        if len(g) > 0:
            acc_parts.append(g)
            acc_n += len(g)

        if acc_n >= min_bin_samples:
            flush_current()

    if acc_left is not None and acc_right is not None:
        tail_df = _concat_parts(acc_parts)
        if len(tail_df) > 0:
            if merged_rows and len(tail_df) < min_bin_samples:
                prev = merged_rows[-1]
                merged_df = pd.concat([prev["_samples"], tail_df], ignore_index=True)
                prev["P_right"] = float(acc_right)
                prev["P_med"] = float(merged_df["_P"].median())
                prev["n"] = int(len(merged_df))
                prev["base_bin_count"] = int(prev["base_bin_count"] + acc_base_bin_count)
                for q in quantiles:
                    prev[f"Q{int(round(q * 100)):02d}"] = float(np.nanquantile(merged_df["_L"].values, q))
                prev["_samples"] = merged_df
            else:
                flush_current()

    if not merged_rows:
        return pd.DataFrame()

    out = pd.DataFrame(merged_rows)
    if "_samples" in out.columns:
        out = out.drop(columns=["_samples"])
    return out.sort_values("P_left").reset_index(drop=True)


def build_piecewise_quantile_bins(
    df: pd.DataFrame,
    p_col: str,
    threshold: float,
    quantiles: Tuple[float, ...] = EMPIRICAL_QUANTILES,
    ultra_low_power_max_mw: Optional[float] = None,
    ultra_low_power_loss_ratio_min: float = ULTRA_LOW_POWER_LOSS_RATIO_MIN,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    low_df = df[np.isfinite(df[p_col]) & np.isfinite(df["L"]) & (df[p_col] > 0) & (df[p_col] <= threshold)].copy()
    high_df = df[np.isfinite(df[p_col]) & np.isfinite(df["L"]) & (df[p_col] > threshold)].copy()

    parts = []
    if len(low_df) > 0:
        low_bins = build_fixed_width_quantile_bins(
            x_vals=low_df[p_col].values,
            y_vals=low_df["L"].values,
            bin_width=LOW_BIN_WIDTH,
            min_bin_samples=MIN_BIN_SAMPLES,
            quantiles=quantiles,
            range_start=0.0,
            range_end=float(threshold),
        )
        if len(low_bins) > 0:
            low_bins["region"] = "low"
            parts.append(low_bins)

    if len(high_df) > 0:
        high_bins = build_fixed_width_quantile_bins(
            x_vals=high_df[p_col].values,
            y_vals=high_df["L"].values,
            bin_width=HIGH_BIN_WIDTH,
            min_bin_samples=MIN_BIN_SAMPLES,
            quantiles=quantiles,
            range_start=float(threshold),
        )
        if len(high_bins) > 0:
            high_bins["region"] = "high"
            parts.append(high_bins)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True).sort_values(["P_left", "P_right"]).reset_index(drop=True)

    # 风机总功率模型：超低功率区间直接使用物理规则经验带
    # 规则：0~ULTRA_LOW_POWER_MAX_MW 内，正常样本应满足 ULTRA_LOW_POWER_LOSS_RATIO_MIN * P <= L <= P
    # 因此直接覆盖经验分位上下界，避免 (0,0) 附近因按箱经验分位而误判
    if p_col == "FAN_eff" and ultra_low_power_max_mw is not None and len(out) > 0:
        q_cols = [f"Q{int(round(q * 100)):02d}" for q in quantiles]
        for idx, row in out.iterrows():
            if float(row["P_right"]) <= float(ultra_low_power_max_mw):
                p_med = float(row["P_med"])
                q_map = {}
                for q in quantiles:
                    if q <= 0.05:
                        q_map[f"Q{int(round(q * 100)):02d}"] = ultra_low_power_loss_ratio_min * p_med
                    elif q >= 0.95:
                        q_map[f"Q{int(round(q * 100)):02d}"] = p_med
                    else:
                        # 中位等中间分位取上下界线性插值
                        alpha = (q - 0.05) / max(0.95 - 0.05, 1e-9)
                        lower = ultra_low_power_loss_ratio_min * p_med
                        upper = p_med
                        q_map[f"Q{int(round(q * 100)):02d}"] = lower + alpha * (upper - lower)
                for col in q_cols:
                    if col in q_map:
                        out.at[idx, col] = q_map[col]
                out.at[idx, "region"] = "ultra_low"
                out.at[idx, "quantile_rule"] = "ultra_low_power_physical_rule"
            else:
                out.at[idx, "quantile_rule"] = "empirical_from_hard_anomaly_removed_only"
    elif len(out) > 0:
        out["quantile_rule"] = "empirical_from_hard_anomaly_removed_only"

    return out



def find_fan_power_columns(columns) -> List[str]:
    fan_cols = []
    pattern = re.compile(FAN_POWER_COL_PATTERN)
    for col in columns:
        col = str(col).strip()
        if pattern.match(col):
            fan_cols.append(col)
    return sorted(fan_cols, key=lambda x: int(x.split("#")[1]))


def find_station_main_csv(station_dir: str) -> str:
    skip_keywords = [
        "_duplicate_timestamps", "_missing_timestamps", "_process_summary", "_nulls",
        "_seconds_not_zero", "_invalid_timestamps", "_check_summary",
        "_timestamp_fixed", "_timestamp_changes",
    ]
    csv_files = []
    for file_name in os.listdir(station_dir):
        if not file_name.endswith(".csv"):
            continue
        if any(k in file_name for k in skip_keywords):
            continue
        csv_files.append(os.path.join(station_dir, file_name))
    if not csv_files:
        raise FileNotFoundError(f"未在场站目录中找到主 CSV：{station_dir}")
    if len(csv_files) > 1:
        raise RuntimeError(f"场站目录下检测到多个候选主 CSV，请手动确认：{csv_files}")
    return csv_files[0]


# ============================================================
# 数据加载
# ============================================================
def load_scada(station_dir: str, rated_power_mw: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 70)
    print("  加载场站主 CSV ...")
    print("=" * 70)

    main_csv = find_station_main_csv(station_dir)
    print(f"  读取主文件: {main_csv}")

    df = pd.read_csv(main_csv, parse_dates=["timestamp"])
    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)

    if "timestamp" not in df.columns:
        raise ValueError("主 CSV 中缺少 timestamp 列")
    if STATION_CT_COL not in df.columns:
        raise ValueError(f"主 CSV 中缺少 {STATION_CT_COL} 列")

    fan_cols = find_fan_power_columns(df.columns)
    if not fan_cols:
        raise ValueError("主 CSV 中未识别到风机有功列 ACTIVE_POWER_#数字")

    # 单位统一到 MW
    df["CT_eff"] = pd.to_numeric(df[STATION_CT_COL], errors="coerce").clip(lower=0)
    fan_df = df[fan_cols].apply(pd.to_numeric, errors="coerce").clip(lower=0)
    df["FAN_SUM"] = fan_df.sum(axis=1) / 1000.0

    # 额定功率硬筛选
    rated_limit = rated_power_mw * RATED_POWER_MARGIN
    df["is_bad_fan_sum"] = np.isfinite(df["FAN_SUM"]) & (df["FAN_SUM"] > rated_limit)
    df["is_bad_ct_eff"] = np.isfinite(df["CT_eff"]) & (df["CT_eff"] > rated_limit)
    df["is_physical_outlier"] = df["is_bad_fan_sum"] | df["is_bad_ct_eff"]

    bad_physical_df = df.loc[df["is_physical_outlier"], [
        "timestamp", STATION_CT_COL, "CT_eff", "FAN_SUM", "is_bad_fan_sum", "is_bad_ct_eff", "is_physical_outlier"
    ]].copy()

    scada = df.loc[~df["is_physical_outlier"], ["timestamp", STATION_CT_COL, "CT_eff", "FAN_SUM"]].copy()
    scada.sort_values("timestamp", inplace=True)
    scada.drop_duplicates(subset="timestamp", inplace=True)
    scada.reset_index(drop=True, inplace=True)

    print(f"  数据行数(硬筛前): {len(df):,}")
    print(f"  明显异常点数(超过额定上限): {len(bad_physical_df):,}")
    print(f"  数据行数(硬筛后): {len(scada):,}")
    print(f"  自动识别风机列数量: {len(fan_cols)}")
    print(f"  CT_eff 范围(MW): {scada['CT_eff'].min():.4f} ~ {scada['CT_eff'].max():.4f}")
    print(f"  FAN_SUM 范围(MW): {scada['FAN_SUM'].min():.4f} ~ {scada['FAN_SUM'].max():.4f}")
    print(f"  额定功率阈值(MW): {rated_limit:.4f}")

    return scada, bad_physical_df


def _read_csv_with_fallback(file_path: str) -> pd.DataFrame:
    """按常见中文编码读取 CSV。"""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取 CSV：{file_path}，最后一次错误：{last_err}")


def load_fan_joint_anomaly_segments(station_dir: str) -> pd.DataFrame:
    """读取风机联合连续相同检测结果。"""
    anomaly_file = os.path.join(station_dir, "风机联合连续相同检测", "联合重复值检测结果.xlsx")
    if not os.path.exists(anomaly_file):
        raise FileNotFoundError(f"未找到异常段文件：{anomaly_file}")

    anom_df = pd.read_excel(anomaly_file, engine="openpyxl")
    print(f"  ✅ 已读取联合重复值检测结果.xlsx（{len(anom_df):,} 条）")

    required_cols = {"开始时间", "结束时间"}
    missing = required_cols - set(anom_df.columns)
    if missing:
        raise ValueError(f"联合重复值检测结果.xlsx 缺少必要列：{missing}")

    out = anom_df[["开始时间", "结束时间"]].copy()
    out["开始时间"] = pd.to_datetime(out["开始时间"], errors="coerce")
    out["结束时间"] = pd.to_datetime(out["结束时间"], errors="coerce")
    out = out.dropna(subset=["开始时间", "结束时间"])
    out["hard_anomaly_source"] = "fan_joint_repeat"
    return out.drop_duplicates().reset_index(drop=True)


def load_station_power_repeat_segments(
    station_dir: str,
    min_length: int = STATION_REPEAT_MIN_LENGTH,
    value_min: float = STATION_REPEAT_VALUE_MIN,
    enable: bool = ENABLE_STATION_REPEAT_HARD_ANOMALY,
) -> pd.DataFrame:
    """
    读取站端 ACTIVE_POWER_STATION 连续重复检测结果，并转成硬异常时间段。

    规则：
    - 字段名 == ACTIVE_POWER_STATION
    - 重复值 > value_min
    - 持续长度 > min_length

    注意：这里按“严格大于”处理持续长度，因为需求是“持续长度大于 3（可设置）”。
    """
    empty = pd.DataFrame(columns=["开始时间", "结束时间", "hard_anomaly_source", "字段名", "重复值", "持续长度"])
    if not enable:
        print("  ℹ️ 站端功率连续重复硬异常规则未启用")
        return empty

    repeat_file = os.path.join(station_dir, "连续相同检测", "每列连续重复检测结果.csv")
    if not os.path.exists(repeat_file):
        print(f"  ⚠️ 未找到站端连续重复检测结果，跳过：{repeat_file}")
        return empty

    repeat_df = _read_csv_with_fallback(repeat_file)
    repeat_df.columns = repeat_df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    print(f"  ✅ 已读取每列连续重复检测结果.csv（{len(repeat_df):,} 条）")

    required_cols = {"字段名", "重复值", "开始时间", "结束时间", "持续长度"}
    missing = required_cols - set(repeat_df.columns)
    if missing:
        raise ValueError(f"每列连续重复检测结果.csv 缺少必要列：{missing}")

    work = repeat_df.copy()
    work["字段名"] = work["字段名"].astype(str).str.strip()
    work["重复值"] = pd.to_numeric(work["重复值"], errors="coerce")
    work["持续长度"] = pd.to_numeric(work["持续长度"], errors="coerce")
    work["开始时间"] = pd.to_datetime(work["开始时间"], errors="coerce")
    work["结束时间"] = pd.to_datetime(work["结束时间"], errors="coerce")

    mask = (
        (work["字段名"] == STATION_CT_COL) &
        np.isfinite(work["重复值"]) & (work["重复值"] > float(value_min)) &
        np.isfinite(work["持续长度"]) & (work["持续长度"] > int(min_length)) &
        work["开始时间"].notna() & work["结束时间"].notna()
    )
    out = work.loc[mask, ["字段名", "重复值", "开始时间", "结束时间", "持续长度"]].copy()
    out["hard_anomaly_source"] = "station_active_power_repeat"
    print(
        f"  ✅ 站端 {STATION_CT_COL} 连续重复硬异常段: {len(out):,} 条 "
        f"(重复值 > {value_min}, 持续长度 > {min_length})"
    )
    return out.drop_duplicates(subset=["开始时间", "结束时间", "字段名", "重复值", "持续长度"]).reset_index(drop=True)


def load_anomaly_segments(
    station_dir: str,
    station_repeat_min_length: int = STATION_REPEAT_MIN_LENGTH,
    station_repeat_value_min: float = STATION_REPEAT_VALUE_MIN,
    enable_station_repeat_hard_anomaly: bool = ENABLE_STATION_REPEAT_HARD_ANOMALY,
) -> pd.DataFrame:
    """合并所有硬异常段来源。"""
    print("\n  加载硬异常段结果 ...")
    parts = []

    fan_joint_df = load_fan_joint_anomaly_segments(station_dir)
    if len(fan_joint_df) > 0:
        parts.append(fan_joint_df)

    station_repeat_df = load_station_power_repeat_segments(
        station_dir=station_dir,
        min_length=station_repeat_min_length,
        value_min=station_repeat_value_min,
        enable=enable_station_repeat_hard_anomaly,
    )
    if len(station_repeat_df) > 0:
        parts.append(station_repeat_df)

    if not parts:
        return pd.DataFrame(columns=["开始时间", "结束时间", "hard_anomaly_source"])

    out = pd.concat(parts, ignore_index=True, sort=False)
    out["开始时间"] = pd.to_datetime(out["开始时间"], errors="coerce")
    out["结束时间"] = pd.to_datetime(out["结束时间"], errors="coerce")
    out = out.dropna(subset=["开始时间", "结束时间"])
    out = out.drop_duplicates(subset=["开始时间", "结束时间", "hard_anomaly_source"]).reset_index(drop=True)

    source_counts = out["hard_anomaly_source"].value_counts(dropna=False).to_dict()
    print(f"  ✅ 合并硬异常段总数: {len(out):,} 条 | 来源统计: {source_counts}")
    return out


# ============================================================
# 插值建模与预测
# ============================================================
def build_high_power_lookup_from_xy(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    bin_width: float = HIGH_BIN_WIDTH,
    min_bin_samples: int = MIN_BIN_SAMPLES,
    range_start: Optional[float] = None,
    range_end: Optional[float] = None,
    loss_min: float = MODEL_LOSS_MIN,
) -> Tuple[pd.DataFrame, int]:
    """
    构造高功率区分箱中位数插值点。

    说明：本脚本已完全舍弃二次拟合函数，高功率区和低功率区一样，
    都只通过分箱中位数点 + 一维线性插值来计算模型中心线。
    """
    x, y = filter_model_xy(x_vals, y_vals, loss_min=loss_min)
    model_samples = len(x)
    if model_samples < 5:
        return _empty_bin_df(), model_samples

    bins_df = build_fixed_width_median_bins(
        x_vals=x, y_vals=y, bin_width=bin_width, min_bin_samples=min_bin_samples,
        range_start=range_start, range_end=range_end, loss_min=loss_min,
    )
    return bins_df, model_samples


def build_low_power_lookup(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    threshold: float,
    bin_width: float = LOW_BIN_WIDTH,
    min_bin_samples: int = MIN_BIN_SAMPLES,
    loss_min: float = MODEL_LOSS_MIN,
):
    x, y = filter_model_xy(x_vals, y_vals, loss_min=loss_min)
    mask = (x > 0) & (x <= threshold)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return _empty_bin_df()
    return build_fixed_width_median_bins(
        x_vals=x, y_vals=y, bin_width=bin_width, min_bin_samples=min_bin_samples,
        range_start=0.0, range_end=float(threshold), loss_min=loss_min,
    )


def _prepare_interp_point_frame(low_points_df: Optional[pd.DataFrame], high_points_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    parts = []
    if low_points_df is not None and len(low_points_df) > 0:
        low = low_points_df.copy()
        low["region"] = "low"
        parts.append(low)
    if high_points_df is not None and len(high_points_df) > 0:
        high = high_points_df.copy()
        high["region"] = "high"
        parts.append(high)
    if not parts:
        return pd.DataFrame(columns=["region", "P_left", "P_right", "P_med", "L_med"])
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.dropna(subset=["P_med"]).sort_values(["P_med", "P_left", "P_right"]).reset_index(drop=True)
    if len(out) > 1 and out["P_med"].duplicated().any():
        if "n" in out.columns:
            out = out.sort_values(["P_med", "n"], ascending=[True, False]).drop_duplicates(subset=["P_med"], keep="first")
        else:
            agg_cols = {c: "first" for c in out.columns if c not in ["P_med", "L_med"]}
            agg_cols["L_med"] = "mean"
            out = out.groupby("P_med", as_index=False).agg(agg_cols)
    return out.sort_values(["P_med", "P_left", "P_right"]).reset_index(drop=True)


def interp_1d_with_hold(values: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    out = np.full(len(values), np.nan)
    keep = np.isfinite(xp) & np.isfinite(fp)
    xp = xp[keep]
    fp = fp[keep]
    if len(xp) == 0:
        return out
    if len(xp) == 1:
        out[np.isfinite(values)] = fp[0]
        return out
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    valid_values = np.isfinite(values)

    if np.any(valid_values):
        out[valid_values] = np.interp(values[valid_values], xp, fp, left=fp[0], right=fp[-1])
    return out


def predict_value_from_point_table(
    p_vals: np.ndarray,
    point_df: Optional[pd.DataFrame],
    value_col: str,
) -> np.ndarray:
    if point_df is None or len(point_df) == 0 or value_col not in point_df.columns:
        return np.full(len(np.asarray(p_vals, dtype=float)), np.nan)
    tmp = point_df[["P_med", value_col]].copy()
    tmp = tmp[np.isfinite(tmp["P_med"]) & np.isfinite(tmp[value_col])].sort_values("P_med")
    if len(tmp) == 0:
        return np.full(len(np.asarray(p_vals, dtype=float)), np.nan)
    return interp_1d_with_hold(np.asarray(p_vals, dtype=float), tmp["P_med"].values, tmp[value_col].values)


def predict_sigma_from_table(p_vals: np.ndarray, sigma_table_df: Optional[pd.DataFrame]) -> np.ndarray:
    if sigma_table_df is None or len(sigma_table_df) == 0:
        return np.full(len(np.asarray(p_vals, dtype=float)), np.nan)
    tmp = sigma_table_df[["P_med", "sigma_local"]].copy()
    tmp = tmp[np.isfinite(tmp["P_med"]) & np.isfinite(tmp["sigma_local"])].sort_values("P_med")
    if len(tmp) == 0:
        return np.full(len(np.asarray(p_vals, dtype=float)), np.nan)
    return interp_1d_with_hold(np.asarray(p_vals, dtype=float), tmp["P_med"].values, tmp["sigma_local"].values)


def predict_loss_from_interp_points(p_vals: np.ndarray, low_points_df: Optional[pd.DataFrame], high_points_df: Optional[pd.DataFrame]) -> np.ndarray:
    point_df = _prepare_interp_point_frame(low_points_df, high_points_df)
    if len(point_df) == 0:
        return np.full(len(np.asarray(p_vals, dtype=float)), np.nan)
    return interp_1d_with_hold(np.asarray(p_vals, dtype=float), point_df["P_med"].values, point_df["L_med"].values)


def build_local_sigma_table_piecewise(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    threshold: float,
    low_lookup_df: Optional[pd.DataFrame],
    high_bins_df: Optional[pd.DataFrame],
    loss_min: float = MODEL_LOSS_MIN,
):
    x, y = filter_model_xy(x_vals, y_vals, loss_min=loss_min)
    empty_cols = [
        "region", "bin_source", "sigma_switch_threshold", "P_left", "P_right", "P_med",
        "sigma_local", "n", "base_bin_count", "bin_width", "min_bin_samples",
    ]
    if len(x) < 10:
        return pd.DataFrame(columns=empty_cols)

    y_hat = predict_loss_from_interp_points(p_vals=x, low_points_df=low_lookup_df, high_points_df=high_bins_df)
    tmp = pd.DataFrame({"_P": x, "_L": y, "_L_hat": y_hat})
    tmp = tmp[np.isfinite(tmp["_P"]) & np.isfinite(tmp["_L_hat"])].copy()
    if len(tmp) < 5:
        return pd.DataFrame(columns=empty_cols)
    tmp["_resid"] = tmp["_L"] - tmp["_L_hat"]

    rows = []

    def append_sigma_rows(ref_df: Optional[pd.DataFrame], region: str, bin_source: str):
        if ref_df is None or len(ref_df) == 0:
            return
        ref_df_sorted = ref_df.sort_values("P_left").reset_index(drop=True)
        for _, row in ref_df_sorted.iterrows():
            left = float(row["P_left"])
            right = float(row["P_right"])
            p_med = float(row["P_med"])
            g = tmp[(tmp["_P"] > left) & (tmp["_P"] <= right)]
            rows.append({
                "region": region,
                "bin_source": bin_source,
                "sigma_switch_threshold": float(threshold) if np.isfinite(threshold) else np.nan,
                "P_left": left,
                "P_right": right,
                "P_med": p_med,
                "sigma_local": float(np.std(g["_resid"].values)) if len(g) >= 3 else np.nan,
                "n": int(len(g)),
                "base_bin_count": int(row.get("base_bin_count", np.nan)) if pd.notna(row.get("base_bin_count", np.nan)) else np.nan,
                "bin_width": float(row.get("bin_width", np.nan)) if pd.notna(row.get("bin_width", np.nan)) else np.nan,
                "min_bin_samples": int(row.get("min_bin_samples", np.nan)) if pd.notna(row.get("min_bin_samples", np.nan)) else np.nan,
            })

    append_sigma_rows(low_lookup_df, region="low", bin_source="low_lookup")
    append_sigma_rows(high_bins_df, region="high", bin_source="high_lookup")
    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows).sort_values(["P_left", "P_right"]).reset_index(drop=True)


# ============================================================
# 粗筛
# ============================================================
def coarse_filter_by_residual_mad(
    df: pd.DataFrame,
    p_col: str,
    low_lookup_df: Optional[pd.DataFrame],
    high_bins_df: Optional[pd.DataFrame],
    min_samples: int = COARSE_FILTER_MIN_SAMPLES,
    k: float = COARSE_FILTER_K,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["L_hat_init"] = predict_loss_from_interp_points(
        p_vals=work[p_col].values.astype(float),
        low_points_df=low_lookup_df,
        high_points_df=high_bins_df,
    )
    work["residual_init"] = work["L"] - work["L_hat_init"]
    work["coarse_removed"] = False
    work["coarse_reason"] = ""

    ref_parts = []
    if low_lookup_df is not None and len(low_lookup_df) > 0:
        low = low_lookup_df[["P_left", "P_right", "P_med"]].copy()
        low["region"] = "low"
        ref_parts.append(low)
    if high_bins_df is not None and len(high_bins_df) > 0:
        high = high_bins_df[["P_left", "P_right", "P_med"]].copy()
        high["region"] = "high"
        ref_parts.append(high)
    if not ref_parts:
        return work.copy(), work.iloc[0:0].copy(), pd.DataFrame()

    ref_df = pd.concat(ref_parts, ignore_index=True).sort_values(["P_left", "P_right"]).reset_index(drop=True)
    summary_rows = []

    for _, row in ref_df.iterrows():
        left = float(row["P_left"])
        right = float(row["P_right"])
        region = row["region"]
        mask = (work[p_col] > left) & (work[p_col] <= right) & np.isfinite(work["residual_init"])
        g = work.loc[mask].copy()

        if len(g) < min_samples:
            summary_rows.append({
                "region": region, "P_left": left, "P_right": right,
                "n_before": len(g), "n_removed": 0, "n_after": len(g),
                "median_resid": np.nan, "mad": np.nan, "threshold": np.nan,
            })
            continue

        med = float(np.median(g["residual_init"]))
        mad = float(np.median(np.abs(g["residual_init"] - med)))
        if not np.isfinite(mad) or mad == 0:
            summary_rows.append({
                "region": region, "P_left": left, "P_right": right,
                "n_before": len(g), "n_removed": 0, "n_after": len(g),
                "median_resid": med, "mad": mad, "threshold": np.nan,
            })
            continue

        robust_sigma = 1.4826 * mad
        threshold_val = k * robust_sigma
        remove_mask = np.abs(g["residual_init"] - med) > threshold_val
        removed_index = g.loc[remove_mask].index

        work.loc[removed_index, "coarse_removed"] = True
        work.loc[removed_index, "coarse_reason"] = f"{region}_bin_residual_mad"

        summary_rows.append({
            "region": region, "P_left": left, "P_right": right,
            "n_before": len(g), "n_removed": int(remove_mask.sum()),
            "n_after": int((~remove_mask).sum()), "median_resid": med,
            "mad": mad, "threshold": threshold_val,
        })

    kept_df = work.loc[~work["coarse_removed"]].copy()
    removed_df = work.loc[work["coarse_removed"]].copy()
    summary_df = pd.DataFrame(summary_rows)
    return kept_df, removed_df, summary_df


def build_coarse_filter_impact_summary(raw_df: pd.DataFrame, kept_df: pd.DataFrame, removed_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "raw_normal_samples": len(raw_df),
        "coarse_kept_samples": len(kept_df),
        "coarse_removed_samples": len(removed_df),
        "coarse_removed_ratio": len(removed_df) / len(raw_df) if len(raw_df) > 0 else np.nan,
    }])


# ============================================================
# 方案建模（仅保留两个分段插值方案）
# ============================================================
def fit_segmented_model(normal_df: pd.DataFrame, fan_threshold_b: float, mode: str, enable_coarse_filter: bool):
    assert mode in ["STATION_SEG", "FAN_SEG"]

    fit_df_low = normal_df[
        np.isfinite(normal_df["FAN_eff"]) &
        np.isfinite(normal_df["L"]) &
        (normal_df["FAN_eff"] > 0) &
        (normal_df["FAN_eff"] <= fan_threshold_b)
    ].copy()

    fit_df_high = normal_df[
        np.isfinite(normal_df["FAN_eff"]) &
        np.isfinite(normal_df["L"]) &
        (normal_df["FAN_eff"] > fan_threshold_b)
    ].copy()

    if mode == "STATION_SEG":
        low_threshold = float(np.nanmax(fit_df_low["CT_eff"].values)) if len(fit_df_low) else fan_threshold_b
        low_lookup = build_low_power_lookup(
            x_vals=fit_df_low["CT_eff"].values,
            y_vals=fit_df_low["L"].values,
            threshold=low_threshold,
            bin_width=LOW_BIN_WIDTH,
            min_bin_samples=MIN_BIN_SAMPLES,
        )
        bins, model_samples = build_high_power_lookup_from_xy(
            fit_df_high["CT_eff"].values,
            fit_df_high["L"].values,
            bin_width=HIGH_BIN_WIDTH,
            min_bin_samples=MIN_BIN_SAMPLES,
        )
        threshold = float(low_lookup["P_right"].max()) if len(low_lookup) > 0 else low_threshold
        p_col = "CT_eff"
    else:
        low_lookup = build_low_power_lookup(
            x_vals=fit_df_low["FAN_eff"].values,
            y_vals=fit_df_low["L"].values,
            threshold=fan_threshold_b,
            bin_width=LOW_BIN_WIDTH,
            min_bin_samples=MIN_BIN_SAMPLES,
        )
        bins, model_samples = build_high_power_lookup_from_xy(
            fit_df_high["FAN_eff"].values,
            fit_df_high["L"].values,
            bin_width=HIGH_BIN_WIDTH,
            min_bin_samples=MIN_BIN_SAMPLES,
            range_start=fan_threshold_b,
        )
        threshold = fan_threshold_b
        p_col = "FAN_eff"

    sigma_table = build_local_sigma_table_piecewise(
        x_vals=np.concatenate([fit_df_low[p_col].values, fit_df_high[p_col].values]) if (len(fit_df_low) + len(fit_df_high)) > 0 else np.array([]),
        y_vals=np.concatenate([fit_df_low["L"].values, fit_df_high["L"].values]) if (len(fit_df_low) + len(fit_df_high)) > 0 else np.array([]),
        threshold=threshold,
        low_lookup_df=low_lookup,
        high_bins_df=bins,
    )

    coarse_removed_df = pd.DataFrame()
    coarse_summary_df = pd.DataFrame()
    coarse_impact_df = pd.DataFrame()

    if enable_coarse_filter:
        base_train_df = pd.concat([fit_df_low, fit_df_high], ignore_index=True)
        coarse_kept_df, coarse_removed_df, coarse_summary_df = coarse_filter_by_residual_mad(
            df=base_train_df,
            p_col=p_col,
            low_lookup_df=low_lookup,
            high_bins_df=bins,
            min_samples=COARSE_FILTER_MIN_SAMPLES,
            k=COARSE_FILTER_K,
        )
        coarse_impact_df = build_coarse_filter_impact_summary(base_train_df, coarse_kept_df, coarse_removed_df)

        if len(coarse_removed_df) > 0:
            fit_df_low = coarse_kept_df[(coarse_kept_df["FAN_eff"] > 0) & (coarse_kept_df["FAN_eff"] <= fan_threshold_b)].copy()
            fit_df_high = coarse_kept_df[coarse_kept_df["FAN_eff"] > fan_threshold_b].copy()

            if mode == "STATION_SEG":
                low_threshold = float(np.nanmax(fit_df_low["CT_eff"].values)) if len(fit_df_low) else fan_threshold_b
                low_lookup = build_low_power_lookup(
                    x_vals=fit_df_low["CT_eff"].values,
                    y_vals=fit_df_low["L"].values,
                    threshold=low_threshold,
                    bin_width=LOW_BIN_WIDTH,
                    min_bin_samples=MIN_BIN_SAMPLES,
                )
                bins, model_samples = build_high_power_lookup_from_xy(
                    fit_df_high["CT_eff"].values,
                    fit_df_high["L"].values,
                    bin_width=HIGH_BIN_WIDTH,
                    min_bin_samples=MIN_BIN_SAMPLES,
                )
                threshold = float(low_lookup["P_right"].max()) if len(low_lookup) > 0 else low_threshold
                p_col = "CT_eff"
            else:
                low_lookup = build_low_power_lookup(
                    x_vals=fit_df_low["FAN_eff"].values,
                    y_vals=fit_df_low["L"].values,
                    threshold=fan_threshold_b,
                    bin_width=LOW_BIN_WIDTH,
                    min_bin_samples=MIN_BIN_SAMPLES,
                )
                bins, model_samples = build_high_power_lookup_from_xy(
                    fit_df_high["FAN_eff"].values,
                    fit_df_high["L"].values,
                    bin_width=HIGH_BIN_WIDTH,
                    min_bin_samples=MIN_BIN_SAMPLES,
                    range_start=fan_threshold_b,
                )
                threshold = fan_threshold_b
                p_col = "FAN_eff"

            sigma_table = build_local_sigma_table_piecewise(
                x_vals=np.concatenate([fit_df_low[p_col].values, fit_df_high[p_col].values]) if (len(fit_df_low) + len(fit_df_high)) > 0 else np.array([]),
                y_vals=np.concatenate([fit_df_low["L"].values, fit_df_high["L"].values]) if (len(fit_df_low) + len(fit_df_high)) > 0 else np.array([]),
                threshold=threshold,
                low_lookup_df=low_lookup,
                        high_bins_df=bins,
            )

    return {
        "fit_df_low": fit_df_low,
        "fit_df_high": fit_df_high,
        "low_lookup": low_lookup,
        "bins": bins,
        "model_samples": model_samples,
        "threshold": threshold,
        "sigma_table": sigma_table,
        "coarse_removed_df": coarse_removed_df,
        "coarse_filter_summary_df": coarse_summary_df,
        "coarse_filter_impact_df": coarse_impact_df,
    }


# ============================================================
# 整站准备数据
# ============================================================
def prepare_station_data(
    scada: pd.DataFrame,
    anom_df: pd.DataFrame,
    station_name: str,
    fan_threshold_b: float,
    bad_physical_df: Optional[pd.DataFrame] = None,
    rated_power_mw: float = STATION_RATED_POWER_MW,
    station_repeat_min_length: int = STATION_REPEAT_MIN_LENGTH,
    station_repeat_value_min: float = STATION_REPEAT_VALUE_MIN,
    enable_station_repeat_hard_anomaly: bool = ENABLE_STATION_REPEAT_HARD_ANOMALY,
) -> dict:
    print("\n" + "=" * 70)
    print("  开始插值建模并准备 Dash 数据")
    print("=" * 70)
    print(f"  分箱设置：低功率 {LOW_BIN_WIDTH} MW，高功率 {HIGH_BIN_WIDTH} MW，最小样本数 {MIN_BIN_SAMPLES}")

    ts_np = scada["timestamp"].values.astype("datetime64[ns]")
    segs = anom_df[["开始时间", "结束时间"]].drop_duplicates().reset_index(drop=True)
    exc_mask = build_exclude_mask(ts_np, segs)

    if "hard_anomaly_source" in anom_df.columns:
        fan_joint_segs = anom_df.loc[anom_df["hard_anomaly_source"] == "fan_joint_repeat", ["开始时间", "结束时间"]].drop_duplicates().reset_index(drop=True)
        station_repeat_segs = anom_df.loc[anom_df["hard_anomaly_source"] == "station_active_power_repeat", ["开始时间", "结束时间"]].drop_duplicates().reset_index(drop=True)
    else:
        fan_joint_segs = pd.DataFrame(columns=["开始时间", "结束时间"])
        station_repeat_segs = pd.DataFrame(columns=["开始时间", "结束时间"])
    fan_joint_mask = build_exclude_mask(ts_np, fan_joint_segs)
    station_repeat_mask = build_exclude_mask(ts_np, station_repeat_segs)

    station_df = scada[["timestamp", STATION_CT_COL, "CT_eff", "FAN_SUM"]].copy()
    station_df.rename(columns={STATION_CT_COL: "CT"}, inplace=True)
    station_df["is_anomaly"] = exc_mask
    station_df["is_fan_joint_repeat_anomaly"] = fan_joint_mask
    station_df["is_station_power_repeat_anomaly"] = station_repeat_mask
    station_df["FAN_eff"] = station_df["FAN_SUM"]
    station_df["L"] = station_df["FAN_eff"] - station_df["CT_eff"]
    station_df["is_negative_loss"] = np.isfinite(station_df["L"]) & (station_df["L"] < MODEL_LOSS_MIN)

    if NEGATIVE_LOSS_AS_ANOMALY:
        station_df["is_model_abnormal"] = station_df["is_anomaly"] | station_df["is_negative_loss"]
    else:
        station_df["is_model_abnormal"] = station_df["is_anomaly"]

    normal_df = station_df[(~station_df["is_model_abnormal"])].copy()
    anomaly_df = station_df[(station_df["is_model_abnormal"])].copy()
    negative_loss_df = station_df[(station_df["is_negative_loss"])].copy()

    station_seg = fit_segmented_model(normal_df, fan_threshold_b, mode="STATION_SEG", enable_coarse_filter=ENABLE_COARSE_FILTER)
    fan_seg = fit_segmented_model(normal_df, fan_threshold_b, mode="FAN_SEG", enable_coarse_filter=ENABLE_COARSE_FILTER)

    # 经验分位带：只剔除硬异常后计算，不剔除粗筛点
    station_seg["empirical_quantile_bins"] = build_piecewise_quantile_bins(
        df=normal_df,
        p_col="CT_eff",
        threshold=station_seg["threshold"],
        quantiles=EMPIRICAL_QUANTILES,
    )
    fan_seg["empirical_quantile_bins"] = build_piecewise_quantile_bins(
        df=normal_df,
        p_col="FAN_eff",
        threshold=fan_seg["threshold"],
        quantiles=EMPIRICAL_QUANTILES,
        ultra_low_power_max_mw=ULTRA_LOW_POWER_MAX_MW,
        ultra_low_power_loss_ratio_min=ULTRA_LOW_POWER_LOSS_RATIO_MIN,
    )

    ultra_low_bin_count = 0
    if len(fan_seg["empirical_quantile_bins"]) > 0 and "quantile_rule" in fan_seg["empirical_quantile_bins"].columns:
        ultra_low_bin_count = int((fan_seg["empirical_quantile_bins"]["quantile_rule"] == "ultra_low_power_physical_rule").sum())

    print(f"  硬异常段数(合并后): {len(segs):,}")
    print(f"  风机联合连续相同硬异常段数: {len(fan_joint_segs):,} | 覆盖时刻: {int(fan_joint_mask.sum()):,}")
    print(f"  站端功率连续重复硬异常段数: {len(station_repeat_segs):,} | 覆盖时刻: {int(station_repeat_mask.sum()):,} | 规则: 重复值 > {station_repeat_value_min}, 持续长度 > {station_repeat_min_length}, 启用={enable_station_repeat_hard_anomaly}")
    print(f"  明显异常点数(超过额定上限): {0 if bad_physical_df is None else len(bad_physical_df):,}")
    print(f"  正常样本(仅剔除硬异常后): {len(normal_df):,}")
    print(f"  异常样本: {len(anomaly_df):,}")
    print(f"  负损耗样本(仅标记): {len(negative_loss_df):,}")
    print(f"  {SCHEME_STATION_SEG} 建模样本数: {station_seg['model_samples']:,} | 低功率箱数: {len(station_seg['low_lookup']):,} | 高功率箱数: {len(station_seg['bins']):,}")
    print(f"  {SCHEME_FAN_SEG} 建模样本数: {fan_seg['model_samples']:,} | 低功率箱数: {len(fan_seg['low_lookup']):,} | 高功率箱数: {len(fan_seg['bins']):,}")
    print(f"  {SCHEME_FAN_SEG} 超低功率经验带规则: 0~{ULTRA_LOW_POWER_MAX_MW} MW 内，{ULTRA_LOW_POWER_LOSS_RATIO_MIN:.2%}×P <= L <= P | 覆盖箱数: {ultra_low_bin_count}")
    if ENABLE_COARSE_FILTER:
        print(f"  {SCHEME_STATION_SEG} 粗筛删除点数: {len(station_seg['coarse_removed_df']):,}")
        print(f"  {SCHEME_FAN_SEG} 粗筛删除点数: {len(fan_seg['coarse_removed_df']):,}")

    return {
        "station_name": station_name,
        "station_df": station_df,
        "normal_df": normal_df,
        "anomaly_df": anomaly_df,
        "negative_loss_df": negative_loss_df,
        "bad_physical_df": bad_physical_df if bad_physical_df is not None else pd.DataFrame(),
        "fan_threshold_b": fan_threshold_b,
        "rated_power_mw": rated_power_mw,
        "station_repeat_min_length": station_repeat_min_length,
        "station_repeat_value_min": station_repeat_value_min,
        "enable_station_repeat_hard_anomaly": enable_station_repeat_hard_anomaly,
        "hard_anomaly_segments": anom_df.copy(),
        "fan_joint_hard_anomaly_segments": fan_joint_segs.copy(),
        "station_power_repeat_hard_anomaly_segments": station_repeat_segs.copy(),
        "scatter_normal": sample_df(normal_df),
        "scatter_anomaly": sample_df(anomaly_df),

        "station_seg": station_seg,
        "fan_seg": fan_seg,
    }


def build_interp_points_output(
    station_code: str,
    station_name: str,
    scheme_code: str,
    scheme_name: str,
    power_col: str,
    threshold_axis: str,
    switch_threshold: float,
    fan_threshold_b: float,
    low_lookup_df: Optional[pd.DataFrame],
    high_bins_df: Optional[pd.DataFrame],
    sigma_table_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    parts = []
    if low_lookup_df is not None and len(low_lookup_df) > 0:
        low = low_lookup_df.copy()
        low["region"] = "low"
        low["bin_source"] = "low_lookup"
        parts.append(low)
    if high_bins_df is not None and len(high_bins_df) > 0:
        high = high_bins_df.copy()
        high["region"] = "high"
        high["bin_source"] = "high_lookup"
        parts.append(high)
    if not parts:
        return pd.DataFrame()

    point_df = pd.concat(parts, ignore_index=True, sort=False)
    point_df = point_df.sort_values(["P_med", "P_left", "P_right"]).reset_index(drop=True)

    sigma_cols = ["region", "P_left", "P_right", "P_med", "sigma_local", "n"]
    if sigma_table_df is not None and len(sigma_table_df) > 0:
        sigma_use = sigma_table_df[sigma_cols].copy().rename(columns={"n": "sigma_sample_n"})
        point_df = point_df.merge(sigma_use, on=["region", "P_left", "P_right", "P_med"], how="left")
    else:
        point_df["sigma_local"] = np.nan
        point_df["sigma_sample_n"] = np.nan

    point_df["var_local"] = point_df["sigma_local"].astype(float) ** 2
    point_df["station_code"] = station_code
    point_df["station_name"] = station_name
    point_df["scheme_code"] = scheme_code
    point_df["scheme_name"] = scheme_name
    point_df["power_col_for_prediction"] = power_col
    point_df["power_threshold_axis"] = threshold_axis
    point_df["power_switch_threshold"] = switch_threshold
    point_df["fan_threshold_b"] = fan_threshold_b
    point_df["interp_order"] = np.arange(1, len(point_df) + 1)
    return point_df


# ============================================================
# 保存结果
# ============================================================
def save_fit_results(results: dict, output_dir: str):
    fit_summary_csv = os.path.join(output_dir, "fit_model_summary.csv")
    fit_sigma_bins_csv = os.path.join(output_dir, "fit_model_sigma_bins.csv")
    fit_low_lookup_csv = os.path.join(output_dir, "fit_model_low_power_lookup.csv")
    fit_high_bins_csv = os.path.join(output_dir, "fit_model_high_power_bins.csv")
    fit_interp_points_csv = os.path.join(output_dir, "fit_model_interp_points.csv")
    fit_empirical_quantile_csv = os.path.join(output_dir, "fit_empirical_quantile_bins.csv")

    station_code = os.path.basename(os.path.normpath(output_dir))
    station_name = results["station_name"]

    scheme_defs = [
        ("STATION_SEG", SCHEME_STATION_SEG, results["station_seg"], "CT_eff", "CT_eff"),
        ("FAN_SEG", SCHEME_FAN_SEG, results["fan_seg"], "FAN_eff", "FAN_eff"),
    ]

    summary_rows = []
    sigma_rows = []
    low_rows = []
    high_rows = []
    interp_rows = []
    empirical_rows = []

    for scheme_code, scheme_name, scheme_res, power_col, threshold_axis in scheme_defs:
        low_lookup = scheme_res["low_lookup"]
        high_bins = scheme_res["bins"]
        sigma_table = scheme_res["sigma_table"]
        switch_threshold = scheme_res["threshold"]

        summary_rows.append({
            "station_code": station_code,
            "station_name": station_name,
            "scheme_code": scheme_code,
            "scheme_name": scheme_name,
            "power_col_for_prediction": power_col,
            "power_threshold_axis": threshold_axis,
            "power_switch_threshold": switch_threshold,
            "fan_threshold_b": results["fan_threshold_b"],
            "negative_loss_as_anomaly": NEGATIVE_LOSS_AS_ANOMALY,
            "coarse_filter_enabled": ENABLE_COARSE_FILTER,
            "coarse_filter_k": COARSE_FILTER_K,
            "rated_power_mw": results["rated_power_mw"],
            "rated_power_margin": RATED_POWER_MARGIN,
            "hard_removed_physical_outliers": len(results["bad_physical_df"]),
            "fan_joint_hard_anomaly_segments": len(results.get("fan_joint_hard_anomaly_segments", pd.DataFrame())),
            "station_power_repeat_hard_anomaly_segments": len(results.get("station_power_repeat_hard_anomaly_segments", pd.DataFrame())),
            "station_power_repeat_rule_enabled": results.get("enable_station_repeat_hard_anomaly", ENABLE_STATION_REPEAT_HARD_ANOMALY),
            "station_power_repeat_min_length": results.get("station_repeat_min_length", STATION_REPEAT_MIN_LENGTH),
            "station_power_repeat_value_min": results.get("station_repeat_value_min", STATION_REPEAT_VALUE_MIN),
            "coarse_removed_samples": len(scheme_res["coarse_removed_df"]),
            "normal_samples": len(results["normal_df"]),
            "anomaly_samples": len(results["anomaly_df"]),
            "negative_loss_samples": len(results["negative_loss_df"]),
            "model_type": "median_bin_interpolation",
            "model_samples": scheme_res["model_samples"],
            "sigma_bins_count": len(sigma_table) if sigma_table is not None else 0,
            "low_bins_count": len(low_lookup) if low_lookup is not None else 0,
            "high_bins_count": len(high_bins) if high_bins is not None else 0,
        })

        if sigma_table is not None and len(sigma_table) > 0:
            tmp_sigma = sigma_table.copy()
            tmp_sigma["station_code"] = station_code
            tmp_sigma["station_name"] = station_name
            tmp_sigma["scheme_code"] = scheme_code
            tmp_sigma["scheme_name"] = scheme_name
            tmp_sigma["power_col_for_prediction"] = power_col
            tmp_sigma["power_threshold_axis"] = threshold_axis
            tmp_sigma["power_switch_threshold"] = switch_threshold
            tmp_sigma["fan_threshold_b"] = results["fan_threshold_b"]
            sigma_rows.append(tmp_sigma)

        if low_lookup is not None and len(low_lookup) > 0:
            tmp_low = low_lookup.copy()
            tmp_low["station_code"] = station_code
            tmp_low["station_name"] = station_name
            tmp_low["scheme_code"] = scheme_code
            tmp_low["scheme_name"] = scheme_name
            tmp_low["power_col_for_prediction"] = power_col
            tmp_low["power_threshold_axis"] = threshold_axis
            tmp_low["power_switch_threshold"] = switch_threshold
            tmp_low["fan_threshold_b"] = results["fan_threshold_b"]
            low_rows.append(tmp_low)

        if high_bins is not None and len(high_bins) > 0:
            tmp_high = high_bins.copy()
            tmp_high["station_code"] = station_code
            tmp_high["station_name"] = station_name
            tmp_high["scheme_code"] = scheme_code
            tmp_high["scheme_name"] = scheme_name
            tmp_high["power_col_for_prediction"] = power_col
            tmp_high["power_threshold_axis"] = threshold_axis
            tmp_high["power_switch_threshold"] = switch_threshold
            tmp_high["fan_threshold_b"] = results["fan_threshold_b"]
            high_rows.append(tmp_high)

        interp_point_df = build_interp_points_output(
            station_code=station_code,
            station_name=station_name,
            scheme_code=scheme_code,
            scheme_name=scheme_name,
            power_col=power_col,
            threshold_axis=threshold_axis,
            switch_threshold=switch_threshold,
            fan_threshold_b=results["fan_threshold_b"],
            low_lookup_df=low_lookup,
            high_bins_df=high_bins,
            sigma_table_df=sigma_table,
        )
        if len(interp_point_df) > 0:
            interp_rows.append(interp_point_df)

        empirical_df = scheme_res.get("empirical_quantile_bins", pd.DataFrame())
        if empirical_df is not None and len(empirical_df) > 0:
            tmp_emp = empirical_df.copy()
            tmp_emp["station_code"] = station_code
            tmp_emp["station_name"] = station_name
            tmp_emp["scheme_code"] = scheme_code
            tmp_emp["scheme_name"] = scheme_name
            tmp_emp["power_col_for_prediction"] = power_col
            tmp_emp["power_threshold_axis"] = threshold_axis
            tmp_emp["power_switch_threshold"] = switch_threshold
            tmp_emp["fan_threshold_b"] = results["fan_threshold_b"]
            tmp_emp["quantile_basis"] = "hard_anomaly_removed_only"
            if "quantile_rule" not in tmp_emp.columns:
                tmp_emp["quantile_rule"] = "empirical_from_hard_anomaly_removed_only"
            tmp_emp["ultra_low_power_max_mw"] = ULTRA_LOW_POWER_MAX_MW
            tmp_emp["ultra_low_power_loss_ratio_min"] = ULTRA_LOW_POWER_LOSS_RATIO_MIN
            empirical_rows.append(tmp_emp)

    pd.DataFrame(summary_rows).to_csv(fit_summary_csv, index=False, encoding="utf-8-sig")
    (pd.concat(sigma_rows, ignore_index=True) if sigma_rows else pd.DataFrame()).to_csv(fit_sigma_bins_csv, index=False, encoding="utf-8-sig")
    (pd.concat(low_rows, ignore_index=True) if low_rows else pd.DataFrame()).to_csv(fit_low_lookup_csv, index=False, encoding="utf-8-sig")
    (pd.concat(high_rows, ignore_index=True) if high_rows else pd.DataFrame()).to_csv(fit_high_bins_csv, index=False, encoding="utf-8-sig")
    (pd.concat(interp_rows, ignore_index=True) if interp_rows else pd.DataFrame()).to_csv(fit_interp_points_csv, index=False, encoding="utf-8-sig")
    (pd.concat(empirical_rows, ignore_index=True) if empirical_rows else pd.DataFrame()).to_csv(fit_empirical_quantile_csv, index=False, encoding="utf-8-sig")

    print("\n已保存插值建模结果：")
    print(f"  - {fit_summary_csv}")
    print(f"  - {fit_sigma_bins_csv}")
    print(f"  - {fit_low_lookup_csv}")
    print(f"  - {fit_high_bins_csv}")
    print(f"  - {fit_interp_points_csv}")
    print(f"  - {fit_empirical_quantile_csv}")


def save_timeseries_loss_detail(results: dict, output_dir: str):
    detail_file = os.path.join(output_dir, "timeseries_loss_detail.csv")
    station_df = results["station_df"].copy()
    keep_cols = [
        "timestamp", "CT", "CT_eff", "FAN_SUM", "FAN_eff", "L",
        "is_anomaly", "is_fan_joint_repeat_anomaly", "is_station_power_repeat_anomaly",
        "is_negative_loss", "is_model_abnormal",
    ]
    detail_df = station_df[[c for c in keep_cols if c in station_df.columns]].copy()
    detail_df.to_csv(detail_file, index=False, encoding="utf-8-sig")
    print(f"  - {detail_file}")


def save_bad_physical_outliers(results: dict, output_dir: str):
    bad_df = results.get("bad_physical_df", pd.DataFrame())
    file_path = os.path.join(output_dir, "bad_physical_outliers.csv")
    if bad_df is not None and len(bad_df) > 0:
        bad_df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"  - {file_path}")


def save_hard_anomaly_segments(results: dict, output_dir: str):
    segs = results.get("hard_anomaly_segments", pd.DataFrame())
    file_path = os.path.join(output_dir, "hard_anomaly_segments_used.csv")
    if segs is not None and len(segs) > 0:
        segs.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"  - {file_path}")


def save_coarse_filter_outputs(results: dict, output_dir: str):
    pairs = [
        ("STATION_SEG", results["station_seg"]),
        ("FAN_SEG", results["fan_seg"]),
    ]
    for scheme_code, scheme_res in pairs:
        removed_df = scheme_res["coarse_removed_df"]
        summary_df = scheme_res["coarse_filter_summary_df"]
        impact_df = scheme_res["coarse_filter_impact_df"]

        if removed_df is not None and len(removed_df) > 0:
            fp = os.path.join(output_dir, f"coarse_filtered_points_{scheme_code}.csv")
            removed_df.to_csv(fp, index=False, encoding="utf-8-sig")
            print(f"  - {fp}")

        if summary_df is not None and len(summary_df) > 0:
            fp = os.path.join(output_dir, f"coarse_filter_summary_{scheme_code}.csv")
            summary_df.to_csv(fp, index=False, encoding="utf-8-sig")
            print(f"  - {fp}")

        if impact_df is not None and len(impact_df) > 0:
            fp = os.path.join(output_dir, f"coarse_filter_impact_summary_{scheme_code}.csv")
            impact_df.to_csv(fp, index=False, encoding="utf-8-sig")
            print(f"  - {fp}")


# ============================================================
# Dash 页面
# ============================================================
def build_dash_app(results: dict) -> Dash:
    app = Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "padding": "16px"},
        children=[
            html.H2("场站整体传输损耗插值建模"),
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "12px"},
                children=[
                    html.Div([
                        html.Label("选择建模方案"),
                        dcc.Dropdown(
                            id="scheme-dd",
                            options=[
                                {"label": SCHEME_STATION_SEG, "value": "STATION_SEG"},
                                {"label": SCHEME_FAN_SEG, "value": "FAN_SEG"},
                            ],
                            value="FAN_SEG",
                            clearable=False,
                        ),
                    ], style={"width": "360px"}),
                    html.Div([
                        html.Label("显示模式"),
                        dcc.Dropdown(
                            id="view-dd",
                            options=[
                                {"label": "正常/异常点", "value": "base"},
                                {"label": "显示粗筛删除点", "value": "coarse"},
                            ],
                            value="coarse",
                            clearable=False,
                        ),
                    ], style={"width": "220px"}),
                    html.Div([
                        html.Label("经验带下限"),
                        dcc.Dropdown(
                            id="emp-lower-dd",
                            options=get_empirical_lower_options(EMPIRICAL_QUANTILES),
                            value=default_empirical_lower_values(),
                            multi=True,
                            clearable=True,
                            placeholder="选择 P01~P10 等下限",
                        ),
                    ], style={"width": "240px"}),
                    html.Div([
                        html.Label("经验带上限"),
                        dcc.Dropdown(
                            id="emp-upper-dd",
                            options=get_empirical_upper_options(EMPIRICAL_QUANTILES),
                            value=default_empirical_upper_values(),
                            multi=True,
                            clearable=True,
                            placeholder="选择 P90~P99 等上限",
                        ),
                    ], style={"width": "240px"}),
                    html.Div([
                        html.Label("局部 sigma 正常区间"),
                        dcc.Dropdown(
                            id="sigma-band-dd",
                            options=[
                                {"label": "不显示", "value": "0"},
                                {"label": "±1σ（硬异常+粗筛后）", "value": "1"},
                                {"label": "±2σ（硬异常+粗筛后）", "value": "2"},
                                {"label": "±3σ（硬异常+粗筛后）", "value": "3"},
                            ],
                            value="2",
                            clearable=False,
                        ),
                    ], style={"width": "260px"}),
                ],
            ),
            html.Div(id="metrics-box", style={"marginBottom": "12px", "fontSize": "15px"}),
            dcc.Graph(id="loss-graph", style={"height": "760px"}),
        ],
    )

    @app.callback(
        Output("loss-graph", "figure"),
        Output("metrics-box", "children"),
        Input("scheme-dd", "value"),
        Input("view-dd", "value"),
        Input("emp-lower-dd", "value"),
        Input("emp-upper-dd", "value"),
        Input("sigma-band-dd", "value"),
    )
    def update_graph(scheme: str, view_mode: str, emp_lower_values, emp_upper_values, sigma_band_mode: str):
        normal_sc = results["scatter_normal"]
        anom_sc = results["scatter_anomaly"]

        if scheme == "STATION_SEG":
            scheme_res = results["station_seg"]
            scheme_label = SCHEME_STATION_SEG
            x_normal = normal_sc["CT_eff"]
            x_anom = anom_sc["CT_eff"]
            x_label = "P = max(ACTIVE_POWER_STATION, 0) (MW)"
            p_col_removed = "CT_eff"
        else:
            scheme_res = results["fan_seg"]
            scheme_label = SCHEME_FAN_SEG
            x_normal = normal_sc["FAN_eff"]
            x_anom = anom_sc["FAN_eff"]
            x_label = "P = FAN_SUM (MW)"
            p_col_removed = "FAN_eff"

        bins = scheme_res["bins"]
        low_lookup = scheme_res["low_lookup"]
        sigma_table = scheme_res["sigma_table"]
        empirical_q = scheme_res.get("empirical_quantile_bins", pd.DataFrame())
        threshold = scheme_res["threshold"]
        model_samples = scheme_res["model_samples"]
        removed_df = scheme_res["coarse_removed_df"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_normal, y=normal_sc["L"], mode="markers", name="正常点",
            marker=dict(size=4, opacity=0.35, color="#1f77b4"),
        ))
        fig.add_trace(go.Scatter(
            x=x_anom, y=anom_sc["L"], mode="markers", name="异常点（硬异常）",
            marker=dict(size=4, opacity=0.35, color="#d62728"),
        ))

        if view_mode == "coarse" and removed_df is not None and len(removed_df) > 0:
            fig.add_trace(go.Scatter(
                x=removed_df[p_col_removed], y=removed_df["L"], mode="markers", name="粗筛删除点",
                marker=dict(size=8, color="orange", symbol="x"),
            ))

        if len(bins) > 0:
            fig.add_trace(go.Scatter(
                x=bins["P_med"], y=bins["L_med"], mode="markers+lines", name="高功率区分箱中位数",
                marker=dict(size=7, color="#111111"), line=dict(color="#111111", width=1),
            ))
        if low_lookup is not None and len(low_lookup) > 0:
            fig.add_trace(go.Scatter(
                x=low_lookup["P_med"], y=low_lookup["L_med"], mode="markers+lines", name="低功率区中位数插值点",
                marker=dict(size=7, color="#9467bd"), line=dict(color="#9467bd", width=2, dash="dot"),
            ))

        x_all = pd.concat([x_normal, x_anom], ignore_index=True)
        if len(x_all) > 0:
            xmin = max(0.0, float(np.nanmin(x_all)))
            xmax = float(np.nanmax(x_all))
            if xmax > xmin:
                x_curve = np.linspace(xmin, xmax, 500)

                if empirical_q is not None and len(empirical_q) > 0:
                    selected_lowers = normalize_dropdown_values(emp_lower_values)
                    selected_uppers = normalize_dropdown_values(emp_upper_values)

                    band_styles = [
                        ("rgba(99,110,250,0.25)", "rgba(99,110,250,0.15)"),
                        ("rgba(44,160,44,0.25)", "rgba(44,160,44,0.12)"),
                        ("rgba(148,103,189,0.25)", "rgba(148,103,189,0.10)"),
                        ("rgba(214,39,40,0.22)", "rgba(214,39,40,0.08)"),
                        ("rgba(23,190,207,0.22)", "rgba(23,190,207,0.08)"),
                        ("rgba(188,189,34,0.22)", "rgba(188,189,34,0.08)"),
                    ]

                    selected_pairs = []
                    for q_low_col in selected_lowers:
                        for q_high_col in selected_uppers:
                            if q_low_col not in empirical_q.columns or q_high_col not in empirical_q.columns:
                                continue
                            try:
                                low_pct = int(q_low_col.replace("Q", ""))
                                high_pct = int(q_high_col.replace("Q", ""))
                            except ValueError:
                                continue
                            if low_pct >= high_pct:
                                continue
                            selected_pairs.append((q_low_col, q_high_col, empirical_band_label(q_low_col, q_high_col)))

                    for band_i, (q_low_col, q_high_col, band_label) in enumerate(selected_pairs):
                        q_low_curve = predict_value_from_point_table(x_curve, empirical_q, q_low_col)
                        q_high_curve = predict_value_from_point_table(x_curve, empirical_q, q_high_col)

                        # 风机总功率模型下，0~ULTRA_LOW_POWER_MAX_MW 区间直接使用规则，不在 DASH 中绘制经验带
                        if scheme == "FAN_SEG":
                            band_mask = x_curve > ULTRA_LOW_POWER_MAX_MW
                            q_low_curve = np.where(band_mask, q_low_curve, np.nan)
                            q_high_curve = np.where(band_mask, q_high_curve, np.nan)

                        line_color, fill_color = band_styles[band_i % len(band_styles)]
                        fig.add_trace(go.Scatter(
                            x=x_curve, y=q_high_curve, mode="lines", name=f"经验{band_label}上界",
                            line=dict(color=line_color, width=1),
                            hoverinfo="skip",
                            showlegend=False,
                        ))
                        fig.add_trace(go.Scatter(
                            x=x_curve, y=q_low_curve, mode="lines", name=f"经验{band_label}带",
                            line=dict(color=line_color, width=1),
                            fill="tonexty",
                            fillcolor=fill_color,
                        ))
                y_curve = predict_loss_from_interp_points(
                    p_vals=x_curve,
                    low_points_df=low_lookup,
                    high_points_df=bins,
                )

                sigma_mult = float(sigma_band_mode)
                if sigma_mult > 0 and sigma_table is not None and len(sigma_table) > 0:
                    sigma_curve = predict_sigma_from_table(x_curve, sigma_table)
                    upper = y_curve + sigma_mult * sigma_curve
                    lower = y_curve - sigma_mult * sigma_curve
                    # 风机总功率模型下，0~ULTRA_LOW_POWER_MAX_MW 区间直接使用规则，不在 DASH 中绘制模型 sigma 区间
                    if scheme == "FAN_SEG":
                        sigma_mask = x_curve > ULTRA_LOW_POWER_MAX_MW
                        upper = np.where(sigma_mask, upper, np.nan)
                        lower = np.where(sigma_mask, lower, np.nan)
                    fig.add_trace(go.Scatter(
                        x=x_curve, y=upper, mode="lines", name=f"模型±{sigma_mult:.0f}σ上界",
                        line=dict(color="rgba(255,127,14,0.20)", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_curve, y=lower, mode="lines", name=f"模型±{sigma_mult:.0f}σ区间",
                        line=dict(color="rgba(255,127,14,0.20)", width=1),
                        fill="tonexty",
                        fillcolor="rgba(255,127,14,0.12)",
                    ))

                fig.add_trace(go.Scatter(
                    x=x_curve, y=y_curve, mode="lines", name="分箱中位数插值线",
                    line=dict(color="#ff7f0e", width=3),
                ))

        fig.update_layout(
            template="plotly_white",
            title=f"{results['station_name']} 损耗插值建模（{scheme_label}）",
            xaxis_title=x_label,
            yaxis_title="L = FAN_SUM − max(ACTIVE_POWER_STATION, 0) (MW)",
            legend=dict(orientation="h", y=1.08, x=0),
        )

        metrics = (
            f"{results['station_name']} | {scheme_label} | "
            f"正常样本 = {len(results['normal_df']):,} | "
            f"硬异常样本 = {len(results['anomaly_df']):,} | "
            f"风机联合重复段 = {len(results.get('fan_joint_hard_anomaly_segments', [])):,} | "
            f"站端功率重复段 = {len(results.get('station_power_repeat_hard_anomaly_segments', [])):,} | "
            f"明显异常点(超额定) = {len(results['bad_physical_df']):,} | "
            f"负损耗样本 = {len(results['negative_loss_df']):,} | "
            f"粗筛删除点 = {len(removed_df):,} | "
            f"建模样本数 = {model_samples:,} | "
            f"局部σ箱数 = {len(sigma_table):,} | "
            f"分段阈值 = {threshold:.4f} MW | "
            f"额定功率阈值 = {results['rated_power_mw'] * RATED_POWER_MARGIN:.4f} MW | "
            f"超低功率经验规则 = 0~{ULTRA_LOW_POWER_MAX_MW}MW, L≥{ULTRA_LOW_POWER_LOSS_RATIO_MIN:.2%}×P | "
            f"经验分位带口径 = 仅剔除硬异常 | "
            f"局部sigma区间口径 = 硬异常+粗筛后 | "
            f"负损耗是否当异常 = {NEGATIVE_LOSS_AS_ANOMALY} | "
            f"粗筛启用 = {ENABLE_COARSE_FILTER} | "
            f"模型类型 = 分箱中位数插值"
        )
        return fig, metrics

    return app


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--station-dir",
        type=str,
        default=r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\江苏\场站数据\JMZSFD",
        help=r"场站目录，例如 G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\江苏\场站数据\JMZSFD"
    )
    parser.add_argument("--output-dir", type=str, default=r"场站数据\JMZSFD\step1-插值损耗模型", help="输出目录")
    parser.add_argument("--dash", action="store_true", help="启动 Dash 页面")
    parser.add_argument("--port", type=int, default=8050, help="Dash 端口")
    parser.add_argument("--station-name", type=str, default="", help="场站名称（用于展示，默认取文件夹名）")
    parser.add_argument("--fan-threshold-b", type=float, default=FAN_THRESHOLD_B, help="低/高功率阈值")
    parser.add_argument("--rated-power-mw", type=float, default=STATION_RATED_POWER_MW, help="全站额定功率(MW)")
    parser.add_argument("--station-repeat-min-length", type=int, default=STATION_REPEAT_MIN_LENGTH, help="站端 ACTIVE_POWER_STATION 重复值硬异常：持续长度需严格大于该值")
    parser.add_argument("--station-repeat-value-min", type=float, default=STATION_REPEAT_VALUE_MIN, help="站端 ACTIVE_POWER_STATION 重复值硬异常：重复值需严格大于该值")
    parser.add_argument("--disable-station-repeat-hard-anomaly", action="store_true", help="禁用站端 ACTIVE_POWER_STATION 连续重复硬异常规则")
    args = parser.parse_args()


    station_dir = args.station_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    station_name = args.station_name.strip() if args.station_name else os.path.basename(os.path.normpath(station_dir))
    fan_threshold_b = args.fan_threshold_b

    scada, bad_physical_df = load_scada(station_dir, args.rated_power_mw)
    enable_station_repeat_hard_anomaly = not args.disable_station_repeat_hard_anomaly
    anom_df = load_anomaly_segments(
        station_dir,
        station_repeat_min_length=args.station_repeat_min_length,
        station_repeat_value_min=args.station_repeat_value_min,
        enable_station_repeat_hard_anomaly=enable_station_repeat_hard_anomaly,
    )
    results = prepare_station_data(
        scada,
        anom_df,
        station_name=station_name,
        fan_threshold_b=fan_threshold_b,
        bad_physical_df=bad_physical_df,
        rated_power_mw=args.rated_power_mw,
        station_repeat_min_length=args.station_repeat_min_length,
        station_repeat_value_min=args.station_repeat_value_min,
        enable_station_repeat_hard_anomaly=enable_station_repeat_hard_anomaly,
    )

    save_fit_results(results, output_dir)
    save_timeseries_loss_detail(results, output_dir)
    save_bad_physical_outliers(results, output_dir)
    save_hard_anomaly_segments(results, output_dir)
    save_coarse_filter_outputs(results, output_dir)

    if args.dash:
        app = build_dash_app(results)
        print(f"\n🚀 Dash 已启动: http://127.0.0.1:{args.port}")
        app.run(debug=False, port=args.port)
    else:
        print("\n已完成插值建模。")
        print("未指定 --dash，因此不启动页面。")


if __name__ == "__main__":
    main()
