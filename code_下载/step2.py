#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP2：江苏场站全站功率异常检测（P05-P95 经验百分位带版）

适配新版 STEP1：
- 不再按 BING / DING / WU 集电线路处理，只处理单个场站全站功率。
- 读取场站主 CSV 中的 ACTIVE_POWER_STATION 与所有 ACTIVE_POWER_#数字 风机列。
- 读取 STEP1 输出的 fit_empirical_quantile_bins.csv，用经验百分位带判异。
- 默认使用 P05-P95 作为异常判定带。
- 超低功率区间使用物理规则：ratio_min * P <= L <= P。
- 保留原始 / 修正后双口径：
    原始口径：FAN_SUM
    修正后口径：FAN_SUM - 当前分钟重复风机功率之和
- 输出字段尽量中文且精简。

默认输出：
- station_anomaly_detail.csv：逐分钟异常明细
- station_anomaly_summary.csv：汇总统计
- hard_anomaly_segments_used_step2.csv：硬异常段清单

可选输出：
- repeat_power_timeseries.csv：使用 --save-repeat-power-timeseries 时输出
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 默认配置
# ============================================================
STATION_CT_COL = "ACTIVE_POWER_STATION"
FAN_POWER_COL_PATTERN = r"^ACTIVE_POWER_#\d+$"

DEFAULT_MODEL_SUMMARY = "fit_model_summary.csv"
DEFAULT_MODEL_EMPIRICAL = "fit_empirical_quantile_bins.csv"

DEFAULT_FAN_JOINT_REPEAT_REL = os.path.join("风机联合连续相同检测", "联合重复值检测结果.xlsx")
DEFAULT_STATION_REPEAT_REL = os.path.join("连续相同检测", "每列连续重复检测结果.csv")

DEFAULT_SCHEME_CODE = "FAN_SEG"
DEFAULT_ULTRA_LOW_POWER_MAX_MW = 3.0
DEFAULT_ULTRA_LOW_LOSS_RATIO_MIN = 0.99
DEFAULT_RATED_POWER_MW = 400.0
DEFAULT_RATED_POWER_MARGIN = 1.05

DEFAULT_QUANTILE_LOWER_PCT = 5
DEFAULT_QUANTILE_UPPER_PCT = 95
DEFAULT_SEVERITY_BANDWIDTH_FLOOR = 0.05

DEFAULT_STATION_REPEAT_MIN_LENGTH = 3
DEFAULT_STATION_REPEAT_VALUE_MIN = 0.0

SKIP_CSV_KEYWORDS = [
    "_duplicate_timestamps", "_missing_timestamps", "_process_summary", "_nulls",
    "_seconds_not_zero", "_invalid_timestamps", "_check_summary",
    "_timestamp_fixed", "_timestamp_changes",
]


# ============================================================
# 通用工具
# ============================================================
def read_csv_auto(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"读取失败：{path}\n最后一次报错：{last_err}")


def quantile_col_name(pct: int | float) -> str:
    return f"Q{int(round(float(pct))):02d}"


def bool_to_cn(x) -> str:
    return "是" if bool(x) else "否"


def source_to_cn(x: str) -> str:
    return {
        "fan_joint_repeat": "风机联合连续重复",
        "station_active_power_repeat": "站端功率连续重复",
        "physical_outlier": "物理超额定",
    }.get(str(x), str(x))


def find_station_main_csv(station_dir: str | Path) -> Path:
    station_dir = Path(station_dir)
    if not station_dir.exists():
        raise FileNotFoundError(f"场站目录不存在：{station_dir}")

    csv_files = []
    for p in station_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        if any(k in p.name for k in SKIP_CSV_KEYWORDS):
            continue
        csv_files.append(p)

    if not csv_files:
        raise FileNotFoundError(f"未在场站目录中找到主 CSV：{station_dir}")
    if len(csv_files) > 1:
        raise RuntimeError(f"场站目录下检测到多个候选主 CSV，请手动确认：{csv_files}")
    return csv_files[0]


def find_fan_power_columns(columns) -> List[str]:
    pattern = re.compile(FAN_POWER_COL_PATTERN)
    fan_cols = []
    for col in columns:
        col = str(col).strip()
        if pattern.match(col):
            fan_cols.append(col)
    return sorted(fan_cols, key=lambda x: int(str(x).split("#")[1]))


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
    valid = np.isfinite(values)
    out[valid] = np.interp(values[valid], xp, fp, left=fp[0], right=fp[-1])
    return out


def _to_float_or_default(val, default: float) -> float:
    try:
        x = float(val)
        return x if np.isfinite(x) else float(default)
    except Exception:  # noqa: BLE001
        return float(default)


# ============================================================
# 场站数据加载
# ============================================================
def load_station_scada(
    station_dir: str | Path,
    rated_power_mw: float,
    rated_power_margin: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    main_csv = find_station_main_csv(station_dir)
    df = pd.read_csv(main_csv, parse_dates=["timestamp"])
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    if "timestamp" not in df.columns:
        raise ValueError("主 CSV 中缺少 timestamp 列")
    if STATION_CT_COL not in df.columns:
        raise ValueError(f"主 CSV 中缺少 {STATION_CT_COL} 列")

    fan_cols = find_fan_power_columns(df.columns)
    if not fan_cols:
        raise ValueError("主 CSV 中未识别到风机有功列 ACTIVE_POWER_#数字")

    out = df[["timestamp", STATION_CT_COL] + fan_cols].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).copy()
    out = out.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    out["CT功率MW"] = pd.to_numeric(out[STATION_CT_COL], errors="coerce")
    out["CT有效功率MW"] = out["CT功率MW"].clip(lower=0)
    fan_df = out[fan_cols].apply(pd.to_numeric, errors="coerce").clip(lower=0)
    out["原风机汇总功率MW"] = fan_df.sum(axis=1) / 1000.0
    out["原损耗MW"] = out["原风机汇总功率MW"] - out["CT有效功率MW"]

    rated_limit = float(rated_power_mw) * float(rated_power_margin)
    out["风机汇总超额定"] = np.isfinite(out["原风机汇总功率MW"]) & (out["原风机汇总功率MW"] > rated_limit)
    out["站端功率超额定"] = np.isfinite(out["CT有效功率MW"]) & (out["CT有效功率MW"] > rated_limit)
    out["是否物理越限硬异常"] = out["风机汇总超额定"] | out["站端功率超额定"]

    bad_physical_df = out.loc[out["是否物理越限硬异常"], [
        "timestamp", STATION_CT_COL, "CT功率MW", "CT有效功率MW", "原风机汇总功率MW",
        "风机汇总超额定", "站端功率超额定", "是否物理越限硬异常",
    ]].copy()

    return out, bad_physical_df, fan_cols


# ============================================================
# 重复值 / 硬异常段读取
# ============================================================
def parse_combo(combo) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        vals = str(combo).strip("()").split(",")
        vals = [v.strip() for v in vals]

        def to_float(x):
            if x is None or x == "":
                return None
            return float(x)

        status = to_float(vals[0]) if len(vals) > 0 else None
        active_power = to_float(vals[1]) if len(vals) > 1 else None
        windspeed = to_float(vals[3]) if len(vals) > 3 else None
        return status, active_power, windspeed
    except Exception:  # noqa: BLE001
        return None, None, None


def safe_nonnegative_power_kw(x) -> float:
    if pd.isna(x):
        return 0.0
    try:
        return max(float(x), 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def load_fan_joint_repeat_segments(station_dir: str | Path, rel_path: str) -> pd.DataFrame:
    path = Path(station_dir) / rel_path
    if not path.exists():
        print(f"⚠️ 未找到风机联合重复值检测结果：{path}，将按无风机重复段处理。")
        return pd.DataFrame(columns=["开始时间", "结束时间", "风机编号", "重复功率MW", "状态码", "冻结风速ms", "硬异常来源"])

    df = pd.read_excel(path, engine="openpyxl")
    required = {"开始时间", "结束时间", "风机编号", "重复值组合"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"风机联合重复值检测结果缺少必要列：{missing}，文件：{path}")

    df["开始时间"] = pd.to_datetime(df["开始时间"], errors="coerce")
    df["结束时间"] = pd.to_datetime(df["结束时间"], errors="coerce")
    df = df.dropna(subset=["开始时间", "结束时间"]).copy()
    df = df[df["结束时间"] >= df["开始时间"]].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["开始时间", "结束时间", "风机编号", "重复功率MW", "状态码", "冻结风速ms", "硬异常来源"])

    parsed = df["重复值组合"].apply(parse_combo)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=["状态码", "冻结有功kW", "冻结风速ms"], index=df.index)
    df = pd.concat([df, parsed_df], axis=1)
    df["重复功率MW"] = df["冻结有功kW"].apply(safe_nonnegative_power_kw) / 1000.0
    df["硬异常来源"] = "fan_joint_repeat"
    return df[["开始时间", "结束时间", "风机编号", "重复功率MW", "状态码", "冻结风速ms", "硬异常来源"]].copy()


def load_station_power_repeat_segments(
    station_dir: str | Path,
    rel_path: str,
    min_length: int,
    value_min: float,
    enabled: bool,
) -> pd.DataFrame:
    if not enabled:
        return pd.DataFrame(columns=["开始时间", "结束时间", "字段名", "重复值", "持续长度", "硬异常来源"])

    path = Path(station_dir) / rel_path
    if not path.exists():
        print(f"⚠️ 未找到站端连续重复检测结果：{path}，将按无站端重复硬异常处理。")
        return pd.DataFrame(columns=["开始时间", "结束时间", "字段名", "重复值", "持续长度", "硬异常来源"])

    df = read_csv_auto(path)
    required = {"字段名", "重复值", "开始时间", "结束时间", "持续长度"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"站端连续重复检测结果缺少必要列：{missing}，文件：{path}")

    df = df.copy()
    df["字段名"] = df["字段名"].astype(str).str.strip()
    df["重复值"] = pd.to_numeric(df["重复值"], errors="coerce")
    df["持续长度"] = pd.to_numeric(df["持续长度"], errors="coerce")
    df["开始时间"] = pd.to_datetime(df["开始时间"], errors="coerce")
    df["结束时间"] = pd.to_datetime(df["结束时间"], errors="coerce")
    df = df.dropna(subset=["开始时间", "结束时间"]).copy()
    df = df[df["结束时间"] >= df["开始时间"]].copy()

    mask = (
        (df["字段名"] == STATION_CT_COL)
        & np.isfinite(df["重复值"])
        & (df["重复值"] > float(value_min))
        & np.isfinite(df["持续长度"])
        & (df["持续长度"] > int(min_length))
    )
    out = df.loc[mask, ["开始时间", "结束时间", "字段名", "重复值", "持续长度"]].copy()
    out["硬异常来源"] = "station_active_power_repeat"
    return out.reset_index(drop=True)


def build_interval_mask(full_timestamps: pd.Series, seg_df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(full_timestamps, errors="coerce")
    t_int = ts.values.astype("datetime64[ns]").astype("int64")
    mask = np.zeros(len(ts), dtype=bool)
    if seg_df is None or len(seg_df) == 0:
        return mask

    for _, row in seg_df.iterrows():
        start = pd.to_datetime(row["开始时间"], errors="coerce")
        end = pd.to_datetime(row["结束时间"], errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        s = start.to_datetime64().astype("datetime64[ns]").astype("int64")
        e = end.to_datetime64().astype("datetime64[ns]").astype("int64")
        mask |= (t_int >= s) & (t_int <= e)
    return mask


def build_repeat_power_series(full_timestamps: pd.Series, fan_repeat_df: pd.DataFrame) -> pd.Series:
    full_index = pd.DatetimeIndex(pd.to_datetime(full_timestamps, errors="coerce")).dropna().drop_duplicates().sort_values()
    if len(full_index) == 0:
        return pd.Series(dtype=float)
    if fan_repeat_df is None or len(fan_repeat_df) == 0:
        return pd.Series(0.0, index=full_index, dtype=float)

    delta_map: Dict[pd.Timestamp, float] = defaultdict(float)
    for _, row in fan_repeat_df.iterrows():
        power_mw = float(row.get("重复功率MW", 0.0))
        if not np.isfinite(power_mw) or power_mw <= 0:
            continue
        start = pd.to_datetime(row["开始时间"], errors="coerce")
        end = pd.to_datetime(row["结束时间"], errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        delta_map[start] += power_mw
        delta_map[end + pd.Timedelta(minutes=1)] -= power_mw

    if not delta_map:
        return pd.Series(0.0, index=full_index, dtype=float)

    delta_s = pd.Series(delta_map, dtype=float).sort_index()
    power_s = delta_s.cumsum().reindex(full_index, method="ffill").fillna(0.0)
    return power_s.clip(lower=0.0).astype(float)


# ============================================================
# STEP1 模型读取
# ============================================================
def load_model_files(model_dir: str | Path, scheme_code: str, lower_pct: int, upper_pct: int) -> Tuple[pd.Series, pd.DataFrame]:
    model_dir = Path(model_dir)
    summary_path = model_dir / DEFAULT_MODEL_SUMMARY
    empirical_path = model_dir / DEFAULT_MODEL_EMPIRICAL

    summary_df = read_csv_auto(summary_path)
    empirical_df = read_csv_auto(empirical_path)

    if "scheme_code" not in summary_df.columns:
        raise ValueError(f"{summary_path} 缺少 scheme_code 列。请确认使用新版 STEP1 输出。")
    if "scheme_code" not in empirical_df.columns:
        raise ValueError(f"{empirical_path} 缺少 scheme_code 列。请确认使用新版 STEP1 输出。")

    q_low_col = quantile_col_name(lower_pct)
    q_high_col = quantile_col_name(upper_pct)
    missing_q = {q_low_col, q_high_col} - set(empirical_df.columns)
    if missing_q:
        raise ValueError(
            f"{empirical_path} 缺少经验分位列：{missing_q}。\n"
            f"请确认 STEP1 的 EMPIRICAL_QUANTILES 包含 {lower_pct/100:.2f} 和 {upper_pct/100:.2f}。"
        )

    summary_use = summary_df[summary_df["scheme_code"].astype(str) == str(scheme_code)].copy()
    if len(summary_use) == 0:
        raise RuntimeError(f"模型汇总文件中没有找到 scheme_code={scheme_code} 的记录：{summary_path}")
    if len(summary_use) > 1:
        print(f"⚠️ scheme_code={scheme_code} 在 summary 中有多行，将使用第一行。")
    model_row = summary_use.iloc[0].copy()

    empirical_use = empirical_df[empirical_df["scheme_code"].astype(str) == str(scheme_code)].copy()
    if len(empirical_use) == 0:
        raise RuntimeError(f"经验分位文件中没有找到 scheme_code={scheme_code} 的记录：{empirical_path}")

    for col in ["P_med", q_low_col, q_high_col]:
        empirical_use[col] = pd.to_numeric(empirical_use[col], errors="coerce")
    empirical_use = empirical_use[np.isfinite(empirical_use["P_med"])].copy()
    empirical_use = empirical_use.sort_values("P_med").reset_index(drop=True)
    if len(empirical_use) == 0:
        raise ValueError(f"经验分位文件中没有可用的 P_med 记录：{empirical_path}")

    if empirical_use["P_med"].duplicated().any():
        agg = {q_low_col: "mean", q_high_col: "mean"}
        for c in empirical_use.columns:
            if c not in agg and c != "P_med":
                agg[c] = "first"
        empirical_use = empirical_use.groupby("P_med", as_index=False).agg(agg).sort_values("P_med").reset_index(drop=True)

    return model_row, empirical_use


def infer_ultra_low_params(
    model_row: pd.Series,
    empirical_df: pd.DataFrame,
    default_power_max: float,
    default_loss_ratio_min: float,
) -> Tuple[float, float]:
    power_max = _to_float_or_default(model_row.get("ultra_low_power_max_mw", np.nan), default_power_max)
    ratio_min = _to_float_or_default(model_row.get("ultra_low_power_loss_ratio_min", np.nan), default_loss_ratio_min)

    if "ultra_low_power_max_mw" in empirical_df.columns:
        vals = pd.to_numeric(empirical_df["ultra_low_power_max_mw"], errors="coerce").dropna()
        if len(vals) > 0:
            power_max = float(vals.iloc[0])
    if "ultra_low_power_loss_ratio_min" in empirical_df.columns:
        vals = pd.to_numeric(empirical_df["ultra_low_power_loss_ratio_min"], errors="coerce").dropna()
        if len(vals) > 0:
            ratio_min = float(vals.iloc[0])
    return power_max, ratio_min


# ============================================================
# 经验带判异与异常程度
# ============================================================
def compute_quantile_track(
    p_vals: np.ndarray,
    loss_vals: np.ndarray,
    empirical_df: pd.DataFrame,
    lower_pct: int,
    upper_pct: int,
    ultra_low_power_max_mw: float,
    ultra_low_loss_ratio_min: float,
    severity_bandwidth_floor: float,
    enable_ultra_low_rule: bool,
) -> Dict[str, np.ndarray]:
    p = np.asarray(p_vals, dtype=float)
    loss = np.asarray(loss_vals, dtype=float)
    q_low_col = quantile_col_name(lower_pct)
    q_high_col = quantile_col_name(upper_pct)

    q_low = interp_1d_with_hold(p, empirical_df["P_med"].values, empirical_df[q_low_col].values)
    q_high = interp_1d_with_hold(p, empirical_df["P_med"].values, empirical_df[q_high_col].values)

    ultra_mask = (
        bool(enable_ultra_low_rule)
        & np.isfinite(p)
        & np.isfinite(loss)
        & np.isfinite(ultra_low_power_max_mw)
        & (p >= 0)
        & (p <= float(ultra_low_power_max_mw))
    )

    lower = q_low.copy()
    upper = q_high.copy()
    lower[ultra_mask] = float(ultra_low_loss_ratio_min) * p[ultra_mask]
    upper[ultra_mask] = p[ultra_mask]

    valid = np.isfinite(p) & np.isfinite(loss) & np.isfinite(lower) & np.isfinite(upper)
    # 避免上界下界反转导致误判；若反转则统一置为无效。
    valid &= upper >= lower

    below_amount = np.where(valid & (loss < lower), lower - loss, 0.0)
    above_amount = np.where(valid & (loss > upper), loss - upper, 0.0)
    exceed_amount = np.maximum(below_amount, above_amount)
    model_anomaly = valid & (exceed_amount > 0)

    bandwidth = np.where(valid, upper - lower, np.nan)
    denom = np.maximum(bandwidth, float(severity_bandwidth_floor))
    severity = np.where(valid, exceed_amount / denom, np.nan)

    direction = np.full(len(p), "无效", dtype=object)
    direction[valid & (exceed_amount == 0)] = "正常"
    direction[valid & (below_amount > 0)] = "低于经验下界"
    direction[valid & (above_amount > 0)] = "高于经验上界"

    rule = np.full(len(p), "无效点或无可用经验带", dtype=object)
    rule[valid & ultra_mask] = "超低功率物理规则"
    rule[valid & (~ultra_mask)] = f"P{lower_pct:02d}-P{upper_pct:02d}经验带"

    return {
        "下界": lower,
        "上界": upper,
        "带宽": bandwidth,
        "是否有效": valid,
        "判异规则": rule,
        "异常方向": direction,
        "异常越界量MW": exceed_amount,
        "异常程度": severity,
        "模型是否异常": model_anomaly,
        "是否超低功率规则区": ultra_mask,
    }


def apply_quantile_detection(
    scada_df: pd.DataFrame,
    model_row: pd.Series,
    empirical_df: pd.DataFrame,
    ultra_low_power_max_mw: float,
    ultra_low_loss_ratio_min: float,
    lower_pct: int,
    upper_pct: int,
    severity_bandwidth_floor: float,
) -> pd.DataFrame:
    out = scada_df.copy()
    power_col = str(model_row.get("power_col_for_prediction", "FAN_eff"))
    enable_ultra_low_rule = power_col == "FAN_eff"

    if power_col == "CT_eff":
        p_raw = out["CT有效功率MW"].values.astype(float)
        p_corr = out["CT有效功率MW"].values.astype(float)
    elif power_col == "FAN_eff":
        p_raw = out["原风机汇总功率MW"].values.astype(float)
        p_corr = out["修正后风机汇总功率MW"].values.astype(float)
    else:
        raise ValueError(f"暂不支持 power_col_for_prediction={power_col}，请使用 FAN_eff 或 CT_eff。")

    raw_track = compute_quantile_track(
        p_vals=p_raw,
        loss_vals=out["原损耗MW"].values.astype(float),
        empirical_df=empirical_df,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
        ultra_low_power_max_mw=ultra_low_power_max_mw,
        ultra_low_loss_ratio_min=ultra_low_loss_ratio_min,
        severity_bandwidth_floor=severity_bandwidth_floor,
        enable_ultra_low_rule=enable_ultra_low_rule,
    )
    corr_track = compute_quantile_track(
        p_vals=p_corr,
        loss_vals=out["修正后损耗MW"].values.astype(float),
        empirical_df=empirical_df,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
        ultra_low_power_max_mw=ultra_low_power_max_mw,
        ultra_low_loss_ratio_min=ultra_low_loss_ratio_min,
        severity_bandwidth_floor=severity_bandwidth_floor,
        enable_ultra_low_rule=enable_ultra_low_rule,
    )

    out["方案代码"] = str(model_row.get("scheme_code", ""))
    out["方案名称"] = str(model_row.get("scheme_name", ""))
    out["预测功率口径"] = power_col
    out["经验带下限P"] = int(lower_pct)
    out["经验带上限P"] = int(upper_pct)
    out["超低功率阈值MW"] = float(ultra_low_power_max_mw)
    out["超低功率损耗比例阈值"] = float(ultra_low_loss_ratio_min)

    mapping = {
        "下界": "经验下界MW",
        "上界": "经验上界MW",
        "带宽": "经验带宽MW",
        "是否有效": "是否有效检测",
        "判异规则": "判异规则",
        "异常方向": "异常方向",
        "异常越界量MW": "异常越界量MW",
        "异常程度": "异常程度",
        "模型是否异常": "模型是否异常",
        "是否超低功率规则区": "是否超低功率规则区",
    }
    for prefix, track in [("原", raw_track), ("修正后", corr_track)]:
        for key, name in mapping.items():
            out[f"{prefix}{name}"] = track[key]

    return out


def combine_final_anomaly_flags(
    df: pd.DataFrame,
    fan_joint_hard_mode: str,
    station_repeat_hard_mode: str,
    physical_outlier_hard_mode: str,
) -> pd.DataFrame:
    out = df.copy()
    hard = np.zeros(len(out), dtype=bool)
    if fan_joint_hard_mode == "final":
        hard |= out["是否风机联合重复"].fillna(False).astype(bool).values
    if station_repeat_hard_mode == "final":
        hard |= out["是否站端功率重复硬异常"].fillna(False).astype(bool).values
    if physical_outlier_hard_mode == "final":
        hard |= out["是否物理越限硬异常"].fillna(False).astype(bool).values

    out["是否最终合并硬异常"] = hard
    out["原是否异常"] = out["原模型是否异常"].fillna(False).astype(bool) | hard
    out["是否异常"] = out["修正后模型是否异常"].fillna(False).astype(bool) | hard

    def source_col(model_col: str) -> List[str]:
        res = []
        for i in range(len(out)):
            parts = []
            if bool(out.iloc[i][model_col]):
                parts.append("经验带/物理规则")
            if fan_joint_hard_mode == "final" and bool(out.iloc[i]["是否风机联合重复"]):
                parts.append("风机联合连续重复")
            if station_repeat_hard_mode == "final" and bool(out.iloc[i]["是否站端功率重复硬异常"]):
                parts.append("站端功率连续重复")
            if physical_outlier_hard_mode == "final" and bool(out.iloc[i]["是否物理越限硬异常"]):
                parts.append("物理超额定")
            res.append("、".join(parts) if parts else "正常")
        return res

    out["原异常来源"] = source_col("原模型是否异常")
    out["异常来源"] = source_col("修正后模型是否异常")
    return out


# ============================================================
# 输出整理
# ============================================================
def build_detail_df(df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """构造逐分钟明细输出。

    物理越限点仍保留在明细中，但通过“是否纳入汇总统计”标明不会进入 summary 统计。
    """
    out = pd.DataFrame()
    out["时间"] = df["timestamp"]
    out["场站"] = station_name

    out["站端有功功率MW（ACTIVE_POWER_STATION）"] = df.get(STATION_CT_COL, np.nan)
    out["原始风机功率之和MW"] = df.get("原风机汇总功率MW", np.nan)
    out["重复风机功率扣减量MW"] = df.get("重复风机功率之和MW", np.nan)
    out["修正后风机功率之和MW"] = df.get("修正后风机汇总功率MW", np.nan)
    out["原损耗MW"] = df.get("原损耗MW", np.nan)
    out["修正后损耗MW"] = df.get("修正后损耗MW", np.nan)

    out["原经验带下界MW"] = df.get("原经验下界MW", np.nan)
    out["原经验带上界MW"] = df.get("原经验上界MW", np.nan)
    out["原判异规则"] = df.get("原判异规则", "")
    out["原是否异常"] = df.get("原是否异常", False)
    out["原异常方向"] = df.get("原异常方向", "")
    out["原异常越界量MW"] = df.get("原异常越界量MW", np.nan)
    out["原异常程度"] = df.get("原异常程度", np.nan)

    out["修正后经验带下界MW"] = df.get("修正后经验下界MW", np.nan)
    out["修正后经验带上界MW"] = df.get("修正后经验上界MW", np.nan)
    out["修正后判异规则"] = df.get("修正后判异规则", "")
    # 保留“是否异常”列名，兼容后续 STEP3；含义为“修正后是否异常”。
    out["是否异常"] = df.get("是否异常", False)
    out["修正后异常方向"] = df.get("修正后异常方向", "")
    out["修正后异常越界量MW"] = df.get("修正后异常越界量MW", np.nan)
    out["修正后异常程度"] = df.get("修正后异常程度", np.nan)

    out["是否风机联合重复标记"] = df.get("是否风机联合重复", False)
    out["是否站端功率重复硬异常"] = df.get("是否站端功率重复硬异常", False)
    out["是否物理越限硬异常"] = df.get("是否物理越限硬异常", False)
    physical = pd.Series(df.get("是否物理越限硬异常", False), index=df.index).fillna(False).astype(bool)
    out["是否纳入汇总统计"] = ~physical
    out["异常来源（修正后口径）"] = df.get("异常来源", "")

    bool_cols = [
        "原是否异常", "是否异常", "是否风机联合重复标记",
        "是否站端功率重复硬异常", "是否物理越限硬异常", "是否纳入汇总统计",
    ]
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].map(bool_to_cn)
    return out


def build_summary_df(df: pd.DataFrame, station_name: str, args: argparse.Namespace) -> pd.DataFrame:
    """构造汇总输出。

    风机功率之和等物理越限点只在明细中保留，不参与本汇总的异常率、损耗和异常程度统计，
    避免极端值把汇总指标拉得失真。
    """
    total_rows = len(df)
    physical_mask = df["是否物理越限硬异常"].fillna(False).astype(bool) if total_rows else pd.Series(dtype=bool)
    stat_df = df.loc[~physical_mask].copy() if total_rows else df.copy()
    stat_rows = len(stat_df)
    excluded_physical_rows = int(physical_mask.sum()) if total_rows else 0

    raw_valid = int(stat_df["原是否有效检测"].sum()) if stat_rows else 0
    corr_valid = int(stat_df["修正后是否有效检测"].sum()) if stat_rows else 0
    raw_anom = int(stat_df["原是否异常"].sum()) if stat_rows else 0
    corr_anom = int(stat_df["是否异常"].sum()) if stat_rows else 0

    raw_sev = pd.to_numeric(stat_df.get("原异常程度", pd.Series(dtype=float)), errors="coerce").fillna(0)
    corr_sev = pd.to_numeric(stat_df.get("修正后异常程度", pd.Series(dtype=float)), errors="coerce").fillna(0)
    raw_anom_mask = stat_df["原是否异常"].fillna(False).astype(bool) if stat_rows else pd.Series(dtype=bool)
    corr_anom_mask = stat_df["是否异常"].fillna(False).astype(bool) if stat_rows else pd.Series(dtype=bool)
    raw_anom_sev = raw_sev[raw_anom_mask]
    corr_anom_sev = corr_sev[corr_anom_mask]

    raw_loss = pd.to_numeric(stat_df.get("原损耗MW", pd.Series(dtype=float)), errors="coerce")
    corr_loss = pd.to_numeric(stat_df.get("修正后损耗MW", pd.Series(dtype=float)), errors="coerce")

    row = {
        "场站": station_name,
        "方案代码": str(df["方案代码"].iloc[0]) if total_rows and "方案代码" in df.columns else args.scheme_code,
        "方案名称": str(df["方案名称"].iloc[0]) if total_rows and "方案名称" in df.columns else "",
        "预测功率口径": str(df["预测功率口径"].iloc[0]) if total_rows and "预测功率口径" in df.columns else "",
        "原始总行数": total_rows,
        "纳入汇总统计行数": stat_rows,
        "剔除物理越限行数": excluded_physical_rows,
        "物理越限剔除说明": "物理越限点保留在明细中，但不参与本汇总统计",
        "重复功率修正行数（纳入统计）": int((stat_df["重复风机功率之和MW"] > 0).sum()) if stat_rows else 0,
        "原有效检测行数（纳入统计）": raw_valid,
        "原异常行数（纳入统计）": raw_anom,
        "原异常率（纳入统计）": raw_anom / raw_valid if raw_valid > 0 else np.nan,
        "修正后有效检测行数（纳入统计）": corr_valid,
        "修正后异常行数（纳入统计）": corr_anom,
        "修正后异常率（纳入统计）": corr_anom / corr_valid if corr_valid > 0 else np.nan,
        "原异常程度总和（纳入统计）": float(raw_anom_sev.sum()),
        "修正后异常程度总和（纳入统计）": float(corr_anom_sev.sum()),
        "异常程度下降量（原-修正后）": float(raw_anom_sev.sum() - corr_anom_sev.sum()),
        "原平均异常程度（异常点）": float(raw_anom_sev.mean()) if len(raw_anom_sev) > 0 else 0.0,
        "修正后平均异常程度（异常点）": float(corr_anom_sev.mean()) if len(corr_anom_sev) > 0 else 0.0,
        "原最大异常程度（异常点）": float(raw_anom_sev.max()) if len(raw_anom_sev) > 0 else 0.0,
        "修正后最大异常程度（异常点）": float(corr_anom_sev.max()) if len(corr_anom_sev) > 0 else 0.0,
        "原损耗MW最大值（纳入统计）": float(raw_loss.max()) if raw_loss.notna().any() else np.nan,
        "修正后损耗MW最大值（纳入统计）": float(corr_loss.max()) if corr_loss.notna().any() else np.nan,
        "风机联合重复标记分钟数（纳入统计）": int(stat_df["是否风机联合重复"].sum()) if stat_rows else 0,
        "站端功率重复硬异常分钟数（纳入统计）": int(stat_df["是否站端功率重复硬异常"].sum()) if stat_rows else 0,
        "物理越限硬异常分钟数（已剔除）": excluded_physical_rows,
        "最终合并硬异常分钟数（纳入统计）": int(stat_df["是否最终合并硬异常"].sum()) if stat_rows else 0,
        "经验带下限百分位": f"P{int(args.quantile_lower_pct):02d}",
        "经验带上限百分位": f"P{int(args.quantile_upper_pct):02d}",
        "超低功率阈值MW": float(df["超低功率阈值MW"].iloc[0]) if total_rows else args.ultra_low_power_max_mw,
        "超低功率损耗比例阈值": float(df["超低功率损耗比例阈值"].iloc[0]) if total_rows else args.ultra_low_loss_ratio_min,
        "异常程度带宽下限MW": args.severity_bandwidth_floor,
        "风机联合重复并入最终异常模式": args.fan_joint_repeat_hard_mode,
        "站端重复并入最终异常模式": args.station_repeat_hard_mode,
        "物理越限并入最终异常模式": args.physical_outlier_hard_mode,
    }
    return pd.DataFrame([row])


def write_hard_segments_used(
    fan_repeat_df: pd.DataFrame,
    station_repeat_df: pd.DataFrame,
    bad_physical_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """输出硬异常段清单，字段保持简洁且语义明确。"""
    parts = []
    if fan_repeat_df is not None and len(fan_repeat_df) > 0:
        tmp = fan_repeat_df.copy()
        tmp["硬异常来源"] = "fan_joint_repeat"
        tmp["硬异常来源说明"] = "风机联合连续重复；用于计算重复风机功率扣减量，是否并入最终异常由参数控制"
        parts.append(tmp[["开始时间", "结束时间", "硬异常来源", "硬异常来源说明"]])
    if station_repeat_df is not None and len(station_repeat_df) > 0:
        tmp = station_repeat_df.copy()
        tmp["硬异常来源"] = "station_active_power_repeat"
        tmp["硬异常来源说明"] = "ACTIVE_POWER_STATION 重复值大于阈值且持续长度大于阈值"
        parts.append(tmp[["开始时间", "结束时间", "硬异常来源", "硬异常来源说明"]])
    if bad_physical_df is not None and len(bad_physical_df) > 0:
        tmp = bad_physical_df.rename(columns={"timestamp": "开始时间"}).copy()
        tmp["结束时间"] = tmp["开始时间"]
        tmp["硬异常来源"] = "physical_outlier"
        tmp["硬异常来源说明"] = "风机功率之和或 ACTIVE_POWER_STATION 超过额定功率上限；明细保留但汇总剔除"
        parts.append(tmp[["开始时间", "结束时间", "硬异常来源", "硬异常来源说明"]])

    out = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame(
        columns=["开始时间", "结束时间", "持续分钟数", "硬异常来源", "硬异常来源说明"]
    )
    if len(out) > 0:
        out["开始时间"] = pd.to_datetime(out["开始时间"], errors="coerce")
        out["结束时间"] = pd.to_datetime(out["结束时间"], errors="coerce")
        out["持续分钟数"] = ((out["结束时间"] - out["开始时间"]).dt.total_seconds() / 60 + 1).clip(lower=1)
        out["持续分钟数"] = out["持续分钟数"].round(0).astype("Int64")
        out["硬异常来源"] = out["硬异常来源"].map(source_to_cn)
        out = out[["开始时间", "结束时间", "持续分钟数", "硬异常来源", "硬异常来源说明"]].sort_values(["开始时间", "结束时间"]).reset_index(drop=True)
    out.to_csv(output_dir / "hard_anomaly_segments_used_step2.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 参数与主流程
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STEP2：江苏场站全站功率异常检测（P05-P95 经验带版）")
    parser.add_argument("--station-dir", default="场站数据\JMZSFD", help="场站目录，内部应包含主 CSV 和重复值检测结果子目录")
    parser.add_argument("--model-dir", default="场站数据\JMZSFD\step1-插值损耗模型",  help="STEP1 输出目录，包含 fit_model_summary.csv / fit_model_interp_points.csv")
    parser.add_argument("--output-dir", default="场站数据\JMZSFD\step2-异常检测",  help="STEP2 输出目录")
    parser.add_argument("--station-name", default="JMZSFD", help="场站名称，默认取 station-dir 文件夹名")
    parser.add_argument("--scheme-code", default=DEFAULT_SCHEME_CODE, help="使用 STEP1 哪个方案，默认 FAN_SEG")

    parser.add_argument("--quantile-lower-pct", type=int, default=DEFAULT_QUANTILE_LOWER_PCT, help="经验带下限百分位，默认 5，即 P05")
    parser.add_argument("--quantile-upper-pct", type=int, default=DEFAULT_QUANTILE_UPPER_PCT, help="经验带上限百分位，默认 95，即 P95")
    parser.add_argument("--severity-bandwidth-floor", type=float, default=DEFAULT_SEVERITY_BANDWIDTH_FLOOR, help="异常程度归一化带宽下限 MW，默认 0.05")

    parser.add_argument("--rated-power-mw", type=float, default=None, help="全站额定功率 MW；默认优先读取模型汇总，否则 400")
    parser.add_argument("--rated-power-margin", type=float, default=None, help="额定功率硬筛裕度；默认优先读取模型汇总，否则 1.05")
    parser.add_argument("--ultra-low-power-max-mw", type=float, default=DEFAULT_ULTRA_LOW_POWER_MAX_MW, help="超低功率规则上限，若模型文件未提供则使用该值")
    parser.add_argument("--ultra-low-loss-ratio-min", type=float, default=DEFAULT_ULTRA_LOW_LOSS_RATIO_MIN, help="超低功率规则下界比例，若模型文件未提供则使用该值")

    parser.add_argument("--fan-joint-repeat-rel", default=DEFAULT_FAN_JOINT_REPEAT_REL, help="相对 station-dir 的风机联合重复结果路径")
    parser.add_argument("--station-repeat-rel", default=DEFAULT_STATION_REPEAT_REL, help="相对 station-dir 的站端连续重复结果路径")
    parser.add_argument("--station-repeat-min-length", type=int, default=DEFAULT_STATION_REPEAT_MIN_LENGTH, help="站端连续重复硬异常：持续长度需大于该值")
    parser.add_argument("--station-repeat-value-min", type=float, default=DEFAULT_STATION_REPEAT_VALUE_MIN, help="站端连续重复硬异常：重复值需大于该值")
    parser.add_argument("--disable-station-repeat-hard-anomaly", action="store_true", help="不读取/不使用站端连续重复硬异常")

    parser.add_argument(
        "--fan-joint-repeat-hard-mode",
        choices=["marker", "final"],
        default="marker",
        help="风机联合重复段如何进入最终异常：marker=只标记不强制并入；final=强制并入。默认 marker，便于分析修正效果。",
    )
    parser.add_argument(
        "--station-repeat-hard-mode",
        choices=["off", "marker", "final"],
        default="final",
        help="站端功率连续重复段如何进入最终异常：默认 final。",
    )
    parser.add_argument(
        "--physical-outlier-hard-mode",
        choices=["marker", "final"],
        default="final",
        help="超额定物理异常如何进入最终异常：默认 final。",
    )
    parser.add_argument("--save-repeat-power-timeseries", action="store_true", help="额外输出 repeat_power_timeseries.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.quantile_lower_pct < args.quantile_upper_pct <= 100):
        raise ValueError("请保证 0 <= quantile-lower-pct < quantile-upper-pct <= 100")

    station_dir = Path(args.station_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    station_name = args.station_name.strip() if args.station_name else station_dir.name

    print("=" * 72)
    print("STEP2：江苏场站全站功率异常检测（经验百分位带版）")
    print("=" * 72)
    print(f"场站目录: {station_dir}")
    print(f"模型目录: {model_dir}")
    print(f"输出目录: {output_dir}")
    print(f"场站名称: {station_name}")
    print(f"使用方案: {args.scheme_code}")
    print(f"判异经验带: P{args.quantile_lower_pct:02d}-P{args.quantile_upper_pct:02d}")

    model_row, empirical_df = load_model_files(
        model_dir=model_dir,
        scheme_code=args.scheme_code,
        lower_pct=args.quantile_lower_pct,
        upper_pct=args.quantile_upper_pct,
    )

    rated_power_mw = args.rated_power_mw
    rated_power_margin = args.rated_power_margin
    if rated_power_mw is None:
        rated_power_mw = _to_float_or_default(model_row.get("rated_power_mw", np.nan), DEFAULT_RATED_POWER_MW)
    if rated_power_margin is None:
        rated_power_margin = _to_float_or_default(model_row.get("rated_power_margin", np.nan), DEFAULT_RATED_POWER_MARGIN)

    ultra_low_power_max_mw, ultra_low_loss_ratio_min = infer_ultra_low_params(
        model_row=model_row,
        empirical_df=empirical_df,
        default_power_max=args.ultra_low_power_max_mw,
        default_loss_ratio_min=args.ultra_low_loss_ratio_min,
    )

    scada_df, bad_physical_df, fan_cols = load_station_scada(
        station_dir=station_dir,
        rated_power_mw=rated_power_mw,
        rated_power_margin=rated_power_margin,
    )
    print(f"读取主数据行数: {len(scada_df):,}")
    print(f"识别风机功率列数: {len(fan_cols):,}")
    print(f"物理越限点数: {len(bad_physical_df):,}")

    fan_repeat_df = load_fan_joint_repeat_segments(station_dir, args.fan_joint_repeat_rel)
    station_repeat_df = load_station_power_repeat_segments(
        station_dir=station_dir,
        rel_path=args.station_repeat_rel,
        min_length=args.station_repeat_min_length,
        value_min=args.station_repeat_value_min,
        enabled=not args.disable_station_repeat_hard_anomaly and args.station_repeat_hard_mode != "off",
    )
    print(f"风机联合重复段数: {len(fan_repeat_df):,}")
    print(f"站端功率连续重复硬异常段数: {len(station_repeat_df):,}")

    repeat_power_s = build_repeat_power_series(scada_df["timestamp"], fan_repeat_df)
    scada_df["重复风机功率之和MW"] = repeat_power_s.reindex(pd.to_datetime(scada_df["timestamp"])).fillna(0.0).values
    scada_df["重复风机功率之和MW"] = pd.to_numeric(scada_df["重复风机功率之和MW"], errors="coerce").fillna(0.0).clip(lower=0.0)

    scada_df["修正后风机汇总功率MW"] = np.where(
        np.isfinite(scada_df["原风机汇总功率MW"]),
        np.maximum(scada_df["原风机汇总功率MW"] - scada_df["重复风机功率之和MW"], 0.0),
        np.nan,
    )
    scada_df["修正后损耗MW"] = scada_df["修正后风机汇总功率MW"] - scada_df["CT有效功率MW"]
    scada_df["是否风机联合重复"] = build_interval_mask(scada_df["timestamp"], fan_repeat_df)
    scada_df["是否站端功率重复硬异常"] = build_interval_mask(scada_df["timestamp"], station_repeat_df)

    detected_df = apply_quantile_detection(
        scada_df=scada_df,
        model_row=model_row,
        empirical_df=empirical_df,
        ultra_low_power_max_mw=ultra_low_power_max_mw,
        ultra_low_loss_ratio_min=ultra_low_loss_ratio_min,
        lower_pct=args.quantile_lower_pct,
        upper_pct=args.quantile_upper_pct,
        severity_bandwidth_floor=args.severity_bandwidth_floor,
    )
    detected_df = combine_final_anomaly_flags(
        detected_df,
        fan_joint_hard_mode=args.fan_joint_repeat_hard_mode,
        station_repeat_hard_mode=args.station_repeat_hard_mode,
        physical_outlier_hard_mode=args.physical_outlier_hard_mode,
    )

    detail_df = build_detail_df(detected_df, station_name=station_name)
    summary_df = build_summary_df(detected_df, station_name=station_name, args=args)

    detail_path = output_dir / "station_anomaly_detail.csv"
    summary_path = output_dir / "station_anomaly_summary.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_hard_segments_used(fan_repeat_df, station_repeat_df, bad_physical_df, output_dir)

    if args.save_repeat_power_timeseries:
        repeat_ts_path = output_dir / "repeat_power_timeseries.csv"
        repeat_ts_df = detected_df[["timestamp", "重复风机功率之和MW", "原风机汇总功率MW", "修正后风机汇总功率MW"]].rename(columns={
            "timestamp": "时间",
            "重复风机功率之和MW": "重复风机功率扣减量MW",
            "原风机汇总功率MW": "原始风机功率之和MW",
            "修正后风机汇总功率MW": "修正后风机功率之和MW",
        })
        repeat_ts_df.to_csv(repeat_ts_path, index=False, encoding="utf-8-sig")

    print("\n检测完成：")
    print(f"  - 异常明细: {detail_path}")
    print(f"  - 汇总结果: {summary_path}")
    print(f"  - 硬异常段清单: {output_dir / 'hard_anomaly_segments_used_step2.csv'}")
    if args.save_repeat_power_timeseries:
        print(f"  - 重复功率时序: {output_dir / 'repeat_power_timeseries.csv'}")

    if len(summary_df) > 0:
        row = summary_df.iloc[0]
        print("\n关键统计（已剔除物理越限点）：")
        print(f"  原始总行数: {int(row['原始总行数']):,} | 纳入汇总统计行数: {int(row['纳入汇总统计行数']):,} | 剔除物理越限行数: {int(row['剔除物理越限行数']):,}")
        print(f"  原异常行数: {int(row['原异常行数（纳入统计）']):,} / 原有效检测行数: {int(row['原有效检测行数（纳入统计）']):,}")
        print(f"  修正后异常行数: {int(row['修正后异常行数（纳入统计）']):,} / 修正后有效检测行数: {int(row['修正后有效检测行数（纳入统计）']):,}")
        print(f"  原异常程度总和: {float(row['原异常程度总和（纳入统计）']):.4f}")
        print(f"  修正后异常程度总和: {float(row['修正后异常程度总和（纳入统计）']):.4f}")
        print(f"  异常程度下降量: {float(row['异常程度下降量（原-修正后）']):.4f}")
        print(f"  发生重复功率修正的行数: {int(row['重复功率修正行数（纳入统计）']):,}")
        print(f"  站端连续重复硬异常分钟数: {int(row['站端功率重复硬异常分钟数（纳入统计）']):,}")
        print(f"  风机联合重复标记分钟数: {int(row['风机联合重复标记分钟数（纳入统计）']):,}")
        print(f"  硬异常合并模式: fan_joint={args.fan_joint_repeat_hard_mode}, station_repeat={args.station_repeat_hard_mode}, physical={args.physical_outlier_hard_mode}")


if __name__ == "__main__":
    main()
