#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP8：江苏场站 15min 预测实验：全站直接预测 vs 单机预测汇总。

输入：
    STEP7 输出的 15min 建模宽表：
        场站数据/<场站名>/step7-15min建模数据/<场站名>_15min_model_wide.csv

核心筛选：
    1. DATA_15MIN_USE_RESULT == “可用”
    2. 默认剔除限电时刻：
       非限电 = LIMIT_POWER_15MIN 为空、<=0、或 >= curtail_ratio * rated_power_mw
       限电 = 0 < LIMIT_POWER_15MIN < curtail_ratio * rated_power_mw

模型：
    lightgbm、xgboost、random_forest、ridge

方案：
    station_direct:
        直接预测未来 ACTIVE_POWER_STATION_15MIN

    turbine_sum_raw:
        每台风机单独预测 FINAL_ACTIVE_POWER_#n_15MIN，然后求和

    turbine_sum_loss_q50:
        每台风机单独预测后求和，再用 STEP1 损耗中位数曲线 Q50 插值得到损耗：
        station_pred = sum_fan_pred - loss_q50_interp(sum_fan_pred)

损耗曲线：
    优先读取 STEP1 输出的 P_med / Q50。
    若 STEP1 文件同时包含“站端功率分段模型”和“风机总功率分段模型”，
    STEP8 默认只使用“风机总功率分段模型”。
    若未指定 --loss-curve-file，会在场站目录下自动搜索包含 P_med 和 Q50 的 csv。
    若找不到，可用 --allow-train-loss-fallback 从训练集重新构造中位数损耗曲线作为兜底。

输出：
    metrics_summary.csv
    predictions_all.csv
    curtail_summary.csv
    sample_summary.csv
    loss_curve_used.csv
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 默认配置
# ============================================================
DEFAULT_BASE_DIR = r"场站数据"
DEFAULT_STATION_NAME = "LGXRFD"
DEFAULT_STEP7_DIR_NAME = "step7-15min建模数据"
DEFAULT_STEP8_DIR_NAME = "step8-预测实验"

RATED_POWER_BY_STATION = {
    "JMZSFD": 300.0,
    "LGXRFD": 400.0,
}

DEFAULT_MODELS = ["lightgbm", "xgboost", "random_forest", "ridge"]

DEFAULT_M = 16
DEFAULT_N_LIST = [1, 2,3,4,5,6,7, 8,9,10,11, 12,13,14,15,16,17,18,19,20,21,22,23,24]
DEFAULT_FREQ_MINUTES = 15
DEFAULT_CURTAIL_RATIO = 0.95
DEFAULT_FAN_UPPER_FACTOR = 1.05
DEFAULT_LOSS_MODEL_TYPE = "风机总功率分段模型"
RANDOM_SEED = 42

TARGET_COL = "ACTIVE_POWER_STATION_15MIN"
LIMIT_COL = "LIMIT_POWER_15MIN"
DATA_USE_COL = "DATA_15MIN_USE_RESULT"
FINAL_FAN_SUM_COL = "FINAL_FAN_POWER_SUM_MW_15MIN"
FINAL_LOSS_COL = "FINAL_LOSS_MW_15MIN"
STATION_FEATURE_COLS = [
    "ACTIVE_POWER_STATION_15MIN",
    "FINAL_FAN_POWER_SUM_MW_15MIN",
    "FINAL_LOSS_MW_15MIN",
]


# ============================================================
# 通用工具
# ============================================================
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


def safe_threshold_str(x: float) -> str:
    return f"{float(x):.10g}".replace(".", "p").replace("-", "minus")


def build_station_dir(base_dir: str | Path, station_name: str) -> Path:
    return Path(base_dir) / str(station_name)


def default_step7_file(base_dir: str | Path, station_name: str, step7_dir_name: str = DEFAULT_STEP7_DIR_NAME) -> Path:
    return build_station_dir(base_dir, station_name) / step7_dir_name / f"{station_name}_15min_model_wide.csv"


def default_output_dir(base_dir: str | Path, station_name: str, step8_dir_name: str = DEFAULT_STEP8_DIR_NAME) -> Path:
    return build_station_dir(base_dir, station_name) / step8_dir_name


def parse_n_list(s: str) -> List[int]:
    if not s:
        return DEFAULT_N_LIST.copy()
    out = []
    for part in str(s).replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def parse_models(s: str) -> List[str]:
    if not s:
        return DEFAULT_MODELS.copy()
    return [x.strip() for x in str(s).replace("，", ",").split(",") if x.strip()]


def extract_fan_power_cols(columns: Iterable[str]) -> List[str]:
    pattern = re.compile(r"^FINAL_ACTIVE_POWER_#(\d+)_15MIN$")
    pairs = []
    for c in columns:
        m = pattern.match(str(c).strip())
        if m:
            pairs.append((int(m.group(1)), str(c).strip()))
    return [c for _, c in sorted(pairs, key=lambda x: x[0])]


def fan_unit_to_mw_scale(unit: str) -> float:
    """
    返回“原始单机功率数值 × scale = MW”的系数。
    - kW: 1 kW = 0.001 MW
    - MW: 1 MW = 1 MW
    """
    u = str(unit).strip().lower()
    if u in ["kw", "kW".lower()]:
        return 0.001
    if u == "mw":
        return 1.0
    raise ValueError(f"未知单机功率单位: {unit}，仅支持 kW/MW")


def fan_mw_to_native_scale(unit: str) -> float:
    """
    返回“MW × scale = 单机列原始单位”的系数。
    """
    return 1.0 / fan_unit_to_mw_scale(unit)


def normalize_power_units_for_step8(
    df: pd.DataFrame,
    fan_cols: List[str],
    fan_power_unit: str = "kW",
    station_power_unit: str = "MW",
) -> pd.DataFrame:
    """
    STEP8 内部统一约定：
    - 全站功率、风机功率和、损耗曲线使用 MW
    - 单机模型的目标列仍使用原始单机单位，例如 kW
      这样单机模型预测后先求和，再换算为 MW 与全站功率比较。

    如果 fan_power_unit=kW：
    - FINAL_ACTIVE_POWER_#n_15MIN 保持 kW，用于单机建模
    - FINAL_FAN_POWER_SUM_MW_15MIN 会在 STEP8 内重新计算为 sum(fan kW)/1000
    - FINAL_LOSS_MW_15MIN 会重新计算为 FINAL_FAN_POWER_SUM_MW_15MIN - ACTIVE_POWER_STATION_15MIN
    """
    out = df.copy()
    fan_to_mw = fan_unit_to_mw_scale(fan_power_unit)

    if station_power_unit.upper() != "MW":
        raise ValueError("当前 STEP8 只支持全站功率单位为 MW")

    if fan_cols:
        fan_sum_mw = out[fan_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True) * fan_to_mw
        out[FINAL_FAN_SUM_COL] = fan_sum_mw

        if TARGET_COL in out.columns:
            station_power = pd.to_numeric(out[TARGET_COL], errors="coerce")
            out[FINAL_LOSS_COL] = out[FINAL_FAN_SUM_COL] - station_power

    return out


def clean_df_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    if "timestamp" not in df.columns:
        raise ValueError("输入缺少 timestamp 列")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


# ============================================================
# 限电识别与数据筛选
# ============================================================
def add_curtail_flag(
    df: pd.DataFrame,
    rated_power_mw: float,
    curtail_ratio: float = DEFAULT_CURTAIL_RATIO,
    mode: str = "limit_threshold",
) -> pd.DataFrame:
    """
    默认限电规则：
        0 < LIMIT_POWER_15MIN < curtail_ratio * rated_power_mw
    认为限电。
    """
    out = df.copy()
    if LIMIT_COL not in out.columns:
        out["IS_CURTAILED"] = False
        out["CURTAIL_REASON"] = "无LIMIT_POWER_15MIN列，按非限电处理"
        return out

    limit = pd.to_numeric(out[LIMIT_COL], errors="coerce")
    threshold = float(curtail_ratio) * float(rated_power_mw)

    if mode == "off":
        is_curtail = pd.Series(False, index=out.index)
        reason = "未启用限电剔除"
    elif mode == "limit_positive":
        is_curtail = limit.fillna(0) > 0
        reason = "LIMIT_POWER_15MIN>0"
    elif mode == "limit_threshold":
        is_curtail = limit.notna() & (limit > 0) & (limit < threshold)
        reason = f"0<LIMIT_POWER_15MIN<{threshold:g}MW"
    else:
        raise ValueError(f"未知 curtail-mode: {mode}")

    out["IS_CURTAILED"] = is_curtail.fillna(False)
    out["CURTAIL_REASON"] = np.where(out["IS_CURTAILED"], reason, "非限电")
    return out


def filter_model_data(
    df: pd.DataFrame,
    exclude_curtail: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    if DATA_USE_COL not in out.columns:
        raise ValueError(f"输入缺少 {DATA_USE_COL} 列，请确认输入为 STEP7 输出")
    out = out[out[DATA_USE_COL].astype(str).str.strip().eq("可用")].copy()
    if exclude_curtail and "IS_CURTAILED" in out.columns:
        out = out[~out["IS_CURTAILED"].fillna(False)].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def apply_physical_filters(
    df: pd.DataFrame,
    fan_cols: List[str],
    rated_power_mw: float,
    station_upper_factor: float = 1.20,
    fan_upper_factor: float = DEFAULT_FAN_UPPER_FACTOR,
    loss_abs_upper_factor: float = 0.50,
    fan_power_unit: str = "kW",
    fan_rated_power_mw: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对 STEP7 15min 建模表做物理范围过滤，避免单机模型被极端异常值带崩。

    注意：
    - 不直接删除整行所有信息，而是将越界的风机功率置为 NaN；
      后续构造该风机样本时会自动跳过包含 NaN 的窗口。
    - 对全站关键字段越界的行直接剔除。
    """
    out = df.copy()
    before_rows = len(out)

    station_upper = rated_power_mw * station_upper_factor
    fan_count = max(len(fan_cols), 1)
    if fan_rated_power_mw is None or not np.isfinite(fan_rated_power_mw) or fan_rated_power_mw <= 0:
        fan_rated_power_mw = rated_power_mw / fan_count
    fan_upper_mw = fan_rated_power_mw * fan_upper_factor
    fan_upper = fan_upper_mw * fan_mw_to_native_scale(fan_power_unit)
    loss_abs_upper = rated_power_mw * loss_abs_upper_factor

    summary = []

    def add_summary(item, value):
        summary.append({"统计项": item, "数值": value})

    add_summary("物理过滤前建模行数", before_rows)
    add_summary("场站功率上限MW", station_upper)
    add_summary("单台风机额定功率MW", fan_rated_power_mw)
    add_summary("单台风机功率上限MW", fan_upper_mw)
    add_summary(f"单台风机功率上限{fan_power_unit}", fan_upper)
    add_summary("损耗绝对值上限MW", loss_abs_upper)

    # 全站目标/汇总功率越界：整行剔除
    row_mask = pd.Series(True, index=out.index)

    if TARGET_COL in out.columns:
        y = pd.to_numeric(out[TARGET_COL], errors="coerce")
        bad = y.notna() & ((y < 0) | (y > station_upper))
        add_summary("剔除：ACTIVE_POWER_STATION_15MIN越界行数", int(bad.sum()))
        row_mask &= ~bad

    if FINAL_FAN_SUM_COL in out.columns:
        p_sum = pd.to_numeric(out[FINAL_FAN_SUM_COL], errors="coerce")
        bad = p_sum.notna() & ((p_sum < 0) | (p_sum > station_upper))
        add_summary("剔除：FINAL_FAN_POWER_SUM_MW_15MIN越界行数", int(bad.sum()))
        row_mask &= ~bad

    if FINAL_LOSS_COL in out.columns:
        loss = pd.to_numeric(out[FINAL_LOSS_COL], errors="coerce")
        bad = loss.notna() & (loss.abs() > loss_abs_upper)
        add_summary("剔除：FINAL_LOSS_MW_15MIN绝对值越界行数", int(bad.sum()))
        row_mask &= ~bad

    out = out.loc[row_mask].copy()

    # 单机功率越界：置 NaN，不整行删除
    total_bad_fan_points = 0
    for col in fan_cols:
        s = pd.to_numeric(out[col], errors="coerce")
        bad = s.notna() & ((s < 0) | (s > fan_upper))
        n_bad = int(bad.sum())
        if n_bad:
            out.loc[bad, col] = np.nan
            total_bad_fan_points += n_bad

    add_summary("置空：单机FINAL_ACTIVE_POWER越界点数", total_bad_fan_points)
    add_summary("物理过滤后建模行数", len(out))
    add_summary("物理过滤剔除行数", before_rows - len(out))

    return out.reset_index(drop=True), pd.DataFrame(summary)


def make_curtail_summary(df: pd.DataFrame, filtered_df: pd.DataFrame, rated_power_mw: float, curtail_ratio: float) -> pd.DataFrame:
    limit = pd.to_numeric(df.get(LIMIT_COL, pd.Series(index=df.index, dtype=float)), errors="coerce")
    rows = [
        {"统计项": "场站额定功率MW", "数值": rated_power_mw},
        {"统计项": "限电阈值比例", "数值": curtail_ratio},
        {"统计项": "限电阈值MW", "数值": rated_power_mw * curtail_ratio},
        {"统计项": "STEP7总15min窗口数", "数值": len(df)},
        {"统计项": "DATA_15MIN_USE_RESULT=可用窗口数", "数值": int(df.get(DATA_USE_COL, pd.Series(dtype=str)).astype(str).eq("可用").sum()) if DATA_USE_COL in df.columns else np.nan},
        {"统计项": "LIMIT_POWER_15MIN为空窗口数", "数值": int(limit.isna().sum())},
        {"统计项": "LIMIT_POWER_15MIN<=0窗口数", "数值": int((limit <= 0).fillna(False).sum())},
        {"统计项": "LIMIT_POWER_15MIN>0窗口数", "数值": int((limit > 0).fillna(False).sum())},
        {"统计项": "识别为限电窗口数", "数值": int(df.get("IS_CURTAILED", pd.Series(False, index=df.index)).fillna(False).sum())},
        {"统计项": "最终建模窗口数", "数值": len(filtered_df)},
    ]
    return pd.DataFrame(rows)


# ============================================================
# 损耗曲线：STEP1 Q50 插值
# ============================================================
def find_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def try_read_loss_curve_file(
    path: Path,
    loss_model_type: str = DEFAULT_LOSS_MODEL_TYPE,
    strict_model_type: bool = True,
) -> Optional[pd.DataFrame]:
    """
    读取 STEP1 损耗中位数曲线。

    关键点：
    fit_empirical_quantile_bins.csv 可能同时包含：
      - 站端功率分段模型
      - 风机总功率分段模型

    STEP8 的 turbine_sum_loss_q50 逻辑是：
        预测风机总功率 -> 查损耗 -> 扣损耗
    因此必须使用“风机总功率分段模型”。

    如果文件中存在模型类型列，则按 loss_model_type 筛选。
    如果不存在模型类型列：
      - strict_model_type=True 时认为不可靠，返回 None；
      - strict_model_type=False 时退化为直接读取 P_med/Q50。
    """
    try:
        df = read_csv_auto(path)
        df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

        model_col = find_existing_col(df, [
            "scheme_name", "分段模型", "模型类型", "功率分段模型",
            "model_type", "model", "curve_type", "分箱模型"
        ])

        if model_col is not None:
            model_s = df[model_col].astype(str).str.strip()
            target = str(loss_model_type).strip()
            print(f"损耗曲线模型列: {model_col}；目标模型: {target}")
            mask = model_s.eq(target)

            # 兼容有些文件写法略有差异，只要同时包含“风机”和“总功率”也认为是目标模型。
            if not mask.any() and target == "风机总功率分段模型":
                mask = model_s.str.contains("风机", na=False) & model_s.str.contains("总功率", na=False)

            if not mask.any():
                available = sorted(model_s.dropna().unique().tolist())
                print(f"⚠️ 损耗曲线文件 {path} 中没有目标模型：{target}；可用模型：{available}")
                return None

            df = df.loc[mask].copy()
        elif strict_model_type:
            print(f"⚠️ 损耗曲线文件 {path} 缺少模型类型列，无法确认是否为“{loss_model_type}”，跳过。")
            return None

        p_col = find_existing_col(df, [
            "P_med", "功率分箱中位数MW", "功率中位数MW", "P_mid", "bin_mid", "power_mid",
            "风机总功率分箱中位数MW", "风机总功率中位数MW"
        ])
        q50_col = find_existing_col(df, [
            "Q50", "P50", "损耗中位数MW", "loss_median", "median_loss", "中位数"
        ])

        if p_col is None or q50_col is None:
            return None

        out = pd.DataFrame({
            "P_med": pd.to_numeric(df[p_col], errors="coerce"),
            "Q50": pd.to_numeric(df[q50_col], errors="coerce"),
        }).dropna()

        out = out[np.isfinite(out["P_med"]) & np.isfinite(out["Q50"])]
        out = out.sort_values("P_med").drop_duplicates(subset=["P_med"], keep="last").reset_index(drop=True)

        if len(out) < 2:
            return None

        out["loss_model_type"] = loss_model_type
        out["loss_curve_file"] = str(path)
        out["loss_model_column"] = str(model_col) if model_col is not None else ""
        return out
    except Exception as exc:
        print(f"⚠️ 读取损耗曲线失败：{path}；{exc}")
        return None


def find_loss_curve_file(
    station_dir: Path,
    loss_model_type: str = DEFAULT_LOSS_MODEL_TYPE,
    strict_model_type: bool = True,
) -> Optional[Path]:
    patterns = [
        "*fit_empirical_quantile_bins*.csv",
        "*empirical*quantile*.csv",
        "*quantile*.csv",
        "*分位*.csv",
        "*经验*.csv",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(station_dir.rglob(pat))
    candidates = [p for p in candidates if p.is_file() and "summary" not in p.name.lower()]
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if try_read_loss_curve_file(p, loss_model_type=loss_model_type, strict_model_type=strict_model_type) is not None:
            return p
    return None



def build_train_loss_curve(
    train_df: pd.DataFrame,
    bin_count: int = 80,
    min_bin_count: int = 10,
) -> pd.DataFrame:
    p = pd.to_numeric(train_df[FINAL_FAN_SUM_COL], errors="coerce")
    loss = pd.to_numeric(train_df[FINAL_LOSS_COL], errors="coerce")
    tmp = pd.DataFrame({"P": p, "loss": loss}).dropna()
    tmp = tmp[np.isfinite(tmp["P"]) & np.isfinite(tmp["loss"])]
    if len(tmp) < max(50, min_bin_count * 3):
        raise ValueError("训练集样本太少，无法兜底构建损耗曲线")

    tmp["bin"] = pd.qcut(tmp["P"], q=min(bin_count, max(3, len(tmp) // min_bin_count)), duplicates="drop")
    g = tmp.groupby("bin", observed=True)
    curve = g.agg(P_med=("P", "median"), Q50=("loss", "median"), count=("loss", "size")).reset_index(drop=True)
    curve = curve[curve["count"] >= min_bin_count].copy()
    curve = curve.sort_values("P_med").reset_index(drop=True)
    if len(curve) < 2:
        raise ValueError("有效损耗分箱太少，无法构建损耗曲线")
    return curve[["P_med", "Q50", "count"]]


def load_loss_curve(
    station_dir: Path,
    loss_curve_file: Optional[str | Path] = None,
    train_df_for_fallback: Optional[pd.DataFrame] = None,
    allow_train_fallback: bool = False,
    fallback_bin_count: int = 80,
    loss_model_type: str = DEFAULT_LOSS_MODEL_TYPE,
    strict_loss_model_type: bool = True,
) -> Tuple[pd.DataFrame, str]:
    if loss_curve_file:
        p = Path(loss_curve_file)
        curve = try_read_loss_curve_file(
            p,
            loss_model_type=loss_model_type,
            strict_model_type=strict_loss_model_type,
        )
        if curve is None:
            raise ValueError(f"指定的损耗曲线文件无法识别目标模型 {loss_model_type} 的 P_med/Q50：{p}")
        return curve, f"{p} | 模型={loss_model_type}"

    found = find_loss_curve_file(
        station_dir,
        loss_model_type=loss_model_type,
        strict_model_type=strict_loss_model_type,
    )
    if found is not None:
        curve = try_read_loss_curve_file(
            found,
            loss_model_type=loss_model_type,
            strict_model_type=strict_loss_model_type,
        )
        if curve is not None:
            return curve, f"{found} | 模型={loss_model_type}"

    if allow_train_fallback and train_df_for_fallback is not None:
        curve = build_train_loss_curve(train_df_for_fallback, bin_count=fallback_bin_count)
        curve["loss_model_type"] = "训练集兜底：风机总功率"
        return curve, "训练集兜底构建：FINAL_FAN_POWER_SUM_MW_15MIN -> FINAL_LOSS_MW_15MIN"

    raise FileNotFoundError(
        f"未找到 STEP1 损耗中位数曲线文件中的目标模型：{loss_model_type}。"
        "请用 --loss-curve-file 指定，或加 --allow-train-loss-fallback 使用训练集兜底构建。"
    )



def interp_loss(loss_curve: pd.DataFrame, p_sum: np.ndarray) -> np.ndarray:
    curve = loss_curve.dropna(subset=["P_med", "Q50"]).sort_values("P_med")
    x = curve["P_med"].to_numpy(dtype=float)
    y = curve["Q50"].to_numpy(dtype=float)
    if len(x) < 2:
        return np.zeros_like(p_sum, dtype=float)
    return np.interp(p_sum.astype(float), x, y, left=y[0], right=y[-1])


# ============================================================
# 样本构造与时间切分
# ============================================================
def prepare_continuity_flags(timestamps: pd.Series, max_gap_seconds: int) -> np.ndarray:
    diffs = timestamps.diff().dt.total_seconds()
    return (diffs <= max_gap_seconds).fillna(True).to_numpy(dtype=bool)


def prepare_break_prefix_sum(continuity_flags: np.ndarray) -> np.ndarray:
    breaks = (~continuity_flags).astype(np.int32)
    return np.concatenate([[0], np.cumsum(breaks)])


def get_split_times(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.Timestamp, pd.Timestamp]:
    ts = df["timestamp"].sort_values().reset_index(drop=True)
    if len(ts) < 10:
        raise ValueError("建模数据太少，无法切分")
    train_idx = int(len(ts) * train_ratio)
    val_idx = int(len(ts) * (train_ratio + val_ratio))
    train_idx = min(max(train_idx, 1), len(ts) - 2)
    val_idx = min(max(val_idx, train_idx + 1), len(ts) - 1)
    return ts.iloc[train_idx], ts.iloc[val_idx]


def split_name_for_ts(ts: pd.Timestamp, split_val_time: pd.Timestamp, split_test_time: pd.Timestamp) -> str:
    if ts < split_val_time:
        return "train"
    if ts < split_test_time:
        return "val"
    return "test"


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    splits: np.ndarray


def build_supervised_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    M: int,
    N: int,
    freq_minutes: int,
    split_val_time: pd.Timestamp,
    split_test_time: pd.Timestamp,
) -> Dataset:
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"].reset_index(drop=True)
    max_gap_seconds = int(freq_minutes * 60)

    values_x = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values_y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)

    ts_to_idx: Dict[pd.Timestamp, int] = {pd.Timestamp(t): i for i, t in enumerate(timestamps)}
    continuity = prepare_continuity_flags(timestamps, max_gap_seconds)
    break_prefix = prepare_break_prefix_sum(continuity)

    upper = max(0, len(df) - M)
    X_list, y_list, ts_list, split_list = [], [], [], []
    target_delta = pd.Timedelta(minutes=N * freq_minutes)

    for i in range(upper):
        hist_end_idx = i + M - 1
        # 历史窗口 [i, i+M) 内不能有时间断点
        if break_prefix[i + M] - break_prefix[i + 1] > 0:
            continue

        target_time = pd.Timestamp(timestamps.iloc[hist_end_idx]) + target_delta
        target_idx = ts_to_idx.get(target_time)
        if target_idx is None:
            continue

        hist = values_x[i : i + M, :]
        target = values_y[target_idx]
        if not np.isfinite(hist).all() or not np.isfinite(target):
            continue

        X_list.append(hist.reshape(-1).astype(np.float32))
        y_list.append(float(target))
        ts_list.append(target_time)
        split_list.append(split_name_for_ts(target_time, split_val_time, split_test_time))

    if not X_list:
        return Dataset(
            X=np.empty((0, len(feature_cols) * M), dtype=np.float32),
            y=np.empty((0,), dtype=np.float32),
            timestamps=np.array([], dtype="datetime64[ns]"),
            splits=np.array([], dtype=object),
        )

    return Dataset(
        X=np.vstack(X_list).astype(np.float32),
        y=np.asarray(y_list, dtype=np.float32),
        timestamps=np.asarray(ts_list),
        splits=np.asarray(split_list, dtype=object),
    )


# ============================================================
# 模型
# ============================================================
def make_model(model_name: str):
    name = model_name.lower()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_SEED))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise ImportError(f"lightgbm 未安装或不可用：{exc}")
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError(f"xgboost 未安装或不可用：{exc}")
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        )
    raise ValueError(f"未知模型：{model_name}")


def fit_predict_model(model_name: str, ds: Dataset) -> pd.DataFrame:
    train_mask = ds.splits == "train"
    eval_mask = ds.splits != "train"

    if train_mask.sum() < 10 or eval_mask.sum() < 1:
        raise ValueError(f"样本不足：train={train_mask.sum()}, eval={eval_mask.sum()}")

    model = make_model(model_name)
    model.fit(ds.X[train_mask], ds.y[train_mask])
    pred = model.predict(ds.X[eval_mask])

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(ds.timestamps[eval_mask]),
        "split": ds.splits[eval_mask],
        "true_value": ds.y[eval_mask].astype(float),
        "predicted_value": pred.astype(float),
    })
    return out


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, rated_power_mw: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "NMAE_by_capacity": np.nan, "sample_count": 0}
    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan
    nmae = mae / rated_power_mw if rated_power_mw and rated_power_mw > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "NMAE_by_capacity": nmae, "sample_count": len(y_true)}


def append_metrics(
    rows: List[dict],
    pred_df: pd.DataFrame,
    model_name: str,
    horizon_steps: int,
    scheme: str,
    rated_power_mw: float,
    freq_minutes: int = DEFAULT_FREQ_MINUTES,
) -> None:
    """
    统一计算指标。

    兼容两类列名：
    - 新版方案统一列：true_station_power / pred_station_power
    - 老版模型输出列：true_value / predicted_value
    """
    if {"true_station_power", "pred_station_power"}.issubset(pred_df.columns):
        true_col = "true_station_power"
        pred_col = "pred_station_power"
    elif {"true_value", "predicted_value"}.issubset(pred_df.columns):
        true_col = "true_value"
        pred_col = "predicted_value"
    else:
        raise KeyError(
            "预测结果缺少指标计算列；需要 true_station_power/pred_station_power "
            "或 true_value/predicted_value"
        )

    for split in ["val", "test"]:
        sub = pred_df[pred_df["split"] == split]
        m = calc_metrics(sub[true_col].to_numpy(), sub[pred_col].to_numpy(), rated_power_mw)
        rows.append({
            "model": model_name,
            "horizon_steps": horizon_steps,
            "horizon_minutes": horizon_steps * freq_minutes,
            "scheme": scheme,
            "split": split,
            **m,
        })


# ============================================================
# 方案运行
# ============================================================
def run_station_direct(
    df: pd.DataFrame,
    model_name: str,
    M: int,
    N: int,
    freq_minutes: int,
    split_val_time: pd.Timestamp,
    split_test_time: pd.Timestamp,
) -> pd.DataFrame:
    feature_cols = [c for c in STATION_FEATURE_COLS if c in df.columns]
    if TARGET_COL not in df.columns:
        raise ValueError(f"缺少目标列：{TARGET_COL}")
    if not feature_cols:
        raise ValueError("没有可用的全站直接预测特征列")

    ds = build_supervised_dataset(df, feature_cols, TARGET_COL, M, N, freq_minutes, split_val_time, split_test_time)
    pred = fit_predict_model(model_name, ds)
    pred = pred.rename(columns={"true_value": "true_station_power", "predicted_value": "pred_station_power"})
    pred["scheme"] = "station_direct"
    return pred


def run_turbine_models(
    df: pd.DataFrame,
    fan_cols: List[str],
    model_name: str,
    M: int,
    N: int,
    freq_minutes: int,
    split_val_time: pd.Timestamp,
    split_test_time: pd.Timestamp,
    fan_pred_min_mw: float = 0.0,
    fan_pred_max_mw: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    每台风机单独训练模型，返回：
    - sum_pred_df：timestamp/split/pred_fan_sum
    - turbine_sample_summary：每台风机样本数统计
    """
    pred_parts = []
    sample_rows = []

    for idx, col in enumerate(fan_cols, start=1):
        try:
            ds = build_supervised_dataset(df, [col], col, M, N, freq_minutes, split_val_time, split_test_time)
            train_count = int((ds.splits == "train").sum())
            val_count = int((ds.splits == "val").sum())
            test_count = int((ds.splits == "test").sum())
            sample_rows.append({
                "fan_col": col,
                "train_samples": train_count,
                "val_samples": val_count,
                "test_samples": test_count,
            })
            if train_count < 10 or (val_count + test_count) < 1:
                print(f"  ⚠️ 风机 {col} 样本不足，跳过")
                continue

            pred = fit_predict_model(model_name, ds)
            pred = pred.rename(columns={"predicted_value": col + "_pred"})

            # 单机预测必须满足物理范围，避免少数模型外推到天文数值。
            pred_col = col + "_pred"
            pred[pred_col] = pd.to_numeric(pred[pred_col], errors="coerce")
            pred[pred_col] = pred[pred_col].clip(lower=fan_pred_min_mw)
            if fan_pred_max_mw is not None and np.isfinite(fan_pred_max_mw):
                pred[pred_col] = pred[pred_col].clip(upper=fan_pred_max_mw)

            pred = pred[["timestamp", "split", pred_col]]
            pred_parts.append(pred)
        except Exception as exc:
            print(f"  ⚠️ 风机 {col} 训练失败：{exc}")
            sample_rows.append({"fan_col": col, "error": str(exc)})

        if idx % 20 == 0:
            print(f"  已完成风机模型 {idx}/{len(fan_cols)}")
            gc.collect()

    if not pred_parts:
        raise RuntimeError("没有任何风机模型成功生成预测")

    # 按 timestamp/split 内连接，只有所有成功风机都有预测的时刻才参与求和。
    merged = pred_parts[0]
    for part in pred_parts[1:]:
        merged = merged.merge(part, on=["timestamp", "split"], how="inner")

    pred_cols = [c for c in merged.columns if c.endswith("_pred")]
    merged["pred_fan_sum_native"] = merged[pred_cols].sum(axis=1)
    sum_pred = merged[["timestamp", "split", "pred_fan_sum_native"]].copy()
    sample_summary = pd.DataFrame(sample_rows)
    return sum_pred, sample_summary


def attach_true_station(df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    truth = df[["timestamp", TARGET_COL]].copy()
    truth["timestamp"] = pd.to_datetime(truth["timestamp"])
    out = pred_df.merge(truth, on="timestamp", how="left")
    out = out.rename(columns={TARGET_COL: "true_station_power"})
    return out


# ============================================================
# main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="STEP8：比较全站直接预测 vs 单机预测汇总的15min预测实验")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="场站数据根目录")
    parser.add_argument("--station-name", default=DEFAULT_STATION_NAME, help="场站名称，例如 LGXRFD 或 JMZSFD")
    parser.add_argument("--step7-dir-name", default=DEFAULT_STEP7_DIR_NAME, help="STEP7 输出目录名")
    parser.add_argument("--step8-dir-name", default=DEFAULT_STEP8_DIR_NAME, help="STEP8 输出目录名")
    parser.add_argument("--input-file", default=None, help="STEP7 15min宽表路径；不填则按场站名自动拼接")
    parser.add_argument("--output-dir", default=None, help="输出目录；不填则按场站名自动拼接")
    parser.add_argument("--rated-power-mw", type=float, default=None, help="场站额定功率MW；不填则按场站名自动识别")
    parser.add_argument("--curtail-ratio", type=float, default=DEFAULT_CURTAIL_RATIO, help="限电阈值比例，默认0.95")
    parser.add_argument("--curtail-mode", default="limit_threshold", choices=["off", "limit_positive", "limit_threshold"], help="限电识别模式")
    parser.add_argument("--include-curtail", action="store_true", help="若指定，则保留限电时刻参与建模")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="模型列表，逗号分隔")
    parser.add_argument("--M", type=int, default=DEFAULT_M, help="历史窗口长度，单位为15min步")
    parser.add_argument("--N-list", default=",".join(map(str, DEFAULT_N_LIST)), help="预测步长列表，单位为15min步，例如 1,4,8,12")
    parser.add_argument("--freq-minutes", type=int, default=DEFAULT_FREQ_MINUTES, help="数据频率，默认15min")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="训练集时间比例")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集时间比例")
    parser.add_argument("--loss-curve-file", default=None, help="STEP1 损耗中位数曲线文件；需包含目标模型的 P_med 和 Q50")
    parser.add_argument("--loss-model-type", default=DEFAULT_LOSS_MODEL_TYPE, help="STEP1损耗曲线中的模型类型；方案C必须使用“风机总功率分段模型”")
    parser.add_argument("--non-strict-loss-model-type", action="store_true", help="若损耗曲线文件缺少模型类型列，则允许直接读取P_med/Q50；不建议")
    parser.add_argument("--allow-train-loss-fallback", action="store_true", help="找不到STEP1损耗曲线时，允许用训练集兜底构建")
    parser.add_argument("--loss-fallback-bin-count", type=int, default=80, help="训练集兜底损耗曲线分箱数")
    parser.add_argument("--station-upper-factor", type=float, default=1.20, help="场站功率/风机功率和物理过滤上限倍率，默认1.2倍额定功率")
    parser.add_argument("--fan-power-unit", default="MW", choices=["kW", "MW"], help="STEP7中 FINAL_ACTIVE_POWER_#n_15MIN 的单位；单位最终版STEP7输出为MW")
    parser.add_argument("--station-power-unit", default="MW", choices=["MW"], help="全站功率单位，当前仅支持MW")
    parser.add_argument("--fan-rated-power-mw", type=float, default=None, help="单台风机额定功率MW；不填则使用场站额定功率/风机数量")
    parser.add_argument("--fan-upper-factor", type=float, default=DEFAULT_FAN_UPPER_FACTOR, help="单台风机功率物理过滤/预测裁剪上限倍率，默认1.05倍单机额定")
    parser.add_argument("--loss-abs-upper-factor", type=float, default=0.50, help="损耗绝对值过滤上限倍率，默认0.5倍场站额定")
    parser.add_argument("--disable-physical-filter", action="store_true", help="关闭物理范围过滤，不建议")
    parser.add_argument("--disable-pred-clip", action="store_true", help="关闭预测值物理裁剪，不建议")
    args = parser.parse_args()

    station_name = args.station_name
    station_dir = build_station_dir(args.base_dir, station_name)
    input_file = Path(args.input_file) if args.input_file else default_step7_file(args.base_dir, station_name, args.step7_dir_name)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.base_dir, station_name, args.step8_dir_name)
    ensure_dir(output_dir)

    rated_power = args.rated_power_mw
    if rated_power is None:
        rated_power = RATED_POWER_BY_STATION.get(station_name.upper())
    if rated_power is None:
        raise ValueError("未指定场站额定功率，请使用 --rated-power-mw")

    models = parse_models(args.models)
    n_list = parse_n_list(args.N_list)

    print("=" * 88)
    print("STEP8：15min预测实验")
    print("=" * 88)
    print(f"场站: {station_name}")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"额定功率: {rated_power:g} MW")
    print(f"限电规则: mode={args.curtail_mode}, ratio={args.curtail_ratio}, include_curtail={args.include_curtail}")
    print(f"模型: {models}")
    print(f"M={args.M}, N_LIST={n_list}")
    print(f"损耗曲线目标模型: {args.loss_model_type}")
    print(f"单机功率单位: {args.fan_power_unit}; 单台额定功率: {args.fan_rated_power_mw if args.fan_rated_power_mw else '自动'} MW; 单机上限倍率: {args.fan_upper_factor}")

    df0 = read_csv_auto(input_file)
    df0 = clean_df_for_model(df0)
    df0 = add_curtail_flag(df0, rated_power_mw=rated_power, curtail_ratio=args.curtail_ratio, mode=args.curtail_mode)
    df_model = filter_model_data(df0, exclude_curtail=(not args.include_curtail))

    if len(df_model) < 100:
        raise ValueError(f"最终建模数据太少：{len(df_model)} 行")

    fan_cols = extract_fan_power_cols(df_model.columns)
    if not fan_cols:
        raise ValueError("未识别到 FINAL_ACTIVE_POWER_#n_15MIN 风机功率列")

    if args.fan_rated_power_mw is None:
        args.fan_rated_power_mw = rated_power / max(len(fan_cols), 1)

    # 单机列通常是 kW，而全站/损耗曲线是 MW。
    # 在 STEP8 内部重新计算 FINAL_FAN_POWER_SUM_MW_15MIN 和 FINAL_LOSS_MW_15MIN，保证单位一致。
    df_model = normalize_power_units_for_step8(
        df_model,
        fan_cols=fan_cols,
        fan_power_unit=args.fan_power_unit,
        station_power_unit=args.station_power_unit,
    )

    physical_summary = pd.DataFrame()
    if not args.disable_physical_filter:
        df_model, physical_summary = apply_physical_filters(
            df_model,
            fan_cols=fan_cols,
            rated_power_mw=rated_power,
            station_upper_factor=args.station_upper_factor,
            fan_upper_factor=args.fan_upper_factor,
            loss_abs_upper_factor=args.loss_abs_upper_factor,
            fan_power_unit=args.fan_power_unit,
            fan_rated_power_mw=args.fan_rated_power_mw,
        )

    if len(df_model) < 100:
        raise ValueError(f"物理过滤后最终建模数据太少：{len(df_model)} 行")

    split_val_time, split_test_time = get_split_times(df_model, args.train_ratio, args.val_ratio)
    print(f"切分时间: val_start={split_val_time}, test_start={split_test_time}")
    print(f"最终建模窗口数: {len(df_model):,}; 风机列数: {len(fan_cols)}")

    # 先用训练段加载/兜底构建损耗曲线，避免验证/测试泄露。
    train_df_for_loss = df_model[df_model["timestamp"] < split_val_time].copy()
    loss_curve, loss_curve_source = load_loss_curve(
        station_dir=station_dir,
        loss_curve_file=args.loss_curve_file,
        train_df_for_fallback=train_df_for_loss,
        allow_train_fallback=args.allow_train_loss_fallback,
        fallback_bin_count=args.loss_fallback_bin_count,
        loss_model_type=args.loss_model_type,
        strict_loss_model_type=(not args.non_strict_loss_model_type),
    )
    print(f"损耗曲线来源: {loss_curve_source}")
    print(f"损耗曲线点数: {len(loss_curve)}")

    all_predictions = []
    metrics_rows = []
    turbine_sample_summaries = []

    start_all = time.time()

    for model_name in models:
        print(f"\n{'#' * 80}")
        print(f"# 模型: {model_name}")
        print(f"{'#' * 80}")

        # 先检查模型是否可创建，缺库就跳过。
        try:
            _ = make_model(model_name)
        except ImportError as exc:
            print(f"⚠️ 跳过模型 {model_name}: {exc}")
            continue

        for N in n_list:
            print(f"\n--- horizon N={N} ({N * args.freq_minutes} min) ---")
            t0 = time.time()

            # Scheme A：全站直接预测
            try:
                pred_station = run_station_direct(
                    df_model, model_name, args.M, N, args.freq_minutes, split_val_time, split_test_time
                )
                if not args.disable_pred_clip:
                    pred_station["pred_station_power"] = pd.to_numeric(pred_station["pred_station_power"], errors="coerce").clip(
                        lower=0.0,
                        upper=rated_power * args.station_upper_factor,
                    )
                pred_station["model"] = model_name
                pred_station["horizon_steps"] = N
                pred_station["horizon_minutes"] = N * args.freq_minutes
                pred_station["error"] = pred_station["pred_station_power"] - pred_station["true_station_power"]
                pred_station["abs_error"] = pred_station["error"].abs()
                all_predictions.append(pred_station)
                append_metrics(metrics_rows, pred_station, model_name, N, "station_direct", rated_power)
                print(f"  station_direct 预测完成: {len(pred_station):,} 条")
            except Exception as exc:
                print(f"  ❌ station_direct 失败: {exc}")

            # Scheme B/C：单机预测汇总
            try:
                fan_pred_max = None
                if not args.disable_pred_clip:
                    fan_pred_max = args.fan_rated_power_mw * args.fan_upper_factor * fan_mw_to_native_scale(args.fan_power_unit)

                fan_sum_pred, fan_sample_summary = run_turbine_models(
                    df_model,
                    fan_cols,
                    model_name,
                    args.M,
                    N,
                    args.freq_minutes,
                    split_val_time,
                    split_test_time,
                    fan_pred_min_mw=0.0,
                    fan_pred_max_mw=fan_pred_max,
                )
                fan_sample_summary["model"] = model_name
                fan_sample_summary["horizon_steps"] = N
                turbine_sample_summaries.append(fan_sample_summary)

                fan_sum_pred = attach_true_station(df_model, fan_sum_pred)
                fan_sum_pred["pred_fan_sum_mw"] = pd.to_numeric(
                    fan_sum_pred["pred_fan_sum_native"], errors="coerce"
                ) * fan_unit_to_mw_scale(args.fan_power_unit)

                # B: 不扣损耗
                pred_raw = fan_sum_pred.copy()
                pred_raw["scheme"] = "turbine_sum_raw"
                pred_raw["model"] = model_name
                pred_raw["horizon_steps"] = N
                pred_raw["horizon_minutes"] = N * args.freq_minutes
                if not args.disable_pred_clip:
                    pred_raw["pred_fan_sum_mw"] = pd.to_numeric(pred_raw["pred_fan_sum_mw"], errors="coerce").clip(
                        lower=0.0,
                        upper=rated_power * args.station_upper_factor,
                    )
                pred_raw["pred_station_power"] = pred_raw["pred_fan_sum_mw"]
                if not args.disable_pred_clip:
                    pred_raw["pred_station_power"] = pred_raw["pred_station_power"].clip(
                        lower=0.0,
                        upper=rated_power * args.station_upper_factor,
                    )
                pred_raw["error"] = pred_raw["pred_station_power"] - pred_raw["true_station_power"]
                pred_raw["abs_error"] = pred_raw["error"].abs()
                all_predictions.append(pred_raw[[
                    "timestamp", "split", "true_station_power", "pred_station_power",
                    "pred_fan_sum_native", "pred_fan_sum_mw", "scheme", "model", "horizon_steps", "horizon_minutes", "error", "abs_error"
                ]])
                append_metrics(metrics_rows, pred_raw.rename(columns={}), model_name, N, "turbine_sum_raw", rated_power)

                # C: 扣 STEP1 Q50 损耗
                pred_loss = fan_sum_pred.copy()
                if not args.disable_pred_clip:
                    pred_loss["pred_fan_sum_mw"] = pd.to_numeric(pred_loss["pred_fan_sum_mw"], errors="coerce").clip(
                        lower=0.0,
                        upper=rated_power * args.station_upper_factor,
                    )
                pred_loss["pred_loss_q50"] = interp_loss(loss_curve, pred_loss["pred_fan_sum_mw"].to_numpy(dtype=float))
                pred_loss["pred_station_power"] = pred_loss["pred_fan_sum_mw"] - pred_loss["pred_loss_q50"]
                if not args.disable_pred_clip:
                    pred_loss["pred_station_power"] = pred_loss["pred_station_power"].clip(
                        lower=0.0,
                        upper=rated_power * args.station_upper_factor,
                    )
                pred_loss["scheme"] = "turbine_sum_loss_q50"
                pred_loss["model"] = model_name
                pred_loss["horizon_steps"] = N
                pred_loss["horizon_minutes"] = N * args.freq_minutes
                pred_loss["error"] = pred_loss["pred_station_power"] - pred_loss["true_station_power"]
                pred_loss["abs_error"] = pred_loss["error"].abs()
                all_predictions.append(pred_loss[[
                    "timestamp", "split", "true_station_power", "pred_station_power",
                    "pred_fan_sum_native", "pred_fan_sum_mw", "pred_loss_q50",
                    "scheme", "model", "horizon_steps", "horizon_minutes", "error", "abs_error"
                ]])
                append_metrics(metrics_rows, pred_loss, model_name, N, "turbine_sum_loss_q50", rated_power)

                print(f"  turbine_sum 预测完成: {len(fan_sum_pred):,} 条")
            except Exception as exc:
                print(f"  ❌ turbine_sum 失败: {exc}")

            print(f"  用时: {time.time() - t0:.1f}s")
            gc.collect()

    if not all_predictions:
        raise RuntimeError("没有任何预测结果生成")

    predictions_all = pd.concat(all_predictions, ignore_index=True)
    metrics_summary = pd.DataFrame(metrics_rows)
    curtail_summary = make_curtail_summary(df0, df_model, rated_power, args.curtail_ratio)

    sample_summary_rows = [
        {"统计项": "输入STEP7窗口数", "数值": len(df0)},
        {"统计项": "建模窗口数", "数值": len(df_model)},
        {"统计项": "风机列数", "数值": len(fan_cols)},
        {"统计项": "单机功率列单位", "数值": args.fan_power_unit},
        {"统计项": "单台风机额定功率MW", "数值": args.fan_rated_power_mw},
        {"统计项": "单机功率上限倍率", "数值": args.fan_upper_factor},
        {"统计项": "单机功率预测/过滤上限原始单位", "数值": args.fan_rated_power_mw * args.fan_upper_factor * fan_mw_to_native_scale(args.fan_power_unit)},
        {"统计项": "训练开始时间", "数值": str(df_model["timestamp"].min())},
        {"统计项": "验证开始时间", "数值": str(split_val_time)},
        {"统计项": "测试开始时间", "数值": str(split_test_time)},
        {"统计项": "结束时间", "数值": str(df_model["timestamp"].max())},
        {"统计项": "损耗曲线来源", "数值": loss_curve_source},
        {"统计项": "损耗曲线目标模型", "数值": args.loss_model_type},
        {"统计项": "总运行秒数", "数值": round(time.time() - start_all, 2)},
    ]
    sample_summary = pd.DataFrame(sample_summary_rows)
    if physical_summary is not None and len(physical_summary) > 0:
        sample_summary = pd.concat([sample_summary, physical_summary], ignore_index=True)

    # 输出
    metrics_path = output_dir / "metrics_summary.csv"
    pred_path = output_dir / "predictions_all.csv"
    curtail_path = output_dir / "curtail_summary.csv"
    sample_path = output_dir / "sample_summary.csv"
    loss_curve_path = output_dir / "loss_curve_used.csv"
    turbine_summary_path = output_dir / "turbine_sample_summary.csv"

    metrics_summary.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions_all.to_csv(pred_path, index=False, encoding="utf-8-sig")
    curtail_summary.to_csv(curtail_path, index=False, encoding="utf-8-sig")
    sample_summary.to_csv(sample_path, index=False, encoding="utf-8-sig")
    loss_curve.to_csv(loss_curve_path, index=False, encoding="utf-8-sig")
    if turbine_sample_summaries:
        pd.concat(turbine_sample_summaries, ignore_index=True).to_csv(turbine_summary_path, index=False, encoding="utf-8-sig")

    print("\n完成：")
    print(f"- 指标汇总: {metrics_path}")
    print(f"- 预测明细: {pred_path}")
    print(f"- 限电统计: {curtail_path}")
    print(f"- 样本统计: {sample_path}")
    print(f"- 使用的损耗曲线: {loss_curve_path}")
    if turbine_sample_summaries:
        print(f"- 单机样本统计: {turbine_summary_path}")

    print("\n指标预览：")
    sort_cols = [c for c in ["split", "horizon_steps", "model", "scheme"] if c in metrics_summary.columns]
    print(metrics_summary.sort_values(sort_cols).to_string(index=False))


if __name__ == "__main__":
    main()
