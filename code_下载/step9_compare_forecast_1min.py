#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP9：分钟级预测实验：全站直接预测 vs 单机预测汇总。

目的：
    不聚合到15min，直接在 STEP6 分钟级清洗宽表上做预测实验，
    用于比较：
        1. 分钟级直接建模是否可行；
        2. 状态码特征是否能提升预测效果；
        3. 单机预测求和方案 vs 全站直接预测方案。

输入：
    STEP6 输出的分钟级宽表，优先自动查找：
        场站数据/<场站名>/step6-风机宽表修正结果-仅正功率/*data_use_unit_mw*.csv

单位规则：
    - STEP6 中 FINAL_ACTIVE_POWER_#n 通常为 kW；
    - STEP9 内部统一将单机风机功率转换为 MW；
    - ACTIVE_POWER_STATION 默认是 MW；
    - FINAL_FAN_POWER_SUM_MW / FINAL_LOSS_MW 在 STEP9 内部重新计算，避免历史文件单位口径不一致。

分钟级清洗规则：
    - DATA_USE_RESULT == 可用；
    - 默认剔除限电时刻：
        限电 = 0 < LIMIT_POWER < curtail_ratio * rated_power_mw
    - 风机功率 < 0：按0处理；
    - 风机功率 > 场站单机功率上限：置为 NaN，不参与该风机建模和风机总功率计算；
        JMZSFD 默认 7000 kW；
        LGXRFD 默认 4500 kW；
    - 状态码只保留允许集合：
        JMZSFD: 0,1,2,3,4,5,6
        LGXRFD: 0,1,2,3,8,11

预测设置：
    默认 M=120，即使用过去120分钟；
    默认 N=15,60,120,180，即预测未来15分钟、1小时、2小时、3小时。

方案：
    station_direct:
        直接预测未来 ACTIVE_POWER_STATION

    turbine_sum_raw:
        每台风机单独预测未来 FINAL_ACTIVE_POWER_#n_MW，然后求和

    turbine_sum_loss_q50:
        单机预测求和后，使用 STEP1 中 scheme_name=风机总功率分段模型 的 Q50 损耗曲线扣损耗

特征组：
    power_only:
        只用功率历史

    power_status:
        station_direct:
            使用全站功率特征 + 当前分钟场站状态比例 STATION_STATUS_RATIO_<code>
        turbine_sum:
            树模型 lightgbm/xgboost/random_forest 使用 单机功率 + STATUS_#n
            ridge 暂不直接使用 STATUS_#n，因为状态码是类别，数值大小未必有序

输出：
    场站数据/<场站名>/step9-分钟级预测实验/
        metrics_summary.csv
        predictions_all.csv
        bias_diagnostics_summary.csv
        feature_gain_summary.csv
        turbine_metrics_by_fan.csv
        turbine_sample_summary.csv
        curtail_summary.csv
        sample_summary.csv
        loss_curve_used.csv
"""

from __future__ import annotations

import argparse
import gc
import re
import time
import warnings
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
DEFAULT_STEP6_DIR_NAME = "step6-风机宽表修正结果-仅正功率"
DEFAULT_STEP9_DIR_NAME = "step9-分钟级预测实验-lightgbm"

RATED_POWER_BY_STATION = {
    "JMZSFD": 300.0,
    "LGXRFD": 400.0,
}

FAN_POWER_UPPER_KW_BY_STATION = {
    "JMZSFD": 7000.0,
    "LGXRFD": 4500.0,
}

ALLOWED_STATUS_CODES_BY_STATION = {
    "JMZSFD": [0, 1, 2, 3, 4, 5, 6],
    "LGXRFD": [0, 1, 2, 3, 8, 11],
}

DEFAULT_MODELS = ["ridge", "lightgbm"]
DEFAULT_FEATURE_SETS = ["power_only", "power_status"]
DEFAULT_M = 60
DEFAULT_N_LIST = [15, 60, 120, 180]
DEFAULT_FREQ_MINUTES = 1
DEFAULT_CURTAIL_RATIO = 0.95
DEFAULT_LOSS_MODEL_TYPE = "风机总功率分段模型"
RANDOM_SEED = 42

TIMESTAMP_COL = "timestamp"
DATA_USE_COL = "DATA_USE_RESULT"
TARGET_COL = "ACTIVE_POWER_STATION"
LIMIT_COL = "LIMIT_POWER"
FINAL_POWER_PREFIX = "FINAL_ACTIVE_POWER_#"
STATUS_PREFIX = "STATUS_#"

STATION_FEATURE_COLS = [
    "ACTIVE_POWER_STATION",
    "FINAL_FAN_POWER_SUM_MW_STEP9",
    "FINAL_LOSS_MW_STEP9",
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


def build_station_dir(base_dir: str | Path, station_name: str) -> Path:
    return Path(base_dir) / str(station_name)


def build_step6_dir(base_dir: str | Path, station_name: str, step6_dir_name: str = DEFAULT_STEP6_DIR_NAME) -> Path:
    return build_station_dir(base_dir, station_name) / step6_dir_name


def default_output_dir(base_dir: str | Path, station_name: str, step9_dir_name: str = DEFAULT_STEP9_DIR_NAME) -> Path:
    return build_station_dir(base_dir, station_name) / step9_dir_name


def csv_has_cols(path: Path, required_cols: List[str]) -> bool:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            header = pd.read_csv(path, encoding=enc, nrows=0)
            cols = set(header.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False))
            return all(c in cols for c in required_cols)
        except Exception:
            continue
    return False


def find_latest_step6_file(step6_dir: Path) -> Path:
    patterns = [
        "*data_use_unit_mw*.csv",
        "*fan_wide_correction_data_use*.csv",
        "*fan_wide_correction_threshold_*.csv",
        "*_fan_wide_correction_*.csv",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(step6_dir.rglob(pat))

    candidates = [
        p for p in candidates
        if p.is_file()
        and "summary" not in p.name.lower()
        and "quality" not in p.name.lower()
    ]
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)

    valid = [p for p in candidates if csv_has_cols(p, [TIMESTAMP_COL, DATA_USE_COL])]
    if valid:
        return valid[0]
    if candidates:
        print("⚠️ 未找到包含 DATA_USE_RESULT 的STEP6文件，将使用最近候选文件，但可能报错。")
        return candidates[0]
    raise FileNotFoundError(f"未找到 STEP6 输出文件：{step6_dir}")


def parse_n_list(s: str) -> List[int]:
    if not s:
        return DEFAULT_N_LIST.copy()
    out = []
    for part in str(s).replace("，", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


def parse_models(s: str) -> List[str]:
    if not s:
        return DEFAULT_MODELS.copy()
    return [x.strip() for x in str(s).replace("，", ",").split(",") if x.strip()]


def parse_feature_sets(s: str) -> List[str]:
    if not s:
        return DEFAULT_FEATURE_SETS.copy()
    allowed = {"power_only", "power_status"}
    out = []
    for part in str(s).replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part not in allowed:
            raise ValueError(f"未知 feature_set={part}，仅支持 {sorted(allowed)}")
        out.append(part)
    return out


def parse_int_list(s: str | None, default: List[int]) -> List[int]:
    if s is None or str(s).strip() == "":
        return list(default)
    out = []
    for part in str(s).replace("，", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(float(part)))
    return sorted(set(out))


def extract_fan_nums(columns: Iterable[str]) -> List[int]:
    nums = []
    pat = re.compile(r"^FINAL_ACTIVE_POWER_#(\d+)$")
    for c in columns:
        m = pat.match(str(c).strip())
        if m:
            nums.append(int(m.group(1)))
    return sorted(set(nums))


def fan_col(n: int) -> str:
    return f"{FINAL_POWER_PREFIX}{n}"


def fan_mw_col(n: int) -> str:
    return f"FINAL_ACTIVE_POWER_#{n}_MW_STEP9"


def status_col(n: int) -> str:
    return f"{STATUS_PREFIX}{n}"


def fan_unit_to_mw_scale(unit: str) -> float:
    u = str(unit).strip().lower()
    if u == "kw":
        return 0.001
    if u == "mw":
        return 1.0
    raise ValueError(f"未知风机功率单位：{unit}，仅支持 kW/MW")


def get_fan_power_upper_kw(station_name: str, user_value: Optional[float]) -> float:
    if user_value is not None and np.isfinite(user_value):
        return float(user_value)
    v = FAN_POWER_UPPER_KW_BY_STATION.get(station_name.upper())
    if v is None:
        raise ValueError(f"未配置场站 {station_name} 的风机功率上限，请用 --fan-power-upper-kw 指定")
    return float(v)


def get_rated_power(station_name: str, user_value: Optional[float]) -> float:
    if user_value is not None and np.isfinite(user_value):
        return float(user_value)
    v = RATED_POWER_BY_STATION.get(station_name.upper())
    if v is None:
        raise ValueError(f"未配置场站 {station_name} 的额定功率，请用 --rated-power-mw 指定")
    return float(v)


def get_allowed_status_codes(station_name: str, user_value: Optional[str]) -> List[int]:
    default = ALLOWED_STATUS_CODES_BY_STATION.get(station_name.upper())
    if default is None:
        default = []
    codes = parse_int_list(user_value, default)
    if not codes:
        raise ValueError(f"未配置场站 {station_name} 的允许状态码，请用 --allowed-status-codes 指定")
    return codes


def normalize_status_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    rounded = x.round()
    mask = x.notna() & np.isclose(x, rounded, atol=1e-9)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[mask] = rounded.loc[mask].astype(float)
    return out


# ============================================================
# 数据准备
# ============================================================
def clean_step6_for_minute_experiment(
    raw: pd.DataFrame,
    station_name: str,
    fan_power_unit: str = "kW",
    fan_power_upper_kw: float = 4500.0,
    allowed_status_codes: List[int] | None = None,
) -> Tuple[pd.DataFrame, List[int]]:
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"输入缺少 {TIMESTAMP_COL} 列")
    if DATA_USE_COL not in df.columns:
        raise ValueError(f"输入缺少 {DATA_USE_COL} 列，请确认输入为 STEP6 数据可用标签版输出")
    if TARGET_COL not in df.columns:
        raise ValueError(f"输入缺少 {TARGET_COL} 列")

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL]).copy()
    df = df.sort_values(TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL], keep="last").reset_index(drop=True)

    fan_nums = extract_fan_nums(df.columns)
    if not fan_nums:
        raise ValueError("未识别到 FINAL_ACTIVE_POWER_#n 风机功率列")

    scale = fan_unit_to_mw_scale(fan_power_unit)
    allowed_status_codes = allowed_status_codes or []

    # 风机功率清洗并转MW
    for n in fan_nums:
        c = fan_col(n)
        s = pd.to_numeric(df[c], errors="coerce")

        # 负值按0；超过上限认为非法，不参与。
        s = s.mask(s < 0, 0.0)

        if fan_power_unit.lower() == "kw":
            upper_native = fan_power_upper_kw
        else:
            upper_native = fan_power_upper_kw / 1000.0
        s = s.mask(s > upper_native, np.nan)

        df[fan_mw_col(n)] = s * scale

    fan_mw_cols = [fan_mw_col(n) for n in fan_nums]
    df["FINAL_FAN_POWER_SUM_MW_STEP9"] = df[fan_mw_cols].sum(axis=1, skipna=True)
    df["FINAL_STATION_POWER_MW_STEP9"] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df["FINAL_LOSS_MW_STEP9"] = df["FINAL_FAN_POWER_SUM_MW_STEP9"] - df["FINAL_STATION_POWER_MW_STEP9"]

    # 状态码清洗
    allowed_set = set(int(x) for x in allowed_status_codes)
    status_valid_count = pd.Series(0, index=df.index, dtype=float)

    for code in allowed_status_codes:
        df[f"STATION_STATUS_COUNT_{int(code)}_STEP9"] = 0.0

    for n in fan_nums:
        sc = status_col(n)
        clean_col = f"STATUS_#{n}_CLEAN_STEP9"
        if sc not in df.columns:
            df[clean_col] = np.nan
            continue

        st = normalize_status_series(df[sc])
        st = st.where(st.isin(allowed_set), np.nan)
        df[clean_col] = st

        valid = st.notna()
        status_valid_count += valid.astype(float)
        for code in allowed_status_codes:
            df[f"STATION_STATUS_COUNT_{int(code)}_STEP9"] += st.eq(int(code)).astype(float)

    df["STATION_STATUS_VALID_FAN_COUNT_STEP9"] = status_valid_count
    for code in allowed_status_codes:
        cnt_col = f"STATION_STATUS_COUNT_{int(code)}_STEP9"
        ratio_col = f"STATION_STATUS_RATIO_{int(code)}_STEP9"
        df[ratio_col] = np.where(status_valid_count > 0, df[cnt_col] / status_valid_count, np.nan)

    return df, fan_nums


def add_curtail_flag(
    df: pd.DataFrame,
    rated_power_mw: float,
    curtail_ratio: float = DEFAULT_CURTAIL_RATIO,
    mode: str = "limit_threshold",
) -> pd.DataFrame:
    out = df.copy()
    if LIMIT_COL not in out.columns:
        out["IS_CURTAILED"] = False
        out["CURTAIL_REASON"] = "无LIMIT_POWER列，按非限电处理"
        return out

    limit = pd.to_numeric(out[LIMIT_COL], errors="coerce")
    threshold = rated_power_mw * curtail_ratio

    if mode == "off":
        is_curtail = pd.Series(False, index=out.index)
        reason = "未启用限电识别"
    elif mode == "limit_positive":
        is_curtail = limit.fillna(0) > 0
        reason = "LIMIT_POWER>0"
    elif mode == "limit_threshold":
        is_curtail = limit.notna() & (limit > 0) & (limit < threshold)
        reason = f"0<LIMIT_POWER<{threshold:g}MW"
    else:
        raise ValueError(f"未知 curtail-mode={mode}")

    out["IS_CURTAILED"] = is_curtail.fillna(False)
    out["CURTAIL_REASON"] = np.where(out["IS_CURTAILED"], reason, "非限电")
    return out


def filter_model_data(df: pd.DataFrame, exclude_curtail: bool = True) -> pd.DataFrame:
    out = df.copy()
    out = out[out[DATA_USE_COL].astype(str).str.strip().eq("可用")].copy()
    if exclude_curtail and "IS_CURTAILED" in out.columns:
        out = out[~out["IS_CURTAILED"].fillna(False)].copy()
    out = out.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return out


def make_curtail_summary(df: pd.DataFrame, filtered_df: pd.DataFrame, rated_power_mw: float, curtail_ratio: float) -> pd.DataFrame:
    limit = pd.to_numeric(df.get(LIMIT_COL, pd.Series(index=df.index, dtype=float)), errors="coerce")
    rows = [
        {"统计项": "场站额定功率MW", "数值": rated_power_mw},
        {"统计项": "限电阈值比例", "数值": curtail_ratio},
        {"统计项": "限电阈值MW", "数值": rated_power_mw * curtail_ratio},
        {"统计项": "STEP6分钟总行数", "数值": len(df)},
        {"统计项": "DATA_USE_RESULT=可用分钟数", "数值": int(df[DATA_USE_COL].astype(str).eq("可用").sum()) if DATA_USE_COL in df.columns else np.nan},
        {"统计项": "LIMIT_POWER为空分钟数", "数值": int(limit.isna().sum())},
        {"统计项": "LIMIT_POWER<=0分钟数", "数值": int((limit <= 0).fillna(False).sum())},
        {"统计项": "LIMIT_POWER>0分钟数", "数值": int((limit > 0).fillna(False).sum())},
        {"统计项": "识别为限电分钟数", "数值": int(df.get("IS_CURTAILED", pd.Series(False, index=df.index)).fillna(False).sum())},
        {"统计项": "最终建模分钟数", "数值": len(filtered_df)},
    ]
    return pd.DataFrame(rows)


# ============================================================
# 损耗曲线
# ============================================================
def find_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def try_read_loss_curve_file(path: Path, loss_model_type: str = DEFAULT_LOSS_MODEL_TYPE, strict_model_type: bool = True) -> Optional[pd.DataFrame]:
    try:
        df = read_csv_auto(path)
        df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

        model_col = find_existing_col(df, [
            "scheme_name", "分段模型", "模型类型", "功率分段模型",
            "model_type", "model", "curve_type", "分箱模型",
        ])

        if model_col is not None:
            model_s = df[model_col].astype(str).str.strip()
            target = str(loss_model_type).strip()
            mask = model_s.eq(target)
            if not mask.any() and target == "风机总功率分段模型":
                mask = model_s.str.contains("风机", na=False) & model_s.str.contains("总功率", na=False)
            if not mask.any():
                available = sorted(model_s.dropna().unique().tolist())
                print(f"⚠️ 损耗曲线文件 {path} 中没有目标模型：{target}；可用模型：{available}")
                return None
            print(f"损耗曲线模型列: {model_col}；目标模型: {target}")
            df = df.loc[mask].copy()
        elif strict_model_type:
            print(f"⚠️ 损耗曲线文件 {path} 缺少模型类型列，无法确认是否为 {loss_model_type}，跳过。")
            return None

        p_col = find_existing_col(df, [
            "P_med", "功率分箱中位数MW", "功率中位数MW", "P_mid", "bin_mid", "power_mid",
            "风机总功率分箱中位数MW", "风机总功率中位数MW",
        ])
        q50_col = find_existing_col(df, ["Q50", "P50", "损耗中位数MW", "loss_median", "median_loss", "中位数"])

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
        return out
    except Exception as exc:
        print(f"⚠️ 读取损耗曲线失败：{path}；{exc}")
        return None


def find_loss_curve_file(station_dir: Path, loss_model_type: str = DEFAULT_LOSS_MODEL_TYPE, strict_model_type: bool = True) -> Optional[Path]:
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


def build_train_loss_curve(train_df: pd.DataFrame, bin_count: int = 80, min_bin_count: int = 30) -> pd.DataFrame:
    p = pd.to_numeric(train_df["FINAL_FAN_POWER_SUM_MW_STEP9"], errors="coerce")
    loss = pd.to_numeric(train_df["FINAL_LOSS_MW_STEP9"], errors="coerce")
    tmp = pd.DataFrame({"P": p, "loss": loss}).dropna()
    tmp = tmp[np.isfinite(tmp["P"]) & np.isfinite(tmp["loss"])]
    if len(tmp) < max(100, min_bin_count * 3):
        raise ValueError("训练集样本太少，无法兜底构建损耗曲线")
    q = min(bin_count, max(3, len(tmp) // min_bin_count))
    tmp["bin"] = pd.qcut(tmp["P"], q=q, duplicates="drop")
    curve = tmp.groupby("bin", observed=True).agg(P_med=("P", "median"), Q50=("loss", "median"), count=("loss", "size")).reset_index(drop=True)
    curve = curve[curve["count"] >= min_bin_count].sort_values("P_med").reset_index(drop=True)
    if len(curve) < 2:
        raise ValueError("有效损耗分箱太少，无法构建损耗曲线")
    return curve[["P_med", "Q50", "count"]]


def load_loss_curve(
    station_dir: Path,
    loss_curve_file: Optional[str | Path] = None,
    train_df_for_fallback: Optional[pd.DataFrame] = None,
    allow_train_fallback: bool = False,
    loss_model_type: str = DEFAULT_LOSS_MODEL_TYPE,
    strict_loss_model_type: bool = True,
) -> Tuple[pd.DataFrame, str]:
    if loss_curve_file:
        p = Path(loss_curve_file)
        curve = try_read_loss_curve_file(p, loss_model_type=loss_model_type, strict_model_type=strict_loss_model_type)
        if curve is None:
            raise ValueError(f"指定的损耗曲线文件无法识别目标模型 {loss_model_type} 的 P_med/Q50：{p}")
        return curve, f"{p} | 模型={loss_model_type}"

    found = find_loss_curve_file(station_dir, loss_model_type=loss_model_type, strict_model_type=strict_loss_model_type)
    if found is not None:
        curve = try_read_loss_curve_file(found, loss_model_type=loss_model_type, strict_model_type=strict_loss_model_type)
        if curve is not None:
            return curve, f"{found} | 模型={loss_model_type}"

    if allow_train_fallback and train_df_for_fallback is not None:
        curve = build_train_loss_curve(train_df_for_fallback)
        curve["loss_model_type"] = "训练集兜底：风机总功率"
        return curve, "训练集兜底构建：FINAL_FAN_POWER_SUM_MW_STEP9 -> FINAL_LOSS_MW_STEP9"

    raise FileNotFoundError(
        f"未找到 STEP1 损耗曲线中的目标模型：{loss_model_type}。"
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
# 样本构造
# ============================================================
def get_split_times(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.Timestamp, pd.Timestamp]:
    ts = df[TIMESTAMP_COL].sort_values().reset_index(drop=True)
    if len(ts) < 100:
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


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, timestamps: np.ndarray, splits: np.ndarray):
        self.X = X
        self.y = y
        self.timestamps = timestamps
        self.splits = splits


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
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    timestamps = df[TIMESTAMP_COL].reset_index(drop=True)
    max_gap_seconds = int(freq_minutes * 60)

    values_x = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values_y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)

    ts_to_idx: Dict[pd.Timestamp, int] = {pd.Timestamp(t): i for i, t in enumerate(timestamps)}
    diffs = timestamps.diff().dt.total_seconds()
    continuity = (diffs <= max_gap_seconds).fillna(True).to_numpy(dtype=bool)
    breaks = (~continuity).astype(np.int32)
    break_prefix = np.concatenate([[0], np.cumsum(breaks)])

    upper = max(0, len(df) - M)
    X_list, y_list, ts_list, split_list = [], [], [], []
    target_delta = pd.Timedelta(minutes=N * freq_minutes)

    for i in range(upper):
        hist_end_idx = i + M - 1
        # 历史窗口内不能有时间断点
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
# 模型与指标
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

    return pd.DataFrame({
        "timestamp": pd.to_datetime(ds.timestamps[eval_mask]),
        "split": ds.splits[eval_mask],
        "true_value": ds.y[eval_mask].astype(float),
        "predicted_value": pred.astype(float),
    })


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, rated_power_mw: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "NMAE_by_capacity": np.nan, "sample_count": 0}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan
    nmae = mae / rated_power_mw if rated_power_mw and rated_power_mw > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "NMAE_by_capacity": nmae, "sample_count": len(y_true)}


def append_metrics(
    rows: List[dict],
    pred_df: pd.DataFrame,
    model_name: str,
    feature_set: str,
    horizon_steps: int,
    scheme: str,
    rated_power_mw: float,
    freq_minutes: int = DEFAULT_FREQ_MINUTES,
) -> None:
    for split in ["val", "test"]:
        sub = pred_df[pred_df["split"] == split]
        m = calc_metrics(sub["true_station_power"].to_numpy(), sub["pred_station_power"].to_numpy(), rated_power_mw)
        rows.append({
            "model": model_name,
            "feature_set": feature_set,
            "horizon_steps": horizon_steps,
            "horizon_minutes": horizon_steps * freq_minutes,
            "scheme": scheme,
            "split": split,
            **m,
        })


def build_bias_diagnostics(predictions_all: pd.DataFrame, rated_power_mw: float) -> pd.DataFrame:
    rows = []
    required = {"model", "feature_set", "horizon_steps", "horizon_minutes", "scheme", "split", "true_station_power", "pred_station_power"}
    if not required.issubset(predictions_all.columns):
        return pd.DataFrame()

    group_cols = ["model", "feature_set", "horizon_steps", "horizon_minutes", "scheme", "split"]
    for keys, g in predictions_all.groupby(group_cols, dropna=False):
        y = pd.to_numeric(g["true_station_power"], errors="coerce")
        p = pd.to_numeric(g["pred_station_power"], errors="coerce")
        err = p - y
        valid = np.isfinite(y) & np.isfinite(p)
        y = y[valid]
        p = p[valid]
        err = err[valid]
        if len(err) == 0:
            continue
        row = dict(zip(group_cols, keys))
        row.update({
            "sample_count": int(len(err)),
            "bias_MW": float(err.mean()),
            "bias_abs_MW": float(abs(err.mean())),
            "bias_pct_capacity": float(err.mean() / rated_power_mw) if rated_power_mw > 0 else np.nan,
            "MAE": float(err.abs().mean()),
            "RMSE": float(np.sqrt(np.mean(np.square(err)))),
            "true_mean_MW": float(y.mean()),
            "pred_mean_MW": float(p.mean()),
            "pred_minus_true_mean_MW": float(p.mean() - y.mean()),
            "error_p05_MW": float(err.quantile(0.05)),
            "error_p25_MW": float(err.quantile(0.25)),
            "error_p50_MW": float(err.quantile(0.50)),
            "error_p75_MW": float(err.quantile(0.75)),
            "error_p95_MW": float(err.quantile(0.95)),
            "overestimate_ratio": float((err > 0).mean()),
            "underestimate_ratio": float((err < 0).mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_gain_summary(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    if metrics_summary is None or len(metrics_summary) == 0 or "feature_set" not in metrics_summary.columns:
        return pd.DataFrame()

    key_cols = ["model", "horizon_steps", "horizon_minutes", "scheme", "split"]
    need_cols = key_cols + ["feature_set", "MAE", "RMSE", "R2", "sample_count"]
    if not set(need_cols).issubset(metrics_summary.columns):
        return pd.DataFrame()

    tmp = metrics_summary[need_cols].copy()
    p = tmp[tmp["feature_set"].eq("power_only")]
    s = tmp[tmp["feature_set"].eq("power_status")]
    merged = p.merge(s, on=key_cols, how="inner", suffixes=("_power_only", "_power_status"))
    if len(merged) == 0:
        return merged
    merged["MAE_improvement"] = merged["MAE_power_only"] - merged["MAE_power_status"]
    merged["RMSE_improvement"] = merged["RMSE_power_only"] - merged["RMSE_power_status"]
    merged["R2_improvement"] = merged["R2_power_status"] - merged["R2_power_only"]
    merged["MAE_improvement_ratio"] = merged["MAE_improvement"] / merged["MAE_power_only"]
    return merged


def calc_one_metric_row(y_true: pd.Series, y_pred: pd.Series, rated_power: float) -> dict:
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(y_pred, errors="coerce")
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if len(y) == 0:
        return {
            "sample_count": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "bias_MW": np.nan,
            "abs_bias_MW": np.nan,
            "NMAE_by_fan_capacity": np.nan,
            "true_mean_MW": np.nan,
            "pred_mean_MW": np.nan,
            "true_max_MW": np.nan,
            "pred_max_MW": np.nan,
        }
    err = p - y
    try:
        r2 = r2_score(y, p)
    except Exception:
        r2 = np.nan
    return {
        "sample_count": int(len(y)),
        "MAE": float(err.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(err)))),
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "bias_MW": float(err.mean()),
        "abs_bias_MW": float(abs(err.mean())),
        "NMAE_by_fan_capacity": float(err.abs().mean() / rated_power) if rated_power and rated_power > 0 else np.nan,
        "true_mean_MW": float(y.mean()),
        "pred_mean_MW": float(p.mean()),
        "true_max_MW": float(y.max()),
        "pred_max_MW": float(p.max()),
    }


# ============================================================
# 特征列
# ============================================================
def get_station_feature_cols(df: pd.DataFrame, feature_set: str, allowed_status_codes: List[int]) -> List[str]:
    cols = [c for c in STATION_FEATURE_COLS if c in df.columns]
    if feature_set == "power_status":
        for code in allowed_status_codes:
            c = f"STATION_STATUS_RATIO_{int(code)}_STEP9"
            if c in df.columns:
                cols.append(c)
    return list(dict.fromkeys(cols))


def get_turbine_feature_cols(df: pd.DataFrame, fan_no: int, model_name: str, feature_set: str) -> List[str]:
    cols = [fan_mw_col(fan_no)]
    if feature_set == "power_status" and str(model_name).lower() in {"lightgbm", "xgboost", "random_forest"}:
        c = f"STATUS_#{fan_no}_CLEAN_STEP9"
        if c in df.columns:
            cols.append(c)
    return list(dict.fromkeys(cols))


# ============================================================
# 方案运行
# ============================================================
def run_station_direct(
    df: pd.DataFrame,
    model_name: str,
    feature_set: str,
    allowed_status_codes: List[int],
    M: int,
    N: int,
    freq_minutes: int,
    split_val_time: pd.Timestamp,
    split_test_time: pd.Timestamp,
) -> pd.DataFrame:
    feature_cols = get_station_feature_cols(df, feature_set, allowed_status_codes)
    ds = build_supervised_dataset(df, feature_cols, TARGET_COL, M, N, freq_minutes, split_val_time, split_test_time)
    pred = fit_predict_model(model_name, ds)
    pred = pred.rename(columns={"true_value": "true_station_power", "predicted_value": "pred_station_power"})
    pred["scheme"] = "station_direct"
    pred["model"] = model_name
    pred["feature_set"] = feature_set
    pred["feature_count"] = len(feature_cols)
    return pred


def run_turbine_models(
    df: pd.DataFrame,
    fan_nums: List[int],
    model_name: str,
    feature_set: str,
    M: int,
    N: int,
    freq_minutes: int,
    split_val_time: pd.Timestamp,
    split_test_time: pd.Timestamp,
    fan_pred_min_mw: float = 0.0,
    fan_pred_max_mw: Optional[float] = None,
    fan_capacity_mw: Optional[float] = None,
    save_fan_prediction_detail: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_parts = []
    sample_rows = []
    metric_rows = []
    detail_parts = []

    if fan_capacity_mw is None:
        fan_capacity_mw = np.nan

    for idx, fan_no in enumerate(fan_nums, start=1):
        target_col = fan_mw_col(fan_no)
        try:
            feature_cols = get_turbine_feature_cols(df, fan_no, model_name, feature_set)
            ds = build_supervised_dataset(df, feature_cols, target_col, M, N, freq_minutes, split_val_time, split_test_time)

            train_count = int((ds.splits == "train").sum())
            val_count = int((ds.splits == "val").sum())
            test_count = int((ds.splits == "test").sum())
            sample_rows.append({
                "fan_no": fan_no,
                "fan_col": target_col,
                "model": model_name,
                "feature_set": feature_set,
                "horizon_steps": N,
                "horizon_minutes": N * freq_minutes,
                "feature_count": len(feature_cols),
                "feature_cols": "|".join(feature_cols),
                "train_samples": train_count,
                "val_samples": val_count,
                "test_samples": test_count,
            })

            if train_count < 10 or (val_count + test_count) < 1:
                print(f"  ⚠️ 风机 #{fan_no} 样本不足，跳过")
                continue

            pred = fit_predict_model(model_name, ds)
            pred = pred.rename(columns={"predicted_value": f"fan_{fan_no}_pred"})
            pred_col = f"fan_{fan_no}_pred"

            pred[pred_col] = pd.to_numeric(pred[pred_col], errors="coerce").clip(lower=fan_pred_min_mw)
            if fan_pred_max_mw is not None and np.isfinite(fan_pred_max_mw):
                pred[pred_col] = pred[pred_col].clip(upper=fan_pred_max_mw)

            for split in ["val", "test"]:
                sub = pred[pred["split"] == split]
                metric = calc_one_metric_row(sub["true_value"], sub[pred_col], fan_capacity_mw)
                metric_rows.append({
                    "fan_no": fan_no,
                    "fan_col": target_col,
                    "model": model_name,
                    "feature_set": feature_set,
                    "horizon_steps": N,
                    "horizon_minutes": N * freq_minutes,
                    "split": split,
                    **metric,
                })

            if save_fan_prediction_detail:
                detail = pred[["timestamp", "split", "true_value", pred_col]].copy()
                detail = detail.rename(columns={"true_value": "true_fan_power", pred_col: "pred_fan_power"})
                detail["fan_no"] = fan_no
                detail["fan_col"] = target_col
                detail["model"] = model_name
                detail["feature_set"] = feature_set
                detail["horizon_steps"] = N
                detail["horizon_minutes"] = N * freq_minutes
                detail["error"] = detail["pred_fan_power"] - detail["true_fan_power"]
                detail["abs_error"] = detail["error"].abs()
                detail_parts.append(detail)

            pred_parts.append(pred[["timestamp", "split", pred_col]].copy())

        except Exception as exc:
            print(f"  ⚠️ 风机 #{fan_no} 训练失败：{exc}")
            sample_rows.append({
                "fan_no": fan_no,
                "fan_col": target_col,
                "model": model_name,
                "feature_set": feature_set,
                "horizon_steps": N,
                "horizon_minutes": N * freq_minutes,
                "error": str(exc),
            })

        if idx % 20 == 0:
            print(f"  已完成风机模型 {idx}/{len(fan_nums)}")
            gc.collect()

    if not pred_parts:
        raise RuntimeError("没有任何风机模型成功生成预测")

    # 当前沿用 STEP8 的内连接，保证所有成功风机在同一timestamp都有预测。
    merged = pred_parts[0]
    for part in pred_parts[1:]:
        merged = merged.merge(part, on=["timestamp", "split"], how="inner")

    pred_cols = [c for c in merged.columns if c.endswith("_pred")]
    merged["pred_fan_sum_mw"] = merged[pred_cols].sum(axis=1)
    merged["pred_turbine_count"] = len(pred_cols)
    merged["pred_turbine_coverage"] = len(pred_cols) / max(len(fan_nums), 1)

    sum_pred = merged[["timestamp", "split", "pred_fan_sum_mw", "pred_turbine_count", "pred_turbine_coverage"]].copy()
    sum_pred["model"] = model_name
    sum_pred["feature_set"] = feature_set

    sample_summary = pd.DataFrame(sample_rows)
    turbine_metrics = pd.DataFrame(metric_rows)
    turbine_detail = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    return sum_pred, sample_summary, turbine_metrics, turbine_detail


def attach_true_station(df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    truth = df[[TIMESTAMP_COL, TARGET_COL]].copy()
    truth[TIMESTAMP_COL] = pd.to_datetime(truth[TIMESTAMP_COL])
    out = pred_df.merge(truth, left_on="timestamp", right_on=TIMESTAMP_COL, how="left")
    if TIMESTAMP_COL in out.columns and TIMESTAMP_COL != "timestamp":
        out = out.drop(columns=[TIMESTAMP_COL])
    out = out.rename(columns={TARGET_COL: "true_station_power"})
    return out


# ============================================================
# main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="STEP9：分钟级预测实验")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="场站数据根目录")
    parser.add_argument("--station-name", default=DEFAULT_STATION_NAME, help="场站名称，例如 LGXRFD 或 JMZSFD")
    parser.add_argument("--step6-dir-name", default=DEFAULT_STEP6_DIR_NAME, help="STEP6输出目录名")
    parser.add_argument("--step9-dir-name", default=DEFAULT_STEP9_DIR_NAME, help="STEP9输出目录名")
    parser.add_argument("--input-file", default=None, help="STEP6分钟级宽表路径；不填则自动查找")
    parser.add_argument("--output-dir", default=None, help="输出目录；不填则按场站名自动拼接")

    parser.add_argument("--rated-power-mw", type=float, default=None, help="场站额定功率MW；不填则按场站名自动识别")
    parser.add_argument("--curtail-ratio", type=float, default=DEFAULT_CURTAIL_RATIO, help="限电阈值比例，默认0.95")
    parser.add_argument("--curtail-mode", default="limit_threshold", choices=["off", "limit_positive", "limit_threshold"], help="限电识别模式")
    parser.add_argument("--include-curtail", action="store_true", help="若指定，则保留限电时刻参与建模")

    parser.add_argument("--fan-power-unit", default="kW", choices=["kW", "MW"], help="STEP6风机功率列单位，江苏通常为kW")
    parser.add_argument("--fan-power-upper-kw", type=float, default=None, help="分钟级单机风机功率上限kW；不填按场站默认")
    parser.add_argument("--allowed-status-codes", default=None, help="允许状态码，逗号分隔；不填按场站默认")

    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="模型列表，逗号分隔")
    parser.add_argument("--feature-sets", default=",".join(DEFAULT_FEATURE_SETS), help="特征组，逗号分隔：power_only,power_status")
    parser.add_argument("--M", type=int, default=DEFAULT_M, help="历史窗口长度，单位为1min步")
    parser.add_argument("--N-list", default=",".join(map(str, DEFAULT_N_LIST)), help="预测步长列表，单位为1min步，例如 15,60,120,180")
    parser.add_argument("--freq-minutes", type=int, default=DEFAULT_FREQ_MINUTES, help="数据频率，默认1min")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="训练集时间比例")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集时间比例")

    parser.add_argument("--loss-curve-file", default=None, help="STEP1 损耗中位数曲线文件；需包含目标模型的 P_med 和 Q50")
    parser.add_argument("--loss-model-type", default=DEFAULT_LOSS_MODEL_TYPE, help="STEP1损耗曲线模型类型；默认风机总功率分段模型")
    parser.add_argument("--non-strict-loss-model-type", action="store_true", help="若损耗曲线文件缺少模型类型列，允许直接读取P_med/Q50；不建议")
    parser.add_argument("--allow-train-loss-fallback", action="store_true", help="找不到STEP1损耗曲线时，允许用训练集兜底构建")

    parser.add_argument("--station-upper-factor", type=float, default=1.20, help="全站预测值裁剪上限倍率，默认1.2倍额定")
    parser.add_argument("--disable-pred-clip", action="store_true", help="关闭预测值裁剪，不建议")
    parser.add_argument("--save-turbine-predictions-by-fan", action="store_true", help="输出每台风机逐时刻预测明细，文件可能较大")
    args = parser.parse_args()

    station_name = args.station_name
    station_dir = build_station_dir(args.base_dir, station_name)
    step6_dir = build_step6_dir(args.base_dir, station_name, args.step6_dir_name)
    input_file = Path(args.input_file) if args.input_file else find_latest_step6_file(step6_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.base_dir, station_name, args.step9_dir_name)
    ensure_dir(output_dir)

    rated_power = get_rated_power(station_name, args.rated_power_mw)
    fan_power_upper_kw = get_fan_power_upper_kw(station_name, args.fan_power_upper_kw)
    allowed_status_codes = get_allowed_status_codes(station_name, args.allowed_status_codes)
    models = parse_models(args.models)
    feature_sets = parse_feature_sets(args.feature_sets)
    n_list = parse_n_list(args.N_list)

    print("=" * 88)
    print("STEP9：分钟级预测实验")
    print("=" * 88)
    print(f"场站: {station_name}")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"额定功率: {rated_power:g} MW")
    print(f"风机功率单位: {args.fan_power_unit}; 上限: {fan_power_upper_kw:g} kW")
    print(f"允许状态码: {allowed_status_codes}")
    print(f"限电规则: mode={args.curtail_mode}, ratio={args.curtail_ratio}, include_curtail={args.include_curtail}")
    print(f"模型: {models}")
    print(f"特征组: {feature_sets}")
    print(f"M={args.M}, N_LIST={n_list}")

    raw = read_csv_auto(input_file)
    df0, fan_nums = clean_step6_for_minute_experiment(
        raw,
        station_name=station_name,
        fan_power_unit=args.fan_power_unit,
        fan_power_upper_kw=fan_power_upper_kw,
        allowed_status_codes=allowed_status_codes,
    )
    df0 = add_curtail_flag(df0, rated_power_mw=rated_power, curtail_ratio=args.curtail_ratio, mode=args.curtail_mode)
    df_model = filter_model_data(df0, exclude_curtail=(not args.include_curtail))

    if len(df_model) < 1000:
        raise ValueError(f"最终分钟级建模数据太少：{len(df_model)} 行")

    split_val_time, split_test_time = get_split_times(df_model, args.train_ratio, args.val_ratio)
    print(f"切分时间: val_start={split_val_time}, test_start={split_test_time}")
    print(f"最终建模分钟数: {len(df_model):,}; 风机数: {len(fan_nums)}")

    train_df_for_loss = df_model[df_model[TIMESTAMP_COL] < split_val_time].copy()
    loss_curve, loss_curve_source = load_loss_curve(
        station_dir=station_dir,
        loss_curve_file=args.loss_curve_file,
        train_df_for_fallback=train_df_for_loss,
        allow_train_fallback=args.allow_train_loss_fallback,
        loss_model_type=args.loss_model_type,
        strict_loss_model_type=(not args.non_strict_loss_model_type),
    )
    print(f"损耗曲线来源: {loss_curve_source}")
    print(f"损耗曲线点数: {len(loss_curve)}")

    fan_capacity_mw = rated_power / max(len(fan_nums), 1)
    fan_pred_max_mw = fan_power_upper_kw / 1000.0

    all_predictions = []
    metrics_rows = []
    turbine_sample_summaries = []
    turbine_metrics_by_fan_all = []
    turbine_predictions_by_fan_all = []

    start_all = time.time()

    for model_name in models:
        print(f"\n{'#' * 80}")
        print(f"# 模型: {model_name}")
        print(f"{'#' * 80}")

        try:
            _ = make_model(model_name)
        except ImportError as exc:
            print(f"⚠️ 跳过模型 {model_name}: {exc}")
            continue

        for feature_set in feature_sets:
            print(f"\n{'=' * 70}")
            print(f"特征组: {feature_set}")
            print(f"{'=' * 70}")

            for N in n_list:
                print(f"\n--- horizon N={N} ({N * args.freq_minutes} min), feature_set={feature_set} ---")
                t0 = time.time()

                # Scheme A: station_direct
                try:
                    pred_station = run_station_direct(
                        df_model, model_name, feature_set, allowed_status_codes,
                        args.M, N, args.freq_minutes, split_val_time, split_test_time,
                    )
                    if not args.disable_pred_clip:
                        pred_station["pred_station_power"] = pd.to_numeric(pred_station["pred_station_power"], errors="coerce").clip(
                            lower=0.0,
                            upper=rated_power * args.station_upper_factor,
                        )
                    pred_station["horizon_steps"] = N
                    pred_station["horizon_minutes"] = N * args.freq_minutes
                    pred_station["error"] = pred_station["pred_station_power"] - pred_station["true_station_power"]
                    pred_station["abs_error"] = pred_station["error"].abs()
                    all_predictions.append(pred_station)
                    append_metrics(metrics_rows, pred_station, model_name, feature_set, N, "station_direct", rated_power, args.freq_minutes)
                    print(f"  station_direct 预测完成: {len(pred_station):,} 条")
                except Exception as exc:
                    print(f"  ❌ station_direct 失败: {exc}")

                # Scheme B/C: turbine_sum
                try:
                    fan_sum_pred, fan_sample_summary, fan_metrics, fan_detail = run_turbine_models(
                        df_model,
                        fan_nums,
                        model_name,
                        feature_set,
                        args.M,
                        N,
                        args.freq_minutes,
                        split_val_time,
                        split_test_time,
                        fan_pred_min_mw=0.0,
                        fan_pred_max_mw=(None if args.disable_pred_clip else fan_pred_max_mw),
                        fan_capacity_mw=fan_capacity_mw,
                        save_fan_prediction_detail=args.save_turbine_predictions_by_fan,
                    )

                    if len(fan_sample_summary) > 0:
                        turbine_sample_summaries.append(fan_sample_summary)
                    if fan_metrics is not None and len(fan_metrics) > 0:
                        turbine_metrics_by_fan_all.append(fan_metrics)
                    if fan_detail is not None and len(fan_detail) > 0:
                        turbine_predictions_by_fan_all.append(fan_detail)

                    fan_sum_pred = attach_true_station(df_model, fan_sum_pred)

                    # B: raw
                    pred_raw = fan_sum_pred.copy()
                    pred_raw["scheme"] = "turbine_sum_raw"
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
                        "pred_fan_sum_mw", "pred_turbine_count", "pred_turbine_coverage",
                        "scheme", "model", "feature_set", "horizon_steps", "horizon_minutes", "error", "abs_error"
                    ]])
                    append_metrics(metrics_rows, pred_raw, model_name, feature_set, N, "turbine_sum_raw", rated_power, args.freq_minutes)

                    # C: loss q50
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
                    pred_loss["horizon_steps"] = N
                    pred_loss["horizon_minutes"] = N * args.freq_minutes
                    pred_loss["error"] = pred_loss["pred_station_power"] - pred_loss["true_station_power"]
                    pred_loss["abs_error"] = pred_loss["error"].abs()
                    all_predictions.append(pred_loss[[
                        "timestamp", "split", "true_station_power", "pred_station_power",
                        "pred_fan_sum_mw", "pred_turbine_count", "pred_turbine_coverage", "pred_loss_q50",
                        "scheme", "model", "feature_set", "horizon_steps", "horizon_minutes", "error", "abs_error"
                    ]])
                    append_metrics(metrics_rows, pred_loss, model_name, feature_set, N, "turbine_sum_loss_q50", rated_power, args.freq_minutes)

                    print(f"  turbine_sum 预测完成: {len(fan_sum_pred):,} 条")
                except Exception as exc:
                    print(f"  ❌ turbine_sum 失败: {exc}")

                print(f"  用时: {time.time() - t0:.1f}s")
                gc.collect()

    if not all_predictions:
        raise RuntimeError("没有任何预测结果生成")

    predictions_all = pd.concat(all_predictions, ignore_index=True)
    metrics_summary = pd.DataFrame(metrics_rows)
    bias_diagnostics = build_bias_diagnostics(predictions_all, rated_power)
    feature_gain_summary = build_feature_gain_summary(metrics_summary)
    turbine_metrics_by_fan = pd.concat(turbine_metrics_by_fan_all, ignore_index=True) if turbine_metrics_by_fan_all else pd.DataFrame()
    turbine_predictions_by_fan = pd.concat(turbine_predictions_by_fan_all, ignore_index=True) if turbine_predictions_by_fan_all else pd.DataFrame()
    curtail_summary = make_curtail_summary(df0, df_model, rated_power, args.curtail_ratio)

    sample_summary = pd.DataFrame([
        {"统计项": "输入STEP6分钟数", "数值": len(df0)},
        {"统计项": "建模分钟数", "数值": len(df_model)},
        {"统计项": "风机数量", "数值": len(fan_nums)},
        {"统计项": "风机功率单位", "数值": args.fan_power_unit},
        {"统计项": "风机功率上限kW", "数值": fan_power_upper_kw},
        {"统计项": "允许状态码", "数值": ",".join(map(str, allowed_status_codes))},
        {"统计项": "模型", "数值": ",".join(models)},
        {"统计项": "特征组", "数值": ",".join(feature_sets)},
        {"统计项": "M", "数值": args.M},
        {"统计项": "N-list", "数值": ",".join(map(str, n_list))},
        {"统计项": "训练开始时间", "数值": str(df_model[TIMESTAMP_COL].min())},
        {"统计项": "验证开始时间", "数值": str(split_val_time)},
        {"统计项": "测试开始时间", "数值": str(split_test_time)},
        {"统计项": "结束时间", "数值": str(df_model[TIMESTAMP_COL].max())},
        {"统计项": "损耗曲线来源", "数值": loss_curve_source},
        {"统计项": "总运行秒数", "数值": round(time.time() - start_all, 2)},
    ])

    # 输出
    metrics_path = output_dir / "metrics_summary.csv"
    pred_path = output_dir / "predictions_all.csv"
    bias_diag_path = output_dir / "bias_diagnostics_summary.csv"
    feature_gain_path = output_dir / "feature_gain_summary.csv"
    turbine_metrics_path = output_dir / "turbine_metrics_by_fan.csv"
    turbine_detail_path = output_dir / "turbine_predictions_by_fan.csv"
    turbine_sample_path = output_dir / "turbine_sample_summary.csv"
    curtail_path = output_dir / "curtail_summary.csv"
    sample_path = output_dir / "sample_summary.csv"
    loss_curve_path = output_dir / "loss_curve_used.csv"

    metrics_summary.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions_all.to_csv(pred_path, index=False, encoding="utf-8-sig")
    bias_diagnostics.to_csv(bias_diag_path, index=False, encoding="utf-8-sig")
    feature_gain_summary.to_csv(feature_gain_path, index=False, encoding="utf-8-sig")
    curtail_summary.to_csv(curtail_path, index=False, encoding="utf-8-sig")
    sample_summary.to_csv(sample_path, index=False, encoding="utf-8-sig")
    loss_curve.to_csv(loss_curve_path, index=False, encoding="utf-8-sig")

    if turbine_sample_summaries:
        pd.concat(turbine_sample_summaries, ignore_index=True).to_csv(turbine_sample_path, index=False, encoding="utf-8-sig")
    if len(turbine_metrics_by_fan) > 0:
        turbine_metrics_by_fan.to_csv(turbine_metrics_path, index=False, encoding="utf-8-sig")
    if len(turbine_predictions_by_fan) > 0:
        turbine_predictions_by_fan.to_csv(turbine_detail_path, index=False, encoding="utf-8-sig")

    print("\n完成：")
    print(f"- 指标汇总: {metrics_path}")
    print(f"- 预测明细: {pred_path}")
    print(f"- 整体偏差诊断: {bias_diag_path}")
    print(f"- 状态特征提升对比: {feature_gain_path}")
    print(f"- 每台风机预测指标: {turbine_metrics_path}")
    print(f"- 单机样本统计: {turbine_sample_path}")
    print(f"- 限电统计: {curtail_path}")
    print(f"- 样本统计: {sample_path}")
    print(f"- 使用的损耗曲线: {loss_curve_path}")
    if len(turbine_predictions_by_fan) > 0:
        print(f"- 每台风机预测明细: {turbine_detail_path}")

    print("\n指标预览：")
    sort_cols = [c for c in ["split", "horizon_steps", "model", "feature_set", "scheme"] if c in metrics_summary.columns]
    print(metrics_summary.sort_values(sort_cols).to_string(index=False))

    if len(feature_gain_summary) > 0:
        print("\n状态特征提升预览（test，MAE_improvement>0表示变好）：")
        sub = feature_gain_summary[feature_gain_summary["split"].eq("test")].copy()
        cols = ["model", "horizon_steps", "scheme", "MAE_power_only", "MAE_power_status", "MAE_improvement", "MAE_improvement_ratio"]
        cols = [c for c in cols if c in sub.columns]
        print(sub.sort_values(["horizon_steps", "scheme", "model"])[cols].to_string(index=False))


if __name__ == "__main__":
    main()
