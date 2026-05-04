#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP6：基于原始风机宽表 + STEP3/STEP4 结果，生成风机宽表修正结果。

输出结构：
    timestamp
    DATA_USE_RESULT
    DATA_USE_REASON
    DATA_USE_SCENE
    ACTIVE_POWER_STATION
    LIMIT_POWER
    RAW_FAN_POWER_SUM_MW
    FINAL_FAN_POWER_SUM_MW
    FINAL_LOSS_MW
    FINAL_STATION_POWER_MW
    POINT_OUTLIER_RESULT
    POINT_OUTLIER_REASON
    STATUS_#1
    ACTIVE_POWER_#1
    WINDSPEED_#1
    FINAL_ACTIVE_POWER_#1
    PROCESS_RESULT_#1
    STATUS_#2
    ACTIVE_POWER_#2
    WINDSPEED_#2
    FINAL_ACTIVE_POWER_#2
    PROCESS_RESULT_#2
    ...

核心规则：
1. PROCESS_RESULT_#n 不再压缩为“正常/修正/异常”，而是直接使用 STEP4 的“最终处理细分类型”。
2. 重复正功率>0分段：
   - STEP3 明细中的重复风机：使用 STEP4 的最终处理细分类型。
   - 同一分段内未重复的其他风机：标记为“重复正功率>0｜非重复风机”。
3. 重复正功率=0且冻结风机数=0分段：
   - 该时间段所有风机均标记为 STEP4 的最终处理细分类型：
     “重复正功率=0且冻结风机数=0｜正常”
     或“重复正功率=0且冻结风机数=0｜标记为异常分段”。
4. 只有满足以下两个条件时，才修正功率：
   - 最终处理细分类型 == “重复正功率>0｜修正后保留”
   - 该风机该时刻 ACTIVE_POWER_#n > 0
   此时 FINAL_ACTIVE_POWER_#n = 0。
5. 其他所有情况：
   FINAL_ACTIVE_POWER_#n = ACTIVE_POWER_#n。
6. 若某时间戳完全未被 STEP3 分段覆盖，则兜底标记为“未匹配STEP3分段”。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 默认参数
# ============================================================
DEFAULT_STATION_DIR = r"场站数据\LGXRFD"
DEFAULT_STEP3_DIR = r"场站数据\LGXRFD\step3-重复冻结分段-仅正功率"
DEFAULT_STEP4_DIR = r"场站数据\LGXRFD\step4-阈值分析-细分版-仅正功率"
DEFAULT_OUTPUT_DIR = r"场站数据\LGXRFD\step6-风机宽表修正结果-仅正功率"

DEFAULT_STEP3_DETAIL_NAME = "freeze_count_segments_detail.csv"
DEFAULT_STEP3_SEGMENT_NAME = "freeze_count_segments.csv"
DEFAULT_STEP4_SCORED_NAME = "threshold_scan_scored.csv"
DEFAULT_STEP2_DETAIL_NAME = "station_anomaly_detail.csv"

STATUS_PREFIX = "STATUS_#"
POWER_PREFIX = "ACTIVE_POWER_#"
WIND_PREFIX = "WINDSPEED_#"

FINAL_POWER_PREFIX = "FINAL_ACTIVE_POWER_#"
PROCESS_RESULT_PREFIX = "PROCESS_RESULT_#"

# 需要从原始主 CSV 保留的场站级列。
# 若某个场站不存在其中某列，输出中会保留该列但填充为空值，避免后续脚本字段缺失。
STATION_LEVEL_OUTPUT_COLS = ["ACTIVE_POWER_STATION", "LIMIT_POWER"]

UNMATCHED_STEP3_SUBTYPE = "未匹配STEP3分段"
POSITIVE_NON_REPEAT_SUBTYPE = "重复正功率>0｜非重复风机"

DEFAULT_PROCESS_RESULT = UNMATCHED_STEP3_SUBTYPE

DETAIL_POS_KEEP_RAW = "重复正功率>0｜保留（不修正）"
CORRECTED_SUBTYPE = "重复正功率>0｜修正后保留"
DETAIL_POS_ABNORMAL = "重复正功率>0｜标记为异常分段"

ZERO_WITH_FAN_NORMAL_SUBTYPE = "重复正功率=0且冻结风机数>0｜正常"
ZERO_WITH_FAN_ABNORMAL_SUBTYPE = "重复正功率=0且冻结风机数>0｜标记为异常分段"

ZERO_NO_FAN_ABNORMAL_SUBTYPE = "重复正功率=0且冻结风机数=0｜标记为异常分段"
ZERO_NO_FAN_NORMAL_SUBTYPE = "重复正功率=0且冻结风机数=0｜正常"

POSITIVE_SEGMENT_SUBTYPES = {
    DETAIL_POS_KEEP_RAW,
    CORRECTED_SUBTYPE,
    DETAIL_POS_ABNORMAL,
}

# 点级明显离群默认阈值；超低功率区间暂不标记离群。
POINT_OUTLIER_POWER_MIN_MW = 3.0
POINT_OUTLIER_SEVERITY_THRESHOLD = 2.0

DATA_USE_AVAILABLE = "可用"
DATA_USE_UNAVAILABLE = "不可用"
POINT_NORMAL = "正常点"
POINT_OUTLIER = "点级明显离群"


# ============================================================
# 通用读取
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


def parse_thresholds_for_match(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(10)


def safe_threshold_str(x: float) -> str:
    s = f"{float(x):.10g}".replace(".", "p")
    return s.replace("-", "minus")


def first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def find_existing_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = False, desc: str = "") -> Optional[str]:
    col = first_existing_col(df, candidates)
    if required and col is None:
        label = desc or "字段"
        raise ValueError(f"未找到必要字段：{label}，候选列名：{list(candidates)}")
    return col


def load_step2_detail_for_point_outliers(step2_detail_path: Path) -> pd.DataFrame:
    """
    读取 STEP2 逐分钟明细，用于 STEP6 生成场站级点级离群标签。

    仅保留点级离群判断需要的字段：
      timestamp
      原始风机功率之和MW / 修正后风机功率之和MW
      原损耗MW / 修正后损耗MW
      原异常程度 / 修正后异常程度
    """
    if step2_detail_path is None or not Path(step2_detail_path).exists():
        print(f"⚠️ 未找到 STEP2 明细文件，点级离群标签将全部按正常点处理：{step2_detail_path}")
        return pd.DataFrame(columns=[
            "timestamp", "STEP2_原始风机功率之和MW", "STEP2_修正后风机功率之和MW",
            "STEP2_原损耗MW", "STEP2_修正后损耗MW",
            "STEP2_原异常程度", "STEP2_修正后异常程度",
        ])

    df = read_csv_auto(step2_detail_path)
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    time_col = find_existing_col(df, ["时间", "timestamp", "Timestamp"], required=True, desc="时间")
    raw_p_col = find_existing_col(df, ["原始风机功率之和MW", "原风机汇总功率MW", "原风机汇总功率", "FAN_eff_raw"], required=False)
    corr_p_col = find_existing_col(df, ["修正后风机功率之和MW", "修正后风机汇总功率MW", "修正后风机汇总功率", "FAN_eff"], required=False)
    raw_loss_col = find_existing_col(df, ["原损耗MW", "原损耗", "原始损耗MW"], required=False)
    corr_loss_col = find_existing_col(df, ["修正后损耗MW", "修正后损耗", "损耗MW"], required=False)
    raw_sev_col = find_existing_col(df, ["原异常程度", "原始异常程度"], required=False)
    corr_sev_col = find_existing_col(df, ["修正后异常程度", "异常程度"], required=False)

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    out["STEP2_原始风机功率之和MW"] = pd.to_numeric(df[raw_p_col], errors="coerce") if raw_p_col else np.nan
    out["STEP2_修正后风机功率之和MW"] = pd.to_numeric(df[corr_p_col], errors="coerce") if corr_p_col else np.nan
    out["STEP2_原损耗MW"] = pd.to_numeric(df[raw_loss_col], errors="coerce") if raw_loss_col else np.nan
    out["STEP2_修正后损耗MW"] = pd.to_numeric(df[corr_loss_col], errors="coerce") if corr_loss_col else np.nan
    out["STEP2_原异常程度"] = pd.to_numeric(df[raw_sev_col], errors="coerce").fillna(0.0) if raw_sev_col else 0.0
    out["STEP2_修正后异常程度"] = pd.to_numeric(df[corr_sev_col], errors="coerce").fillna(0.0) if corr_sev_col else 0.0

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return out


def classify_row_scene(result_values: List[str]) -> str:
    """
    根据一行所有 PROCESS_RESULT_#n，提炼场站时刻级 DATA_USE_SCENE。
    """
    vals = [str(x).strip() for x in result_values if pd.notna(x)]
    s = set(vals)

    if CORRECTED_SUBTYPE in s:
        return CORRECTED_SUBTYPE
    if DETAIL_POS_KEEP_RAW in s:
        return DETAIL_POS_KEEP_RAW
    if DETAIL_POS_ABNORMAL in s:
        return DETAIL_POS_ABNORMAL
    if ZERO_WITH_FAN_ABNORMAL_SUBTYPE in s:
        return ZERO_WITH_FAN_ABNORMAL_SUBTYPE
    if ZERO_NO_FAN_ABNORMAL_SUBTYPE in s:
        return ZERO_NO_FAN_ABNORMAL_SUBTYPE
    if ZERO_WITH_FAN_NORMAL_SUBTYPE in s:
        return ZERO_WITH_FAN_NORMAL_SUBTYPE
    if ZERO_NO_FAN_NORMAL_SUBTYPE in s:
        return ZERO_NO_FAN_NORMAL_SUBTYPE
    if POSITIVE_NON_REPEAT_SUBTYPE in s:
        return POSITIVE_NON_REPEAT_SUBTYPE
    if UNMATCHED_STEP3_SUBTYPE in s:
        return UNMATCHED_STEP3_SUBTYPE
    if vals:
        return "其他"
    return UNMATCHED_STEP3_SUBTYPE


def add_station_use_labels(
    out: pd.DataFrame,
    fan_nums: List[int],
    step2_detail: Optional[pd.DataFrame] = None,
    point_power_min_mw: float = POINT_OUTLIER_POWER_MIN_MW,
    point_severity_threshold: float = POINT_OUTLIER_SEVERITY_THRESHOLD,
) -> pd.DataFrame:
    """
    增加后续建模/验证所需的场站级标签和汇总列。

    不改变任何风机级 FINAL_ACTIVE_POWER_#n，只新增：
      DATA_USE_RESULT / DATA_USE_REASON / DATA_USE_SCENE
      RAW_FAN_POWER_SUM_MW / FINAL_FAN_POWER_SUM_MW / FINAL_LOSS_MW / FINAL_STATION_POWER_MW
      POINT_OUTLIER_*
    """
    out = out.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    raw_power_cols = [f"{POWER_PREFIX}{fan}" for fan in fan_nums if f"{POWER_PREFIX}{fan}" in out.columns]
    final_power_cols = [f"{FINAL_POWER_PREFIX}{fan}" for fan in fan_nums if f"{FINAL_POWER_PREFIX}{fan}" in out.columns]
    result_cols = [f"{PROCESS_RESULT_PREFIX}{fan}" for fan in fan_nums if f"{PROCESS_RESULT_PREFIX}{fan}" in out.columns]

    out["RAW_FAN_POWER_SUM_MW"] = out[raw_power_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True) if raw_power_cols else np.nan
    out["FINAL_FAN_POWER_SUM_MW"] = out[final_power_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True) if final_power_cols else np.nan

    if "ACTIVE_POWER_STATION" in out.columns:
        out["FINAL_STATION_POWER_MW"] = pd.to_numeric(out["ACTIVE_POWER_STATION"], errors="coerce")
    else:
        out["FINAL_STATION_POWER_MW"] = np.nan
    out["FINAL_LOSS_MW"] = out["FINAL_FAN_POWER_SUM_MW"] - out["FINAL_STATION_POWER_MW"]

    if result_cols:
        out["DATA_USE_SCENE"] = out[result_cols].apply(lambda row: classify_row_scene(row.tolist()), axis=1)
    else:
        out["DATA_USE_SCENE"] = UNMATCHED_STEP3_SUBTYPE

    # 合并 STEP2 明细，补充点级离群所需的异常程度和损耗字段。
    if step2_detail is not None and len(step2_detail) > 0:
        out = out.merge(step2_detail, on="timestamp", how="left")
    else:
        for col in [
            "STEP2_原始风机功率之和MW", "STEP2_修正后风机功率之和MW",
            "STEP2_原损耗MW", "STEP2_修正后损耗MW",
            "STEP2_原异常程度", "STEP2_修正后异常程度",
        ]:
            out[col] = np.nan

    out["POINT_OUTLIER_RESULT"] = POINT_NORMAL
    out["POINT_OUTLIER_REASON"] = ""
    out["POINT_OUTLIER_SOURCE_TYPE"] = ""
    out["POINT_OUTLIER_POWER_MW"] = np.nan
    out["POINT_OUTLIER_LOSS_MW"] = np.nan
    out["POINT_OUTLIER_SEVERITY"] = np.nan

    raw_power_for_outlier = pd.to_numeric(out["STEP2_原始风机功率之和MW"], errors="coerce").fillna(out["RAW_FAN_POWER_SUM_MW"])
    corr_power_for_outlier = pd.to_numeric(out["STEP2_修正后风机功率之和MW"], errors="coerce").fillna(out["FINAL_FAN_POWER_SUM_MW"])
    raw_loss_for_outlier = pd.to_numeric(out["STEP2_原损耗MW"], errors="coerce")
    corr_loss_for_outlier = pd.to_numeric(out["STEP2_修正后损耗MW"], errors="coerce").fillna(out["FINAL_LOSS_MW"])
    raw_sev = pd.to_numeric(out["STEP2_原异常程度"], errors="coerce").fillna(0.0)
    corr_sev = pd.to_numeric(out["STEP2_修正后异常程度"], errors="coerce").fillna(0.0)

    raw_keep_mask = (
        out["DATA_USE_SCENE"].eq(DETAIL_POS_KEEP_RAW)
        & (raw_power_for_outlier > point_power_min_mw)
        & (raw_sev >= point_severity_threshold)
    )
    raw_zero_normal_mask = (
        out["DATA_USE_SCENE"].eq(ZERO_NO_FAN_NORMAL_SUBTYPE)
        & (raw_power_for_outlier > point_power_min_mw)
        & (raw_sev >= point_severity_threshold)
    )
    corr_keep_mask = (
        out["DATA_USE_SCENE"].eq(CORRECTED_SUBTYPE)
        & (corr_power_for_outlier > point_power_min_mw)
        & (corr_sev >= point_severity_threshold)
    )

    # 优先级：修正后可用数据 > 原始保留不修正 > 原始零重复正常
    def set_outlier(mask: pd.Series, source_type: str, power_vals: pd.Series, loss_vals: pd.Series, sev_vals: pd.Series, reason: str):
        idx = mask.fillna(False)
        out.loc[idx, "POINT_OUTLIER_RESULT"] = POINT_OUTLIER
        out.loc[idx, "POINT_OUTLIER_SOURCE_TYPE"] = source_type
        out.loc[idx, "POINT_OUTLIER_POWER_MW"] = power_vals.loc[idx]
        out.loc[idx, "POINT_OUTLIER_LOSS_MW"] = loss_vals.loc[idx]
        out.loc[idx, "POINT_OUTLIER_SEVERITY"] = sev_vals.loc[idx]
        out.loc[idx, "POINT_OUTLIER_REASON"] = reason

    set_outlier(
        raw_zero_normal_mask,
        f"原始｜{ZERO_NO_FAN_NORMAL_SUBTYPE}",
        raw_power_for_outlier,
        raw_loss_for_outlier,
        raw_sev,
        f"原始口径功率>{point_power_min_mw:g}MW且原异常程度>={point_severity_threshold:g}",
    )
    set_outlier(
        raw_keep_mask,
        f"原始｜{DETAIL_POS_KEEP_RAW}",
        raw_power_for_outlier,
        raw_loss_for_outlier,
        raw_sev,
        f"原始口径功率>{point_power_min_mw:g}MW且原异常程度>={point_severity_threshold:g}",
    )
    set_outlier(
        corr_keep_mask,
        f"修正后｜{CORRECTED_SUBTYPE}",
        corr_power_for_outlier,
        corr_loss_for_outlier,
        corr_sev,
        f"修正后口径功率>{point_power_min_mw:g}MW且修正后异常程度>={point_severity_threshold:g}",
    )

    out["DATA_USE_RESULT"] = DATA_USE_AVAILABLE
    out["DATA_USE_REASON"] = "正常可用"

    # 场站时刻级不可用标签。点级离群最高优先级。
    abnormal_scene = out["DATA_USE_SCENE"].astype(str).str.contains("标记为异常分段", na=False)
    unmatched_scene = out["DATA_USE_SCENE"].eq(UNMATCHED_STEP3_SUBTYPE)
    point_outlier = out["POINT_OUTLIER_RESULT"].eq(POINT_OUTLIER)

    out.loc[out["DATA_USE_SCENE"].eq(CORRECTED_SUBTYPE), "DATA_USE_REASON"] = "正功率重复已修正后可用"
    out.loc[out["DATA_USE_SCENE"].eq(DETAIL_POS_KEEP_RAW), "DATA_USE_REASON"] = "正功率重复保留不修正，可用"
    out.loc[out["DATA_USE_SCENE"].eq(ZERO_NO_FAN_NORMAL_SUBTYPE), "DATA_USE_REASON"] = "无正功率重复且分段正常，可用"

    out.loc[abnormal_scene, "DATA_USE_RESULT"] = DATA_USE_UNAVAILABLE
    out.loc[abnormal_scene, "DATA_USE_REASON"] = "异常分段，不建议使用"

    out.loc[unmatched_scene, "DATA_USE_RESULT"] = DATA_USE_UNAVAILABLE
    out.loc[unmatched_scene, "DATA_USE_REASON"] = "未匹配STEP3分段，不建议使用"

    out.loc[point_outlier, "DATA_USE_RESULT"] = DATA_USE_UNAVAILABLE
    out.loc[point_outlier, "DATA_USE_REASON"] = "点级明显离群，不建议使用"

    # 将新增场站级字段放到最前面，便于后续实验筛选。
    front_cols = [
        "timestamp",
        "DATA_USE_RESULT", "DATA_USE_REASON", "DATA_USE_SCENE",
        "ACTIVE_POWER_STATION", "LIMIT_POWER",
        "RAW_FAN_POWER_SUM_MW", "FINAL_FAN_POWER_SUM_MW", "FINAL_LOSS_MW", "FINAL_STATION_POWER_MW",
        "POINT_OUTLIER_RESULT", "POINT_OUTLIER_REASON", "POINT_OUTLIER_SOURCE_TYPE",
        "POINT_OUTLIER_POWER_MW", "POINT_OUTLIER_LOSS_MW", "POINT_OUTLIER_SEVERITY",
    ]
    front_cols = [c for c in front_cols if c in out.columns]
    rest_cols = [c for c in out.columns if c not in front_cols and not c.startswith("STEP2_")]
    out = out[front_cols + rest_cols]
    return out


# ============================================================
# 主 CSV 自动识别
# ============================================================
def find_main_csv(station_dir: str | Path) -> Path:
    station_dir = Path(station_dir)
    if not station_dir.exists():
        raise FileNotFoundError(f"场站目录不存在：{station_dir}")

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

    candidates = []
    for p in station_dir.glob("*.csv"):
        name = p.name
        if any(k in name for k in skip_keywords):
            continue
        candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"未在场站目录下找到主 CSV：{station_dir}")

    if len(candidates) > 1:
        # 通常主 CSV 文件最大，优先选择最大者；同时打印提示。
        candidates = sorted(candidates, key=lambda x: x.stat().st_size, reverse=True)
        print("⚠️ 检测到多个候选主 CSV，将选择文件最大的一个：")
        for c in candidates[:5]:
            print(f"  - {c} ({c.stat().st_size / 1024 / 1024:.2f} MB)")
    return candidates[0]


# ============================================================
# 风机列识别
# ============================================================
def extract_fan_numbers(columns: Iterable[str]) -> List[int]:
    cols = [str(c).strip() for c in columns]
    nums = set()

    pattern = re.compile(r"^(STATUS_|ACTIVE_POWER_|WINDSPEED_)#(\d+)$")
    for col in cols:
        m = pattern.match(col)
        if m:
            nums.add(int(m.group(2)))

    valid_nums = []
    col_set = set(cols)
    for n in sorted(nums):
        if f"{STATUS_PREFIX}{n}" in col_set and f"{POWER_PREFIX}{n}" in col_set and f"{WIND_PREFIX}{n}" in col_set:
            valid_nums.append(n)

    return valid_nums


# ============================================================
# STEP4 / STEP3 加载
# ============================================================
def load_step4_scored(step4_scored_path: Path, threshold: float) -> pd.DataFrame:
    scored = read_csv_auto(step4_scored_path)

    required_base = {"分段ID", "阈值"}
    missing = required_base - set(scored.columns)
    if missing:
        raise ValueError(f"STEP4 scored 文件缺少必要列：{missing}，文件：{step4_scored_path}")

    subtype_col = first_existing_col(scored, ["最终处理细分类型", "最终处理方式"])
    if subtype_col is None:
        raise ValueError("STEP4 scored 文件缺少“最终处理细分类型”或“最终处理方式”列")

    scored = scored.copy()
    scored["_阈值_num"] = parse_thresholds_for_match(scored["阈值"])
    threshold_num = round(float(threshold), 10)
    scored = scored[scored["_阈值_num"] == threshold_num].copy()

    if len(scored) == 0:
        available = sorted(pd.to_numeric(read_csv_auto(step4_scored_path)["阈值"], errors="coerce").dropna().unique().tolist())
        raise RuntimeError(
            f"STEP4 scored 文件中没有找到阈值 {threshold}。\n"
            f"可用阈值示例：{available[:20]}"
        )

    out = scored[["分段ID", subtype_col]].copy()
    out = out.rename(columns={subtype_col: "最终处理细分类型"})
    out["最终处理细分类型"] = out["最终处理细分类型"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["分段ID"], keep="last")
    return out


def load_step3_detail(step3_detail_path: Path) -> pd.DataFrame:
    detail = read_csv_auto(step3_detail_path)

    required = {"分段ID", "开始时间", "结束时间", "风机编号"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"STEP3 明细文件缺少必要列：{missing}，文件：{step3_detail_path}")

    out = detail.copy()
    out["开始时间"] = pd.to_datetime(out["开始时间"], errors="coerce")
    out["结束时间"] = pd.to_datetime(out["结束时间"], errors="coerce")
    out["风机编号"] = pd.to_numeric(out["风机编号"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["开始时间", "结束时间", "风机编号"]).copy()
    out = out[out["结束时间"] >= out["开始时间"]].copy()
    out["风机编号"] = out["风机编号"].astype(int)
    return out


def load_step3_segments(step3_segment_path: Path) -> pd.DataFrame:
    """
    读取 STEP3 分段汇总表。
    这里主要用于处理“冻结风机数=0”的空档异常分段：
    这类分段在 freeze_count_segments_detail.csv 中没有风机明细行，
    但如果 STEP4 判定为异常，需要对该时间段内所有风机打标。
    """
    seg = read_csv_auto(step3_segment_path)

    required = {"分段ID", "开始时间", "结束时间"}
    missing = required - set(seg.columns)
    if missing:
        raise ValueError(f"STEP3 分段汇总文件缺少必要列：{missing}，文件：{step3_segment_path}")

    out = seg.copy()
    out["开始时间"] = pd.to_datetime(out["开始时间"], errors="coerce")
    out["结束时间"] = pd.to_datetime(out["结束时间"], errors="coerce")

    if "冻结风机数" in out.columns:
        out["冻结风机数"] = pd.to_numeric(out["冻结风机数"], errors="coerce").fillna(0).astype(int)
    else:
        out["冻结风机数"] = 0

    if "重复正功率之和MW" in out.columns:
        p_col = "重复正功率之和MW"
    elif "重复风机功率之和MW" in out.columns:
        p_col = "重复风机功率之和MW"
    elif "重复功率之和MW" in out.columns:
        p_col = "重复功率之和MW"
    else:
        p_col = None

    if p_col:
        out["重复正功率之和MW"] = pd.to_numeric(out[p_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        out["重复正功率之和MW"] = 0.0

    out = out.dropna(subset=["开始时间", "结束时间"]).copy()
    out = out[out["结束时间"] >= out["开始时间"]].copy()
    return out


def build_fan_time_subtype_map(step3_detail: pd.DataFrame, step4_map: pd.DataFrame) -> pd.DataFrame:
    """
    将 STEP3 分段-风机明细与 STEP4 指定阈值的最终处理细分类型合并，
    并展开到逐分钟、逐风机：
        时间, 风机编号, 最终处理细分类型
    """
    work = step3_detail.merge(step4_map, on="分段ID", how="left")
    work["最终处理细分类型"] = work["最终处理细分类型"].fillna("未匹配STEP4分段").astype(str)

    rows = []
    for _, row in work.iterrows():
        start = pd.to_datetime(row["开始时间"])
        end = pd.to_datetime(row["结束时间"])
        fan = int(row["风机编号"])
        subtype = str(row["最终处理细分类型"]).strip()

        rng = pd.date_range(start=start, end=end, freq="min")
        if len(rng) == 0:
            continue

        rows.append(pd.DataFrame({
            "timestamp": rng,
            "风机编号": fan,
            "最终处理细分类型": subtype,
        }))

    if not rows:
        return pd.DataFrame(columns=["timestamp", "风机编号", "最终处理细分类型"])

    out = pd.concat(rows, ignore_index=True)

    # 如果同一风机同一分钟被多个分段命中，按优先级保留：
    # 修正后保留 > 标记为异常分段 > 其他。
    def priority(s: pd.Series) -> pd.Series:
        text = s.astype(str)
        return np.select(
            [
                text.eq(CORRECTED_SUBTYPE),
                text.str.contains("标记为异常分段", na=False),
            ],
            [3, 2],
            default=1,
        )

    out["_priority"] = priority(out["最终处理细分类型"])
    out = (
        out.sort_values(["timestamp", "风机编号", "_priority"])
        .drop_duplicates(subset=["timestamp", "风机编号"], keep="last")
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )
    return out




def build_all_fan_time_subtype_map_for_positive_segments(
    step3_segments: pd.DataFrame,
    step4_map: pd.DataFrame,
    fan_nums: List[int],
) -> pd.DataFrame:
    """
    对“重复正功率>0”分段建立全风机底图。

    业务含义：
    - 一个正功率重复分段内，只有 STEP3 明细表里的部分风机发生了重复；
    - 但同一时间段内，其他未重复风机也需要明确标识为：
      “重复正功率>0｜非重复风机”；
    - 后续再用 STEP3 明细展开的重复风机标签覆盖这些底图标签。
    """
    if step3_segments is None or len(step3_segments) == 0:
        return pd.DataFrame(columns=["timestamp", "风机编号", "最终处理细分类型"])

    work = step3_segments.merge(step4_map, on="分段ID", how="left")
    work["最终处理细分类型"] = work["最终处理细分类型"].fillna("").astype(str).str.strip()

    repeat_positive = pd.to_numeric(work.get("重复正功率之和MW", 0.0), errors="coerce").fillna(0.0)
    target = work[
        (repeat_positive > 0)
        & (work["最终处理细分类型"].isin(POSITIVE_SEGMENT_SUBTYPES))
    ].copy()

    if len(target) == 0:
        return pd.DataFrame(columns=["timestamp", "风机编号", "最终处理细分类型"])

    rows = []
    fan_nums = [int(x) for x in fan_nums]

    for _, row in target.iterrows():
        start = pd.to_datetime(row["开始时间"])
        end = pd.to_datetime(row["结束时间"])
        rng = pd.date_range(start=start, end=end, freq="min")
        if len(rng) == 0:
            continue

        base = pd.MultiIndex.from_product(
            [rng, fan_nums],
            names=["timestamp", "风机编号"],
        ).to_frame(index=False)
        base["最终处理细分类型"] = POSITIVE_NON_REPEAT_SUBTYPE
        rows.append(base)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "风机编号", "最终处理细分类型"])

    return pd.concat(rows, ignore_index=True)


def build_all_fan_time_subtype_map_for_empty_segments(
    step3_segments: pd.DataFrame,
    step4_map: pd.DataFrame,
    fan_nums: List[int],
) -> pd.DataFrame:
    """
    处理“重复正功率=0且冻结风机数=0”的空档分段。

    这类分段没有风机明细行，无法从 freeze_count_segments_detail.csv 展开。
    但它们仍然属于 STEP4 的最终处理细分类型：
      - 重复正功率=0且冻结风机数=0｜正常
      - 重复正功率=0且冻结风机数=0｜标记为异常分段

    因此需要根据 freeze_count_segments.csv 的时间范围，
    将该时间段内所有风机的 PROCESS_RESULT_#n 标记为对应细分类型。
    功率不做修正，FINAL_ACTIVE_POWER_#n 仍等于 ACTIVE_POWER_#n。
    """
    if step3_segments is None or len(step3_segments) == 0:
        return pd.DataFrame(columns=["timestamp", "风机编号", "最终处理细分类型"])

    work = step3_segments.merge(step4_map, on="分段ID", how="left")
    work["最终处理细分类型"] = work["最终处理细分类型"].fillna("").astype(str).str.strip()

    freeze_count = pd.to_numeric(work.get("冻结风机数", 0), errors="coerce").fillna(0)
    target = work[
        (freeze_count <= 0)
        & (
            work["最终处理细分类型"].eq(ZERO_NO_FAN_NORMAL_SUBTYPE)
            | work["最终处理细分类型"].eq(ZERO_NO_FAN_ABNORMAL_SUBTYPE)
        )
    ].copy()

    if len(target) == 0:
        return pd.DataFrame(columns=["timestamp", "风机编号", "最终处理细分类型"])

    rows = []
    fan_nums = [int(x) for x in fan_nums]

    for _, row in target.iterrows():
        start = pd.to_datetime(row["开始时间"])
        end = pd.to_datetime(row["结束时间"])
        subtype = str(row["最终处理细分类型"]).strip()

        rng = pd.date_range(start=start, end=end, freq="min")
        if len(rng) == 0:
            continue

        # 对该空档分段内的所有风机打标
        base = pd.MultiIndex.from_product(
            [rng, fan_nums],
            names=["timestamp", "风机编号"],
        ).to_frame(index=False)
        base["最终处理细分类型"] = subtype
        rows.append(base)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "风机编号", "最终处理细分类型"])

    return pd.concat(rows, ignore_index=True)


def combine_fan_time_maps(*maps: pd.DataFrame) -> pd.DataFrame:
    """
    合并多类风机-时间标签映射。
    若同一风机同一时刻命中多种类型，优先级：
      1. 重复正功率>0｜修正后保留
      2. 重复正功率>0｜标记为异常分段 / 其他标记异常分段
      3. 重复正功率>0｜保留（不修正）
      4. 重复正功率=0... 的全风机空档标签
      5. 重复正功率>0｜非重复风机
    """
    valid = [m for m in maps if m is not None and len(m) > 0]
    if not valid:
        return pd.DataFrame(columns=["timestamp", "风机编号", "最终处理细分类型"])

    out = pd.concat(valid, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["风机编号"] = pd.to_numeric(out["风机编号"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["timestamp", "风机编号"]).copy()
    out["风机编号"] = out["风机编号"].astype(int)
    out["最终处理细分类型"] = out["最终处理细分类型"].astype(str).str.strip()

    text = out["最终处理细分类型"]
    out["_priority"] = np.select(
        [
            text.eq(CORRECTED_SUBTYPE),
            text.str.contains("标记为异常分段", na=False),
            text.eq(DETAIL_POS_KEEP_RAW),
            text.str.startswith("重复正功率=0", na=False),
            text.eq(POSITIVE_NON_REPEAT_SUBTYPE),
        ],
        [5, 4, 3, 2, 1],
        default=0,
    )

    out = (
        out.sort_values(["timestamp", "风机编号", "_priority"])
        .drop_duplicates(subset=["timestamp", "风机编号"], keep="last")
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )
    return out

# ============================================================
# 宽表构造与回填
# ============================================================
def build_base_wide_output(raw: pd.DataFrame, fan_nums: List[int]) -> pd.DataFrame:
    """
    一次性构造宽表，避免循环 insert 导致 DataFrame fragmentation。

    输出最前面保留场站级原始列：
    - ACTIVE_POWER_STATION
    - LIMIT_POWER
    """
    data = {"timestamp": pd.to_datetime(raw["timestamp"], errors="coerce")}

    for col in STATION_LEVEL_OUTPUT_COLS:
        if col in raw.columns:
            data[col] = pd.to_numeric(raw[col], errors="coerce")
        else:
            data[col] = np.nan
            print(f"⚠️ 原始主CSV缺少场站级列：{col}，输出中将填充为空值。")

    for fan in fan_nums:
        status_col = f"{STATUS_PREFIX}{fan}"
        power_col = f"{POWER_PREFIX}{fan}"
        wind_col = f"{WIND_PREFIX}{fan}"
        final_col = f"{FINAL_POWER_PREFIX}{fan}"
        result_col = f"{PROCESS_RESULT_PREFIX}{fan}"

        data[status_col] = raw[status_col]
        data[power_col] = pd.to_numeric(raw[power_col], errors="coerce")
        data[wind_col] = pd.to_numeric(raw[wind_col], errors="coerce")
        data[final_col] = pd.to_numeric(raw[power_col], errors="coerce")
        data[result_col] = DEFAULT_PROCESS_RESULT

    return pd.DataFrame(data)


def apply_subtype_results(
    out: pd.DataFrame,
    fan_time_map: pd.DataFrame,
    fan_nums: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    回填 PROCESS_RESULT_#n 和 FINAL_ACTIVE_POWER_#n。
    """
    if len(fan_time_map) == 0:
        summary = pd.DataFrame([{
            "说明": "没有可回填的风机-时间分段记录",
            "命中风机时刻数": 0,
            "置零修正风机时刻数": 0,
        }])
        return out, summary

    out = out.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    row_pos_by_time = pd.Series(np.arange(len(out)), index=out["timestamp"]).to_dict()

    # 统计用
    total_hit = 0
    total_corrected_zero = 0
    subtype_counter: Dict[str, int] = {}

    fan_set = set(fan_nums)

    for fan, g in fan_time_map.groupby("风机编号", sort=False):
        fan = int(fan)
        if fan not in fan_set:
            continue

        power_col = f"{POWER_PREFIX}{fan}"
        final_col = f"{FINAL_POWER_PREFIX}{fan}"
        result_col = f"{PROCESS_RESULT_PREFIX}{fan}"

        # 将时间映射到 out 的行位置
        positions = g["timestamp"].map(row_pos_by_time)
        valid_mask = positions.notna()
        if not valid_mask.any():
            continue

        g2 = g.loc[valid_mask].copy()
        pos = positions.loc[valid_mask].astype(int).to_numpy()
        subtypes = g2["最终处理细分类型"].astype(str).to_numpy()

        total_hit += len(pos)
        for subtype, count in pd.Series(subtypes).value_counts().items():
            subtype_counter[subtype] = subtype_counter.get(subtype, 0) + int(count)

        # PROCESS_RESULT_#n 直接使用最终处理细分类型
        out.loc[pos, result_col] = subtypes

        # 仅 “重复正功率>0｜修正后保留” 且 ACTIVE_POWER_#n > 0 时置零
        raw_power = pd.to_numeric(out.loc[pos, power_col], errors="coerce").to_numpy(dtype=float)
        corrected_mask = (subtypes == CORRECTED_SUBTYPE) & np.isfinite(raw_power) & (raw_power > 0)

        if corrected_mask.any():
            corrected_pos = pos[corrected_mask]
            out.loc[corrected_pos, final_col] = 0.0
            total_corrected_zero += int(corrected_mask.sum())

    summary_rows = [
        {
            "统计项": "命中STEP3/STEP4分段的风机时刻数",
            "数量": total_hit,
        },
        {
            "统计项": f"FINAL_ACTIVE_POWER置零数量（{CORRECTED_SUBTYPE} 且原始功率>0）",
            "数量": total_corrected_zero,
        },
    ]
    for subtype, count in sorted(subtype_counter.items(), key=lambda x: (-x[1], x[0])):
        summary_rows.append({
            "统计项": f"PROCESS_RESULT类型：{subtype}",
            "数量": count,
        })

    return out, pd.DataFrame(summary_rows)


def select_output_rows(out: pd.DataFrame, max_output_rows: int, keep_hit_rows: bool = False) -> pd.DataFrame:
    """
    默认输出前 X 行。
    如果 keep_hit_rows=True，则优先保留至少有一个 PROCESS_RESULT 不是“未匹配STEP3分段”的行。
    """
    if max_output_rows is None or max_output_rows <= 0 or len(out) <= max_output_rows:
        return out

    if not keep_hit_rows:
        return out.head(max_output_rows).copy()

    result_cols = [c for c in out.columns if c.startswith(PROCESS_RESULT_PREFIX)]
    if not result_cols:
        return out.head(max_output_rows).copy()

    hit_mask = (out[result_cols] != DEFAULT_PROCESS_RESULT).any(axis=1)
    hit_df = out.loc[hit_mask].copy()
    normal_df = out.loc[~hit_mask].copy()

    if len(hit_df) >= max_output_rows:
        return hit_df.head(max_output_rows).copy()

    need = max_output_rows - len(hit_df)
    return pd.concat([hit_df, normal_df.head(need)], ignore_index=True)


# ============================================================
# main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="STEP6：生成风机宽表修正结果，PROCESS_RESULT直接使用STEP4最终处理细分类型")
    parser.add_argument("--station-dir", default=DEFAULT_STATION_DIR, help="场站目录，用于自动查找主CSV")
    parser.add_argument("--input-csv", default=None, help="场站主CSV路径；不填则自动从station-dir查找")
    parser.add_argument("--step3-dir", default=DEFAULT_STEP3_DIR, help="STEP3输出目录")
    parser.add_argument("--step4-dir", default=DEFAULT_STEP4_DIR, help="STEP4输出目录")
    parser.add_argument("--step3-detail-file", default=None, help="STEP3 freeze_count_segments_detail.csv路径；不填则使用step3-dir默认文件")
    parser.add_argument("--step3-segment-file", default=None, help="STEP3 freeze_count_segments.csv路径；不填则使用step3-dir默认文件")
    parser.add_argument("--step4-scored-file", default=None, help="STEP4 threshold_scan_scored.csv路径；不填则使用step4-dir默认文件")
    parser.add_argument("--step2-detail-file", default=None, help="STEP2 station_anomaly_detail.csv路径；不填则默认使用 station-dir/step2-异常检测-仅正功率/station_anomaly_detail.csv")
    parser.add_argument("--point-outlier-power-min-mw", type=float, default=POINT_OUTLIER_POWER_MIN_MW, help="点级离群识别的当前口径功率下限MW；小于等于该值视作超低功率区间，暂不标记离群")
    parser.add_argument("--point-outlier-severity-threshold", type=float, default=POINT_OUTLIER_SEVERITY_THRESHOLD, help="点级离群识别的异常程度阈值")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.15,required=True, help="使用的STEP4阈值，例如 0.16")
    parser.add_argument("--max-output-rows", type=int, default=0, help="最多输出前X行；0表示不限制")
    parser.add_argument("--keep-hit-rows", action="store_true", help="限制输出行数时，优先保留命中重复分段的时间戳")
    args = parser.parse_args()

    station_dir = Path(args.station_dir)
    input_csv = Path(args.input_csv) if args.input_csv else find_main_csv(station_dir)
    step3_detail_path = Path(args.step3_detail_file) if args.step3_detail_file else Path(args.step3_dir) / DEFAULT_STEP3_DETAIL_NAME
    step3_segment_path = Path(args.step3_segment_file) if args.step3_segment_file else Path(args.step3_dir) / DEFAULT_STEP3_SEGMENT_NAME
    step4_scored_path = Path(args.step4_scored_file) if args.step4_scored_file else Path(args.step4_dir) / DEFAULT_STEP4_SCORED_NAME
    step2_detail_path = Path(args.step2_detail_file) if args.step2_detail_file else station_dir / "step2-异常检测-仅正功率" / DEFAULT_STEP2_DETAIL_NAME
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("STEP6：生成风机宽表修正结果")
    print("=" * 88)
    print(f"主CSV: {input_csv}")
    print(f"STEP3明细: {step3_detail_path}")
    print(f"STEP3分段汇总: {step3_segment_path}")
    print(f"STEP4 scored: {step4_scored_path}")
    print(f"STEP2明细: {step2_detail_path}")
    print(f"阈值: {args.threshold}")
    print(f"点级离群阈值: 当前口径功率 > {args.point_outlier_power_min_mw:g} MW 且异常程度 >= {args.point_outlier_severity_threshold:g}")
    print(f"输出目录: {output_dir}")

    raw = read_csv_auto(input_csv)
    raw.columns = raw.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    if "timestamp" not in raw.columns:
        raise ValueError(f"主CSV缺少 timestamp 列：{input_csv}")

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    fan_nums = extract_fan_numbers(raw.columns)
    if not fan_nums:
        raise RuntimeError("未识别到完整风机列。需要同时存在 STATUS_#n、ACTIVE_POWER_#n、WINDSPEED_#n。")

    print(f"识别到风机数量: {len(fan_nums)}")
    print(f"风机编号范围: {fan_nums[:5]} ... {fan_nums[-5:] if len(fan_nums) >= 5 else fan_nums}")

    step4_map = load_step4_scored(step4_scored_path, threshold=args.threshold)
    step3_detail = load_step3_detail(step3_detail_path)
    step3_segments = load_step3_segments(step3_segment_path)
    fan_time_map_positive_non_repeat = build_all_fan_time_subtype_map_for_positive_segments(step3_segments, step4_map, fan_nums)
    fan_time_map_from_detail = build_fan_time_subtype_map(step3_detail, step4_map)
    fan_time_map_empty_segments = build_all_fan_time_subtype_map_for_empty_segments(step3_segments, step4_map, fan_nums)
    fan_time_map = combine_fan_time_maps(
        fan_time_map_positive_non_repeat,
        fan_time_map_from_detail,
        fan_time_map_empty_segments,
    )

    print(f"STEP3风机明细行数: {len(step3_detail):,}")
    print(f"STEP3分段汇总行数: {len(step3_segments):,}")
    print(f"由正功率重复分段展开的非重复风机底图记录数: {len(fan_time_map_positive_non_repeat):,}")
    print(f"由风机明细展开的重复风机-时间记录数: {len(fan_time_map_from_detail):,}")
    print(f"由空档分段（正常/异常）展开的全风机-时间记录数: {len(fan_time_map_empty_segments):,}")
    print(f"合并后的风机-时间命中记录数: {len(fan_time_map):,}")
    if len(fan_time_map) > 0:
        print("命中记录的最终处理细分类型分布：")
        print(fan_time_map["最终处理细分类型"].value_counts().head(20).to_string())

    step2_detail_for_outlier = load_step2_detail_for_point_outliers(step2_detail_path)

    out = build_base_wide_output(raw, fan_nums)
    out, summary_df = apply_subtype_results(out, fan_time_map, fan_nums)
    out = add_station_use_labels(
        out=out,
        fan_nums=fan_nums,
        step2_detail=step2_detail_for_outlier,
        point_power_min_mw=args.point_outlier_power_min_mw,
        point_severity_threshold=args.point_outlier_severity_threshold,
    )

    # 补充场站级可用性和点级离群统计到 summary。
    extra_summary = []
    if "DATA_USE_RESULT" in out.columns:
        for k, v in out["DATA_USE_RESULT"].value_counts(dropna=False).items():
            extra_summary.append({"统计项": f"DATA_USE_RESULT：{k}", "数量": int(v)})
    if "DATA_USE_SCENE" in out.columns:
        for k, v in out["DATA_USE_SCENE"].value_counts(dropna=False).items():
            extra_summary.append({"统计项": f"DATA_USE_SCENE：{k}", "数量": int(v)})
    if "POINT_OUTLIER_RESULT" in out.columns:
        for k, v in out["POINT_OUTLIER_RESULT"].value_counts(dropna=False).items():
            extra_summary.append({"统计项": f"POINT_OUTLIER_RESULT：{k}", "数量": int(v)})
    if extra_summary:
        summary_df = pd.concat([summary_df, pd.DataFrame(extra_summary)], ignore_index=True)

    output_df = select_output_rows(out, max_output_rows=args.max_output_rows, keep_hit_rows=args.keep_hit_rows)

    threshold_tag = safe_threshold_str(args.threshold)
    stem = input_csv.stem
    output_file = output_dir / f"{stem}_fan_wide_correction_threshold_{threshold_tag}.csv"
    summary_file = output_dir / f"{stem}_fan_wide_correction_threshold_{threshold_tag}_summary.csv"

    output_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print("\n完成：")
    print(f"- 输出宽表: {output_file}")
    print(f"- 输出汇总: {summary_file}")
    print(f"- 原始时间戳行数: {len(out):,}")
    print(f"- 实际输出行数: {len(output_df):,}")
    print("\n汇总：")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
