#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP4 江苏全站版：重复冻结分段阈值分析

输入：
  - STEP3 输出的 freeze_count_segments.csv

输出：
  - threshold_scan_scored.csv
      逐分段、逐阈值打分明细
  - threshold_scan_raw_profile.csv
      修正前原始特征汇总
  - threshold_scan_repair_effect.csv
      重复正功率>0分段的修正作用分类汇总
  - threshold_scan_final_decision.csv
      最终处理方式汇总
  - threshold_recommendation.csv
      简单阈值推荐表

说明：
  - 不再区分 BING / DING / WU 线路。
  - 不再使用厂商字段。
  - 同时考虑“异常分钟数”和“异常程度”。
  - 阈值仍主要作用于异常占比；异常程度用于判断修正是否有效。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# 默认参数
# ============================================================
DEFAULT_STEP3_DIR = r"场站数据\LGXRFD\step3-重复冻结分段"
DEFAULT_OUTPUT_DIR = r"场站数据\LGXRFD\step4-阈值分析-细分版"
DEFAULT_SEGMENT_FILE_NAME = "freeze_count_segments.csv"

DEFAULT_THRESHOLDS = "0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.40,0.50"
DEFAULT_RATIO_EPS = 1e-9
DEFAULT_SEVERITY_EPS = 1e-9

CLASS_REPEAT_POSITIVE = "重复正功率>0分段"
CLASS_REPEAT_ZERO_NO_FAN = "重复正功率=0，且冻结风机数=0"
CLASS_REPEAT_ZERO_WITH_FAN = "重复正功率=0，且冻结风机数>0"
CLASS_ORDER = [CLASS_REPEAT_POSITIVE, CLASS_REPEAT_ZERO_NO_FAN, CLASS_REPEAT_ZERO_WITH_FAN]

DECISION_ABNORMAL = "标记为异常分段"
DECISION_KEEP_CORRECTED = "修正后保留"
DECISION_KEEP_RAW = "保留（不修正）"
DECISION_NORMAL = "正常"
DECISION_ORDER = [DECISION_ABNORMAL, DECISION_KEEP_CORRECTED, DECISION_KEEP_RAW, DECISION_NORMAL]

CASE_1 = "情况1：修正前正常，修正后异常，保留（不修正）"
CASE_2 = "情况2：修正前异常，修正后正常，修正后保留"
CASE_3 = "情况3：修正前正常，修正后正常，修正后改善，修正后保留"
CASE_4 = "情况4：修正前正常，修正后正常，修正后不变，保留（不修正）"
CASE_5 = "情况5：修正前正常，修正后正常，修正后恶化，保留（不修正）"
CASE_6 = "情况6：修正前异常，修正后异常，修正后改善，标记为异常分段"
CASE_7 = "情况7：修正前异常，修正后异常，修正后不变，标记为异常分段"
CASE_8 = "情况8：修正前异常，修正后异常，修正后恶化，标记为异常分段"
CASE_NOT_APPLICABLE = "不适用：重复正功率=0分段，不参与修正作用分类"
CASE_ORDER = [CASE_1, CASE_2, CASE_3, CASE_4, CASE_5, CASE_6, CASE_7, CASE_8]

CASE_TO_DECISION = {
    CASE_1: DECISION_KEEP_RAW,
    CASE_2: DECISION_KEEP_CORRECTED,
    CASE_3: DECISION_KEEP_CORRECTED,
    CASE_4: DECISION_KEEP_RAW,
    CASE_5: DECISION_KEEP_RAW,
    CASE_6: DECISION_ABNORMAL,
    CASE_7: DECISION_ABNORMAL,
    CASE_8: DECISION_ABNORMAL,
}

DETAIL_POS_KEEP_RAW = "重复正功率>0｜保留（不修正）"
DETAIL_POS_KEEP_CORRECTED = "重复正功率>0｜修正后保留"
DETAIL_POS_ABNORMAL = "重复正功率>0｜标记为异常分段"
DETAIL_ZERO_WITH_FAN_NORMAL = "重复正功率=0且冻结风机数>0｜正常"
DETAIL_ZERO_WITH_FAN_ABNORMAL = "重复正功率=0且冻结风机数>0｜标记为异常分段"
DETAIL_ZERO_NO_FAN_NORMAL = "重复正功率=0且冻结风机数=0｜正常"
DETAIL_ZERO_NO_FAN_ABNORMAL = "重复正功率=0且冻结风机数=0｜标记为异常分段"
DETAIL_DECISION_ORDER = [
    DETAIL_POS_KEEP_RAW,
    DETAIL_POS_KEEP_CORRECTED,
    DETAIL_POS_ABNORMAL,
    DETAIL_ZERO_WITH_FAN_NORMAL,
    DETAIL_ZERO_WITH_FAN_ABNORMAL,
    DETAIL_ZERO_NO_FAN_NORMAL,
    DETAIL_ZERO_NO_FAN_ABNORMAL,
]


# ============================================================
# 通用工具
# ============================================================
def read_csv_auto(path: str | Path) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"文件不存在：{path_obj}")

    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path_obj, encoding=enc)
        except Exception as exc:  # pragma: no cover - 用于兼容现场编码
            last_err = exc
    raise RuntimeError(f"读取失败：{path_obj}\n最后一次报错：{last_err}")


def parse_thresholds(raw: str) -> List[float]:
    vals: list[float] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    return sorted(set(vals))


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator and denominator > 0 else np.nan


def to_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def direction_by_diff(diff: float, eps: float = DEFAULT_RATIO_EPS) -> str:
    if diff < -eps:
        return "下降"
    if diff > eps:
        return "上升"
    return "不变"


# ============================================================
# 字段规范化
# ============================================================
def normalize_segment_df(raw_df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """兼容 STEP3 字段名，生成 STEP4 所需标准列。"""
    df = raw_df.copy()

    required = ["分段ID", "开始时间", "结束时间"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"STEP3 分段文件缺少必要列：{missing_required}")

    duration_col = first_existing_col(df, ["持续时长min", "持续时长(min)", "持续时长"])
    repeat_power_col = first_existing_col(df, ["重复正功率之和MW", "重复风机功率之和MW", "重复功率之和MW"])
    freeze_count_col = first_existing_col(df, ["冻结风机数", "重复风机数"])

    if duration_col is None:
        raise ValueError("STEP3 分段文件缺少持续时长列：持续时长min / 持续时长(min)")
    if repeat_power_col is None:
        raise ValueError("STEP3 分段文件缺少重复正功率列：重复正功率之和MW / 重复风机功率之和MW")
    if freeze_count_col is None:
        raise ValueError("STEP3 分段文件缺少冻结风机数列：冻结风机数")

    out = pd.DataFrame(index=df.index)
    out["场站"] = station_name
    out["分段ID"] = df["分段ID"]
    out["开始时间"] = pd.to_datetime(df["开始时间"], errors="coerce")
    out["结束时间"] = pd.to_datetime(df["结束时间"], errors="coerce")
    out["持续时长min"] = pd.to_numeric(df[duration_col], errors="coerce").fillna(0).clip(lower=0)
    out["纳入统计分钟数"] = to_numeric_series(df, "纳入统计分钟数", default=np.nan)
    out["纳入统计分钟数"] = out["纳入统计分钟数"].fillna(out["持续时长min"]).clip(lower=0)
    out["冻结风机数"] = pd.to_numeric(df[freeze_count_col], errors="coerce").fillna(0).clip(lower=0)
    out["重复正功率之和MW"] = pd.to_numeric(df[repeat_power_col], errors="coerce").fillna(0).clip(lower=0)
    # 兼容旧版字段名，方便现有 Dash 或其他脚本继续读取。
    out["重复风机功率之和MW"] = out["重复正功率之和MW"]

    out["原异常分钟数"] = to_numeric_series(df, "原异常分钟数", 0).clip(lower=0)
    out["修正后异常分钟数"] = to_numeric_series(df, "修正后异常分钟数", 0).clip(lower=0)

    if "原异常占比" in df.columns:
        out["原异常占比"] = to_numeric_series(df, "原异常占比", 0).clip(lower=0)
    else:
        out["原异常占比"] = out.apply(lambda r: safe_ratio(r["原异常分钟数"], r["纳入统计分钟数"]), axis=1)

    if "修正后异常占比" in df.columns:
        out["修正后异常占比"] = to_numeric_series(df, "修正后异常占比", 0).clip(lower=0)
    else:
        out["修正后异常占比"] = out.apply(lambda r: safe_ratio(r["修正后异常分钟数"], r["纳入统计分钟数"]), axis=1)

    out["异常分钟数变化量"] = to_numeric_series(df, "异常分钟数变化量", np.nan)
    out["异常分钟数变化量"] = out["异常分钟数变化量"].fillna(out["修正后异常分钟数"] - out["原异常分钟数"])

    out["异常占比变化量"] = to_numeric_series(df, "异常占比变化量", np.nan)
    out["异常占比变化量"] = out["异常占比变化量"].fillna(out["修正后异常占比"] - out["原异常占比"])

    out["原异常程度总和"] = to_numeric_series(df, "原异常程度总和", 0).clip(lower=0)
    out["修正后异常程度总和"] = to_numeric_series(df, "修正后异常程度总和", 0).clip(lower=0)
    out["异常程度变化量"] = to_numeric_series(df, "异常程度变化量", np.nan)
    out["异常程度变化量"] = out["异常程度变化量"].fillna(out["修正后异常程度总和"] - out["原异常程度总和"])

    if "异常程度下降率" in df.columns:
        out["异常程度下降率"] = to_numeric_series(df, "异常程度下降率", np.nan)
    else:
        out["异常程度下降率"] = np.where(
            out["原异常程度总和"] > 0,
            (out["原异常程度总和"] - out["修正后异常程度总和"]) / out["原异常程度总和"],
            np.nan,
        )

    out["原平均异常程度"] = to_numeric_series(df, "原平均异常程度", np.nan)
    out["修正后平均异常程度"] = to_numeric_series(df, "修正后平均异常程度", np.nan)
    out["原最大异常程度"] = to_numeric_series(df, "原最大异常程度", 0).clip(lower=0)
    out["修正后最大异常程度"] = to_numeric_series(df, "修正后最大异常程度", 0).clip(lower=0)

    out["原负损耗分钟数"] = to_numeric_series(df, "原负损耗分钟数", 0).clip(lower=0)
    out["修正后负损耗分钟数"] = to_numeric_series(df, "修正后负损耗分钟数", 0).clip(lower=0)
    out["站端功率重复硬异常分钟数"] = to_numeric_series(df, "站端功率重复硬异常分钟数", 0).clip(lower=0)
    out["物理越限硬异常分钟数"] = to_numeric_series(df, "物理越限硬异常分钟数", 0).clip(lower=0)
    out["风机联合重复标记分钟数"] = to_numeric_series(df, "风机联合重复标记分钟数", 0).clip(lower=0)

    for col in ["风机编号列表", "重复详情"]:
        out[col] = df[col].astype(str) if col in df.columns else ""

    out = out.dropna(subset=["开始时间", "结束时间"]).copy()
    out = out[out["结束时间"] >= out["开始时间"]].reset_index(drop=True)
    return out


# ============================================================
# 分段分类与决策逻辑
# ============================================================
def classify_segment_group(row: pd.Series) -> str:
    repeat_power = float(row.get("重复正功率之和MW", row.get("重复风机功率之和MW", 0)) or 0)
    freeze_count = float(row.get("冻结风机数", 0) or 0)
    if repeat_power > 0:
        return CLASS_REPEAT_POSITIVE
    if freeze_count <= 0:
        return CLASS_REPEAT_ZERO_NO_FAN
    return CLASS_REPEAT_ZERO_WITH_FAN


def is_bad_segment(
    ratio: float,
    abn_minutes: float,
    severity_sum: float,
    avg_severity: float,
    threshold: float,
    min_abn_minutes: int,
    min_severity_sum: float,
    min_avg_severity: float,
) -> bool:
    if pd.isna(ratio) or ratio < threshold:
        return False
    if abn_minutes < min_abn_minutes:
        return False
    if severity_sum < min_severity_sum:
        return False
    if min_avg_severity > 0 and (pd.isna(avg_severity) or avg_severity < min_avg_severity):
        return False
    return True


def build_bad_flags(row: pd.Series, threshold: float, min_abn_minutes: int, min_severity_sum: float, min_avg_severity: float) -> tuple[bool, bool]:
    raw_bad = is_bad_segment(
        ratio=float(row["原异常占比"]),
        abn_minutes=float(row["原异常分钟数"]),
        severity_sum=float(row["原异常程度总和"]),
        avg_severity=float(row["原平均异常程度"]) if pd.notna(row["原平均异常程度"]) else 0.0,
        threshold=threshold,
        min_abn_minutes=min_abn_minutes,
        min_severity_sum=min_severity_sum,
        min_avg_severity=min_avg_severity,
    )
    corr_bad = is_bad_segment(
        ratio=float(row["修正后异常占比"]),
        abn_minutes=float(row["修正后异常分钟数"]),
        severity_sum=float(row["修正后异常程度总和"]),
        avg_severity=float(row["修正后平均异常程度"]) if pd.notna(row["修正后平均异常程度"]) else 0.0,
        threshold=threshold,
        min_abn_minutes=min_abn_minutes,
        min_severity_sum=min_severity_sum,
        min_avg_severity=min_avg_severity,
    )
    return raw_bad, corr_bad


def change_flags(row: pd.Series, ratio_eps: float, severity_eps: float) -> dict[str, bool]:
    """判断修正后相对修正前是改善、不变还是恶化。"""
    minute_diff = float(row["修正后异常分钟数"] - row["原异常分钟数"])
    ratio_diff = float(row["修正后异常占比"] - row["原异常占比"])
    severity_diff = float(row["修正后异常程度总和"] - row["原异常程度总和"])

    minute_reduced = minute_diff < -ratio_eps
    ratio_reduced = ratio_diff < -ratio_eps
    severity_reduced = severity_diff < -severity_eps
    minute_worsened = minute_diff > ratio_eps
    ratio_worsened = ratio_diff > ratio_eps
    severity_worsened = severity_diff > severity_eps

    improved = minute_reduced or ratio_reduced or severity_reduced
    worsened = minute_worsened or ratio_worsened or severity_worsened
    unchanged = (not improved) and (not worsened)
    return {
        "minute_reduced": minute_reduced,
        "ratio_reduced": ratio_reduced,
        "severity_reduced": severity_reduced,
        "minute_worsened": minute_worsened,
        "ratio_worsened": ratio_worsened,
        "severity_worsened": severity_worsened,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
    }


def classify_final_decision_case_and_detail(
    row: pd.Series,
    threshold: float,
    min_abn_minutes: int,
    min_severity_sum: float,
    min_avg_severity: float,
    ratio_eps: float,
    severity_eps: float,
) -> tuple[str, str, str]:
    group = str(row["分段类别"])
    raw_bad, corr_bad = build_bad_flags(row, threshold, min_abn_minutes, min_severity_sum, min_avg_severity)
    flags = change_flags(row, ratio_eps, severity_eps)

    # 重复正功率=0 的两类分段没有“修正后保留/保留不修正”的业务含义，
    # 只按原始口径判断“正常 / 标记为异常分段”。
    if group == CLASS_REPEAT_ZERO_WITH_FAN:
        if raw_bad:
            return DECISION_ABNORMAL, CASE_NOT_APPLICABLE, DETAIL_ZERO_WITH_FAN_ABNORMAL
        return DECISION_NORMAL, CASE_NOT_APPLICABLE, DETAIL_ZERO_WITH_FAN_NORMAL

    if group == CLASS_REPEAT_ZERO_NO_FAN:
        if raw_bad:
            return DECISION_ABNORMAL, CASE_NOT_APPLICABLE, DETAIL_ZERO_NO_FAN_ABNORMAL
        return DECISION_NORMAL, CASE_NOT_APPLICABLE, DETAIL_ZERO_NO_FAN_NORMAL

    # 重复正功率>0 分段：按修正前/后状态 + 改善/不变/恶化拆成 8 类。
    if (not raw_bad) and corr_bad:
        return DECISION_KEEP_RAW, CASE_1, DETAIL_POS_KEEP_RAW
    if raw_bad and (not corr_bad):
        return DECISION_KEEP_CORRECTED, CASE_2, DETAIL_POS_KEEP_CORRECTED
    if (not raw_bad) and (not corr_bad):
        if flags["improved"]:
            return DECISION_KEEP_CORRECTED, CASE_3, DETAIL_POS_KEEP_CORRECTED
        if flags["worsened"]:
            return DECISION_KEEP_RAW, CASE_5, DETAIL_POS_KEEP_RAW
        return DECISION_KEEP_RAW, CASE_4, DETAIL_POS_KEEP_RAW
    if raw_bad and corr_bad:
        if flags["improved"]:
            return DECISION_ABNORMAL, CASE_6, DETAIL_POS_ABNORMAL
        if flags["worsened"]:
            return DECISION_ABNORMAL, CASE_8, DETAIL_POS_ABNORMAL
        return DECISION_ABNORMAL, CASE_7, DETAIL_POS_ABNORMAL

    # 理论上不会走到这里，兜底按异常处理。
    return DECISION_ABNORMAL, CASE_8, DETAIL_POS_ABNORMAL

def build_scored_df(
    base_df: pd.DataFrame,
    threshold: float,
    min_abn_minutes: int,
    min_severity_sum: float,
    min_avg_severity: float,
    ratio_eps: float,
    severity_eps: float,
) -> pd.DataFrame:
    out = base_df.copy()
    out["阈值"] = float(threshold)
    out["最小异常分钟数门槛"] = int(min_abn_minutes)
    out["最小异常程度总和门槛"] = float(min_severity_sum)
    out["最小平均异常程度门槛"] = float(min_avg_severity)
    out["分段类别"] = out.apply(classify_segment_group, axis=1)

    raw_flags = []
    corr_flags = []
    decisions = []
    cases = []
    detail_types = []
    minute_dirs = []
    ratio_dirs = []
    severity_dirs = []

    for _, row in out.iterrows():
        raw_bad, corr_bad = build_bad_flags(row, threshold, min_abn_minutes, min_severity_sum, min_avg_severity)
        decision, case_name, detail_type = classify_final_decision_case_and_detail(
            row=row,
            threshold=threshold,
            min_abn_minutes=min_abn_minutes,
            min_severity_sum=min_severity_sum,
            min_avg_severity=min_avg_severity,
            ratio_eps=ratio_eps,
            severity_eps=severity_eps,
        )
        raw_flags.append(raw_bad)
        corr_flags.append(corr_bad)
        decisions.append(decision)
        cases.append(case_name)
        detail_types.append(detail_type)
        minute_dirs.append(direction_by_diff(float(row["修正后异常分钟数"] - row["原异常分钟数"]), ratio_eps))
        ratio_dirs.append(direction_by_diff(float(row["修正后异常占比"] - row["原异常占比"]), ratio_eps))
        severity_dirs.append(direction_by_diff(float(row["修正后异常程度总和"] - row["原异常程度总和"]), severity_eps))

    out["原是否异常分段"] = raw_flags
    out["修正后是否异常分段"] = corr_flags
    out["异常分钟变化方向"] = minute_dirs
    out["异常占比变化方向"] = ratio_dirs
    out["异常程度变化方向"] = severity_dirs
    out["最终处理方式"] = decisions
    out["最终处理细分类型"] = detail_types
    out["修正作用情况"] = cases

    # 变化量保留有符号口径：修正后 - 原。负数表示改善，正数表示变差。
    out["异常分钟数净变化量（修正后-原）"] = out["修正后异常分钟数"] - out["原异常分钟数"]
    out["异常占比净变化量（修正后-原）"] = out["修正后异常占比"] - out["原异常占比"]
    out["异常程度净变化量（修正后-原）"] = out["修正后异常程度总和"] - out["原异常程度总和"]

    # 改善量只统计“最终采用修正后结果”的分段；
    # 修正后变差、无改善、仍标记异常的分段不计入改善收益。
    effective_repair_mask = (out["分段类别"] == CLASS_REPEAT_POSITIVE) & (out["最终处理方式"] == DECISION_KEEP_CORRECTED)
    out["是否计入有效修正改善统计"] = effective_repair_mask
    out["异常分钟数改善量"] = np.where(
        effective_repair_mask,
        np.maximum(out["原异常分钟数"] - out["修正后异常分钟数"], 0.0),
        0.0,
    )
    out["异常占比改善量"] = np.where(
        effective_repair_mask,
        np.maximum(out["原异常占比"] - out["修正后异常占比"], 0.0),
        0.0,
    )
    out["异常程度改善量"] = np.where(
        effective_repair_mask,
        np.maximum(out["原异常程度总和"] - out["修正后异常程度总和"], 0.0),
        0.0,
    )
    return out


# ============================================================
# 汇总表
# ============================================================
def summarize_raw_profile(scored_df: pd.DataFrame, threshold: float, station_name: str) -> pd.DataFrame:
    rows = []
    for class_name in CLASS_ORDER:
        sub = scored_df[scored_df["分段类别"] == class_name].copy()
        segment_count = int(len(sub))
        abnormal_segment_count = int(sub["原是否异常分段"].sum()) if segment_count else 0
        cover_min = float(sub["持续时长min"].sum())
        stat_min = float(sub["纳入统计分钟数"].sum())
        raw_abn_min = float(sub["原异常分钟数"].sum())
        raw_severity_sum = float(sub["原异常程度总和"].sum())

        rows.append({
            "场站": station_name,
            "阈值": threshold,
            "分段类别": class_name,
            "分段数": segment_count,
            "异常分段数": abnormal_segment_count,
            "异常分段占比": safe_ratio(abnormal_segment_count, segment_count),
            "覆盖分钟数": cover_min,
            "纳入统计分钟数": stat_min,
            "原异常分钟数": raw_abn_min,
            "原异常分钟占纳入统计分钟比例": safe_ratio(raw_abn_min, stat_min),
            "原异常程度总和": raw_severity_sum,
            "原平均异常程度_按异常分钟": safe_ratio(raw_severity_sum, raw_abn_min),
            "原最大异常程度": float(sub["原最大异常程度"].max()) if segment_count else 0.0,
        })
    return pd.DataFrame(rows)


def summarize_repair_effect(scored_df: pd.DataFrame, threshold: float, station_name: str) -> pd.DataFrame:
    sub_all = scored_df[scored_df["分段类别"] == CLASS_REPEAT_POSITIVE].copy()
    total_segments = int(len(sub_all))
    total_cover_min = float(sub_all["持续时长min"].sum())
    total_stat_min = float(sub_all["纳入统计分钟数"].sum())
    total_raw_abn_min = float(sub_all["原异常分钟数"].sum())
    total_corr_abn_min = float(sub_all["修正后异常分钟数"].sum())
    total_raw_severity = float(sub_all["原异常程度总和"].sum())
    total_corr_severity = float(sub_all["修正后异常程度总和"].sum())

    rows = []
    for case_name in CASE_ORDER:
        sub = sub_all[sub_all["修正作用情况"] == case_name].copy()
        segment_count = int(len(sub))
        cover_min = float(sub["持续时长min"].sum())
        stat_min = float(sub["纳入统计分钟数"].sum())
        raw_abn_min = float(sub["原异常分钟数"].sum())
        corr_abn_min = float(sub["修正后异常分钟数"].sum())
        raw_severity = float(sub["原异常程度总和"].sum())
        corr_severity = float(sub["修正后异常程度总和"].sum())
        effective_minute_improvement = float(sub["异常分钟数改善量"].sum()) if "异常分钟数改善量" in sub.columns else 0.0
        effective_severity_improvement = float(sub["异常程度改善量"].sum()) if "异常程度改善量" in sub.columns else 0.0

        rows.append({
            "场站": station_name,
            "阈值": threshold,
            "修正作用情况": case_name,
            "对应最终处理方式": CASE_TO_DECISION.get(case_name),
            "分段数": segment_count,
            "分段占比": safe_ratio(segment_count, total_segments),
            "覆盖分钟数": cover_min,
            "覆盖分钟占比": safe_ratio(cover_min, total_cover_min),
            "纳入统计分钟数": stat_min,
            "原异常分钟数": raw_abn_min,
            "修正后异常分钟数": corr_abn_min,
            "异常分钟数净变化量（修正后-原）": corr_abn_min - raw_abn_min,
            "有效异常分钟数改善量": effective_minute_improvement,
            "原异常程度总和": raw_severity,
            "修正后异常程度总和": corr_severity,
            "异常程度净变化量（修正后-原）": corr_severity - raw_severity,
            "有效异常程度改善量": effective_severity_improvement,
            "有效异常程度改善率": safe_ratio(effective_severity_improvement, raw_severity),
            "重复正功率>0总分段数": total_segments,
            "重复正功率>0总覆盖分钟数": total_cover_min,
            "重复正功率>0总纳入统计分钟数": total_stat_min,
            "重复正功率>0总原异常分钟数": total_raw_abn_min,
            "重复正功率>0总修正后异常分钟数": total_corr_abn_min,
            "重复正功率>0总原异常程度": total_raw_severity,
            "重复正功率>0总修正后异常程度": total_corr_severity,
        })
    return pd.DataFrame(rows)


def summarize_final_decision(scored_df: pd.DataFrame, threshold: float, station_name: str) -> pd.DataFrame:
    """按“最终处理细分类型”汇总，同时保留粗粒度最终处理方式。"""
    total_segments = int(len(scored_df))
    total_cover_min = float(scored_df["持续时长min"].sum())
    total_stat_min = float(scored_df["纳入统计分钟数"].sum())
    total_raw_abn_min = float(scored_df["原异常分钟数"].sum())
    total_corr_abn_min = float(scored_df["修正后异常分钟数"].sum())
    total_raw_severity = float(scored_df["原异常程度总和"].sum())
    total_corr_severity = float(scored_df["修正后异常程度总和"].sum())

    detail_to_group = {
        DETAIL_POS_KEEP_RAW: (CLASS_REPEAT_POSITIVE, DECISION_KEEP_RAW),
        DETAIL_POS_KEEP_CORRECTED: (CLASS_REPEAT_POSITIVE, DECISION_KEEP_CORRECTED),
        DETAIL_POS_ABNORMAL: (CLASS_REPEAT_POSITIVE, DECISION_ABNORMAL),
        DETAIL_ZERO_WITH_FAN_NORMAL: (CLASS_REPEAT_ZERO_WITH_FAN, DECISION_NORMAL),
        DETAIL_ZERO_WITH_FAN_ABNORMAL: (CLASS_REPEAT_ZERO_WITH_FAN, DECISION_ABNORMAL),
        DETAIL_ZERO_NO_FAN_NORMAL: (CLASS_REPEAT_ZERO_NO_FAN, DECISION_NORMAL),
        DETAIL_ZERO_NO_FAN_ABNORMAL: (CLASS_REPEAT_ZERO_NO_FAN, DECISION_ABNORMAL),
    }

    rows = []
    for detail_type in DETAIL_DECISION_ORDER:
        class_name, decision = detail_to_group[detail_type]
        sub = scored_df[scored_df["最终处理细分类型"] == detail_type].copy()
        segment_count = int(len(sub))
        cover_min = float(sub["持续时长min"].sum())
        stat_min = float(sub["纳入统计分钟数"].sum())
        raw_abn_min = float(sub["原异常分钟数"].sum())
        corr_abn_min = float(sub["修正后异常分钟数"].sum())
        raw_severity = float(sub["原异常程度总和"].sum())
        corr_severity = float(sub["修正后异常程度总和"].sum())
        minute_improve = float(sub["异常分钟数改善量"].sum()) if "异常分钟数改善量" in sub.columns else 0.0
        severity_improve = float(sub["异常程度改善量"].sum()) if "异常程度改善量" in sub.columns else 0.0

        rows.append({
            "场站": station_name,
            "阈值": threshold,
            "分段类别": class_name,
            "最终处理方式": decision,
            "最终处理细分类型": detail_type,
            "分段数": segment_count,
            "分段占比": safe_ratio(segment_count, total_segments),
            "覆盖分钟数": cover_min,
            "覆盖分钟占比": safe_ratio(cover_min, total_cover_min),
            "纳入统计分钟数": stat_min,
            "纳入统计分钟占比": safe_ratio(stat_min, total_stat_min),
            "原异常分钟数": raw_abn_min,
            "修正后异常分钟数": corr_abn_min,
            "异常分钟数净变化量（修正后-原）": corr_abn_min - raw_abn_min,
            "有效异常分钟数改善量": minute_improve,
            "原异常程度总和": raw_severity,
            "修正后异常程度总和": corr_severity,
            "异常程度净变化量（修正后-原）": corr_severity - raw_severity,
            "有效异常程度改善量": severity_improve,
            "有效异常程度改善率": safe_ratio(severity_improve, raw_severity),
            "总分段数": total_segments,
            "总覆盖分钟数": total_cover_min,
            "总纳入统计分钟数": total_stat_min,
            "总原异常分钟数": total_raw_abn_min,
            "总修正后异常分钟数": total_corr_abn_min,
            "总原异常程度": total_raw_severity,
            "总修正后异常程度": total_corr_severity,
        })
    return pd.DataFrame(rows)

def build_recommendation(final_df: pd.DataFrame, repair_df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """给出一个简单、透明的阈值排序，供人工参考，不替代人工决策。"""
    if len(final_df) == 0:
        return pd.DataFrame()

    rows = []
    for threshold, final_sub in final_df.groupby("阈值"):
        total_seg = float(final_sub["总分段数"].max()) if "总分段数" in final_sub.columns else np.nan
        abnormal_ratio = float(final_sub.loc[final_sub["最终处理方式"] == DECISION_ABNORMAL, "分段数"].sum() / total_seg) if total_seg and total_seg > 0 else np.nan
        corrected_ratio = float(final_sub.loc[final_sub["最终处理方式"] == DECISION_KEEP_CORRECTED, "分段数"].sum() / total_seg) if total_seg and total_seg > 0 else np.nan
        raw_ratio = float(final_sub.loc[final_sub["最终处理方式"] == DECISION_KEEP_RAW, "分段数"].sum() / total_seg) if total_seg and total_seg > 0 else np.nan
        normal_ratio = float(final_sub.loc[final_sub["最终处理方式"] == DECISION_NORMAL, "分段数"].sum() / total_seg) if total_seg and total_seg > 0 else np.nan

        final_improvement = float(final_sub["有效异常程度改善量"].sum()) if "有效异常程度改善量" in final_sub.columns else 0.0
        final_minute_improvement = float(final_sub["有效异常分钟数改善量"].sum()) if "有效异常分钟数改善量" in final_sub.columns else 0.0

        repair_sub = repair_df[repair_df["阈值"] == threshold].copy() if len(repair_df) else pd.DataFrame()
        repair_improvement = float(repair_sub["有效异常程度改善量"].sum()) if len(repair_sub) and "有效异常程度改善量" in repair_sub.columns else 0.0
        repair_minute_improvement = float(repair_sub["有效异常分钟数改善量"].sum()) if len(repair_sub) and "有效异常分钟数改善量" in repair_sub.columns else 0.0

        score = (
            2.0 * (corrected_ratio if np.isfinite(corrected_ratio) else 0.0)
            - 1.0 * (abnormal_ratio if np.isfinite(abnormal_ratio) else 0.0)
            + 0.02 * np.log1p(max(repair_improvement, 0.0))
            + 0.01 * np.log1p(max(repair_minute_improvement, 0.0))
        )

        rows.append({
            "场站": station_name,
            "阈值": threshold,
            "推荐参考分数": score,
            "异常分段占比": abnormal_ratio,
            "修正后保留分段占比": corrected_ratio,
            "保留不修正分段占比": raw_ratio,
            "正常分段占比": normal_ratio,
            "总分段数": total_seg,
            "最终有效异常程度改善量": final_improvement,
            "最终有效异常分钟数改善量": final_minute_improvement,
            "重复正功率>0分段有效异常程度改善量": repair_improvement,
            "重复正功率>0分段有效异常分钟数改善量": repair_minute_improvement,
            "推荐说明": "分数仅用于排序参考：只把最终处理方式为“修正后保留”的重复正功率>0分段计入有效改善收益；修正后变差/不变/仍异常的分段不计入改善收益；最终阈值仍需结合业务判断。",
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values(["推荐参考分数", "阈值"], ascending=[False, True]).reset_index(drop=True)
        out["推荐排序"] = np.arange(1, len(out) + 1)
        cols = ["推荐排序"] + [c for c in out.columns if c != "推荐排序"]
        out = out[cols]
    return out


# ============================================================
# 主流程
# ============================================================
def resolve_segment_file(step3_dir: str, segment_file: Optional[str]) -> Path:
    if segment_file:
        return Path(segment_file)
    return Path(step3_dir) / DEFAULT_SEGMENT_FILE_NAME


def process_all_thresholds(
    base_df: pd.DataFrame,
    thresholds: list[float],
    station_name: str,
    min_abn_minutes: int,
    min_severity_sum: float,
    min_avg_severity: float,
    ratio_eps: float,
    severity_eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored_parts = []
    raw_parts = []
    repair_parts = []
    final_parts = []

    for threshold in thresholds:
        scored_df = build_scored_df(
            base_df=base_df,
            threshold=threshold,
            min_abn_minutes=min_abn_minutes,
            min_severity_sum=min_severity_sum,
            min_avg_severity=min_avg_severity,
            ratio_eps=ratio_eps,
            severity_eps=severity_eps,
        )
        scored_parts.append(scored_df)
        raw_parts.append(summarize_raw_profile(scored_df, threshold, station_name))
        repair_parts.append(summarize_repair_effect(scored_df, threshold, station_name))
        final_parts.append(summarize_final_decision(scored_df, threshold, station_name))

    scored_all = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    raw_all = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
    repair_all = pd.concat(repair_parts, ignore_index=True) if repair_parts else pd.DataFrame()
    final_all = pd.concat(final_parts, ignore_index=True) if final_parts else pd.DataFrame()
    recommendation = build_recommendation(final_all, repair_all, station_name)
    return scored_all, raw_all, repair_all, final_all, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="STEP4 江苏全站版：对 STEP3 重复冻结分段进行阈值扫描和修正效果分析")
    parser.add_argument("--step3-dir", default=DEFAULT_STEP3_DIR, help="STEP3 输出目录，默认包含 freeze_count_segments.csv")
    parser.add_argument("--segment-file", default=None, help="分段汇总文件路径；不填则使用 step3-dir/freeze_count_segments.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="STEP4 输出目录")
    parser.add_argument("--station-name", default="", help="场站名称；不填则从 step3-dir 的上级目录推断")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="逗号分隔的异常占比阈值列表")
    parser.add_argument("--min-abn-minutes", type=int, default=0, help="判定异常分段所需的最小异常分钟数，默认0")
    parser.add_argument("--min-severity-sum", type=float, default=0.0, help="判定异常分段所需的最小异常程度总和，默认0")
    parser.add_argument("--min-avg-severity", type=float, default=0.0, help="判定异常分段所需的最小平均异常程度，默认0")
    parser.add_argument("--ratio-eps", type=float, default=DEFAULT_RATIO_EPS, help="判断异常占比/分钟数变化方向的容差")
    parser.add_argument("--severity-eps", type=float, default=DEFAULT_SEVERITY_EPS, help="判断异常程度变化方向的容差")
    args = parser.parse_args()

    step3_dir = Path(args.step3_dir)
    segment_path = resolve_segment_file(args.step3_dir, args.segment_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    station_name = args.station_name.strip()
    if not station_name:
        # 典型路径：场站数据\LGXRFD\step3-重复冻结分段，所以取父目录名。
        station_name = step3_dir.parent.name if step3_dir.parent.name else "未知场站"

    thresholds = parse_thresholds(args.thresholds)
    if not thresholds:
        raise ValueError("阈值列表为空，请检查 --thresholds")

    raw_df = read_csv_auto(segment_path)
    base_df = normalize_segment_df(raw_df, station_name=station_name)

    print("=" * 80)
    print("STEP4 江苏全站版：重复冻结分段阈值分析")
    print("=" * 80)
    print(f"场站: {station_name}")
    print(f"输入分段文件: {segment_path}")
    print(f"输出目录: {output_dir}")
    print(f"分段数: {len(base_df):,}")
    print(f"阈值列表: {thresholds}")
    print(f"最小异常分钟数门槛: {args.min_abn_minutes}")
    print(f"最小异常程度总和门槛: {args.min_severity_sum}")
    print(f"最小平均异常程度门槛: {args.min_avg_severity}")

    scored_df, raw_profile_df, repair_effect_df, final_decision_df, recommendation_df = process_all_thresholds(
        base_df=base_df,
        thresholds=thresholds,
        station_name=station_name,
        min_abn_minutes=args.min_abn_minutes,
        min_severity_sum=args.min_severity_sum,
        min_avg_severity=args.min_avg_severity,
        ratio_eps=args.ratio_eps,
        severity_eps=args.severity_eps,
    )

    output_files = {
        "逐分段阈值打分明细": output_dir / "threshold_scan_scored.csv",
        "修正前原始特征汇总": output_dir / "threshold_scan_raw_profile.csv",
        "重复正功率修正作用汇总": output_dir / "threshold_scan_repair_effect.csv",
        "最终处理方式汇总": output_dir / "threshold_scan_final_decision.csv",
        "阈值推荐参考": output_dir / "threshold_recommendation.csv",
    }

    scored_df.to_csv(output_files["逐分段阈值打分明细"], index=False, encoding="utf-8-sig")
    raw_profile_df.to_csv(output_files["修正前原始特征汇总"], index=False, encoding="utf-8-sig")
    repair_effect_df.to_csv(output_files["重复正功率修正作用汇总"], index=False, encoding="utf-8-sig")
    final_decision_df.to_csv(output_files["最终处理方式汇总"], index=False, encoding="utf-8-sig")
    recommendation_df.to_csv(output_files["阈值推荐参考"], index=False, encoding="utf-8-sig")

    print("\n输出文件：")
    for label, path in output_files.items():
        print(f"  - {label}: {path}")

    if len(recommendation_df) > 0:
        print("\n阈值推荐参考 Top 5：")
        show_cols = [
            "推荐排序", "阈值", "推荐参考分数", "异常分段占比", "修正后保留分段占比",
            "重复正功率>0分段有效异常程度改善量", "重复正功率>0分段有效异常分钟数改善量",
        ]
        show_cols = [c for c in show_cols if c in recommendation_df.columns]
        print(recommendation_df[show_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
