#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP3（江苏全站版）：构建风机联合重复冻结分段，并关联 STEP2 全站异常检测结果。

适用场景：
- 江苏场站只有全站功率，不再按 BING/DING/WU 集电线路拆分。
- 江苏场站风机只有一个厂商型号，不再输出/识别厂商字段。
- STEP2 输出来自 step2_detect_station_anomalies_from_step1_quantile.py。

输入：
1) --station-dir
   场站目录，默认读取：
   <station-dir>/风机联合连续相同检测/联合重复值检测结果.xlsx

2) --step2-dir
   STEP2 输出目录，默认读取：
   <step2-dir>/station_anomaly_detail.csv

输出：
1) freeze_count_segments.csv
   全站重复冻结分段汇总表

2) freeze_count_segments_detail.csv
   分段-风机明细表
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_REPEAT_REL = r"风机联合连续相同检测_仅正功率\联合重复值检测结果.xlsx"
DEFAULT_STEP2_DETAIL_NAME = "station_anomaly_detail.csv"
DEFAULT_SEGMENT_OUTPUT = "freeze_count_segments.csv"
DEFAULT_DETAIL_OUTPUT = "freeze_count_segments_detail.csv"


# ============================================================
# 通用读取与解析
# ============================================================
def read_csv_auto(path: str | Path) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"文件不存在：{path_obj}")

    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path_obj, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取失败：{path_obj}\n最后一次报错：{last_err}")


def parse_bool_like(val) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float, np.integer, np.floating)):
        return bool(val)

    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "y", "是", "异常", "true."}:
        return True
    if s in {"false", "0", "no", "n", "否", "正常", "", "nan", "none"}:
        return False
    return False


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_nonnegative_power_kw(x) -> float:
    val = safe_float(x, default=0.0)
    if not np.isfinite(val):
        return 0.0
    return max(val, 0.0)


# ============================================================
# 解析重复值组合
# 预期格式：
# (status, active_power, reactive_power, windspeed, winddirection)
# ============================================================
def parse_combo(combo) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        vals = str(combo).strip().strip("()").split(",")
        vals = [v.strip() for v in vals]

        def to_float(x):
            if x is None or str(x).strip() == "":
                return None
            return float(x)

        status = to_float(vals[0]) if len(vals) > 0 else None
        active_power = to_float(vals[1]) if len(vals) > 1 else None
        windspeed = to_float(vals[3]) if len(vals) > 3 else None
        return status, active_power, windspeed
    except Exception:
        return None, None, None


def format_detail_row(row: pd.Series) -> str:
    fan = row.get("风机编号", "")
    status = row.get("状态码", "")
    p = row.get("冻结有功kW", "")
    ws = row.get("冻结风速ms", "")
    return f"{fan}[status={status},P={p},WS={ws}]"


# ============================================================
# 文件路径解析
# ============================================================
def resolve_repeat_file(station_dir: Path, repeat_file: Optional[str], repeat_rel: str) -> Path:
    if repeat_file:
        p = Path(repeat_file)
        if not p.is_absolute():
            p = station_dir / p
        return p
    return station_dir / repeat_rel


def resolve_step2_detail(step2_dir: Path, anomaly_detail_file: Optional[str]) -> Path:
    if anomaly_detail_file:
        p = Path(anomaly_detail_file)
        if not p.is_absolute():
            p = step2_dir / p
        return p
    return step2_dir / DEFAULT_STEP2_DETAIL_NAME


# ============================================================
# 加载联合重复检测结果
# ============================================================
def load_repeat_events(repeat_file: Path) -> pd.DataFrame:
    if not repeat_file.exists():
        raise FileNotFoundError(f"未找到风机联合重复检测结果：{repeat_file}")

    df = pd.read_excel(repeat_file, engine="openpyxl")
    required = {"开始时间", "结束时间", "风机编号", "重复值组合"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"联合重复值检测结果缺少必要列：{missing}，文件：{repeat_file}")

    out = df.copy()
    out["开始时间"] = pd.to_datetime(out["开始时间"], errors="coerce")
    out["结束时间"] = pd.to_datetime(out["结束时间"], errors="coerce")
    out = out.dropna(subset=["开始时间", "结束时间"]).copy()
    out = out[out["结束时间"] >= out["开始时间"]].copy()
    out = out.reset_index(drop=True)

    parsed = out["重复值组合"].apply(parse_combo)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=["状态码", "冻结有功kW", "冻结风速ms"], index=out.index)
    out = pd.concat([out, parsed_df], axis=1)

    out["风机编号"] = pd.to_numeric(out["风机编号"], errors="coerce").astype("Int64")
    out["冻结有功kW"] = pd.to_numeric(out["冻结有功kW"], errors="coerce")
    out["非负冻结有功kW"] = out["冻结有功kW"].apply(safe_nonnegative_power_kw)
    out["冻结风速ms"] = pd.to_numeric(out["冻结风速ms"], errors="coerce")
    out["状态码"] = pd.to_numeric(out["状态码"], errors="coerce")

    if "持续长度" in out.columns:
        out["原始持续长度min"] = pd.to_numeric(out["持续长度"], errors="coerce")
    else:
        out["原始持续长度min"] = ((out["结束时间"] - out["开始时间"]).dt.total_seconds() / 60 + 1).round(0)

    return out


# ============================================================
# 构建全站冻结分段
# ============================================================
def build_segments_for_station(
    df_repeat: pd.DataFrame,
    full_start: Optional[pd.Timestamp] = None,
    full_end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 即使没有任何重复事件，也要基于 STEP2 完整时间轴生成一个空档分段，
    # 避免原始数据开始时间到第一次重复事件之间出现“未匹配STEP3分段”。
    if df_repeat is None or len(df_repeat) == 0:
        if full_start is None or full_end is None or pd.isna(full_start) or pd.isna(full_end):
            return pd.DataFrame(), pd.DataFrame()

        full_start = pd.to_datetime(full_start).floor("min")
        full_end = pd.to_datetime(full_end).floor("min")
        if full_end < full_start:
            return pd.DataFrame(), pd.DataFrame()

        duration_min = int((full_end - full_start).total_seconds() / 60) + 1
        seg_df = pd.DataFrame([{
            "分段ID": 1,
            "开始时间": full_start,
            "结束时间": full_end,
            "持续时长min": duration_min,
            "冻结风机数": 0,
            "重复风机功率之和MW": 0.0,
            "风机编号列表": "",
            "重复详情": "",
        }])
        detail_df = pd.DataFrame()
        return seg_df, detail_df

    events: Dict[pd.Timestamp, int] = defaultdict(int)
    start_map: Dict[pd.Timestamp, List[int]] = defaultdict(list)
    end_map: Dict[pd.Timestamp, List[int]] = defaultdict(list)

    for idx, row in df_repeat.iterrows():
        start = pd.to_datetime(row["开始时间"])
        end = pd.to_datetime(row["结束时间"])
        events[start] += 1
        events[end + pd.Timedelta(minutes=1)] -= 1
        start_map[start].append(idx)
        end_map[end + pd.Timedelta(minutes=1)].append(idx)

    # 强制加入 STEP2 明细的完整时间边界，使分段覆盖完整检测时间轴。
    # full_start ~ 第一次重复事件前，会成为“冻结风机数=0”的空档分段；
    # 最后一次重复事件后 ~ full_end，也会成为空档分段。
    if full_start is not None and not pd.isna(full_start):
        full_start = pd.to_datetime(full_start).floor("min")
        events[full_start] += 0

    if full_end is not None and not pd.isna(full_end):
        full_end = pd.to_datetime(full_end).floor("min")
        events[full_end + pd.Timedelta(minutes=1)] += 0

    sorted_times = sorted(events.keys())
    if len(sorted_times) < 2:
        return pd.DataFrame(), pd.DataFrame()

    active_set = set()
    segment_rows = []
    detail_rows = []
    next_segment_id = 1

    for i, t in enumerate(sorted_times):
        for idx in start_map.get(t, []):
            active_set.add(idx)
        for idx in end_map.get(t, []):
            active_set.discard(idx)

        if i >= len(sorted_times) - 1:
            continue

        next_t = sorted_times[i + 1]
        if not (t < next_t):
            continue

        seg_start = t
        seg_end = next_t - pd.Timedelta(minutes=1)
        duration_min = int((seg_end - seg_start).total_seconds() / 60) + 1
        active_indices = sorted(
            active_set,
            key=lambda x: int(df_repeat.loc[x, "风机编号"]) if pd.notna(df_repeat.loc[x, "风机编号"]) else 10**9,
        )

        if active_indices:
            active_df = df_repeat.loc[active_indices].copy()
            fan_list = [str(int(x)) for x in active_df["风机编号"].dropna().tolist()]
            detail_list = [format_detail_row(active_df.loc[idx]) for idx in active_df.index]
            power_sum_mw = round(float(active_df["非负冻结有功kW"].sum()) / 1000.0, 6)
        else:
            # 保留相邻重复冻结事件之间的“空档段”。
            # 这类分段用于 STEP4 的“重复功率=0分段，且冻结风机数=0”类别分析。
            fan_list = []
            detail_list = []
            power_sum_mw = 0.0

        segment_rows.append({
            "分段ID": next_segment_id,
            "开始时间": seg_start,
            "结束时间": seg_end,
            "持续时长min": duration_min,
            "冻结风机数": int(len(active_indices)),
            "重复风机功率之和MW": power_sum_mw,
            "风机编号列表": "、".join(fan_list),
            "重复详情": " 、 ".join(detail_list),
        })

        for idx in active_indices:
            r = df_repeat.loc[idx]
            detail_rows.append({
                "分段ID": next_segment_id,
                "开始时间": seg_start,
                "结束时间": seg_end,
                "持续时长min": duration_min,
                "冻结风机数": int(len(active_indices)),
                "重复风机功率之和MW": power_sum_mw,
                "风机编号": int(r["风机编号"]) if pd.notna(r["风机编号"]) else np.nan,
                "状态码": r.get("状态码", np.nan),
                "冻结有功kW": r.get("冻结有功kW", np.nan),
                "非负冻结有功kW": r.get("非负冻结有功kW", np.nan),
                "冻结风速ms": r.get("冻结风速ms", np.nan),
                "重复值组合": r.get("重复值组合", ""),
                "原始开始时间": r.get("开始时间", pd.NaT),
                "原始结束时间": r.get("结束时间", pd.NaT),
                "原始持续长度min": r.get("原始持续长度min", np.nan),
            })

        next_segment_id += 1

    seg_df = pd.DataFrame(segment_rows)
    detail_df = pd.DataFrame(detail_rows)
    if len(seg_df) > 0:
        seg_df = seg_df.sort_values(["开始时间", "结束时间", "分段ID"]).reset_index(drop=True)
    if len(detail_df) > 0:
        detail_df = detail_df.sort_values(["开始时间", "结束时间", "分段ID", "风机编号"], na_position="last").reset_index(drop=True)
    return seg_df, detail_df


# ============================================================
# 加载 STEP2 异常明细
# ============================================================
def _find_required_column(df: pd.DataFrame, candidates: List[str], logical_name: str, required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"STEP2 异常明细缺少必要列：{logical_name}。候选列：{candidates}")
    return None


def load_step2_minute_table(detail_file: Path) -> pd.DataFrame:
    if not detail_file.exists():
        raise FileNotFoundError(f"未找到 STEP2 异常明细：{detail_file}")

    df = read_csv_auto(detail_file)

    time_col = _find_required_column(df, ["时间", "timestamp", "Timestamp"], "时间")
    raw_abn_col = _find_required_column(df, ["原是否异常", "原始口径是否异常", "原异常"], "原是否异常")
    corr_abn_col = _find_required_column(df, ["是否异常", "修正后是否异常", "修正后口径是否异常"], "是否异常")
    raw_loss_col = _find_required_column(df, ["原损耗MW", "原损耗", "原始损耗MW"], "原损耗", required=False)
    corr_loss_col = _find_required_column(df, ["修正后损耗MW", "修正后损耗", "损耗MW"], "修正后损耗", required=False)
    raw_sev_col = _find_required_column(df, ["原异常程度", "原始异常程度"], "原异常程度", required=False)
    corr_sev_col = _find_required_column(df, ["异常程度", "修正后异常程度"], "异常程度", required=False)
    raw_dev_col = _find_required_column(df, ["原偏离程度", "原始偏离程度"], "原偏离程度", required=False)
    corr_dev_col = _find_required_column(df, ["修正后偏离程度", "偏离程度"], "修正后偏离程度", required=False)
    included_col = _find_required_column(df, ["是否纳入汇总统计", "纳入汇总统计"], "是否纳入汇总统计", required=False)
    station_repeat_col = _find_required_column(df, ["是否站端功率重复硬异常", "站端功率重复硬异常"], "是否站端功率重复硬异常", required=False)
    physical_col = _find_required_column(df, ["是否物理越限硬异常", "物理越限硬异常"], "是否物理越限硬异常", required=False)
    fan_repeat_col = _find_required_column(df, ["是否风机联合重复标记", "是否风机联合重复", "风机联合重复标记"], "是否风机联合重复标记", required=False)

    out = pd.DataFrame()
    out["时间"] = pd.to_datetime(df[time_col], errors="coerce")
    out["原是否异常"] = df[raw_abn_col].apply(parse_bool_like).astype(int)
    out["是否异常"] = df[corr_abn_col].apply(parse_bool_like).astype(int)

    if raw_loss_col:
        out["原损耗MW"] = pd.to_numeric(df[raw_loss_col], errors="coerce")
    else:
        out["原损耗MW"] = np.nan

    if corr_loss_col:
        out["修正后损耗MW"] = pd.to_numeric(df[corr_loss_col], errors="coerce")
    else:
        out["修正后损耗MW"] = np.nan

    if raw_sev_col:
        out["原异常程度"] = pd.to_numeric(df[raw_sev_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        out["原异常程度"] = 0.0

    if corr_sev_col:
        out["异常程度"] = pd.to_numeric(df[corr_sev_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        out["异常程度"] = 0.0

    if raw_dev_col:
        out["原偏离程度"] = pd.to_numeric(df[raw_dev_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        out["原偏离程度"] = 0.0

    if corr_dev_col:
        out["修正后偏离程度"] = pd.to_numeric(df[corr_dev_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        out["修正后偏离程度"] = 0.0

    if included_col:
        out["是否纳入汇总统计"] = df[included_col].apply(parse_bool_like).astype(int)
    else:
        out["是否纳入汇总统计"] = 1

    if station_repeat_col:
        out["是否站端功率重复硬异常"] = df[station_repeat_col].apply(parse_bool_like).astype(int)
    else:
        out["是否站端功率重复硬异常"] = 0

    if physical_col:
        out["是否物理越限硬异常"] = df[physical_col].apply(parse_bool_like).astype(int)
    else:
        out["是否物理越限硬异常"] = 0

    if fan_repeat_col:
        out["是否风机联合重复标记"] = df[fan_repeat_col].apply(parse_bool_like).astype(int)
    else:
        out["是否风机联合重复标记"] = 0

    out = out.dropna(subset=["时间"]).copy()
    out = out.sort_values("时间").drop_duplicates(subset=["时间"], keep="last").reset_index(drop=True)

    out["原负损耗"] = (out["原损耗MW"] < 0).fillna(False).astype(int)
    out["修正后负损耗"] = (out["修正后损耗MW"] < 0).fillna(False).astype(int)

    return out


# ============================================================
# 分段统计
# ============================================================
def _safe_ratio(n: float, d: float) -> float:
    return float(n / d) if d and d > 0 else np.nan


def _sum_between(cum_s: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if len(cum_s) == 0:
        return 0.0
    end_val = float(cum_s.loc[end]) if end in cum_s.index else 0.0
    prev_t = start - pd.Timedelta(minutes=1)
    start_val = float(cum_s.loc[prev_t]) if prev_t in cum_s.index else 0.0
    return end_val - start_val


def _max_between(minute_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, col: str, included_only: bool = False) -> float:
    if col not in minute_df.columns or len(minute_df) == 0:
        return 0.0
    g = minute_df.loc[(minute_df.index >= start) & (minute_df.index <= end)]
    if included_only and "是否纳入汇总统计" in g.columns:
        g = g[g["是否纳入汇总统计"] == 1]
    if len(g) == 0:
        return 0.0
    return float(pd.to_numeric(g[col], errors="coerce").fillna(0.0).max())


def append_segment_anomaly_stats(segment_df: pd.DataFrame, minute_df: pd.DataFrame) -> pd.DataFrame:
    if segment_df is None or len(segment_df) == 0:
        return segment_df.copy() if segment_df is not None else pd.DataFrame()

    stat_cols = [
        "纳入统计分钟数", "原异常分钟数", "修正后异常分钟数", "异常分钟数变化量",
        "原异常占比", "修正后异常占比", "异常占比变化量",
        "原负损耗分钟数", "修正后负损耗分钟数",
        "原异常程度总和", "修正后异常程度总和", "异常程度变化量", "异常程度下降率",
        "原平均异常程度", "修正后平均异常程度", "原最大异常程度", "修正后最大异常程度",
        "原偏离程度总和", "修正后偏离程度总和", "偏离程度变化量", "偏离程度下降率",
        "原平均偏离程度", "修正后平均偏离程度", "原最大偏离程度", "修正后最大偏离程度",
        "站端功率重复硬异常分钟数", "物理越限硬异常分钟数", "风机联合重复标记分钟数",
        "是否异常分钟减少", "是否异常程度下降", "是否偏离程度下降", "是否全部修正为正常",
    ]

    if minute_df is None or len(minute_df) == 0:
        out = segment_df.copy()
        for col in stat_cols:
            out[col] = 0
        return out

    minute_df = minute_df.set_index("时间").sort_index()

    seg_min = pd.to_datetime(segment_df["开始时间"]).min()
    seg_max = pd.to_datetime(segment_df["结束时间"]).max()
    full_index = pd.date_range(start=seg_min, end=seg_max, freq="min")
    minute_df = minute_df.reindex(full_index)

    fill_zero_cols = [
        "原是否异常", "是否异常", "原异常程度", "异常程度", "原偏离程度", "修正后偏离程度",
        "原负损耗", "修正后负损耗",
        "是否纳入汇总统计", "是否站端功率重复硬异常", "是否物理越限硬异常", "是否风机联合重复标记",
    ]
    for col in fill_zero_cols:
        if col not in minute_df.columns:
            minute_df[col] = 0
        minute_df[col] = pd.to_numeric(minute_df[col], errors="coerce").fillna(0.0)

    # 异常程度与偏离程度均只对纳入汇总统计的分钟聚合，避免物理越限极端值拉爆。
    included = (minute_df["是否纳入汇总统计"] == 1).astype(int)
    minute_df["原异常程度_纳入统计"] = minute_df["原异常程度"] * included
    minute_df["异常程度_纳入统计"] = minute_df["异常程度"] * included
    minute_df["原偏离程度_纳入统计"] = minute_df["原偏离程度"] * included
    minute_df["修正后偏离程度_纳入统计"] = minute_df["修正后偏离程度"] * included

    cum_cols = [
        "是否纳入汇总统计", "原是否异常", "是否异常", "原负损耗", "修正后负损耗",
        "原异常程度_纳入统计", "异常程度_纳入统计",
        "原偏离程度_纳入统计", "修正后偏离程度_纳入统计",
        "是否站端功率重复硬异常", "是否物理越限硬异常", "是否风机联合重复标记",
    ]
    cum_df = minute_df[cum_cols].cumsum()

    out = segment_df.copy()
    for c in stat_cols:
        out[c] = 0.0 if ("占比" in c or "程度" in c or "率" in c) else 0

    for idx_row, row in out.iterrows():
        start = pd.to_datetime(row["开始时间"])
        end = pd.to_datetime(row["结束时间"])
        duration = int(row["持续时长min"])
        if pd.isna(start) or pd.isna(end) or duration <= 0:
            continue

        included_min = int(round(_sum_between(cum_df["是否纳入汇总统计"], start, end)))
        raw_abn = int(round(_sum_between(cum_df["原是否异常"], start, end)))
        corr_abn = int(round(_sum_between(cum_df["是否异常"], start, end)))
        raw_neg = int(round(_sum_between(cum_df["原负损耗"], start, end)))
        corr_neg = int(round(_sum_between(cum_df["修正后负损耗"], start, end)))

        raw_sev_sum = float(_sum_between(cum_df["原异常程度_纳入统计"], start, end))
        corr_sev_sum = float(_sum_between(cum_df["异常程度_纳入统计"], start, end))
        raw_dev_sum = float(_sum_between(cum_df["原偏离程度_纳入统计"], start, end))
        corr_dev_sum = float(_sum_between(cum_df["修正后偏离程度_纳入统计"], start, end))

        station_repeat_min = int(round(_sum_between(cum_df["是否站端功率重复硬异常"], start, end)))
        physical_min = int(round(_sum_between(cum_df["是否物理越限硬异常"], start, end)))
        fan_repeat_marker_min = int(round(_sum_between(cum_df["是否风机联合重复标记"], start, end)))

        raw_max_sev = _max_between(minute_df, start, end, "原异常程度", included_only=True)
        corr_max_sev = _max_between(minute_df, start, end, "异常程度", included_only=True)
        raw_max_dev = _max_between(minute_df, start, end, "原偏离程度", included_only=True)
        corr_max_dev = _max_between(minute_df, start, end, "修正后偏离程度", included_only=True)

        raw_ratio = _safe_ratio(raw_abn, duration)
        corr_ratio = _safe_ratio(corr_abn, duration)

        raw_avg_sev = _safe_ratio(raw_sev_sum, included_min)
        corr_avg_sev = _safe_ratio(corr_sev_sum, included_min)
        sev_change = corr_sev_sum - raw_sev_sum
        sev_drop_rate = _safe_ratio(raw_sev_sum - corr_sev_sum, raw_sev_sum)

        raw_avg_dev = _safe_ratio(raw_dev_sum, included_min)
        corr_avg_dev = _safe_ratio(corr_dev_sum, included_min)
        dev_change = corr_dev_sum - raw_dev_sum
        dev_drop_rate = _safe_ratio(raw_dev_sum - corr_dev_sum, raw_dev_sum)

        out.at[idx_row, "纳入统计分钟数"] = included_min
        out.at[idx_row, "原异常分钟数"] = raw_abn
        out.at[idx_row, "修正后异常分钟数"] = corr_abn
        out.at[idx_row, "异常分钟数变化量"] = corr_abn - raw_abn
        out.at[idx_row, "原异常占比"] = round(raw_ratio, 6) if np.isfinite(raw_ratio) else np.nan
        out.at[idx_row, "修正后异常占比"] = round(corr_ratio, 6) if np.isfinite(corr_ratio) else np.nan
        out.at[idx_row, "异常占比变化量"] = round(corr_ratio - raw_ratio, 6) if np.isfinite(raw_ratio) and np.isfinite(corr_ratio) else np.nan
        out.at[idx_row, "原负损耗分钟数"] = raw_neg
        out.at[idx_row, "修正后负损耗分钟数"] = corr_neg

        out.at[idx_row, "原异常程度总和"] = round(raw_sev_sum, 6)
        out.at[idx_row, "修正后异常程度总和"] = round(corr_sev_sum, 6)
        out.at[idx_row, "异常程度变化量"] = round(sev_change, 6)
        out.at[idx_row, "异常程度下降率"] = round(sev_drop_rate, 6) if np.isfinite(sev_drop_rate) else np.nan
        out.at[idx_row, "原平均异常程度"] = round(raw_avg_sev, 6) if np.isfinite(raw_avg_sev) else np.nan
        out.at[idx_row, "修正后平均异常程度"] = round(corr_avg_sev, 6) if np.isfinite(corr_avg_sev) else np.nan
        out.at[idx_row, "原最大异常程度"] = round(raw_max_sev, 6)
        out.at[idx_row, "修正后最大异常程度"] = round(corr_max_sev, 6)

        out.at[idx_row, "原偏离程度总和"] = round(raw_dev_sum, 6)
        out.at[idx_row, "修正后偏离程度总和"] = round(corr_dev_sum, 6)
        out.at[idx_row, "偏离程度变化量"] = round(dev_change, 6)
        out.at[idx_row, "偏离程度下降率"] = round(dev_drop_rate, 6) if np.isfinite(dev_drop_rate) else np.nan
        out.at[idx_row, "原平均偏离程度"] = round(raw_avg_dev, 6) if np.isfinite(raw_avg_dev) else np.nan
        out.at[idx_row, "修正后平均偏离程度"] = round(corr_avg_dev, 6) if np.isfinite(corr_avg_dev) else np.nan
        out.at[idx_row, "原最大偏离程度"] = round(raw_max_dev, 6)
        out.at[idx_row, "修正后最大偏离程度"] = round(corr_max_dev, 6)

        out.at[idx_row, "站端功率重复硬异常分钟数"] = station_repeat_min
        out.at[idx_row, "物理越限硬异常分钟数"] = physical_min
        out.at[idx_row, "风机联合重复标记分钟数"] = fan_repeat_marker_min
        out.at[idx_row, "是否异常分钟减少"] = bool(corr_abn < raw_abn)
        out.at[idx_row, "是否异常程度下降"] = bool(corr_sev_sum < raw_sev_sum)
        out.at[idx_row, "是否偏离程度下降"] = bool(corr_dev_sum < raw_dev_sum)
        out.at[idx_row, "是否全部修正为正常"] = bool(raw_abn > 0 and corr_abn == 0)

    return out

# ============================================================
# 输出
# ============================================================
def write_outputs(segment_df: pd.DataFrame, detail_df: pd.DataFrame, output_dir: Path, segment_name: str, detail_name: str) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_path = output_dir / segment_name
    detail_path = output_dir / detail_name

    segment_cols = [
        "分段ID", "开始时间", "结束时间", "持续时长min", "纳入统计分钟数",
        "冻结风机数", "重复风机功率之和MW", "风机编号列表",
        "原异常分钟数", "修正后异常分钟数", "异常分钟数变化量",
        "原异常占比", "修正后异常占比", "异常占比变化量",
        "原异常程度总和", "修正后异常程度总和", "异常程度变化量", "异常程度下降率",
        "原平均异常程度", "修正后平均异常程度", "原最大异常程度", "修正后最大异常程度",
        "原偏离程度总和", "修正后偏离程度总和", "偏离程度变化量", "偏离程度下降率",
        "原平均偏离程度", "修正后平均偏离程度", "原最大偏离程度", "修正后最大偏离程度",
        "原负损耗分钟数", "修正后负损耗分钟数",
        "站端功率重复硬异常分钟数", "物理越限硬异常分钟数", "风机联合重复标记分钟数",
        "是否异常分钟减少", "是否异常程度下降", "是否偏离程度下降", "是否全部修正为正常",
        "重复详情",
    ]

    detail_cols = [
        "分段ID", "开始时间", "结束时间", "持续时长min",
        "冻结风机数", "重复风机功率之和MW",
        "风机编号", "状态码", "冻结有功kW", "非负冻结有功kW", "冻结风速ms",
        "重复值组合", "原始开始时间", "原始结束时间", "原始持续长度min",
        "原异常分钟数", "修正后异常分钟数", "异常分钟数变化量",
        "原异常占比", "修正后异常占比", "异常占比变化量",
        "原异常程度总和", "修正后异常程度总和", "异常程度变化量", "异常程度下降率",
        "原偏离程度总和", "修正后偏离程度总和", "偏离程度变化量", "偏离程度下降率",
    ]

    seg_out = segment_df.copy()
    if len(seg_out) == 0:
        seg_out = pd.DataFrame(columns=segment_cols)
    else:
        for c in segment_cols:
            if c not in seg_out.columns:
                seg_out[c] = np.nan
        seg_out = seg_out[segment_cols]

    detail_out = detail_df.copy()
    if len(detail_out) > 0 and len(segment_df) > 0:
        stats_cols = [
            "分段ID", "原异常分钟数", "修正后异常分钟数", "异常分钟数变化量",
            "原异常占比", "修正后异常占比", "异常占比变化量",
            "原异常程度总和", "修正后异常程度总和", "异常程度变化量", "异常程度下降率",
            "原偏离程度总和", "修正后偏离程度总和", "偏离程度变化量", "偏离程度下降率",
        ]
        stats_cols = [c for c in stats_cols if c in segment_df.columns]
        detail_out = detail_out.merge(segment_df[stats_cols], on="分段ID", how="left")

    if len(detail_out) == 0:
        detail_out = pd.DataFrame(columns=detail_cols)
    else:
        for c in detail_cols:
            if c not in detail_out.columns:
                detail_out[c] = np.nan
        detail_out = detail_out[detail_cols]

    seg_out.to_csv(seg_path, index=False, encoding="utf-8-sig")
    detail_out.to_csv(detail_path, index=False, encoding="utf-8-sig")
    return seg_path, detail_path


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="STEP3 江苏全站版：生成风机重复冻结分段，并关联 STEP2 异常结果")
    parser.add_argument("--station-dir", default=r"场站数据\LGXRFD", help="场站目录")
    parser.add_argument("--step2-dir", default=r"场站数据\LGXRFD\step2-异常检测-仅正功率", help="STEP2 输出目录，包含 station_anomaly_detail.csv")
    parser.add_argument("--output-dir", default=r"场站数据\LGXRFD\step3-重复冻结分段-仅正功率", help="STEP3 输出目录")
    parser.add_argument("--repeat-file", default=None, help="联合重复值检测结果.xlsx 路径；不填则使用 station-dir 下的默认相对路径")
    parser.add_argument("--repeat-rel", default=DEFAULT_REPEAT_REL, help="联合重复值检测结果相对 station-dir 的路径")
    parser.add_argument("--anomaly-detail-file", default=None, help="STEP2 station_anomaly_detail.csv 路径；不填则使用 step2-dir/station_anomaly_detail.csv")
    parser.add_argument("--segment-output", default=DEFAULT_SEGMENT_OUTPUT, help="分段汇总输出文件名")
    parser.add_argument("--detail-output", default=DEFAULT_DETAIL_OUTPUT, help="分段明细输出文件名")
    args = parser.parse_args()

    station_dir = Path(args.station_dir)
    step2_dir = Path(args.step2_dir)
    output_dir = Path(args.output_dir)

    repeat_file = resolve_repeat_file(station_dir, args.repeat_file, args.repeat_rel)
    anomaly_detail_file = resolve_step2_detail(step2_dir, args.anomaly_detail_file)

    print("=" * 72)
    print("STEP3 江苏全站版：构建重复冻结分段 + 关联异常程度")
    print("=" * 72)
    print(f"场站目录: {station_dir}")
    print(f"重复检测文件: {repeat_file}")
    print(f"STEP2异常明细: {anomaly_detail_file}")
    print(f"输出目录: {output_dir}")

    repeat_df = load_repeat_events(repeat_file)
    minute_df = load_step2_minute_table(anomaly_detail_file)

    full_start = pd.to_datetime(minute_df["时间"]).min()
    full_end = pd.to_datetime(minute_df["时间"]).max()
    if pd.isna(full_start) or pd.isna(full_end):
        raise RuntimeError("STEP2 异常明细中没有可用时间，无法构建完整分段时间轴。")

    print(f"STEP2完整时间范围: {full_start} ~ {full_end}")

    segment_df, detail_df = build_segments_for_station(
        repeat_df,
        full_start=full_start,
        full_end=full_end,
    )
    segment_df = append_segment_anomaly_stats(segment_df, minute_df)

    seg_path, detail_path = write_outputs(
        segment_df=segment_df,
        detail_df=detail_df,
        output_dir=output_dir,
        segment_name=args.segment_output,
        detail_name=args.detail_output,
    )

    print("\n完成：")
    print(f"- 分段汇总表: {seg_path}")
    print(f"- 分段明细表: {detail_path}")
    print(f"- 原始重复事件数: {len(repeat_df):,}")
    print(f"- 生成分段数: {len(segment_df):,}")
    print(f"- 分段明细行数: {len(detail_df):,}")

    if len(segment_df) > 0:
        print("\n分段表示例：")
        preview_cols = [
            "分段ID", "开始时间", "结束时间", "持续时长min", "冻结风机数", "重复风机功率之和MW",
            "原异常分钟数", "修正后异常分钟数", "原异常程度总和", "修正后异常程度总和",
        ]
        preview_cols = [c for c in preview_cols if c in segment_df.columns]
        print(segment_df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
