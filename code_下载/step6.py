#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP6：基于原始风机宽表 + STEP3/STEP4 结果，生成风机宽表修正结果。

输出结构：
    timestamp
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
2. 未命中任何 STEP3 风机重复明细的风机-时刻，PROCESS_RESULT_#n = “非重复冻结时刻”。
3. 只有满足以下两个条件时，才修正功率：
   - 最终处理细分类型 == “重复正功率>0｜修正后保留”
   - 该风机该时刻 ACTIVE_POWER_#n > 0
   此时 FINAL_ACTIVE_POWER_#n = 0。
4. 其他所有情况：
   FINAL_ACTIVE_POWER_#n = ACTIVE_POWER_#n。
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
DEFAULT_STATION_DIR = r"场站数据\JMZSFD"
DEFAULT_STEP3_DIR = r"场站数据\JMZSFD\step3-重复冻结分段"
DEFAULT_STEP4_DIR = r"场站数据\JMZSFD\step4-阈值分析-细分版"
DEFAULT_OUTPUT_DIR = r"场站数据\JMZSFD\step6-风机宽表修正结果"

DEFAULT_STEP3_DETAIL_NAME = "freeze_count_segments_detail.csv"
DEFAULT_STEP3_SEGMENT_NAME = "freeze_count_segments.csv"
DEFAULT_STEP4_SCORED_NAME = "threshold_scan_scored.csv"

STATUS_PREFIX = "STATUS_#"
POWER_PREFIX = "ACTIVE_POWER_#"
WIND_PREFIX = "WINDSPEED_#"

FINAL_POWER_PREFIX = "FINAL_ACTIVE_POWER_#"
PROCESS_RESULT_PREFIX = "PROCESS_RESULT_#"

DEFAULT_PROCESS_RESULT = "非重复冻结时刻"
CORRECTED_SUBTYPE = "重复正功率>0｜修正后保留"
ZERO_NO_FAN_ABNORMAL_SUBTYPE = "重复正功率=0且冻结风机数=0｜标记为异常分段"
ZERO_NO_FAN_NORMAL_SUBTYPE = "重复正功率=0且冻结风机数=0｜正常"


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
    合并普通风机明细映射和空档异常全风机映射。
    若同一风机同一时刻命中多种类型，优先级：
      1. 重复正功率>0｜修正后保留
      2. 包含“标记为异常分段”的类型
      3. 其他类型
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
        ],
        [3, 2],
        default=1,
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
    """
    data = {"timestamp": pd.to_datetime(raw["timestamp"], errors="coerce")}

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
    如果 keep_hit_rows=True，则优先保留至少有一个 PROCESS_RESULT 不是“非重复冻结时刻”的行。
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
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--threshold", type=float, required=True, help="使用的STEP4阈值，例如 0.16")
    parser.add_argument("--max-output-rows", type=int, default=0, help="最多输出前X行；0表示不限制")
    parser.add_argument("--keep-hit-rows", action="store_true", help="限制输出行数时，优先保留命中重复分段的时间戳")
    args = parser.parse_args()

    station_dir = Path(args.station_dir)
    input_csv = Path(args.input_csv) if args.input_csv else find_main_csv(station_dir)
    step3_detail_path = Path(args.step3_detail_file) if args.step3_detail_file else Path(args.step3_dir) / DEFAULT_STEP3_DETAIL_NAME
    step3_segment_path = Path(args.step3_segment_file) if args.step3_segment_file else Path(args.step3_dir) / DEFAULT_STEP3_SEGMENT_NAME
    step4_scored_path = Path(args.step4_scored_file) if args.step4_scored_file else Path(args.step4_dir) / DEFAULT_STEP4_SCORED_NAME
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("STEP6：生成风机宽表修正结果")
    print("=" * 88)
    print(f"主CSV: {input_csv}")
    print(f"STEP3明细: {step3_detail_path}")
    print(f"STEP3分段汇总: {step3_segment_path}")
    print(f"STEP4 scored: {step4_scored_path}")
    print(f"阈值: {args.threshold}")
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
    fan_time_map_from_detail = build_fan_time_subtype_map(step3_detail, step4_map)
    fan_time_map_empty_segments = build_all_fan_time_subtype_map_for_empty_segments(step3_segments, step4_map, fan_nums)
    fan_time_map = combine_fan_time_maps(fan_time_map_from_detail, fan_time_map_empty_segments)

    print(f"STEP3风机明细行数: {len(step3_detail):,}")
    print(f"STEP3分段汇总行数: {len(step3_segments):,}")
    print(f"由风机明细展开的风机-时间记录数: {len(fan_time_map_from_detail):,}")
    print(f"由空档分段（正常/异常）展开的全风机-时间记录数: {len(fan_time_map_empty_segments):,}")
    print(f"合并后的风机-时间命中记录数: {len(fan_time_map):,}")
    if len(fan_time_map) > 0:
        print("命中记录的最终处理细分类型分布：")
        print(fan_time_map["最终处理细分类型"].value_counts().head(20).to_string())

    out = build_base_wide_output(raw, fan_nums)
    out, summary_df = apply_subtype_results(out, fan_time_map, fan_nums)

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
