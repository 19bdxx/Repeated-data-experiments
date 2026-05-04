# -*- coding: utf-8 -*-
"""
联合重复区段内的 状态码-风速-功率 关系分析

思路：
1. 读取每个场站主 CSV
2. 读取 “风机联合连续相同检测/联合重复值检测结果.xlsx”
3. 基于“风机编号 + 开始时间 + 结束时间”，回到主 CSV 中提取该重复区段对应风机的：
   STATUS_#n, ACTIVE_POWER_#n, WINDSPEED_#n, REACTIVE_POWER_#n, WINDDIRECTION_#n
4. 形成“重复区段级别”明细（每个重复段 1 行）
5. 输出状态码统计、功率类型统计、散点图等

注意：
- 联合重复段本身是“多个字段连续完全相同”的区段，因此在单个重复段内部，
  风速/功率/状态码通常是不变化的。
- 所以这里更适合做“区段级分析”，即：每个重复段视为一个点，
  点大小可以用“持续长度”表示。
"""

import os
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 可配置参数
# =========================
TIMESTAMP_COL = "timestamp"
STATUS_PREFIX = "STATUS_"
POWER_PREFIX = "ACTIVE_POWER_"
WIND_PREFIX = "WINDSPEED_"
REACTIVE_PREFIX = "REACTIVE_POWER_"
WINDDIR_PREFIX = "WINDDIRECTION_"

FIG_DPI = 180
MIN_SEGMENTS_FOR_STATUS_PLOT = 5

# 主 CSV 排除关键词
SKIP_KEYWORDS = [
    "_duplicate_timestamps",
    "_missing_timestamps",
    "_process_summary",
    "_nulls",
    "_seconds_not_zero",
    "_invalid_timestamps",
    "_check_summary",
    "_timestamp_fixed",
    "_timestamp_changes",
    "状态码_风速功率关系分析",
    "联合重复区段状态码分析",
]

# 为了画图不被极端值拉坏，可适当过滤
FILTER_WIND_RANGE = True
MIN_WIND_SPEED = 0
MAX_WIND_SPEED = 30

FILTER_POWER_RANGE = False
MIN_POWER = -1000
MAX_POWER = 10000


# =========================
# 工具函数
# =========================
def setup_chinese_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    for font in candidates:
        try:
            plt.rcParams["font.sans-serif"] = [font]
            return
        except Exception:
            continue


def find_main_csv_files(output_root):
    output_root = Path(output_root)
    main_files = []

    if not output_root.exists():
        raise FileNotFoundError(f"output_root 不存在：{output_root}")

    for station_folder in output_root.iterdir():
        if not station_folder.is_dir():
            continue

        for file_path in station_folder.iterdir():
            if file_path.suffix.lower() != ".csv":
                continue

            if any(k in file_path.name for k in SKIP_KEYWORDS):
                continue

            main_files.append(file_path)

    return sorted(main_files)


def classify_power_type(power, eps=1.0):
    if pd.isna(power):
        return "未知"
    if abs(power) <= eps:
        return "零功率"
    if power < -eps:
        return "负功率"
    return "正功率"


def guess_repeat_status_type(row):
    """
    对“联合重复区段中的状态码”做初步经验判断。
    """
    power_type = row["典型功率类型"]
    median_wind = row["风速中位数"]
    median_power = row["功率中位数"]
    total_minutes = row["重复总分钟数"]
    mean_repeat_len = row["平均重复长度"]

    if power_type == "正功率" and median_power > 0:
        return "疑似正常发电时的重复/冻结"
    if power_type == "零功率" and median_wind >= 3:
        return "疑似停机/故障/维护/待机时的重复"
    if power_type == "零功率" and median_wind < 3:
        return "疑似低风速停机或待机重复"
    if power_type == "负功率":
        return "疑似停机耗电/维护耗电时的重复"
    if mean_repeat_len >= 30 and total_minutes > 100:
        return "疑似长期卡值/通信冻结，需要重点关注"
    return "需结合时段与原始曲线进一步判断"


# =========================
# 核心处理
# =========================
def load_station_main_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"{csv_path} 缺少 {TIMESTAMP_COL} 列")

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL).reset_index(drop=True)

    if FILTER_WIND_RANGE:
        wind_cols = [c for c in df.columns if c.startswith(WIND_PREFIX)]
        for c in wind_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_repeat_segments(repeat_excel_path):
    if not repeat_excel_path.exists():
        return pd.DataFrame()

    df = pd.read_excel(repeat_excel_path)
    if df.empty:
        return df

    # 标准化列名
    rename_map = {
        "风机编号": "fan_no",
        "开始时间": "start_time",
        "结束时间": "end_time",
        "持续长度": "repeat_len",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    required = ["fan_no", "start_time", "end_time", "repeat_len"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{repeat_excel_path} 缺少必要列: {missing}")

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df["fan_no"] = pd.to_numeric(df["fan_no"], errors="coerce")
    df["repeat_len"] = pd.to_numeric(df["repeat_len"], errors="coerce")

    df = df.dropna(subset=["fan_no", "start_time", "end_time", "repeat_len"]).copy()
    df["fan_no"] = df["fan_no"].astype(int)

    return df.sort_values(["fan_no", "start_time"]).reset_index(drop=True)


def extract_segment_value(main_df, fan_no, start_time, end_time):
    """
    对某个重复段，从主 CSV 中提取该风机对应区段的代表值。
    由于该区段本身就是联合重复，理论上这些值在区段内应基本不变，
    直接取首行即可；同时也保留区段内记录条数供核对。
    """
    status_col = f"{STATUS_PREFIX}#{fan_no}"
    power_col = f"{POWER_PREFIX}#{fan_no}"
    wind_col = f"{WIND_PREFIX}#{fan_no}"
    reactive_col = f"{REACTIVE_PREFIX}#{fan_no}"
    winddir_col = f"{WINDDIR_PREFIX}#{fan_no}"

    needed = [status_col, power_col, wind_col]
    for c in needed:
        if c not in main_df.columns:
            return None

    mask = (main_df[TIMESTAMP_COL] >= start_time) & (main_df[TIMESTAMP_COL] <= end_time)
    seg = main_df.loc[mask, [TIMESTAMP_COL] + [c for c in [status_col, power_col, wind_col, reactive_col, winddir_col] if c in main_df.columns]].copy()

    if seg.empty:
        return None

    # 转数值
    for c in seg.columns:
        if c != TIMESTAMP_COL:
            seg[c] = pd.to_numeric(seg[c], errors="coerce")

    first = seg.iloc[0]

    res = {
        "fan_no": fan_no,
        "start_time": start_time,
        "end_time": end_time,
        "covered_rows": len(seg),
        "status": first.get(status_col, np.nan),
        "active_power": first.get(power_col, np.nan),
        "windspeed": first.get(wind_col, np.nan),
        "reactive_power": first.get(reactive_col, np.nan) if reactive_col in seg.columns else np.nan,
        "winddirection": first.get(winddir_col, np.nan) if winddir_col in seg.columns else np.nan,
    }
    return res


def build_repeat_segment_detail(main_df, repeat_df):
    rows = []
    for _, r in repeat_df.iterrows():
        item = extract_segment_value(
            main_df=main_df,
            fan_no=int(r["fan_no"]),
            start_time=r["start_time"],
            end_time=r["end_time"],
        )
        if item is None:
            continue

        item["repeat_len"] = int(r["repeat_len"])
        item["status"] = pd.to_numeric(item["status"], errors="coerce")
        if pd.notna(item["status"]):
            item["status"] = int(item["status"])

        item["power_type"] = classify_power_type(item["active_power"])
        rows.append(item)

    if not rows:
        return pd.DataFrame()

    detail_df = pd.DataFrame(rows)

    if FILTER_WIND_RANGE:
        detail_df = detail_df[
            detail_df["windspeed"].between(MIN_WIND_SPEED, MAX_WIND_SPEED, inclusive="both")
        ].copy()

    if FILTER_POWER_RANGE:
        detail_df = detail_df[
            detail_df["active_power"].between(MIN_POWER, MAX_POWER, inclusive="both")
        ].copy()

    return detail_df.sort_values(["fan_no", "start_time"]).reset_index(drop=True)


def make_status_summary(detail_df):
    if detail_df.empty:
        return pd.DataFrame()

    summary = (
        detail_df.groupby("status")
        .agg(
            重复段数量=("status", "size"),
            重复总分钟数=("repeat_len", "sum"),
            平均重复长度=("repeat_len", "mean"),
            最大重复长度=("repeat_len", "max"),
            涉及风机数=("fan_no", "nunique"),
            风速均值=("windspeed", "mean"),
            风速中位数=("windspeed", "median"),
            功率均值=("active_power", "mean"),
            功率中位数=("active_power", "median"),
            零功率重复段占比=("power_type", lambda x: (x == "零功率").mean()),
            负功率重复段占比=("power_type", lambda x: (x == "负功率").mean()),
            正功率重复段占比=("power_type", lambda x: (x == "正功率").mean()),
        )
        .reset_index()
        .sort_values("重复总分钟数", ascending=False)
    )

    def typical_power_type(subdf):
        vc = subdf["power_type"].value_counts()
        return vc.index[0] if len(vc) > 0 else "未知"

    type_df = (
        detail_df.groupby("status")
        .apply(typical_power_type)
        .reset_index(name="典型功率类型")
    )

    summary = summary.merge(type_df, on="status", how="left")
    summary["初步推测"] = summary.apply(guess_repeat_status_type, axis=1)

    total_seg = summary["重复段数量"].sum()
    total_min = summary["重复总分钟数"].sum()
    summary["重复段占比"] = summary["重复段数量"] / total_seg if total_seg > 0 else 0
    summary["重复分钟占比"] = summary["重复总分钟数"] / total_min if total_min > 0 else 0

    return summary


def make_power_type_summary(detail_df):
    if detail_df.empty:
        return pd.DataFrame()

    out = (
        detail_df.groupby(["status", "power_type"])
        .agg(
            重复段数量=("power_type", "size"),
            重复总分钟数=("repeat_len", "sum"),
            平均重复长度=("repeat_len", "mean"),
            风速均值=("windspeed", "mean"),
            功率均值=("active_power", "mean"),
        )
        .reset_index()
        .sort_values(["status", "重复总分钟数"], ascending=[True, False])
    )
    return out


# =========================
# 画图
# =========================
def scaled_marker_sizes(values, min_size=15, max_size=180):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    if np.all(values == values[0]):
        return np.full_like(values, (min_size + max_size) / 2.0)
    vmin, vmax = values.min(), values.max()
    return min_size + (values - vmin) / (vmax - vmin) * (max_size - min_size)


def plot_all_status_scatter(detail_df, save_path, station_name):
    """
    每个重复段一个点：
    x=风速, y=功率, size=重复长度
    """
    plt.figure(figsize=(10, 7))

    for status in sorted(detail_df["status"].dropna().unique()):
        sub = detail_df[detail_df["status"] == status]
        if len(sub) < MIN_SEGMENTS_FOR_STATUS_PLOT:
            continue

        sizes = scaled_marker_sizes(sub["repeat_len"].values)
        plt.scatter(
            sub["windspeed"],
            sub["active_power"],
            s=sizes,
            alpha=0.35,
            label=f"{status}({len(sub)})"
        )

    plt.xlabel("风速 WINDSPEED")
    plt.ylabel("有功功率 ACTIVE_POWER")
    plt.title(f"{station_name}：联合重复区段内，不同状态码的风速-功率关系\n点大小表示重复长度")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


def plot_one_status_scatter(detail_df, status, save_path, station_name):
    sub = detail_df[detail_df["status"] == status].copy()
    if len(sub) < MIN_SEGMENTS_FOR_STATUS_PLOT:
        return

    sizes = scaled_marker_sizes(sub["repeat_len"].values)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        sub["windspeed"],
        sub["active_power"],
        s=sizes,
        alpha=0.4
    )
    plt.xlabel("风速 WINDSPEED")
    plt.ylabel("有功功率 ACTIVE_POWER")
    plt.title(f"{station_name} 状态码 {status}：联合重复区段散点图\n点大小表示重复长度，重复段数={len(sub)}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


def plot_status_repeat_length_box(detail_df, save_path, station_name, max_status_num=15):
    """
    画“各状态码重复长度分布”箱线图。
    仅画重复总分钟数靠前的若干状态码。
    """
    rank_df = (
        detail_df.groupby("status")["repeat_len"]
        .sum()
        .sort_values(ascending=False)
        .head(max_status_num)
    )
    statuses = list(rank_df.index)
    if not statuses:
        return

    data = [detail_df.loc[detail_df["status"] == s, "repeat_len"].values for s in statuses]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=[str(s) for s in statuses], showfliers=False)
    plt.xlabel("状态码")
    plt.ylabel("重复长度（分钟）")
    plt.title(f"{station_name}：各状态码联合重复长度分布（Top {len(statuses)}）")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


# =========================
# 单场站处理
# =========================
def process_one_station(csv_path):
    csv_path = Path(csv_path)
    station_dir = csv_path.parent
    station_name = station_dir.name

    repeat_excel = station_dir / "风机联合连续相同检测" / "联合重复值检测结果.xlsx"
    if not repeat_excel.exists():
        print(f"[跳过] {station_name} 未找到：{repeat_excel}")
        return

    out_dir = station_dir / "联合重复区段状态码分析"
    fig_dir = out_dir / "图片"
    each_status_dir = fig_dir / "各状态码散点图"

    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    each_status_dir.mkdir(exist_ok=True)

    print(f"\n{'='*100}")
    print(f"开始处理场站：{station_name}")
    print(f"主 CSV：{csv_path}")
    print(f"联合重复结果：{repeat_excel}")

    main_df = load_station_main_csv(csv_path)
    repeat_df = load_repeat_segments(repeat_excel)

    if repeat_df.empty:
        print(f"[跳过] {station_name} 的联合重复结果为空")
        return

    detail_df = build_repeat_segment_detail(main_df, repeat_df)
    if detail_df.empty:
        print(f"[跳过] {station_name} 未提取到任何重复区段明细")
        return

    # 保存明细
    detail_path = out_dir / "联合重复区段_状态码风速功率明细.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"明细已保存：{detail_path}")

    # 汇总表
    status_summary = make_status_summary(detail_df)
    status_summary_path = out_dir / "联合重复区段_状态码统计汇总.csv"
    status_summary.to_csv(status_summary_path, index=False, encoding="utf-8-sig")
    print(f"状态码汇总已保存：{status_summary_path}")

    power_type_summary = make_power_type_summary(detail_df)
    power_type_summary_path = out_dir / "联合重复区段_状态码功率类型汇总.csv"
    power_type_summary.to_csv(power_type_summary_path, index=False, encoding="utf-8-sig")
    print(f"功率类型汇总已保存：{power_type_summary_path}")

    # 图
    all_scatter = fig_dir / "所有状态码_联合重复区段风速功率散点图.png"
    plot_all_status_scatter(detail_df, all_scatter, station_name)
    print(f"总散点图已保存：{all_scatter}")

    repeat_box = fig_dir / "各状态码_联合重复长度分布箱线图.png"
    plot_status_repeat_length_box(detail_df, repeat_box, station_name)
    print(f"重复长度箱线图已保存：{repeat_box}")

    for status in sorted(detail_df["status"].dropna().unique()):
        sub = detail_df[detail_df["status"] == status]
        if len(sub) < MIN_SEGMENTS_FOR_STATUS_PLOT:
            continue
        save_path = each_status_dir / f"状态码_{int(status)}_联合重复区段散点图.png"
        plot_one_status_scatter(detail_df, int(status), save_path, station_name)

    print(f"各状态码散点图已保存：{each_status_dir}")
    print(f"完成：{station_name}")


def main(output_root):
    setup_chinese_font()
    main_csv_files = find_main_csv_files(output_root)

    if not main_csv_files:
        print("没有找到可处理的主 CSV 文件。")
        return

    print(f"共找到 {len(main_csv_files)} 个场站主 CSV")
    for p in main_csv_files:
        print(f" - {p}")

    for csv_path in main_csv_files:
        try:
            process_one_station(csv_path)
        except Exception as e:
            print(f"[失败] {csv_path}: {e}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # 改成你的“场站数据”根目录
    output_root = r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\江苏\场站数据"

    main(output_root)
