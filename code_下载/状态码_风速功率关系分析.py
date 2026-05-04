# -*- coding: utf-8 -*-
"""
风机状态码 - 风速功率关系分析

功能：
1. 自动寻找每个场站主 CSV
2. 自动识别风机编号：STATUS_#1 / ACTIVE_POWER_#1 / WINDSPEED_#1
3. 将所有风机展开成长表：timestamp, fan_no, status, windspeed, active_power
4. 输出：
   - 状态码统计汇总.csv
   - 状态码_风速功率明细.csv
   - 每个状态码一张风速-功率散点图
   - 所有状态码对比图
   - 每个状态码的风速分箱功率中位数曲线
   - 状态码_初步推测.csv

使用方法：
修改最下面的 output_root，然后运行：
python 状态码_风速功率关系分析.py
"""

import os
import re
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 可配置参数
# =========================

# 风机字段名前缀：与你现有代码一致
STATUS_PREFIX = "STATUS_"
POWER_PREFIX = "ACTIVE_POWER_"
WIND_PREFIX = "WINDSPEED_"

TIMESTAMP_COL = "timestamp"

# 图片参数
MAX_POINTS_PER_STATUS = 100000       # 每个状态码最多绘制多少个散点，避免图片太大
MIN_POINTS_FOR_PLOT = 20            # 少于该数量的状态码不单独画图
MIN_POINTS_FOR_BIN_CURVE = 50       # 少于该数量不画分箱中位数曲线
WINDSPEED_BIN_WIDTH = 0.5           # 风速分箱宽度，单位 m/s
FIG_DPI = 180

# 合理范围过滤：只用于画图和统计，避免极端异常值把图拉变形
FILTER_WIND_RANGE = True
MIN_WIND_SPEED = 0
MAX_WIND_SPEED = 30

FILTER_POWER_RANGE = False          # 如果不知道额定功率，建议先 False
MIN_POWER = -500
MAX_POWER = 10000

# 主 CSV 排除关键词：复用你之前找主文件的思路
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
    "状态码_风速功率明细",
    "状态码统计汇总",
]


# =========================
# 2. 工具函数
# =========================

def setup_chinese_font():
    """尽量设置中文字体，避免图片中文乱码。"""
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
    """
    在场站数据输出根目录下寻找主 CSV。
    结构示例：
    output_root/
        LGXRFD/
            LGXRFD_202401-202412.csv
    """
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

            name = file_path.name
            if any(k in name for k in SKIP_KEYWORDS):
                continue

            main_files.append(file_path)

    return sorted(main_files)


def extract_fan_numbers_from_columns(columns):
    """
    从列名中提取风机编号。
    需要同时存在：
    STATUS_#n, ACTIVE_POWER_#n, WINDSPEED_#n
    """
    columns = [str(c).strip().replace("\ufeff", "") for c in columns]
    col_set = set(columns)

    fan_numbers = set()
    pattern = re.compile(r"#(\d+)$")

    for col in columns:
        if col.startswith(STATUS_PREFIX):
            m = pattern.search(col)
            if not m:
                continue
            fan_no = int(m.group(1))

            status_col = f"{STATUS_PREFIX}#{fan_no}"
            power_col = f"{POWER_PREFIX}#{fan_no}"
            wind_col = f"{WIND_PREFIX}#{fan_no}"

            if status_col in col_set and power_col in col_set and wind_col in col_set:
                fan_numbers.add(fan_no)

    return sorted(fan_numbers)


def load_station_long_table(csv_path):
    """
    将单个场站宽表转成长表。
    输出字段：
    timestamp, fan_no, status, windspeed, active_power
    """
    csv_path = Path(csv_path)
    print(f"\n读取场站主文件：{csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"{csv_path} 中没有 {TIMESTAMP_COL} 列")

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL).reset_index(drop=True)

    fan_numbers = extract_fan_numbers_from_columns(df.columns)
    print(f"识别到风机数量：{len(fan_numbers)}，风机编号：{fan_numbers}")

    long_parts = []
    for fan_no in fan_numbers:
        status_col = f"{STATUS_PREFIX}#{fan_no}"
        power_col = f"{POWER_PREFIX}#{fan_no}"
        wind_col = f"{WIND_PREFIX}#{fan_no}"

        tmp = df[[TIMESTAMP_COL, status_col, power_col, wind_col]].copy()
        tmp.columns = ["timestamp", "status", "active_power", "windspeed"]
        tmp["fan_no"] = fan_no

        # 转数值
        tmp["status"] = pd.to_numeric(tmp["status"], errors="coerce")
        tmp["active_power"] = pd.to_numeric(tmp["active_power"], errors="coerce")
        tmp["windspeed"] = pd.to_numeric(tmp["windspeed"], errors="coerce")

        tmp = tmp.dropna(subset=["status", "active_power", "windspeed"])
        tmp["status"] = tmp["status"].astype(int)

        long_parts.append(tmp[["timestamp", "fan_no", "status", "windspeed", "active_power"]])

    if not long_parts:
        return pd.DataFrame(columns=["timestamp", "fan_no", "status", "windspeed", "active_power"])

    long_df = pd.concat(long_parts, ignore_index=True)

    if FILTER_WIND_RANGE:
        long_df = long_df[
            (long_df["windspeed"] >= MIN_WIND_SPEED) &
            (long_df["windspeed"] <= MAX_WIND_SPEED)
        ].copy()

    if FILTER_POWER_RANGE:
        long_df = long_df[
            (long_df["active_power"] >= MIN_POWER) &
            (long_df["active_power"] <= MAX_POWER)
        ].copy()

    return long_df


def make_status_summary(long_df):
    """按状态码汇总统计特征。"""
    if long_df.empty:
        return pd.DataFrame()

    summary = (
        long_df
        .groupby("status")
        .agg(
            样本数=("status", "size"),
            涉及风机数=("fan_no", "nunique"),
            风速均值=("windspeed", "mean"),
            风速中位数=("windspeed", "median"),
            风速P10=("windspeed", lambda x: x.quantile(0.10)),
            风速P90=("windspeed", lambda x: x.quantile(0.90)),
            功率均值=("active_power", "mean"),
            功率中位数=("active_power", "median"),
            功率P10=("active_power", lambda x: x.quantile(0.10)),
            功率P90=("active_power", lambda x: x.quantile(0.90)),
            零功率占比=("active_power", lambda x: (x.abs() <= 1).mean()),
            负功率占比=("active_power", lambda x: (x < 0).mean()),
            正功率占比=("active_power", lambda x: (x > 1).mean()),
        )
        .reset_index()
        .sort_values("样本数", ascending=False)
    )

    total = summary["样本数"].sum()
    summary["样本占比"] = summary["样本数"] / total

    # 调整列顺序
    cols = ["status", "样本数", "样本占比", "涉及风机数",
            "风速均值", "风速中位数", "风速P10", "风速P90",
            "功率均值", "功率中位数", "功率P10", "功率P90",
            "零功率占比", "负功率占比", "正功率占比"]
    return summary[cols]


def guess_status_type(row):
    """
    根据状态码对应的风速-功率分布做初步推测。
    注意：这里只是经验判断，不是最终定义。
    """
    median_power = row["功率中位数"]
    p90_power = row["功率P90"]
    zero_ratio = row["零功率占比"]
    negative_ratio = row["负功率占比"]
    positive_ratio = row["正功率占比"]
    median_wind = row["风速中位数"]

    if positive_ratio >= 0.70 and p90_power > 0:
        return "疑似正常发电/并网运行"
    if zero_ratio >= 0.70 and median_wind >= 3:
        return "疑似停机/待机/故障停机/维护"
    if negative_ratio >= 0.50:
        return "疑似停机耗电/维护耗电/偏航或控制系统耗电"
    if positive_ratio >= 0.30 and zero_ratio >= 0.30:
        return "疑似启停过渡/限功率/状态切换"
    if median_wind < 3 and zero_ratio >= 0.50:
        return "疑似低风速待机"
    return "暂不明确，需要结合曲线和时间段判断"


def make_guess_table(summary_df):
    if summary_df.empty:
        return pd.DataFrame()

    guess_df = summary_df.copy()
    guess_df["初步推测"] = guess_df.apply(guess_status_type, axis=1)
    return guess_df


def sample_for_plot(df, max_points, random_state=42):
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=random_state)


def plot_one_status_scatter(status_df, status, save_path, station_name):
    """单个状态码散点图。"""
    plot_df = sample_for_plot(status_df, MAX_POINTS_PER_STATUS)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        plot_df["windspeed"],
        plot_df["active_power"],
        s=4,
        alpha=0.25
    )
    plt.xlabel("风速 WINDSPEED")
    plt.ylabel("有功功率 ACTIVE_POWER")
    plt.title(f"{station_name} 状态码 {status}：风速-功率散点图\n样本数={len(status_df):,}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


def plot_all_status_scatter(long_df, save_path, station_name):
    """所有状态码放在一张图上对比。"""
    plt.figure(figsize=(10, 7))

    statuses = sorted(long_df["status"].unique())
    for status in statuses:
        status_df = long_df[long_df["status"] == status]
        if len(status_df) < MIN_POINTS_FOR_PLOT:
            continue
        plot_df = sample_for_plot(status_df, max(2000, MAX_POINTS_PER_STATUS // max(len(statuses), 1)))
        plt.scatter(
            plot_df["windspeed"],
            plot_df["active_power"],
            s=4,
            alpha=0.20,
            label=f"{status}({len(status_df):,})"
        )

    plt.xlabel("风速 WINDSPEED")
    plt.ylabel("有功功率 ACTIVE_POWER")
    plt.title(f"{station_name}：不同状态码风速-功率对比")
    plt.grid(True, alpha=0.25)
    plt.legend(markerscale=3, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


def calc_bin_curve(status_df):
    """计算状态码下风速分箱功率中位数。"""
    if status_df.empty:
        return pd.DataFrame()

    df = status_df.copy()
    min_ws = math.floor(df["windspeed"].min() / WINDSPEED_BIN_WIDTH) * WINDSPEED_BIN_WIDTH
    max_ws = math.ceil(df["windspeed"].max() / WINDSPEED_BIN_WIDTH) * WINDSPEED_BIN_WIDTH

    bins = np.arange(min_ws, max_ws + WINDSPEED_BIN_WIDTH, WINDSPEED_BIN_WIDTH)
    if len(bins) < 2:
        return pd.DataFrame()

    df["风速分箱"] = pd.cut(df["windspeed"], bins=bins, include_lowest=True)

    curve = (
        df.groupby("风速分箱", observed=True)
        .agg(
            样本数=("active_power", "size"),
            风速均值=("windspeed", "mean"),
            功率中位数=("active_power", "median"),
            功率P25=("active_power", lambda x: x.quantile(0.25)),
            功率P75=("active_power", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )

    # 分箱样本太少的点不稳定，剔除
    curve = curve[curve["样本数"] >= 10].copy()
    return curve


def plot_bin_curves(long_df, save_path, station_name):
    """所有状态码的风速分箱中位数曲线对比。"""
    plt.figure(figsize=(10, 7))

    all_curves = []
    for status in sorted(long_df["status"].unique()):
        status_df = long_df[long_df["status"] == status]
        if len(status_df) < MIN_POINTS_FOR_BIN_CURVE:
            continue

        curve = calc_bin_curve(status_df)
        if curve.empty:
            continue

        curve["status"] = status
        all_curves.append(curve)

        plt.plot(
            curve["风速均值"],
            curve["功率中位数"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"{status}({len(status_df):,})"
        )

    plt.xlabel("风速分箱均值")
    plt.ylabel("功率中位数")
    plt.title(f"{station_name}：不同状态码风速分箱功率中位数曲线")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()

    if all_curves:
        return pd.concat(all_curves, ignore_index=True)
    return pd.DataFrame()


def process_one_station(csv_path):
    """处理单个场站。"""
    csv_path = Path(csv_path)
    station_name = csv_path.parent.name
    out_dir = csv_path.parent / "状态码_风速功率关系分析"
    fig_dir = out_dir / "图片"
    status_fig_dir = fig_dir / "各状态码散点图"

    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    status_fig_dir.mkdir(exist_ok=True)

    long_df = load_station_long_table(csv_path)
    if long_df.empty:
        print(f"[跳过] {csv_path} 没有可用的状态码/功率/风速数据")
        return

    # 保存长表明细
    detail_path = out_dir / "状态码_风速功率明细.csv"
    long_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"明细已保存：{detail_path}")

    # 汇总表
    summary_df = make_status_summary(long_df)
    summary_path = out_dir / "状态码统计汇总.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"汇总已保存：{summary_path}")

    # 初步推测表
    guess_df = make_guess_table(summary_df)
    guess_path = out_dir / "状态码_初步推测.csv"
    guess_df.to_csv(guess_path, index=False, encoding="utf-8-sig")
    print(f"初步推测已保存：{guess_path}")

    # 总对比散点图
    all_scatter_path = fig_dir / "所有状态码_风速功率散点对比.png"
    plot_all_status_scatter(long_df, all_scatter_path, station_name)
    print(f"总对比散点图已保存：{all_scatter_path}")

    # 分箱曲线图
    bin_curve_path = fig_dir / "所有状态码_风速分箱功率中位数曲线.png"
    curve_df = plot_bin_curves(long_df, bin_curve_path, station_name)
    print(f"分箱曲线图已保存：{bin_curve_path}")

    if not curve_df.empty:
        curve_path = out_dir / "状态码_风速分箱功率曲线数据.csv"
        curve_df.to_csv(curve_path, index=False, encoding="utf-8-sig")
        print(f"分箱曲线数据已保存：{curve_path}")

    # 每个状态码单独散点图
    for status, status_df in long_df.groupby("status"):
        if len(status_df) < MIN_POINTS_FOR_PLOT:
            continue
        save_path = status_fig_dir / f"状态码_{status}_风速功率散点图.png"
        plot_one_status_scatter(status_df, status, save_path, station_name)

    print(f"各状态码散点图已保存：{status_fig_dir}")
    print(f"完成：{station_name}")


def main(output_root):
    setup_chinese_font()

    main_csv_files = find_main_csv_files(output_root)
    if not main_csv_files:
        print("没有找到可处理的主 CSV 文件。")
        return

    print(f"共找到 {len(main_csv_files)} 个场站主 CSV：")
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
