#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dash：STEP4 阈值结果的功率-损耗散点图

功能：
- 读取 STEP2 的 station_anomaly_detail.csv
- 读取 STEP4 的 threshold_scan_scored.csv
- 根据选择的阈值，把每个时刻映射到对应重复冻结分段的“最终处理方式”
- 绘制：横坐标=风机功率之和，纵坐标=损耗
- 颜色=标记为异常分段 / 修正后保留 / 保留（不修正）

默认路径按江苏场站 LGXRFD 设置，可通过命令行参数修改。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

POINT_OUTLIER_POWER_MIN_MW = 3.0
POINT_OUTLIER_SEVERITY_THRESHOLD = 2.0



DEFAULT_STEP2_DETAIL = r"场站数据\LGXRFD\step2-异常检测-仅正功率\station_anomaly_detail.csv"
DEFAULT_STEP4_SCORED = r"场站数据\LGXRFD\step4-阈值分析-细分版-仅正功率\threshold_scan_scored.csv"

DECISION_ORDER = ["标记为异常分段", "修正后保留", "保留（不修正）", "未匹配分段"]

# 不指定具体颜色，保持 plotly 默认配色；这里只固定分类顺序。


def read_csv_auto(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取失败：{path}\n最后一次报错：{last_err}")


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True, desc: str = "") -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # 宽松匹配：去空格后再比对
    norm_map = {str(c).replace(" ", "").strip(): c for c in cols}
    for c in candidates:
        key = str(c).replace(" ", "").strip()
        if key in norm_map:
            return norm_map[key]
    if required:
        raise ValueError(f"缺少必要列{f'（{desc}）' if desc else ''}，候选列名：{candidates}\n实际列名：{cols}")
    return None


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    def parse(v):
        if pd.isna(v):
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        return str(v).strip().lower() in {"true", "1", "yes", "y", "是", "异常"}
    return s.apply(parse).astype(bool)


def normalize_detail_df(df: pd.DataFrame) -> pd.DataFrame:
    time_col = find_col(df, ["时间", "timestamp"], desc="时间")
    station_col = find_col(df, ["场站", "station", "station_name"], required=False)
    ct_col = find_col(df, [
        "站端有功功率MW（ACTIVE_POWER_STATION）", "ACTIVE_POWER_STATION", "站端有功功率MW", "CT有效功率", "CT_eff"
    ], required=False)
    raw_p_col = find_col(df, ["原始风机功率之和MW", "原风机汇总功率MW", "原风机汇总功率", "FAN_SUM", "FAN_eff_raw"], desc="原始风机功率")
    corr_p_col = find_col(df, ["修正后风机功率之和MW", "修正后风机汇总功率MW", "修正后风机汇总功率", "FAN_eff", "FAN_SUM_corr"], desc="修正后风机功率")
    repeat_col = find_col(df, ["重复风机功率扣减量MW", "重复风机功率之和MW", "repeat_power_mw"], required=False)
    raw_l_col = find_col(df, ["原损耗MW", "原损耗", "L_raw"], desc="原损耗")
    corr_l_col = find_col(df, ["修正后损耗MW", "修正后损耗", "L"], desc="修正后损耗")
    raw_abn_col = find_col(df, ["原是否异常", "原始是否异常", "is_anomaly_raw"], required=False)
    corr_abn_col = find_col(df, ["是否异常", "修正后是否异常", "is_anomaly"], required=False)
    raw_sev_col = find_col(df, ["原异常程度", "原始异常程度"], required=False)
    corr_sev_col = find_col(df, ["异常程度", "修正后异常程度"], required=False)
    raw_dev_col = find_col(df, ["原偏离程度", "原始偏离程度"], required=False)
    corr_dev_col = find_col(df, ["修正后偏离程度", "偏离程度"], required=False)
    include_col = find_col(df, ["是否纳入汇总统计", "是否纳入统计", "include_in_summary"], required=False)
    rule_col = find_col(df, ["修正后判异规则", "判异规则", "anomaly_rule"], required=False)

    out = pd.DataFrame()
    out["时间"] = pd.to_datetime(df[time_col], errors="coerce")
    out["场站"] = df[station_col].astype(str) if station_col else ""
    if ct_col:
        out["站端有功功率MW"] = pd.to_numeric(df[ct_col], errors="coerce")
    else:
        out["站端有功功率MW"] = np.nan
    out["原始风机功率之和MW"] = pd.to_numeric(df[raw_p_col], errors="coerce")
    out["修正后风机功率之和MW"] = pd.to_numeric(df[corr_p_col], errors="coerce")
    out["重复风机功率扣减量MW"] = pd.to_numeric(df[repeat_col], errors="coerce").fillna(0.0) if repeat_col else 0.0
    out["原损耗MW"] = pd.to_numeric(df[raw_l_col], errors="coerce")
    out["修正后损耗MW"] = pd.to_numeric(df[corr_l_col], errors="coerce")
    out["原是否异常"] = to_bool_series(df[raw_abn_col]) if raw_abn_col else False
    out["是否异常"] = to_bool_series(df[corr_abn_col]) if corr_abn_col else False
    out["原异常程度"] = pd.to_numeric(df[raw_sev_col], errors="coerce").fillna(0.0) if raw_sev_col else 0.0
    out["异常程度"] = pd.to_numeric(df[corr_sev_col], errors="coerce").fillna(0.0) if corr_sev_col else 0.0
    out["原偏离程度"] = pd.to_numeric(df[raw_dev_col], errors="coerce").fillna(0.0) if raw_dev_col else 0.0
    out["偏离程度"] = pd.to_numeric(df[corr_dev_col], errors="coerce").fillna(0.0) if corr_dev_col else 0.0
    out["是否纳入汇总统计"] = to_bool_series(df[include_col]) if include_col else True
    out["判异规则"] = df[rule_col].astype(str) if rule_col else ""
    out = out.dropna(subset=["时间"]).sort_values("时间").reset_index(drop=True)
    return out


def normalize_scored_df(df: pd.DataFrame) -> pd.DataFrame:
    threshold_col = find_col(df, ["阈值", "threshold"], desc="阈值")
    seg_id_col = find_col(df, ["分段ID", "segment_id"], desc="分段ID")
    start_col = find_col(df, ["开始时间", "start_time"], desc="开始时间")
    end_col = find_col(df, ["结束时间", "end_time"], desc="结束时间")
    decision_col = find_col(df, ["最终处理方式", "final_decision"], required=False)
    subtype_col = find_col(df, ["最终处理细分类型", "detail_type", "final_detail_type"], required=False)
    class_col = find_col(df, ["分段类别", "segment_class"], required=False)
    repeat_power_col = find_col(df, ["重复风机功率之和MW", "重复功率之和MW", "repeat_power_mw"], required=False)
    freeze_cnt_col = find_col(df, ["冻结风机数", "重复风机数", "freeze_fan_count"], required=False)
    raw_ratio_col = find_col(df, ["原异常占比", "raw_anomaly_ratio"], required=False)
    corr_ratio_col = find_col(df, ["修正后异常占比", "corr_anomaly_ratio"], required=False)
    raw_sev_col = find_col(df, ["原异常程度总和", "raw_severity_sum"], required=False)
    corr_sev_col = find_col(df, ["修正后异常程度总和", "corr_severity_sum"], required=False)
    raw_dev_col = find_col(df, ["原偏离程度总和", "raw_deviation_sum"], required=False)
    corr_dev_col = find_col(df, ["修正后偏离程度总和", "corr_deviation_sum"], required=False)
    dev_change_col = find_col(df, ["偏离程度净变化量（修正后-原）", "偏离程度变化量", "deviation_change"], required=False)

    out = pd.DataFrame()
    out["阈值"] = pd.to_numeric(df[threshold_col], errors="coerce")
    out["分段ID"] = df[seg_id_col]
    out["开始时间"] = pd.to_datetime(df[start_col], errors="coerce")
    out["结束时间"] = pd.to_datetime(df[end_col], errors="coerce")
    out["最终处理方式"] = df[decision_col].astype(str).fillna("未匹配分段") if decision_col else "未匹配分段"
    out["最终处理细分类型"] = df[subtype_col].astype(str).fillna("未匹配分段") if subtype_col else out["最终处理方式"]
    out["分段类别"] = df[class_col].astype(str) if class_col else ""
    out["分段重复功率之和MW"] = pd.to_numeric(df[repeat_power_col], errors="coerce") if repeat_power_col else np.nan
    out["冻结风机数"] = pd.to_numeric(df[freeze_cnt_col], errors="coerce") if freeze_cnt_col else np.nan
    out["原异常占比"] = pd.to_numeric(df[raw_ratio_col], errors="coerce") if raw_ratio_col else np.nan
    out["修正后异常占比"] = pd.to_numeric(df[corr_ratio_col], errors="coerce") if corr_ratio_col else np.nan
    out["原异常程度总和"] = pd.to_numeric(df[raw_sev_col], errors="coerce") if raw_sev_col else np.nan
    out["修正后异常程度总和"] = pd.to_numeric(df[corr_sev_col], errors="coerce") if corr_sev_col else np.nan
    out["原偏离程度总和"] = pd.to_numeric(df[raw_dev_col], errors="coerce") if raw_dev_col else np.nan
    out["修正后偏离程度总和"] = pd.to_numeric(df[corr_dev_col], errors="coerce") if corr_dev_col else np.nan
    if dev_change_col:
        out["偏离程度净变化量（修正后-原）"] = pd.to_numeric(df[dev_change_col], errors="coerce")
    else:
        out["偏离程度净变化量（修正后-原）"] = out["修正后偏离程度总和"] - out["原偏离程度总和"]
    out = out.dropna(subset=["阈值", "开始时间", "结束时间"]).sort_values(["阈值", "开始时间"]).reset_index(drop=True)
    return out


def build_threshold_options(scored_df: pd.DataFrame) -> List[Dict[str, str]]:
    vals = sorted(scored_df["阈值"].dropna().unique())
    return [{"label": f"{v:.4g}", "value": f"{v:.12g}"} for v in vals]


def build_category_options(values: pd.Series) -> List[Dict[str, str]]:
    items = [x for x in sorted(values.dropna().astype(str).unique()) if x and x.lower() != "nan"]
    if "未匹配分段" not in items:
        items.append("未匹配分段")
    return [{"label": x, "value": x} for x in items]


def make_interval_join(detail_df: pd.DataFrame, scored_one: pd.DataFrame) -> pd.DataFrame:
    """把逐分钟 STEP2 明细映射到某个阈值下的 STEP4 分段决策。

    不在任何 STEP4 分段内的普通分钟也必须保留，并标记为：
        最终处理方式 = 未匹配分段
        分段类别 = 未匹配分段
    否则 Dash 默认筛选会把大量普通点过滤掉。
    """
    detail = detail_df.sort_values("时间").copy()
    seg = scored_one.sort_values("开始时间").copy()

    if len(seg) == 0:
        out = detail.copy()
        out["分段ID"] = np.nan
        out["最终处理方式"] = "未匹配分段"
        out["最终处理细分类型"] = "未匹配分段"
        out["分段类别"] = "未匹配分段"
        out["分段重复功率之和MW"] = 0.0
        out["冻结风机数"] = 0
        out["偏离程度净变化量（修正后-原）"] = np.nan
        return out

    candidate_cols = [
        "分段ID", "开始时间", "结束时间", "最终处理方式", "最终处理细分类型", "分段类别",
        "分段重复功率之和MW", "冻结风机数", "原异常占比", "修正后异常占比",
        "原异常程度总和", "修正后异常程度总和",
        "原偏离程度总和", "修正后偏离程度总和", "偏离程度净变化量（修正后-原）",
    ]
    use_cols = [c for c in candidate_cols if c in seg.columns]

    joined = pd.merge_asof(
        detail,
        seg[use_cols].sort_values("开始时间"),
        left_on="时间",
        right_on="开始时间",
        direction="backward",
    )

    in_seg = joined["结束时间"].notna() & (joined["时间"] <= joined["结束时间"])

    clear_cols = [
        "分段ID", "最终处理方式", "最终处理细分类型", "分段类别",
        "分段重复功率之和MW", "冻结风机数", "原异常占比", "修正后异常占比",
        "原异常程度总和", "修正后异常程度总和",
        "原偏离程度总和", "修正后偏离程度总和", "偏离程度净变化量（修正后-原）",
    ]
    for col in clear_cols:
        if col in joined.columns:
            joined.loc[~in_seg, col] = np.nan

    for col, default in [
        ("最终处理方式", "未匹配分段"),
        ("最终处理细分类型", "未匹配分段"),
        ("分段类别", "未匹配分段"),
    ]:
        if col not in joined.columns:
            joined[col] = default
        joined[col] = joined[col].fillna(default)

    if "分段重复功率之和MW" not in joined.columns:
        joined["分段重复功率之和MW"] = 0.0
    if "冻结风机数" not in joined.columns:
        joined["冻结风机数"] = 0
    if "偏离程度净变化量（修正后-原）" not in joined.columns:
        joined["偏离程度净变化量（修正后-原）"] = np.nan

    joined["分段重复功率之和MW"] = pd.to_numeric(joined["分段重复功率之和MW"], errors="coerce").fillna(0.0)
    joined["冻结风机数"] = pd.to_numeric(joined["冻结风机数"], errors="coerce").fillna(0)
    return joined


def sample_for_plot(df: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    # 尽量按图中显示类别分层采样，避免小类别被完全淹没。
    parts = []
    if "图中显示类别" in df.columns:
        group_col = "图中显示类别"
    elif "最终处理细分类型" in df.columns:
        group_col = "最终处理细分类型"
    else:
        group_col = "最终处理方式"
    groups = list(df.groupby(group_col, dropna=False))
    base_n = max(max_points // max(len(groups), 1), 1)
    remaining = max_points
    for _, g in groups:
        n = min(len(g), base_n)
        parts.append(g.sample(n=n, random_state=seed) if len(g) > n else g)
        remaining -= n
    used = pd.concat(parts, ignore_index=False) if parts else df.iloc[0:0]
    if remaining > 0:
        rest = df.drop(index=used.index, errors="ignore")
        if len(rest) > 0:
            used = pd.concat([used, rest.sample(n=min(remaining, len(rest)), random_state=seed)], ignore_index=False)
    return used.sort_values("时间")


def create_app(detail_df: pd.DataFrame, scored_df: pd.DataFrame, max_points_default: int) -> Dash:
    app = Dash(__name__)

    threshold_options = build_threshold_options(scored_df)
    default_threshold = threshold_options[0]["value"] if threshold_options else None

    RAW_DISPLAY_SUBTYPES = [
        "重复正功率>0｜保留（不修正）",
        "重复正功率>0｜修正后保留",
        "重复正功率>0｜标记为异常分段",
        "重复正功率=0且冻结风机数=0｜正常",
        "重复正功率=0且冻结风机数=0｜标记为异常分段",
    ]
    CORRECTED_SUBTYPE = "重复正功率>0｜修正后保留"

    RAW_KEEP_CATEGORY = "原始｜重复正功率>0｜保留（不修正）"
    RAW_ZERO_NORMAL_CATEGORY = "原始｜重复正功率=0且冻结风机数=0｜正常"
    CORR_KEEP_CATEGORY = f"修正后｜{CORRECTED_SUBTYPE}"

    RAW_KEEP_OUTLIER_CATEGORY = RAW_KEEP_CATEGORY + "-离群点"
    RAW_ZERO_NORMAL_OUTLIER_CATEGORY = RAW_ZERO_NORMAL_CATEGORY + "-离群点"
    CORR_KEEP_OUTLIER_CATEGORY = CORR_KEEP_CATEGORY + "-离群点"

    display_categories = [
        RAW_KEEP_CATEGORY,
        RAW_KEEP_OUTLIER_CATEGORY,
        "原始｜重复正功率>0｜修正后保留",
        "原始｜重复正功率>0｜标记为异常分段",
        RAW_ZERO_NORMAL_CATEGORY,
        RAW_ZERO_NORMAL_OUTLIER_CATEGORY,
        "原始｜重复正功率=0且冻结风机数=0｜标记为异常分段",
        CORR_KEEP_CATEGORY,
        CORR_KEEP_OUTLIER_CATEGORY,
    ]
    category_options = [{"label": x, "value": x} for x in display_categories]

    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "padding": "16px"},
        children=[
            html.H2("STEP4 阈值结果：原始5类 + 修正后1类 + 点级离群点"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "220px 560px 220px 220px", "gap": "12px", "marginBottom": "12px"},
                children=[
                    html.Div([
                        html.Label("阈值"),
                        dcc.Dropdown(id="threshold-dd", options=threshold_options, value=default_threshold, clearable=False),
                    ]),
                    html.Div([
                        html.Label("图中显示类别"),
                        dcc.Dropdown(id="display-category-dd", options=category_options, value=[x["value"] for x in category_options], multi=True),
                    ]),
                    html.Div([
                        html.Label("是否纳入汇总统计"),
                        dcc.Dropdown(
                            id="include-dd",
                            options=[
                                {"label": "全部显示", "value": "all"},
                                {"label": "只显示纳入统计点", "value": "yes"},
                            ],
                            value="all",
                            clearable=False,
                        ),
                    ]),
                    html.Div([
                        html.Label("点大小"),
                        dcc.Dropdown(
                            id="size-dd",
                            options=[
                                {"label": "固定大小", "value": "none"},
                                {"label": "重复风机功率扣减量MW", "value": "重复风机功率扣减量MW"},
                                {"label": "当前口径异常程度", "value": "当前口径异常程度"},
                                {"label": "当前口径偏离程度", "value": "当前口径偏离程度"},
                                {"label": "分段偏离程度净变化量", "value": "偏离程度净变化量（修正后-原）"},
                            ],
                            value="none",
                            clearable=False,
                        ),
                    ]),
                    html.Div([
                        html.Label("最大显示点数（0=全部）"),
                        dcc.Input(id="max-points-input", type="number", value=max_points_default, min=0, step=1000, style={"width": "100%"}),
                    ]),
                ],
            ),
            html.Div(
                f"说明：前5类使用原始风机功率之和/原损耗绘制；第6类仅针对“重复正功率>0｜修正后保留”，使用修正后风机功率之和/修正后损耗绘制。点级离群仅从三类最终可用数据中识别：原始保留不修正、原始零重复正常、修正后保留。离群规则：当前口径功率 > {POINT_OUTLIER_POWER_MIN_MW:g} MW 且当前口径异常程度 >= {POINT_OUTLIER_SEVERITY_THRESHOLD:g}；超低功率区间暂不标记离群。",
                style={"marginBottom": "8px", "fontSize": "14px", "color": "#555"},
            ),
            html.Div(id="metrics-box", style={"marginBottom": "10px", "fontSize": "15px"}),
            dcc.Graph(id="scatter-graph", style={"height": "760px"}),
        ],
    )

    @app.callback(
        Output("scatter-graph", "figure"),
        Output("metrics-box", "children"),
        Input("threshold-dd", "value"),
        Input("display-category-dd", "value"),
        Input("include-dd", "value"),
        Input("size-dd", "value"),
        Input("max-points-input", "value"),
    )
    def update_graph(threshold_value, display_categories_selected, include_mode, size_col, max_points):
        if threshold_value is None:
            return px.scatter(title="没有可用阈值"), "未读取到阈值。"

        th = float(threshold_value)
        scored_one = scored_df[np.isclose(scored_df["阈值"].astype(float), th)].copy()
        base_df = make_interval_join(detail_df, scored_one)

        total_step2 = len(base_df)

        if include_mode == "yes":
            base_df = base_df[base_df["是否纳入汇总统计"]].copy()

        raw_base = base_df[base_df["最终处理细分类型"].isin(RAW_DISPLAY_SUBTYPES)].copy()

        raw_plot = raw_base.copy()
        raw_plot["图中显示类别"] = "原始｜" + raw_plot["最终处理细分类型"].astype(str)
        raw_plot["图中口径"] = "原始"
        raw_plot["图中风机功率之和MW"] = raw_plot["原始风机功率之和MW"]
        raw_plot["图中损耗MW"] = raw_plot["原损耗MW"]
        raw_plot["当前口径是否异常"] = raw_plot["原是否异常"]
        raw_plot["当前口径异常程度"] = raw_plot["原异常程度"]
        raw_plot["当前口径偏离程度"] = raw_plot["原偏离程度"] if "原偏离程度" in raw_plot.columns else 0.0

        raw_power = pd.to_numeric(raw_plot["图中风机功率之和MW"], errors="coerce")
        raw_severity = pd.to_numeric(raw_plot["当前口径异常程度"], errors="coerce")
        raw_outlier_mask = (
            raw_plot["图中显示类别"].isin([RAW_KEEP_CATEGORY, RAW_ZERO_NORMAL_CATEGORY])
            & (raw_power > POINT_OUTLIER_POWER_MIN_MW)
            & (raw_severity >= POINT_OUTLIER_SEVERITY_THRESHOLD)
        )
        raw_plot["是否点级离群"] = raw_outlier_mask
        raw_plot["点级离群原因"] = np.where(
            raw_outlier_mask,
            f"原始口径功率>{POINT_OUTLIER_POWER_MIN_MW:g}MW且原异常程度>={POINT_OUTLIER_SEVERITY_THRESHOLD:g}",
            "",
        )
        raw_plot.loc[raw_outlier_mask & raw_plot["图中显示类别"].eq(RAW_KEEP_CATEGORY), "图中显示类别"] = RAW_KEEP_OUTLIER_CATEGORY
        raw_plot.loc[raw_outlier_mask & raw_plot["图中显示类别"].eq(RAW_ZERO_NORMAL_CATEGORY), "图中显示类别"] = RAW_ZERO_NORMAL_OUTLIER_CATEGORY

        corr_base = base_df[base_df["最终处理细分类型"].eq(CORRECTED_SUBTYPE)].copy()
        corr_plot = corr_base.copy()
        corr_plot["图中显示类别"] = CORR_KEEP_CATEGORY
        corr_plot["图中口径"] = "修正后"
        corr_plot["图中风机功率之和MW"] = corr_plot["修正后风机功率之和MW"]
        corr_plot["图中损耗MW"] = corr_plot["修正后损耗MW"]
        corr_plot["当前口径是否异常"] = corr_plot["是否异常"]
        corr_plot["当前口径异常程度"] = corr_plot["异常程度"]
        corr_plot["当前口径偏离程度"] = corr_plot["偏离程度"] if "偏离程度" in corr_plot.columns else 0.0

        corr_power = pd.to_numeric(corr_plot["图中风机功率之和MW"], errors="coerce")
        corr_severity = pd.to_numeric(corr_plot["当前口径异常程度"], errors="coerce")
        corr_outlier_mask = (
            (corr_power > POINT_OUTLIER_POWER_MIN_MW)
            & (corr_severity >= POINT_OUTLIER_SEVERITY_THRESHOLD)
        )
        corr_plot["是否点级离群"] = corr_outlier_mask
        corr_plot["点级离群原因"] = np.where(
            corr_outlier_mask,
            f"修正后口径功率>{POINT_OUTLIER_POWER_MIN_MW:g}MW且修正后异常程度>={POINT_OUTLIER_SEVERITY_THRESHOLD:g}",
            "",
        )
        corr_plot.loc[corr_outlier_mask, "图中显示类别"] = CORR_KEEP_OUTLIER_CATEGORY

        plot_df = pd.concat([raw_plot, corr_plot], ignore_index=True)

        if display_categories_selected:
            plot_df = plot_df[plot_df["图中显示类别"].isin(display_categories_selected)].copy()

        before_xy_filter = len(plot_df)
        plot_df = plot_df[
            np.isfinite(pd.to_numeric(plot_df["图中风机功率之和MW"], errors="coerce"))
            & np.isfinite(pd.to_numeric(plot_df["图中损耗MW"], errors="coerce"))
        ].copy()
        dropped_xy = before_xy_filter - len(plot_df)

        plot_df["图中风机功率之和MW"] = pd.to_numeric(plot_df["图中风机功率之和MW"], errors="coerce")
        plot_df["图中损耗MW"] = pd.to_numeric(plot_df["图中损耗MW"], errors="coerce")
        plot_df["悬浮时间"] = plot_df["时间"].dt.strftime("%Y-%m-%d %H:%M")

        total_before_sample = len(plot_df)
        max_points = int(max_points if max_points is not None else max_points_default)
        plot_df = sample_for_plot(plot_df, max_points=max_points)

        actual_size = None if size_col == "none" or size_col not in plot_df.columns else size_col

        fig = px.scatter(
            plot_df,
            x="图中风机功率之和MW",
            y="图中损耗MW",
            color="图中显示类别",
            category_orders={"图中显示类别": display_categories},
            size=actual_size,
            hover_data={
                "悬浮时间": True,
                "图中口径": True,
                "分段ID": True,
                "最终处理细分类型": True,
                "是否点级离群": True,
                "点级离群原因": True,
                "冻结风机数": True,
                "重复风机功率扣减量MW": ":.4f",
                "当前口径是否异常": True,
                "当前口径异常程度": ":.4f",
                "当前口径偏离程度": ":.4f",
                "偏离程度净变化量（修正后-原）": ":.4f",
                "判异规则": True,
                "图中风机功率之和MW": ":.4f",
                "图中损耗MW": ":.4f",
            },
            title=f"阈值={th:.4g} | 原始5类 + 修正后1类 + 点级离群点 | 横坐标=风机功率之和，纵坐标=损耗",
        )
        fig.update_traces(marker=dict(opacity=0.55), selector=dict(mode="markers"))
        fig.update_layout(
            template="plotly_white",
            xaxis_title="风机功率之和MW",
            yaxis_title="损耗MW",
            legend_title="图中显示类别",
        )

        counts = plot_df["图中显示类别"].value_counts().to_dict()
        count_text = " | ".join([f"{k}: {v:,}" for k, v in counts.items()])
        outlier_count = int(plot_df["是否点级离群"].fillna(False).sum()) if "是否点级离群" in plot_df.columns else 0
        avg_dev = float(pd.to_numeric(plot_df.get("当前口径偏离程度", pd.Series(dtype=float)), errors="coerce").mean()) if len(plot_df) else np.nan
        avg_sev = float(pd.to_numeric(plot_df.get("当前口径异常程度", pd.Series(dtype=float)), errors="coerce").mean()) if len(plot_df) else np.nan
        sampling_text = "未采样，显示全部" if max_points <= 0 else f"最大显示点数={max_points:,}"

        metrics = (
            f"阈值={th:.4g} | STEP2总点数={total_step2:,} | "
            f"构造后候选点数={before_xy_filter:,} | 有效XY点数={total_before_sample:,} | "
            f"实际绘制点数={len(plot_df):,} | 点级离群点数={outlier_count:,} | {sampling_text} | "
            f"XY无效丢弃点数={dropped_xy:,} | 平均异常程度={avg_sev:.4f} | 平均偏离程度={avg_dev:.4f} | {count_text}"
        )
        return fig, metrics

    return app


def main():
    parser = argparse.ArgumentParser(description="Dash：原始5类 + 修正后1类 + 点级离群点的风机功率-损耗散点图")
    parser.add_argument("--step2-detail", default=DEFAULT_STEP2_DETAIL, help="STEP2 station_anomaly_detail.csv 路径")
    parser.add_argument("--step4-scored", default=DEFAULT_STEP4_SCORED, help="STEP4 threshold_scan_scored.csv 路径")
    parser.add_argument("--host", default="127.0.0.1", help="Dash host")
    parser.add_argument("--port", type=int, default=8050, help="Dash port")
    parser.add_argument("--max-points", type=int, default=0, help="默认最大绘制点数；0表示显示全部点，不采样")
    args = parser.parse_args()

    detail_raw = read_csv_auto(args.step2_detail)
    scored_raw = read_csv_auto(args.step4_scored)
    detail_df = normalize_detail_df(detail_raw)
    scored_df = normalize_scored_df(scored_raw)

    print("=" * 72)
    print("Dash：STEP4 阈值结果功率-损耗散点图")
    print("=" * 72)
    print(f"STEP2明细: {args.step2_detail}")
    print(f"STEP4阈值明细: {args.step4_scored}")
    print(f"STEP2点数: {len(detail_df):,}")
    print(f"STEP4记录数: {len(scored_df):,}")
    print(f"可选阈值: {sorted(scored_df['阈值'].dropna().unique())}")
    print(f"启动地址: http://{args.host}:{args.port}")

    app = create_app(detail_df=detail_df, scored_df=scored_df, max_points_default=args.max_points)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()