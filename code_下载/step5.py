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


DEFAULT_STEP2_DETAIL = r"场站数据\LGXRFD\step2-异常检测\station_anomaly_detail.csv"
DEFAULT_STEP4_SCORED = r"场站数据\LGXRFD\step4-阈值分析\threshold_scan_scored.csv"

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
    out["是否纳入汇总统计"] = to_bool_series(df[include_col]) if include_col else True
    out["判异规则"] = df[rule_col].astype(str) if rule_col else ""
    out = out.dropna(subset=["时间"]).sort_values("时间").reset_index(drop=True)
    return out


def normalize_scored_df(df: pd.DataFrame) -> pd.DataFrame:
    threshold_col = find_col(df, ["阈值", "threshold"], desc="阈值")
    seg_id_col = find_col(df, ["分段ID", "segment_id"], desc="分段ID")
    start_col = find_col(df, ["开始时间", "start_time"], desc="开始时间")
    end_col = find_col(df, ["结束时间", "end_time"], desc="结束时间")
    decision_col = find_col(df, ["最终处理方式", "final_decision"], desc="最终处理方式")
    class_col = find_col(df, ["分段类别", "segment_class"], required=False)
    repeat_power_col = find_col(df, ["重复风机功率之和MW", "重复功率之和MW", "repeat_power_mw"], required=False)
    freeze_cnt_col = find_col(df, ["冻结风机数", "重复风机数", "freeze_fan_count"], required=False)
    raw_ratio_col = find_col(df, ["原异常占比", "raw_anomaly_ratio"], required=False)
    corr_ratio_col = find_col(df, ["修正后异常占比", "corr_anomaly_ratio"], required=False)
    raw_sev_col = find_col(df, ["原异常程度总和", "raw_severity_sum"], required=False)
    corr_sev_col = find_col(df, ["修正后异常程度总和", "corr_severity_sum"], required=False)

    out = pd.DataFrame()
    out["阈值"] = pd.to_numeric(df[threshold_col], errors="coerce")
    out["分段ID"] = df[seg_id_col]
    out["开始时间"] = pd.to_datetime(df[start_col], errors="coerce")
    out["结束时间"] = pd.to_datetime(df[end_col], errors="coerce")
    out["最终处理方式"] = df[decision_col].astype(str).fillna("未匹配分段")
    out["分段类别"] = df[class_col].astype(str) if class_col else ""
    out["分段重复功率之和MW"] = pd.to_numeric(df[repeat_power_col], errors="coerce") if repeat_power_col else np.nan
    out["冻结风机数"] = pd.to_numeric(df[freeze_cnt_col], errors="coerce") if freeze_cnt_col else np.nan
    out["原异常占比"] = pd.to_numeric(df[raw_ratio_col], errors="coerce") if raw_ratio_col else np.nan
    out["修正后异常占比"] = pd.to_numeric(df[corr_ratio_col], errors="coerce") if corr_ratio_col else np.nan
    out["原异常程度总和"] = pd.to_numeric(df[raw_sev_col], errors="coerce") if raw_sev_col else np.nan
    out["修正后异常程度总和"] = pd.to_numeric(df[corr_sev_col], errors="coerce") if corr_sev_col else np.nan
    out = out.dropna(subset=["阈值", "开始时间", "结束时间"]).sort_values(["阈值", "开始时间"]).reset_index(drop=True)
    return out


def build_threshold_options(scored_df: pd.DataFrame) -> List[Dict[str, str]]:
    vals = sorted(scored_df["阈值"].dropna().unique())
    return [{"label": f"{v:.4g}", "value": f"{v:.12g}"} for v in vals]


def build_category_options(values: pd.Series) -> List[Dict[str, str]]:
    items = [x for x in sorted(values.dropna().astype(str).unique()) if x and x.lower() != "nan"]
    return [{"label": x, "value": x} for x in items]


def make_interval_join(detail_df: pd.DataFrame, scored_one: pd.DataFrame) -> pd.DataFrame:
    """把逐分钟 STEP2 明细映射到某个阈值下的 STEP4 分段决策。"""
    detail = detail_df.sort_values("时间").copy()
    seg = scored_one.sort_values("开始时间").copy()
    if len(seg) == 0:
        out = detail.copy()
        out["分段ID"] = np.nan
        out["最终处理方式"] = "未匹配分段"
        out["分段类别"] = ""
        return out

    use_cols = [
        "分段ID", "开始时间", "结束时间", "最终处理方式", "分段类别",
        "分段重复功率之和MW", "冻结风机数", "原异常占比", "修正后异常占比",
        "原异常程度总和", "修正后异常程度总和",
    ]
    joined = pd.merge_asof(
        detail,
        seg[use_cols].sort_values("开始时间"),
        left_on="时间",
        right_on="开始时间",
        direction="backward",
    )
    in_seg = joined["结束时间"].notna() & (joined["时间"] <= joined["结束时间"])
    for col in ["分段ID", "最终处理方式", "分段类别", "分段重复功率之和MW", "冻结风机数", "原异常占比", "修正后异常占比", "原异常程度总和", "修正后异常程度总和"]:
        joined.loc[~in_seg, col] = np.nan
    joined["最终处理方式"] = joined["最终处理方式"].fillna("未匹配分段")
    joined["分段类别"] = joined["分段类别"].fillna("")
    return joined


def sample_for_plot(df: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    # 尽量按最终处理方式分层采样，避免小类别被完全淹没。
    parts = []
    groups = list(df.groupby("最终处理方式", dropna=False))
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
    decision_options = [{"label": x, "value": x} for x in DECISION_ORDER if x in set(scored_df["最终处理方式"].astype(str)) or x == "未匹配分段"]
    class_options = build_category_options(scored_df["分段类别"])

    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "padding": "16px"},
        children=[
            html.H2("STEP4 阈值结果：风机功率-损耗散点图"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "220px 220px 260px 260px", "gap": "12px", "marginBottom": "12px"},
                children=[
                    html.Div([
                        html.Label("阈值"),
                        dcc.Dropdown(id="threshold-dd", options=threshold_options, value=default_threshold, clearable=False),
                    ]),
                    html.Div([
                        html.Label("功率/损耗口径"),
                        dcc.Dropdown(
                            id="mode-dd",
                            options=[
                                {"label": "修正后口径", "value": "corr"},
                                {"label": "原始口径", "value": "raw"},
                            ],
                            value="corr",
                            clearable=False,
                        ),
                    ]),
                    html.Div([
                        html.Label("最终处理方式"),
                        dcc.Dropdown(id="decision-dd", options=decision_options, value=[x["value"] for x in decision_options if x["value"] != "未匹配分段"], multi=True),
                    ]),
                    html.Div([
                        html.Label("分段类别"),
                        dcc.Dropdown(id="class-dd", options=class_options, value=[x["value"] for x in class_options], multi=True),
                    ]),
                    html.Div([
                        html.Label("是否纳入汇总统计"),
                        dcc.Dropdown(
                            id="include-dd",
                            options=[
                                {"label": "只显示纳入统计点", "value": "yes"},
                                {"label": "全部显示", "value": "all"},
                            ],
                            value="yes",
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
                                {"label": "异常程度", "value": "异常程度"},
                                {"label": "原异常程度", "value": "原异常程度"},
                            ],
                            value="none",
                            clearable=False,
                        ),
                    ]),
                    html.Div([
                        html.Label("最大显示点数"),
                        dcc.Input(id="max-points-input", type="number", value=max_points_default, min=1000, step=1000, style={"width": "100%"}),
                    ]),
                ],
            ),
            html.Div(id="metrics-box", style={"marginBottom": "10px", "fontSize": "15px"}),
            dcc.Graph(id="scatter-graph", style={"height": "760px"}),
        ],
    )

    @app.callback(
        Output("scatter-graph", "figure"),
        Output("metrics-box", "children"),
        Input("threshold-dd", "value"),
        Input("mode-dd", "value"),
        Input("decision-dd", "value"),
        Input("class-dd", "value"),
        Input("include-dd", "value"),
        Input("size-dd", "value"),
        Input("max-points-input", "value"),
    )
    def update_graph(threshold_value, mode, decisions, classes, include_mode, size_col, max_points):
        if threshold_value is None:
            return px.scatter(title="没有可用阈值"), "未读取到阈值。"
        th = float(threshold_value)
        scored_one = scored_df[np.isclose(scored_df["阈值"].astype(float), th)].copy()
        plot_df = make_interval_join(detail_df, scored_one)

        if include_mode == "yes":
            plot_df = plot_df[plot_df["是否纳入汇总统计"]].copy()
        if decisions:
            plot_df = plot_df[plot_df["最终处理方式"].isin(decisions)].copy()
        if classes:
            plot_df = plot_df[plot_df["分段类别"].isin(classes)].copy()

        if mode == "raw":
            x_col = "原始风机功率之和MW"
            y_col = "原损耗MW"
            title_mode = "原始口径"
            anomaly_col = "原是否异常"
            sev_col = "原异常程度"
        else:
            x_col = "修正后风机功率之和MW"
            y_col = "修正后损耗MW"
            title_mode = "修正后口径"
            anomaly_col = "是否异常"
            sev_col = "异常程度"

        plot_df = plot_df[np.isfinite(plot_df[x_col]) & np.isfinite(plot_df[y_col])].copy()
        plot_df["当前口径是否异常"] = plot_df[anomaly_col]
        plot_df["当前口径异常程度"] = plot_df[sev_col]
        plot_df["悬浮时间"] = plot_df["时间"].dt.strftime("%Y-%m-%d %H:%M")

        total_before_sample = len(plot_df)
        max_points = int(max_points or max_points_default)
        plot_df = sample_for_plot(plot_df, max_points=max_points)

        actual_size = None if size_col == "none" or size_col not in plot_df.columns else size_col
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color="最终处理方式",
            category_orders={"最终处理方式": DECISION_ORDER},
            size=actual_size,
            hover_data={
                "悬浮时间": True,
                "分段ID": True,
                "分段类别": True,
                "冻结风机数": True,
                "重复风机功率扣减量MW": ":.4f",
                "当前口径是否异常": True,
                "当前口径异常程度": ":.4f",
                "判异规则": True,
                x_col: ":.4f",
                y_col: ":.4f",
            },
            title=f"阈值={th:.4g} | {title_mode} | 横坐标=风机功率之和，纵坐标=损耗",
        )
        fig.update_traces(marker=dict(opacity=0.55), selector=dict(mode="markers"))
        fig.update_layout(
            template="plotly_white",
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend_title="最终处理方式",
        )

        counts = plot_df["最终处理方式"].value_counts().to_dict()
        count_text = " | ".join([f"{k}: {v:,}" for k, v in counts.items()])
        metrics = (
            f"阈值={th:.4g} | {title_mode} | 筛选后点数={total_before_sample:,} | "
            f"实际绘制点数={len(plot_df):,} | {count_text}"
        )
        return fig, metrics

    return app


def main():
    parser = argparse.ArgumentParser(description="Dash：不同阈值下按最终处理方式显示风机功率-损耗散点图")
    parser.add_argument("--step2-detail", default=DEFAULT_STEP2_DETAIL, help="STEP2 station_anomaly_detail.csv 路径")
    parser.add_argument("--step4-scored", default=DEFAULT_STEP4_SCORED, help="STEP4 threshold_scan_scored.csv 路径")
    parser.add_argument("--host", default="127.0.0.1", help="Dash host")
    parser.add_argument("--port", type=int, default=8050, help="Dash port")
    parser.add_argument("--max-points", type=int, default=60000, help="默认最大绘制点数，避免浏览器卡顿")
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
