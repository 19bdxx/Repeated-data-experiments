# 风电场重复数据分析、异常修正与预测实验 — 项目报告

> 生成时间：2026-05-04  
> 适用仓库：19bdxx/Repeated-data-experiments  
> 报告版本：v1.0

---

## 目录

1. [项目背景](#1-项目背景)
2. [数据问题与业务假设](#2-数据问题与业务假设)
3. [数据处理总流程](#3-数据处理总流程)
4. [各 STEP 代码功能解析](#4-各-step-代码功能解析)
5. [关键字段与标签说明](#5-关键字段与标签说明)
6. [单位口径说明](#6-单位口径说明)
7. [异常识别与修正逻辑](#7-异常识别与修正逻辑)
8. [15min 数据集构造逻辑](#8-15min-数据集构造逻辑)
9. [1min 数据集实验逻辑](#9-1min-数据集实验逻辑)
10. [预测实验设计](#10-预测实验设计)
11. [模型与特征组说明](#11-模型与特征组说明)
12. [损耗模型使用说明](#12-损耗模型使用说明)
13. [已有实验发现](#13-已有实验发现)
14. [潜在风险与代码审计建议](#14-潜在风险与代码审计建议)
15. [后续优化方向](#15-后续优化方向)

---

## 1. 项目背景

本项目围绕江苏地区多个风电场（LGXRFD、JMZSFD 等）的 SCADA / 场站分钟级数据开展端到端的数据清洗、异常识别、异常修正和功率预测实验。

### 场站基本信息

| 场站代号  | 全站额定功率 | 单机额定功率  | 风机台数    |
|-----------|-------------|--------------|------------|
| LGXRFD    | 约 400 MW   | 约 4 MW（4000 kW）| 100 台   |
| JMZSFD    | 约 300 MW   | 约 6~7 MW    | 若干台      |

### 数据来源

- 原始数据：月度分场站 SCADA 点位数据（CSV 格式）
- 每列为一个测点（ACTIVE_POWER_#n、STATUS_#n、WINDSPEED_#n 等）
- 原始时间分辨率：1分钟

### 项目起因

初步分析发现，原始数据中存在以下复杂问题：

- 风机级有功功率**大量连续重复（冻结）**现象
- 部分重复为正功率冻结（真实异常），部分为零/负功率重复（正常停机）
- 全站存在限电、状态异常、负功率等多种复杂工况

**核心业务判断**：正功率重复（`ACTIVE_POWER > 0` 的连续相同值段）更可能代表传感器冻结或数据采集异常，需重点识别和处理；而零功率/负功率重复通常是正常停机或低功率工况，不应作为主要异常处理依据。

因此，**项目从"联合检测所有重复"调整为"仅重点处理正功率重复"**，这是贯穿全部代码的核心设计原则。

---

## 2. 数据问题与业务假设

### 2.1 已识别的数据问题

| 问题类型 | 描述 | 影响 |
|---------|------|------|
| 正功率冻结 | 风机功率在非合理范围内连续保持同一正值 | 直接导致功率数据失真 |
| 负功率记录 | `ACTIVE_POWER_#n < 0`，物理上不合理 | 应按 0 处理或标记为异常 |
| 限电工况 | `0 < LIMIT_POWER < 0.95 × 额定功率` | 不代表真实发电能力，需剔除 |
| 状态码非法 | 状态码不在允许集合内 | 状态特征不可用 |
| 时间戳秒数非零 | 部分时间戳含秒级偏移 | 需按规则修正到整分钟 |
| 缺失时间戳 | 时间序列中存在分钟级断点 | 影响连续时序建模 |
| 重复时间戳 | 同一时刻出现多行记录 | 需去重处理 |
| 超限功率 | 风机功率超过场站单机上限（LGXRFD: 4500 kW；JMZSFD: 7000 kW） | 视为非法值，不参与均值或建模 |

### 2.2 业务假设

1. **正功率重复是异常的主要来源**：连续 ≥4 分钟保持相同正功率，认为是潜在异常冻结。
2. **零/负功率重复是正常的**：停机、待机、限电工况下出现大量相同零/负功率是正常现象。
3. **修正策略**：对于可修正的正功率冻结分段，将重复期间的风机功率置为 0（反映停机状态）。
4. **DATA_USE_RESULT 决定建模可用性**：只有打上"可用"标签的时间戳才能进入后续建模数据集。
5. **限电时刻不用于建模**：限电工况下的功率不代表风机真实能力，应从训练和测试集中剔除。
6. **损耗 = 风机总功率 - 全站功率**：在允许范围内为正值（变压器/线路损耗）。
7. **公平预测比较**：不同预测方案（station_direct / turbine_sum）必须在相同时间戳集合上比较。

---

## 3. 数据处理总流程

```
原始月度 CSV（分场站点位数据）
    │
    ▼
STEP0-第一步：数据合并与时间戳清洗
    │   输出：场站主 CSV（LGXRFD_YYYYMM-YYYYMM.csv）
    │
    ▼
STEP0-第二步（仅正功率版）：连续重复检测
    │   输出：
    │     连续相同检测/每列连续重复检测结果.csv
    │     风机联合连续相同检测_仅正功率/联合重复值检测结果.xlsx
    │
    ▼
STEP1（仅正功率）：经验损耗模型（Dash 可视化）
    │   输出：step1-插值损耗模型-仅正功率/fit_empirical_quantile_bins.csv
    │         scheme_name = 站端功率分段模型 / 风机总功率分段模型
    │
    ▼
STEP2（仅正功率）：全站功率异常检测
    │   输出：station_anomaly_detail.csv（逐分钟异常标记）
    │
    ▼
STEP3（仅正功率）：构建完整时序冻结分段
    │   输出：freeze_count_segments.csv（分段汇总）
    │         freeze_count_segments_detail.csv（分段-风机明细）
    │
    ▼
STEP4（仅正功率）：分段阈值分析与最终处理决策
    │   输出：threshold_scan_scored.csv（分段-阈值-得分明细）
    │         threshold_scan_final_decision.csv（最终处理决策）
    │
    ▼
STEP5（离群点版）：可视化与点级离群标记（Dash）
    │   用于人工验证 STEP4 分段处理结果
    │
    ▼
STEP6（数据可用标签版）：生成分钟级修正宽表
    │   输出：fan_wide_correction_threshold_*.csv
    │   字段：DATA_USE_RESULT / FINAL_ACTIVE_POWER_#n / FINAL_FAN_POWER_SUM_MW 等
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
STEP7（单位状态版 v2）：聚合为15min建模表    STEP9：直接分钟级建模
    │   输出：<场站名>_15min_model_wide.csv        │
    │                                              │
    ▼                                              ▼
STEP8（v8 status_features）：15min预测实验    分钟级预测实验输出
    输出：metrics_summary.csv 等                  输出：metrics_summary.csv 等
```

---

## 4. 各 STEP 代码功能解析

### 4.1 STEP0 — 数据合并与重复检测

#### STEP0 第一步（`step0_第一步.py`）

**目的**：将原始月度数据按场站点位表提取、合并、清洗，生成场站主 CSV。

**输入**：
- 月度总表（每月一个 CSV，包含所有测点）
- 场站点位表（`<场站代号>_点位.xlsx`，含点位→英文列名映射）

**核心处理**：
1. 按场站点位表筛选相关测点列
2. 重命名点位 ID 为英文字段名（如 `ACTIVE_POWER_#1`）
3. 处理时间戳：秒数 ≤30 置0，秒数 >30 则分钟+1
4. 删除无效时间戳行和含空值行
5. 输出重复时间戳检查和缺失时间戳检查报告

**输出**：
- `<场站代号>_YYYYMM-YYYYMM.csv`（清洗后场站主文件）
- `_duplicate_timestamps.csv`（重复时间戳报告）
- `_missing_timestamps.csv`（缺失时间戳报告）
- `_process_summary.csv`（处理汇总）

**运行说明**：通常只需运行一次，不必重复运行。

---

#### STEP0 第二步（推荐版：`step0_第二步_仅正功率版.py`）

**目的**：对场站主 CSV 执行重复检测，但**只保留有功功率 > 0 的联合重复结果**。

**输入**：STEP0-第一步输出的场站主 CSV

**核心处理**：
1. 每列连续重复检测（站端功率、限电功率等）
2. 风机联合连续重复检测：对每台风机的 (STATUS, ACTIVE_POWER, REACTIVE_POWER, WINDSPEED, WINDDIRECTION) 五元组检测连续相同段
3. 筛选：只保留 `ACTIVE_POWER > 0` 的联合重复段
4. 输出全量与正功率对比统计

**输出**：
- `连续相同检测/每列连续重复检测结果.csv`
- `风机联合连续相同检测_仅正功率/联合重复值检测结果.xlsx`
- `风机联合连续相同检测_仅正功率/仅正功率筛选统计.csv`（含总量对比）
- `风机联合连续相同检测_仅正功率/仅正功率筛选统计_按风机.csv`

**关键参数**：`--min-repeat`（默认 4，最少连续几分钟才算重复）

**注意**：旧版 `step0_第二步.py` 保存了包含零/负功率在内的全部联合重复结果，**不再推荐用于后续分析**。

---

### 4.2 STEP1 — 经验损耗模型

**文件**：`step1_仅正功率.py`

**目的**：建立风机总功率 vs. 全站损耗的经验分位数模型，用 Dash 可视化辅助人工判断。

**输入**：场站主 CSV（含 `ACTIVE_POWER_STATION` 和所有 `ACTIVE_POWER_#n`）

**核心处理**：
1. 计算 `FAN_SUM_MW`（所有风机功率之和，kW → MW）
2. 计算 `LOSS_MW = FAN_SUM_MW - ACTIVE_POWER_STATION_MW`
3. 剔除硬异常（站端功率连续重复段 + 风机联合连续正功率重复段）
4. 建立两套分段经验分位数曲线：
   - `站端功率分段模型`（SCHEME_STATION_SEG）：以全站功率分段
   - `风机总功率分段模型`（SCHEME_FAN_SEG）：以风机总功率分段
5. 输出经验百分位带（P01~P99，默认使用 P05~P95 作为正常区间）
6. 超低功率区间（0~3MW）使用物理规则：`LOSS ∈ [0.99 × FAN_SUM, FAN_SUM]`

**输出**：
- `fit_empirical_quantile_bins.csv`（分位数曲线，含 `scheme_name` 列）
- `fit_model_summary.csv`（模型概要）

**重要**：STEP8/STEP9 使用损耗曲线时，**必须筛选 `scheme_name = 风机总功率分段模型`**，不能使用站端功率分段模型（详见 [第12章](#12-损耗模型使用说明) 和 [14.2 节](#142-损耗曲线筛选问题-️高优先级)）。

---

### 4.3 STEP2 — 全站功率异常检测

**文件**：`step2_仅正功率.py`

**目的**：利用 STEP1 的经验百分位带，对每分钟全站功率-损耗关系进行异常判断，区分原始口径和修正后口径。

**输入**：
- 场站主 CSV
- STEP1 输出的 `fit_empirical_quantile_bins.csv`
- 风机联合连续相同检测结果（仅正功率版）
- 站端连续相同检测结果

**核心处理**：
1. 构建原始口径：`FAN_SUM_raw = sum(ACTIVE_POWER_#n)`（kW → MW）
2. 构建修正后口径：`FAN_SUM_corr = FAN_SUM_raw - 当前分钟正功率重复风机之和`（将重复风机功率视为0）
3. 计算异常程度（severity）和偏离程度（deviation）
4. 判断原始/修正后损耗是否落在经验 P05~P95 带内

**输出**：
- `station_anomaly_detail.csv`（逐分钟：原始/修正口径 + 异常标记 + 异常程度）
- `station_anomaly_summary.csv`（汇总统计）
- `hard_anomaly_segments_used_step2.csv`（硬异常段清单）

**注意**：STEP2 的"偏离程度"比"异常程度"更适合用于连续改善指标的评估，不应只看"是否落在带外"。

---

### 4.4 STEP3 — 构建完整时序冻结分段

**文件**：`step3_仅正功率.py`

**目的**：将 STEP0 检测到的风机联合正功率重复记录，映射到完整的时序分段结构，确保**全部时间序列都有分段覆盖**（不仅仅是重复期间）。

**输入**：
- 风机联合重复检测结果（`联合重复值检测结果.xlsx`，仅正功率版）
- STEP2 的 `station_anomaly_detail.csv`

**核心处理**：
1. 从联合重复明细中提取各风机的正功率重复时间区间
2. 构建场站级分段，**覆盖完整时序**（数据开始到结束，重复段和非重复段均有分段）
3. 对每个分段统计：冻结风机数、重复功率值、分段类型
4. 分段类型分为：
   - `重复正功率>0分段`
   - `重复正功率=0，且冻结风机数=0`
   - `重复正功率=0，且冻结风机数>0`
5. 关联 STEP2 逐分钟异常判断结果，计算各分段的异常占比、异常程度等指标

**输出**：
- `freeze_count_segments.csv`（分段汇总，含冻结风机数、异常比例等）
- `freeze_count_segments_detail.csv`（分段-风机明细，含各风机是否参与重复）

**重要设计原则**：分段必须从数据**开始日期**覆盖到**结束日期**，而非只从第一次重复出现开始，否则数据开始到首次重复之间的时间段会没有分段标记，导致 STEP6 中出现大量"未匹配STEP3分段"。

---

### 4.5 STEP4 — 分段阈值分析与最终处理决策

**文件**：`step4_仅正功率.py`

**目的**：对每个重复冻结分段，通过扫描不同异常占比阈值，判断是否应该修正、保留或标记异常。

**输入**：STEP3 输出的 `freeze_count_segments.csv`

**核心处理**：
1. 扫描阈值范围（默认 0.05~0.50）
2. 对每个分段，在每个阈值下判断：
   - 修正前是否正常（异常分钟占比 < 阈值）
   - 修正后是否正常（修正后异常分钟占比 < 阈值）
3. 根据修正前后状态，将每个分段归入 8 种情况（CASE_1 ~ CASE_8）并映射到最终决策
4. 最终处理细分类型：
   - `重复正功率>0｜保留（不修正）`
   - `重复正功率>0｜修正后保留`
   - `重复正功率>0｜标记为异常分段`
   - `重复正功率=0且冻结风机数=0｜正常`
   - `重复正功率=0且冻结风机数=0｜标记为异常分段`

**输出**：
- `threshold_scan_scored.csv`（分段-阈值-得分明细）
- `threshold_scan_raw_profile.csv`（修正前原始特征汇总）
- `threshold_scan_repair_effect.csv`（修正作用分类汇总）
- `threshold_scan_final_decision.csv`（最终处理方式汇总）
- `threshold_recommendation.csv`（简单阈值推荐表）

---

### 4.6 STEP5 — 可视化与点级离群标记

**文件**：`step5_仅正功率_原始5类加修正后1类_离群点.py`

**目的**：使用 Dash 交互式可视化工具，展示 STEP4 分段处理结果，辅助人工验证。同时标记点级明显离群点。

**功能**：
- 展示风机总功率 vs. 损耗的散点图
- 按处理类型着色：标记为异常分段 / 修正后保留 / 保留（不修正）
- 点级离群标记：只在最终可用数据（`保留（不修正）`和`修正后保留`）中识别明显离群点
- 离群判断条件：
  - 全站功率 > `POINT_OUTLIER_POWER_MIN_MW`（默认 3 MW）
  - 异常程度 ≥ `POINT_OUTLIER_SEVERITY_THRESHOLD`（默认 2.0）
- 超低功率区间（0~3 MW）暂不标记离群

---

### 4.7 STEP6 — 生成分钟级修正宽表（核心步骤）

**文件**：`step6_仅正功率_数据可用标签版.py`

**目的**：将 STEP3/STEP4 的分段处理决策，落实到逐分钟逐风机的修正宽表，并为每个时间戳打上数据可用性标签。

**输入**：
- 场站主 CSV（含所有风机功率、状态码、风速）
- STEP3 的 `freeze_count_segments.csv` 和 `freeze_count_segments_detail.csv`
- STEP4 的 `threshold_scan_scored.csv`
- STEP2 的 `station_anomaly_detail.csv`（用于点级离群标记）

**核心处理**：

1. **风机级功率修正**（`FINAL_ACTIVE_POWER_#n`）：
   - 若分段类型为 `重复正功率>0｜修正后保留`，且该时刻风机功率 > 0，则 `FINAL_ACTIVE_POWER_#n = 0`
   - 其他情况：`FINAL_ACTIVE_POWER_#n = ACTIVE_POWER_#n`（保持原始值）

2. **处理结果标记**（`PROCESS_RESULT_#n`）：
   - 重复风机：直接使用 STEP4 最终处理细分类型
   - 同分段内未重复的其他风机：标记为 `重复正功率>0｜非重复风机`
   - 未匹配分段：标记为 `未匹配STEP3分段`

3. **场站级汇总**：
   - `RAW_FAN_POWER_SUM_MW`：原始风机功率之和（kW → MW）
   - `FINAL_FAN_POWER_SUM_MW`：修正后风机功率之和（kW → MW）
   - `FINAL_STATION_POWER_MW`：全站功率（MW，来自 `ACTIVE_POWER_STATION`）
   - `FINAL_LOSS_MW = FINAL_FAN_POWER_SUM_MW - FINAL_STATION_POWER_MW`

4. **DATA_USE_RESULT 标签**：
   - `可用`：时间戳可用于后续预测实验
   - `不可用`：时间戳不应用于建模
   
5. **点级离群标记**（`POINT_OUTLIER_RESULT`）：在可用数据中识别明显离群点

**输出**：`fan_wide_correction_threshold_<阈值>.csv`（完整分钟级宽表）

**单位注意**：
- `FINAL_ACTIVE_POWER_#n`：保持 kW（和原始数据一致）
- `RAW_FAN_POWER_SUM_MW` / `FINAL_FAN_POWER_SUM_MW` / `FINAL_LOSS_MW`：必须是 MW

---

### 4.8 STEP7 — 聚合为 15min 建模数据集

**文件**：`step7_build_15min_model_dataset_按场站名_单位状态版_v2.py`（推荐版本）

**目的**：将 STEP6 分钟级宽表聚合为 15min 均值，生成预测实验所需的建模数据集。

**输入**：STEP6 输出的 `fan_wide_correction_threshold_*.csv`

**核心处理**：

1. **15min 窗口右端点规则**：
   - `00:15` = `00:01 ~ 00:15` 的均值
   - `00:00` = 前一日 `23:46 ~ 00:00` 的均值

2. **可用分钟筛选**：只使用 `DATA_USE_RESULT == 可用` 的分钟

3. **窗口可用性**：可用分钟数 ≥ 14（默认 `--min-valid-minutes 14`）时，该 15min 窗口标记为可用

4. **风机功率均值**（输出为 MW）：
   - 负功率按 0 处理
   - 超过场站单机功率上限的值视为非法，不参与均值（LGXRFD: 4500 kW；JMZSFD: 7000 kW）
   - 最终输出 `FINAL_ACTIVE_POWER_#n_15MIN`（MW）= 均值 / 1000

5. **全站功率列均值**：`ACTIVE_POWER_STATION_15MIN`、`LIMIT_POWER_15MIN`（MW）

6. **派生列（基于 MW 级风机功率）**：
   - `FINAL_FAN_POWER_SUM_MW_15MIN = sum(FINAL_ACTIVE_POWER_#n_15MIN)`
   - `FINAL_LOSS_MW_15MIN = FINAL_FAN_POWER_SUM_MW_15MIN - ACTIVE_POWER_STATION_15MIN`
   - `FINAL_STATION_POWER_MW_15MIN = ACTIVE_POWER_STATION_15MIN`

7. **状态码特征**（v2 版新增）：
   - 风机级：`STATUS_LAST_#n_15MIN`、`STATUS_COUNT_<code>_#n_15MIN`
   - 场站级：`STATION_STATUS_COUNT/RATIO/LAST_COUNT/LAST_RATIO_<code>_15MIN`
   - 只统计允许状态码（LGXRFD: 0,1,2,3,8,11；JMZSFD: 0,1,2,3,4,5,6）

**输出**：`<场站名>_15min_model_wide.csv`

**自动查找文件风险**：若未通过 `--input-file` 显式指定，代码会自动选取**最近修改时间**的 STEP6 宽表。若历史文件未清理，可能误选旧版文件。建议始终使用 `--input-file` 显式指定（详见 [14.3 节](#143-自动文件查找风险-️中优先级) 和 [15.6 节](#156-自动选择输入文件的改进)）。

---

### 4.9 STEP8 — 15min 预测实验

**文件**：`step8_compare_forecast_15min_v8_status_features.py`（推荐版本）

**目的**：基于 STEP7 的 15min 建模数据，比较不同预测方案、模型和特征组的效果。

**输入**：
- STEP7 输出的 `<场站名>_15min_model_wide.csv`
- STEP1 输出的损耗曲线（可选，用于 turbine_sum_loss_q50 方案）

**核心筛选**：
1. `DATA_15MIN_USE_RESULT == 可用`
2. 剔除限电：`0 < LIMIT_POWER_15MIN < curtail_ratio × rated_power_mw`（默认 curtail_ratio=0.95）
3. `LIMIT_POWER_15MIN = 0` 视为非限电

**预测方案**：
- `station_direct`：直接预测 `ACTIVE_POWER_STATION_15MIN`
- `turbine_sum_raw`：逐台风机预测 `FINAL_ACTIVE_POWER_#n_15MIN`，再求和
- `turbine_sum_loss_q50`：逐台风机预测求和，再用 Q50 损耗曲线扣损

**输出**（全部在 `step8-预测实验/` 目录下）：
- `metrics_summary.csv`（各方案/模型/步长的精度指标）
- `predictions_all.csv`（所有预测值）
- `bias_diagnostics_summary.csv`（偏差诊断）
- `feature_gain_summary.csv`（特征重要性）
- `turbine_metrics_by_fan.csv`（单机预测精度汇总）
- `curtail_summary.csv`（限电剔除情况）
- `sample_summary.csv`（样本情况汇总）
- `loss_curve_used.csv`（使用的损耗曲线）

---

### 4.10 STEP9 — 1min 预测实验

**文件**：`step9_compare_forecast_1min.py`

**目的**：不聚合到 15min，直接在分钟级数据上做预测实验，验证分钟级建模是否优于 15min 聚合建模。

**输入**：STEP6 输出的分钟级宽表（自动查找包含 `data_use_unit_mw` 的 CSV）

**核心处理**：
- 内部将 `FINAL_ACTIVE_POWER_#n`（kW）转换为 MW
- 重新计算 `FINAL_FAN_POWER_SUM_MW` 和 `FINAL_LOSS_MW`（避免历史单位口径不一致）
- 预测步长：N=15,60,120,180 分钟
- 历史窗口：M=120 分钟（默认）

**注意**：STEP9 预测的是**未来某一分钟**的功率，与 STEP8 预测未来 15min 均值不完全等价。如需公平比较，需将分钟级预测结果再聚合成 15min 均值后对比。

---

### 4.11 辅助工具脚本

| 文件名 | 用途 |
|--------|------|
| `check_step6_step7_basic_quality.py` | 检查 STEP6/STEP7 输出的基本数据质量（行数、字段、单位范围等）|
| `check_step7_fan_power_range.py` | 专门检查 STEP7 风机功率是否在合理 MW 范围内 |
| `检查文件基础信息.py` | 查看任意 CSV/Excel 文件的行列信息 |
| `状态码_风速功率关系分析.py` | 分析各状态码下的风速-功率分布 |
| `联合重复区段状态码分析.py` | 分析正功率重复区段内的状态码分布 |

---

## 5. 关键字段与标签说明

### 5.1 STEP6 输出字段

| 字段名 | 单位 | 说明 |
|--------|------|------|
| `timestamp` | — | 分钟级时间戳 |
| `DATA_USE_RESULT` | — | `可用` / `不可用`，决定是否进入建模 |
| `DATA_USE_REASON` | — | 不可用原因说明 |
| `DATA_USE_SCENE` | — | 时刻级场景类型（由所有风机 PROCESS_RESULT 提炼） |
| `ACTIVE_POWER_STATION` | MW | 全站功率（原始） |
| `LIMIT_POWER` | MW | 限电功率 |
| `RAW_FAN_POWER_SUM_MW` | MW | 原始风机功率之和 |
| `FINAL_FAN_POWER_SUM_MW` | MW | 修正后风机功率之和 |
| `FINAL_LOSS_MW` | MW | 修正后损耗 |
| `FINAL_STATION_POWER_MW` | MW | 全站功率（等同 ACTIVE_POWER_STATION） |
| `POINT_OUTLIER_RESULT` | — | `正常点` / `点级明显离群` |
| `POINT_OUTLIER_REASON` | — | 离群原因说明 |
| `STATUS_#n` | — | 第 n 台风机状态码（原始） |
| `ACTIVE_POWER_#n` | kW | 第 n 台风机有功功率（原始） |
| `WINDSPEED_#n` | m/s | 第 n 台风机风速（原始） |
| `FINAL_ACTIVE_POWER_#n` | kW | 第 n 台风机修正后有功功率 |
| `PROCESS_RESULT_#n` | — | 第 n 台风机处理结果细分类型 |

### 5.2 PROCESS_RESULT 细分标签

| 标签值 | 含义 |
|--------|------|
| `重复正功率>0｜保留（不修正）` | 该风机本分段为正功率重复，但修正后更差，保留原始值 |
| `重复正功率>0｜修正后保留` | 该风机本分段为正功率重复，修正（置0）后改善，采用修正值 |
| `重复正功率>0｜标记为异常分段` | 该风机本分段为正功率重复，修正无效，整段标记为异常 |
| `重复正功率>0｜非重复风机` | 该风机不参与本分段的重复，但同一分段内其他风机有正功率重复 |
| `重复正功率=0且冻结风机数=0｜正常` | 场站级无正功率重复，也无冻结风机，正常时段 |
| `重复正功率=0且冻结风机数=0｜标记为异常分段` | 场站级无正功率重复，但被标记为异常 |
| `未匹配STEP3分段` | 该时间戳未被任何 STEP3 分段覆盖（属于异常情况，应排查 STEP3） |

### 5.3 DATA_USE_RESULT 可用性判断逻辑

```
DATA_USE_RESULT = 可用，当且仅当：
  1. 没有被标记为"标记为异常分段"的风机
  2. 没有被标记为"点级明显离群"
  3. 时间戳有效（非 NaT）
否则 DATA_USE_RESULT = 不可用
```

### 5.4 STEP7 15min 输出字段（新增）

| 字段名 | 单位 | 说明 |
|--------|------|------|
| `DATA_15MIN_USE_RESULT` | — | `可用` / `不可用` |
| `VALID_MINUTE_COUNT_15MIN` | — | 窗口内可用分钟数 |
| `ACTIVE_POWER_STATION_15MIN` | MW | 全站功率 15min 均值 |
| `LIMIT_POWER_15MIN` | MW | 限电功率 15min 均值 |
| `FINAL_ACTIVE_POWER_#n_15MIN` | MW | 第 n 台风机修正后功率 15min 均值 |
| `FINAL_FAN_POWER_SUM_MW_15MIN` | MW | 风机总功率 15min 均值 |
| `FINAL_LOSS_MW_15MIN` | MW | 损耗 15min 均值 |
| `STATUS_LAST_#n_15MIN` | — | 第 n 台风机窗口内最后一个合法状态码 |
| `STATUS_COUNT_<code>_#n_15MIN` | — | 第 n 台风机状态码 code 出现的分钟数 |
| `STATION_STATUS_RATIO_<code>_15MIN` | — | 场站级状态码 code 占所有有效点的比例 |
| `STATION_STATUS_LAST_RATIO_<code>_15MIN` | — | 场站级末尾状态码 code 的风机占比 |
| `STATION_STATUS_VALID_POINT_COUNT_15MIN` | — | 场站级有效状态码点数 |

---

## 6. 单位口径说明

本项目最容易出错的地方是功率单位。以下是各阶段的单位规范：

### 6.1 单位规范汇总表

| 字段 / 阶段 | 单位 | 备注 |
|------------|------|------|
| `ACTIVE_POWER_#n`（原始） | kW | SCADA 原始数据单位 |
| `ACTIVE_POWER_STATION`（原始） | MW | 全站功率通常已是 MW |
| `STEP6 FINAL_ACTIVE_POWER_#n` | kW | 保持与原始一致 |
| `STEP6 RAW_FAN_POWER_SUM_MW` | MW | = sum(kW) / 1000 |
| `STEP6 FINAL_FAN_POWER_SUM_MW` | MW | = sum(kW) / 1000 |
| `STEP6 FINAL_LOSS_MW` | MW | MW - MW |
| `STEP7 FINAL_ACTIVE_POWER_#n_15MIN` | MW | kW 均值 → MW（除以1000）|
| `STEP7 FINAL_FAN_POWER_SUM_MW_15MIN` | MW | = sum(MW 风机均值) |
| `STEP7 FINAL_LOSS_MW_15MIN` | MW | MW - MW |
| `STEP7 ACTIVE_POWER_STATION_15MIN` | MW | MW 均值 |
| `STEP8/STEP9 建模目标和特征` | MW | 内部统一 MW |

### 6.2 转换关键点

- **STEP6**：汇总 `FINAL_FAN_POWER_SUM_MW` 时，需将风机 kW 除以 1000 再求和。
- **STEP7**：`safe_mean_valid_fan_power_mw_with_upper` 函数中，`source_unit="kW"` 时，均值乘以 `1/1000`（`fan_unit_to_mw_scale("kW") = 0.001`）。
- **STEP9**：内部重新从 STEP6 的 kW 转换为 MW，不依赖历史计算的 `FINAL_FAN_POWER_SUM_MW`。

### 6.3 常见单位错误

| 错误类型 | 后果 |
|---------|------|
| 忘记除以1000（kW→MW） | 风机总功率约为全站功率的1000倍，损耗曲线完全失效 |
| 重复除以1000 | 风机总功率约为全站功率的1/1000，损耗为负数且极大 |
| 混用站端损耗模型 vs 风机总功率损耗模型 | 损耗预测系统性偏差 |

---

## 7. 异常识别与修正逻辑

### 7.1 正功率重复异常识别

**识别条件**（STEP0）：
- 风机 (STATUS, ACTIVE_POWER, REACTIVE_POWER, WINDSPEED, WINDDIRECTION) 五元组连续相同
- 持续时间 ≥ 4 分钟（可配置）
- 有功功率 > 0（正功率才考虑是异常冻结）

**为什么只看正功率**：
- 停机/待机时（功率≤0），连续相同值是正常行为
- 正功率下传感器冻结才需要修正

### 7.2 分段修正策略（STEP4）

每个正功率重复分段，通过以下 8 种情况判断最终处理方式：

| 情况 | 修正前 | 修正后 | 最终决策 |
|------|--------|--------|---------|
| 情况1 | 正常 | 异常 | 保留（不修正）|
| 情况2 | 异常 | 正常 | 修正后保留 |
| 情况3 | 正常 | 正常+改善 | 修正后保留 |
| 情况4 | 正常 | 正常+不变 | 保留（不修正）|
| 情况5 | 正常 | 正常+恶化 | 保留（不修正）|
| 情况6 | 异常 | 异常+改善 | 标记为异常分段 |
| 情况7 | 异常 | 异常+不变 | 标记为异常分段 |
| 情况8 | 异常 | 异常+恶化 | 标记为异常分段 |

"修正"的含义：将正功率重复期间的重复风机功率置为 0（模拟停机状态）。

### 7.3 点级离群标记（STEP6）

在以下条件同时满足时，将某分钟标记为"点级明显离群"：
- 数据场景为"保留（不修正）"或"重复正功率=0且冻结风机数=0｜正常"
- 全站功率 > 3 MW（超低功率区间暂不处理）
- 异常程度 ≥ 2.0

这保证了点级离群只在"本应可用"的数据中识别，不会与"标记为异常分段"的处理重叠。

---

## 8. 15min 数据集构造逻辑

### 8.1 时间窗口定义

采用**右端点规则**：
- 15min 时间戳 `T` 代表 `(T-14min) ~ T` 这 15 分钟的数据
- 例：`00:15` = `00:01 ~ 00:15`；`00:00` = 前一日 `23:46 ~ 00:00`

### 8.2 可用性判断

```
窗口可用 ⟺ 窗口内可用分钟数(DATA_USE_RESULT=可用) ≥ 14
```

分母为**实际可用分钟数**（非固定15），避免稀疏可用分钟拉低均值。

### 8.3 风机功率处理规则（均值计算前）

```python
# 1. 负功率按0处理
power = max(power, 0)

# 2. 超过场站上限的值不参与均值
if power > upper_limit_kw:
    power = NaN  # 不参与均值计算
    
# 3. kW → MW（最终输出）
power_mw = mean(valid_power_kw) / 1000
```

### 8.4 状态码聚合规则（v2 版）

- 只统计允许状态码集合内的状态码
- `STATUS_LAST_#n_15MIN`：窗口内时间序列上最后一个合法状态码
- `STATUS_COUNT_<code>_#n_15MIN`：该风机在窗口内该状态码出现的分钟数
- 场站级统计基于所有风机的有效状态码点

---

## 9. 1min 数据集实验逻辑

### 9.1 目的

验证两个假设：
1. 分钟级直接建模是否比 15min 聚合建模更有预测价值
2. 状态码特征在分钟级是否更有效（相比 15min 聚合后的状态统计）

### 9.2 输入处理

STEP9 从 STEP6 分钟级宽表读取，**在代码内部重新计算**：
- `FINAL_FAN_POWER_SUM_MW`（避免历史文件单位不一致）
- `FINAL_LOSS_MW`

这样可以保证 STEP9 的单位计算是正确的，不依赖 STEP6 的历史输出。

### 9.3 预测方式与 STEP8 的区别

| 维度 | STEP8（15min） | STEP9（1min） |
|------|---------------|--------------|
| 目标变量 | 未来15min均值功率 | 未来某一分钟功率 |
| 历史特征 | 过去 M×15min = 4h 的15min均值 | 过去 M=120 分钟的分钟值 |
| 预测步长 | N=1,4,8,12 个15min（15min~3h）| N=15,60,120,180 分钟 |

**注意**：STEP9 预测的是单分钟功率，不等价于 STEP8 预测的15min均值。若要公平比较，需将 STEP9 的分钟级预测结果再聚合为15min均值。

---

## 10. 预测实验设计

### 10.1 实验维度

| 维度 | 选项 |
|------|------|
| 模型 | lightgbm / xgboost / random_forest / ridge |
| 特征组 | power_only / power_status |
| 预测方案 | station_direct / turbine_sum_raw / turbine_sum_loss_q50 |
| 步长（STEP8） | N=1,4,8,12（对应15min, 1h, 2h, 3h）|
| 步长（STEP9） | N=15,60,120,180（分钟）|

### 10.2 三种预测方案详解

#### 方案1：station_direct（全站直接预测）
```
历史全站功率 [+ 历史损耗 + 状态特征] → 未来全站功率
```
- 最简单直接，但不分解到单机
- 无法利用单机差异化信息

#### 方案2：turbine_sum_raw（单机预测求和）
```
每台风机历史功率 [+ 状态码] → 每台风机未来功率预测
→ 所有风机预测值求和 → 风机总功率预测
（不扣损耗）
```
- 利用了单机级信息
- 不考虑变压器/线路损耗

#### 方案3：turbine_sum_loss_q50（单机预测求和 + 损耗修正）
```
每台风机历史功率 [+ 状态码] → 每台风机未来功率预测
→ 所有风机预测值求和 = sum_fan_pred
→ loss_q50 = 用风机总功率分段模型的Q50损耗曲线插值
→ 全站预测 = sum_fan_pred - loss_q50
```
- 利用了单机信息，并修正了系统性损耗偏差
- 但扣损耗不一定总是更好（见第13章）

### 10.3 样本构造原则

- 样本以15min时间戳为单位（STEP8）
- 训练集/测试集按时间顺序切分（无随机打乱，避免数据泄露）
- `station_direct` 和 `turbine_sum` 方案必须使用**相同的时间戳集合**进行公平比较

### 10.4 特征构造

对于每个预测时间戳 `T`，使用 `T-N` 时刻及之前 `M` 个时间步的历史作为特征（预测 `T` 时刻的值）。

---

## 11. 模型与特征组说明

### 11.1 机器学习模型

| 模型 | 说明 | 适用场景 |
|------|------|---------|
| `ridge` | 岭回归（L2正则线性模型） | 基线模型，特征较少时稳定 |
| `lightgbm` | LightGBM 梯度提升 | 速度快，适合大量特征 |
| `xgboost` | XGBoost 梯度提升 | 精度高，适合中等特征量 |
| `random_forest` | 随机森林 | 鲁棒性好，抗过拟合 |

Ridge 使用 `StandardScaler` + `Ridge` 管道，树模型直接使用各自库。

### 11.2 特征组

#### power_only（纯功率特征）

**station_direct 方案**：
- 过去 M 个时间步的 `ACTIVE_POWER_STATION_15MIN`
- 过去 M 个时间步的 `FINAL_FAN_POWER_SUM_MW_15MIN`
- 过去 M 个时间步的 `FINAL_LOSS_MW_15MIN`

**turbine_sum 方案**：
- 过去 M 个时间步的 `FINAL_ACTIVE_POWER_#n_15MIN`（逐台风机）

#### power_status（功率 + 状态码特征）

**station_direct 方案**：
- power_only 所有特征
- `STATION_STATUS_RATIO_<code>_15MIN`（场站级状态比例）
- `STATION_STATUS_LAST_RATIO_<code>_15MIN`（场站级末尾状态比例）
- 不使用 COUNT 类特征（避免和可用分钟数强相关）

**turbine_sum 方案（树模型）**：
- 逐台风机功率历史
- `STATUS_#n`（分钟级，或 `STATUS_LAST_#n_15MIN`）
- Ridge 暂不加状态码（状态码是类别特征，数值大小未必有序，对线性模型不友好）

---

## 12. 损耗模型使用说明

### 12.1 损耗的定义

```
LOSS = FAN_SUM_MW - ACTIVE_POWER_STATION_MW
```

包含变压器、集电线路等所有电气损耗，通常为正值（少数负值为测量误差）。

### 12.2 损耗曲线来源

STEP1 输出的 `fit_empirical_quantile_bins.csv` 中包含：
- `scheme_name = 站端功率分段模型`（以全站功率分组）
- `scheme_name = 风机总功率分段模型`（以风机总功率分组）

**关键规则**：预测方案 `turbine_sum_loss_q50` 中，**必须使用 `风机总功率分段模型`**。

原因：此方案是用风机总功率预测值来查找对应损耗，分段依据应与输入变量一致（风机总功率），不能用站端功率分组的损耗曲线。

### 12.3 Q50 损耗曲线的使用

```python
# 根据预测的风机总功率，插值查找对应的Q50损耗
loss_q50 = interp(sum_fan_pred, 损耗曲线的P_med, 损耗曲线的Q50)
station_pred = sum_fan_pred - loss_q50
```

### 12.4 损耗修正的适用条件

诊断发现：损耗修正**并不总是有效**：
- N=1（预测15min）：`turbine_sum_raw` 存在约 3MW 系统性高估 → 扣Q50损耗有效
- N≥4（预测1h及以上）：`turbine_sum_raw` 已基本无偏或略低估 → 扣Q50损耗反而造成低估

这意味着损耗应该做动态校准，而非固定使用 Q50。

---

## 13. 已有实验发现

### 13.1 15min 预测实验结论

| 步长 | 最优方案 | 说明 |
|------|---------|------|
| 未来15min（N=1） | `station_direct` 通常最好 | 短步长全站直接预测优势明显 |
| 未来1h（N=4） | `station_direct` ≈ `turbine_sum_raw` | 两种方案接近 |
| 未来2h（N=8） | `turbine_sum_raw` 有时略优 | 长步长单机汇总开始占优 |
| 未来3h（N=12） | `turbine_sum_raw` 有时略优 | 同上 |

### 13.2 损耗修正诊断

```
N=1（15min预测）：
  turbine_sum_raw 高估约 3MW → 扣Q50损耗后改善
  
N≥4（1h及以上预测）：
  turbine_sum_raw 基本无偏或略低估 → 扣Q50损耗后反而低估（变差）
```

**结论**：不能简单认为"扣损耗一定更好"。需要根据步长动态决策，或做损耗校准。

### 13.3 单机预测诊断

整体上 100 台风机预测效果合理，但以下风机在部分步长下误差较高，值得重点关注：

- 风机 #48、#85、#81、#68、#74

这些风机可能存在：
- 较多历史异常/停机工况
- 状态码分布异常
- 功率曲线偏离

建议对这些风机的状态码分布、功率曲线和停机记录做进一步排查。

### 13.4 特征组比较

- `power_status` vs `power_only`：在部分场景下，状态码特征能改善预测效果，但效果依赖于模型类型（树模型 > 线性模型）和预测步长。

---

## 14. 潜在风险与代码审计建议

### 14.1 单位问题 ⚠️（高优先级）

**风险点**：
- STEP6 中 `add_station_use_labels` 函数：`RAW_FAN_POWER_SUM_MW` 和 `FINAL_FAN_POWER_SUM_MW` 必须确认已将 kW 除以 1000
- STEP7 `safe_mean_valid_fan_power_mw_with_upper`：确认 `source_unit="kW"` 时有 `/1000` 转换
- STEP9 内部重算 FAN_SUM 时：确认已将 kW→MW

**验证方法**：
- LGXRFD（100台，单机4MW）：`FINAL_FAN_POWER_SUM_MW` 应在 0~400 范围内（异常情况下最多 ±10%）
- 若出现数万 MW 范围，说明单位未转换
- 可运行 `check_step6_step7_basic_quality.py` 和 `check_step7_fan_power_range.py` 检查

### 14.2 损耗曲线筛选问题 ⚠️（高优先级）

**风险点**：STEP8/STEP9 加载 STEP1 损耗曲线时，文件中可能同时包含两种 `scheme_name`（参见 [第12章](#12-损耗模型使用说明)）。

**正确做法**：
```python
loss_df = loss_df[loss_df["scheme_name"] == "风机总功率分段模型"]
```

STEP8 v8 版本已默认配置 `DEFAULT_LOSS_MODEL_TYPE = "风机总功率分段模型"`，但需要确认代码中筛选逻辑正确生效。

### 14.3 自动文件查找风险 ⚠️（中优先级）

**风险点**：STEP7 和 STEP9 都有自动查找最近 STEP6 输出文件的逻辑（按修改时间，参见 [4.8 节](#48-step7--聚合为-15min-建模数据集)）。

**具体风险**：
- 若同一场站目录下存在多个版本的 STEP6 输出（用不同阈值、不同参数生成），自动选择可能选到旧版本
- STEP7 的 `find_latest_step6_file` 按 `st_mtime` 排序，但旧文件若被打开浏览过也会更新 `atime`（某些系统设置）

**建议**：始终通过 `--input-file` 显式指定文件路径，不依赖自动查找逻辑。

### 14.4 时间窗口连续性 ⚠️（中优先级）

**风险点**：STEP8/STEP9 构造历史特征窗口时，若样本之间存在时间缺口（如缺少某些15min），相邻样本可能在时间上不连续。

**影响**：若代码简单按行号偏移构造历史特征（而不检查时间连续性），会误用跨越缺口的数据作为"连续历史"。

**建议**：确认 STEP8/STEP9 样本构造时，历史窗口内各时间戳是严格连续的（不存在跳跃）。

### 14.5 限电剔除逻辑 ⚠️（中优先级）

**正确规则**：
```python
# 限电条件（需剔除）
is_curtail = (LIMIT_POWER > 0) & (LIMIT_POWER < curtail_ratio * rated_power_mw)

# 非限电（保留）
# LIMIT_POWER = 0 → 视为非限电（无限制）
# LIMIT_POWER >= curtail_ratio * rated_power → 视为非限电（限制值很高）
```

**风险**：若误将 `LIMIT_POWER = 0` 视为限电，会大量剔除正常时段数据，导致训练样本极少。

### 14.6 样本公平性 ⚠️（中优先级）

`station_direct` 和 `turbine_sum` 方案的可用样本可能不完全相同（因为两者对缺失风机数据的处理逻辑不同）。若样本量差异较大，精度比较会失去公平性。

**建议**：在对比前，先统计两方案的样本 timestamp 集合，确认重叠度；若必要，做"共同时间戳"子集对比。

### 14.7 STEP3 分段覆盖完整性 ⚠️（低优先级）

**风险**：STEP3 若只从第一次正功率重复出现时刻开始建立分段，则数据开始日期到第一次重复之间的时段会完全没有分段覆盖，导致 STEP6 中大量 `未匹配STEP3分段` 标签。

**验证方法**：检查 `freeze_count_segments.csv` 中最早分段是否从数据开始时刻覆盖。

### 14.8 输出文件覆盖风险 ⚠️（低优先级）

多次以不同参数运行同一 STEP 时，输出文件名若不包含参数信息（如阈值），旧结果会被覆盖。

**建议**：
- 在输出文件名中加入关键参数（如阈值、场站名、版本号）
- 使用带时间戳的输出目录（如 `step7-15min建模数据-20260401/`）

---

## 15. 后续优化方向

### 15.1 问题风机专项处理

对 #48、#85、#81、#68、#74 等高误差风机：
- 单独分析状态码分布、功率曲线、停机记录
- 考虑对这些风机使用不同的历史窗口长度 M
- 考虑为这些风机单独训练模型（而不是使用统一模型）
- 若某台风机数据质量太差，考虑从 `turbine_sum` 方案中排除，用全站均值替代

### 15.2 引入风速特征

当前建模主要依赖历史功率。风速是风力发电的核心驱动因素，引入风速特征（`WINDSPEED_#n` 或场站平均风速）可能：
- 大幅改善中长步长预测效果
- 减少模型对历史功率的过度依赖

注意：风速数据在冻结期间也可能存在冻结问题，使用前需通过 STEP6 验证数据可用性。

### 15.3 1min 预测聚合验证

当前 STEP9 直接预测未来某一分钟功率，与 STEP8 的 15min 均值预测不可直接比较。

**建议增加**：将 STEP9 的分钟级预测结果，再聚合为 15min 均值，与 STEP8 结果做公平比较。这样可以判断："分钟级建模后聚合" vs "直接15min建模" 哪种方式更好。

### 15.4 损耗动态校准

当前 `turbine_sum_loss_q50` 使用固定的 Q50 损耗曲线。诊断发现不同预测步长下损耗偏差方向不同：

**建议**：
- 对不同步长 N，分别计算历史损耗的分布，确定对应的校准分位数（Q50 不一定最优）
- 或引入"残差损耗"建模：将损耗预测也作为机器学习问题，用历史损耗预测未来损耗

### 15.5 统一配置文件管理场站参数

当前场站参数（额定功率、风机上限、允许状态码）分散在各 STEP 文件中。当增加新场站时，需要修改多处代码。

**建议**：引入集中配置文件（如 `station_config.yaml` 或 `station_config.json`）：

```yaml
LGXRFD:
  rated_power_mw: 400.0
  fan_count: 100
  fan_rated_power_kw: 4000.0
  fan_power_upper_kw: 4500.0
  allowed_status_codes: [0, 1, 2, 3, 8, 11]

JMZSFD:
  rated_power_mw: 300.0
  fan_power_upper_kw: 7000.0
  allowed_status_codes: [0, 1, 2, 3, 4, 5, 6]
```

各 STEP 读取同一配置文件，避免参数不一致。

### 15.6 自动选择输入文件的改进

当前 STEP7/STEP9 自动查找 STEP6 输出时，以文件修改时间排序，存在误选旧文件的风险。

**建议**：
- 改为在 STEP6 输出时，写一个 `latest_output.txt` 记录最新输出文件路径
- 或在 STEP7/STEP9 中强制要求显式指定（移除自动查找，改为报错提示）

### 15.7 跨时间缺口样本过滤

STEP8/STEP9 中历史特征窗口的连续性验证，建议显式加入缺口检查：

```python
def has_gap(timestamps, expected_freq_minutes):
    """检查时间序列中是否有缺口"""
    diffs = timestamps.diff().dt.total_seconds() / 60
    return (diffs > expected_freq_minutes * 1.5).any()
```

过滤掉跨越缺口的样本，确保历史特征的时序完整性。

### 15.8 输出版本管理

建议在每次运行时，将关键参数（场站名、阈值、运行日期）写入输出目录名或 `run_config.json`，便于复现和溯源。

---

## 附录：推荐版本对照表

| STEP | 推荐使用文件 | 不再推荐文件 |
|------|------------|------------|
| STEP0-第二步 | `step0_第二步_仅正功率版.py` | `step0_第二步.py` |
| STEP1 | `step1_仅正功率.py` | `step1.py` |
| STEP2 | `step2_仅正功率.py` | `step2.py` |
| STEP3 | `step3_仅正功率.py` | `step3.py` |
| STEP4 | `step4_仅正功率.py` | `step4.py` |
| STEP5 | `step5_仅正功率_原始5类加修正后1类_离群点.py` | 其他 step5 版本 |
| STEP6 | `step6_仅正功率_数据可用标签版.py` | `step6.py` |
| STEP7 | `step7_build_15min_model_dataset_按场站名_单位状态版_v2.py` | `step7_..._单位最终版.py` |
| STEP8 | `step8_compare_forecast_15min_v8_status_features.py` | v6, v7 版本 |
| STEP9 | `step9_compare_forecast_1min.py` | — |

---

*本报告由 GitHub Copilot 基于仓库代码自动生成。如有更新或错误，请直接修改本文件。*
