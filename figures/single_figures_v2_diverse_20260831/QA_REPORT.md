# B4 v2 主图与数据终检报告

## 交付范围

- 12 张独立主图，每图同时导出 SVG、PDF、600 dpi PNG 和 600 dpi TIFF。
- 12 份主 source CSV，外加 F05 和 F10 的 summary CSV，共 14 份数据表。
- 每份 source CSV 保留原始字段和图中实际使用的派生编码，包括 z-score、log10 残差、排序序号、方向标签和裁决边际。
- 单一 Excel 数据簿共 17 个工作表：README、Figure_Index、Key_Metrics、F01–F12、Field_Dictionary、Claim_Boundary。

## 图形类型分布

| 图 | 图形类型 |
|---:|---|
| F01 | 时序快照热图 |
| F02 | 含误差的相空间散点图 |
| F03 | E0 水平森林图 |
| F04 | E1 水平森林图 |
| F05 | 三行残差热图 |
| F06 | log 时标区间条 |
| F07 | log-log 配对散点图 |
| F08 | 发散棒棒糖图 |
| F09 | 累计阶梯诊断（唯一序列折线） |
| F10 | 置换直方图 |
| F11 | 情景哑铃图 |
| F12 | 单位感知的负载流程图 |

## 科学图件 QA

- Python 源码严格验证：21 PASS，0 WARN，0 FAIL。
- PDF 文本审计：12/12 可审计；最小字号 5.25–6.30 pt，全部高于 5 pt 底线。
- 严格几何碰撞审计：12/12 PASS，0 FAIL，0 WARN。
- 最终 PNG 已逐张人工查看：无裁切、无标签重叠、无字线交叉、无图例遮挡。
- F06 的图例已移至坐标区外；F11 取消了会穿过 p 值标签的纵向网格。

## Excel 数据簿 QA

- 17/17 个工作表均已渲染预览并逐页检查。
- Key_Metrics 使用公式回指 F01–F12 数据表，未发现 `#REF!`、`#DIV/0!`、`#VALUE!`、`#NAME?`、`#NUM!` 或 `#N/A`。
- scheduled_utc 显示格式已固定为 `yyyy-mm-dd hh:mm:ss`，数据字典明确时区为 UTC。
- 数据表均启用筛选、冻结表头、单位/字段字典和声称边界。

## 不可突破的证据边界

- T176 只有 Hardware Session 0；Hardware Session 1 缺失，不宣称注册双会话 all-hardware PASS。
- F05/F06 的 T* 上界碰到观测窗，裁决保持 INCONCLUSIVE。
- F11 是模拟辅助/事后敏感性证据，不代替缺失的真机会话。
- H1/H2 是量子测量定义的环境代理，不等同于温度计、电磁传感器或底层脉冲参数。
