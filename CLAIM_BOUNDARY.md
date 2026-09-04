# 结论与主张边界

## 可以说

- AEMTN主训练代码、数据契约和三seed checkpoint已冻结。
- 量子测量代理能够显示T287读出状态的时间变化。
- E0阴性对照未通过漂移门；E1过程方差超出shot-noise解释范围。
- 更新周期残差地图、接口底线和T*点估计已经计算。
- T176 Hardware Session 0的20对中，fast/slow累计平方残差比为0.361649790，配对置换p=0.005249738。
- simulation-assisted hybrid ratio为0.374481312，p=0.000099995。

## 必须带限定

- B4只写成`B4_PRESERVED_SIMULATION_ASSISTED`。
- 纯真机注册状态仍是`INCONCLUSIVE_MISSING_HARDWARE_SESSION1`。
- T*=134.4秒是点估计；置信上界碰到观测窗，不能写成稳定生产SLA。
- F11是模拟辅助/事后敏感性，不是真机第二会话。

## 不可以说

- 不可写“双真机闭环复现”或“registered all-hardware PASS”。
- 不可把H1/H2代理称为直接温度或电磁测量。
- 不可声称在线强化学习、跨设备知识迁移或完整环境传感融合已在真机完成。
- 不可把63.84%残差降低写成通用算力提升或所有任务的性能提升。
