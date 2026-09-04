# 安全与数据边界

- 包内没有API key、天衍OpenID、token、密码或`.env`。
- 所有平台凭据必须通过进程环境变量注入；不得写入config、命令历史或manifest。
- 原T176 `snapshots.jsonl`未收入ZIP，因为包含88个query ID、raw结果路径和本机路径引用。
- 原raw counts与NPZ未收入ZIP；它们的来源路径和SHA256保存在`manifest/EXTERNAL_PRIVATE_EVIDENCE.json`。
- ZIP包含公开派生的cycle endpoints和hybrid pair rows，因此可以复算核心比值与置换检验。
- 三seed模型只收入`best.pt`，不重复收入内容角色相近的`last.pt`；排除决定不会改变代码或已报告结果。
- 任何真机命令默认阻断，统一入口要求显式`--allow-hardware`。
