# Resume Copilot Agent Instructions

These rules apply to this repository in addition to the parent workspace instructions.

## 禁词表三问 — 硬门禁

新增或扩展任何禁词表、黑名单、关键词排除表前必须回答：

1. 表项是领域或协议固有，还是从观测输出、榜单 badcase、失败样例中抄取或归纳的？后者命中即禁止。
2. 命中后只影响路由、分类或待确认标记，还是会删除、压制、改写事实内容？会造成内容丢弃的命中即禁止。
3. 是否可用 source span、字段类型、record 边界、BBOX、置信度、schema 约束等结构化信号替代？可以替代的命中即禁止。

任一问命中即不得引入该表。已有表命中时也不得继续追加词项，必须改用对应的结构化信号。
