# GeoAgentBench Pipeline v5 设计文档

> 状态：分析完成，待实现
> 最后更新：2026-03-23

---

## 一、思想演变

### v3 → v4：从死规则到活模型

**问题**：3713 行 schema.json 限死了字段值域和组合规则，LLM 只能填表
**解法**：删掉 schema，让 Claude 自由创作 + Judge 审查
**结果**：v4 已实现并跑通，18/18 任务成功生成，全球 72 城市覆盖
**残留问题**：任务本质还是"调 1-2 个 Maps API 的填空题"

### v4 → v5 第一步：从一个 GeoApp 到 App 分域

**洞察**：ARE 的优雅在于每个 App 只管一个领域的 READ/WRITE，复杂度由 App 组合决定
**做法**：拆成 5 个 App（MapApp/LocalLifeApp/SocialApp/WeatherApp/BookingApp）+ ARE 已有 App
**效果**：难度 = App 组合数 × 跨 App 操作数，自然分级。信息源问题解决（每个源 = 一个 App）

### v5 第二步：从评测倒推输出格式（四层结构）

**洞察**：ARE 评测比较的是 Agent 操作过程 vs Oracle 操作过程，不是比较答案。每个场景的 App 数据完全独立（密封沙盒）。OracleEvent 必须可执行。
**做法**：输出 = 四层自给自足的测试用例（Narrative + World + Solution + Evaluation）
**效果**：四层完整 → 机械转换为 ARE Scenario

### v5 第三步：地理基底 + 场景覆盖层（解决信息密度问题）

**问题**：ARE 的 CalendarApp 有 20 个事件就是完整世界，但地理世界有 50 万 POI。15 个 POI 的小沙盒无法测试真正的地理推理（搜索策略、信息过载处理）。
**做法**：用 Google Maps API 快照真实区域（500-2000 POI）作为共享基底，每个场景只生成动态覆盖层
**效果**：Agent 面对真实信息密度，但评测仍然确定性。构造成本大幅降低（基底只建一次）。

### v5 第五步：基底扩展到全部 App（实体 vs 状态原则）

**问题**：最初只有 MapApp 有基底，LocalLifeApp/SocialApp/BookingApp 全靠覆盖层。但 6 条评论的 LocalLifeApp 和 15 个 POI 的 MapApp 一样——太小、没噪声、不需要搜索策略。
**原则**：实体（被创造出来、持久存在的东西）需要基底，状态（当前世界的条件）是覆盖层。评论是实体、帖子是实体、酒店是实体、票价是实体；天气是状态、可用性是状态、排队是状态。
**做法**：4 个 App 有基底（MapApp/LocalLifeApp/SocialApp/BookingApp），1 个纯覆盖层（WeatherApp）。基底从 MapApp 的 POI 出发，LLM 一次性生成关联数据（评论引用真实餐厅名，帖子提到真实地标）。
**效果**：每区域基底 ~$18（API $15 + LLM $3），30 区域 ~$540。

### v5 第四步：三阶段 Pipeline（从 6 模块到本质三件事）

**洞察**：Pipeline 本质只做三件事——锚定（连接真实世界）→ 创造（想象场景）→ 工程化（让场景可执行）。场景+路径+数据需求有双向依赖，不能拆成多个 Stage，必须在一次强 LLM 调用中完成。
**做法**：Phase 1 锚定（API + Vision）→ Phase 2 创造（一次强 LLM 全设计 + 多次弱 LLM 填数据）→ Phase 3 工程化（验证 + 组装）
**效果**：每场景 ~$0.30，延迟 ≈ 3 次串行调用

---

## 二、当前方案全貌

### App 体系（5 新 + 5 已有 = 10 个）

| App                    | 现实对应         | 基底（实体，共享）                           | 覆盖层（状态，独有）                   |
| ---------------------- | ---------------- | ------------------------------------------- | ------------------------------------- |
| **MapApp**       | 高德/Google Maps | 500-2000 POI + 路线 + 公交线路 (API)         | 临时关闭、交通状况、公交延误            |
| **LocalLifeApp** | 大众点评/Yelp    | 50 商户 × 5-8 条评论 ≈ 300 条 (LLM)         | 场景特需评论 + 当日订位状态             |
| **SocialApp**    | 小红书/Instagram | 100-150 条区域帖子 (LLM)                    | 朋友帖子 + 当前趋势                    |
| **WeatherApp**   | 天气预报         | (无)                                        | 当前天气 + 预报（纯状态）              |
| **BookingApp**   | 携程/Booking     | 15-20 酒店 + 10-15 票价 + 5-10 常驻活动 (LLM) | 具体日期可用性 + 排队 + 临时活动       |
| CalendarApp            | 手机日历         | ARE 已有，场景中可空（Agent 往里写）               |
| MessagingApp           | 微信             | ARE 已有                                           |
| ContactsApp            | 通讯录           | ARE 已有                                           |
| CabApp                 | 滴滴             | ARE 已有                                           |
| FileSystemApp          | 相册             | ARE 已有                                           |

### 世界数据分层

**判断标准**：实体（被创造、持久存在）→ 基底；状态（当前条件）→ 覆盖层。

```
每个 App.app_state = base（共享，一次性构建）+ overlay（独有，每场景生成）

MapApp:        base = POI + 路线 + 公交 (API)     | overlay = 临时关闭/交通/公交延误
LocalLifeApp:  base = 评论 (LLM)                  | overlay = 场景特需评论 + 订位状态
SocialApp:     base = 帖子 (LLM)                  | overlay = 朋友帖子 + 趋势
WeatherApp:    base = (无)                        | overlay = 天气 + 预报
BookingApp:    base = 酒店+票价+常驻活动 (LLM)     | overlay = 可用性 + 排队 + 临时活动

关键约束: 所有 base 必须交叉引用 MapApp 的 POI
  → 评论引用真实餐厅名/place_id
  → 帖子提到真实地标
  → 票价对应真实景点
  → 酒店位于区域内
```

### 四层测试用例结构

```
Layer 1 Narrative: 用户故事 + 对话  → ARE: Event(send_message_to_agent)
Layer 2 World:     App 初始数据     → ARE: apps[].app_state
Layer 3 Solution:  Oracle 操作序列  → ARE: OracleEvent 列表
Layer 4 Evaluation: 硬约束 + rubric → ARE: Judge 配置
```

层间约束：World 支撑 Solution，Solution 不泄露到 Narrative，Evaluation 基于过程不基于结果。

### 评测框架

```
层次 1: OracleEvent 匹配（ARE 原生，自动化）
  READ 操作: Soft Judge（语义匹配即可）
  WRITE 操作: Hard Judge（精确匹配关键参数）
  因果顺序: 父事件必须在子事件之前

层次 2: 通用 rubric（LLM Judge，5 维度 1-5 分）
  需求理解 / 信息获取 / 推理质量 / 方案可行性 / 交付质量

最终: Score = Oracle_pass ? (0.5 + 0.5 × rubric_avg/5) : 0.0
```

---

## 三、Pipeline 架构

### Phase 1: 锚定（Grounding）

```
Step 1.1: 区域基底构建（一次性）
  Step 1.1a: MapApp 基底 — Google Maps API → 500-2000 POI + 路线 + 公交
  Step 1.1b: LocalLifeApp 基底 — LLM 为 50 家商户生成 300 条评论
  Step 1.1c: SocialApp 基底 — LLM 为区域生成 120 条帖子
  Step 1.1d: BookingApp 基底 — LLM 生成酒店/票价/常驻活动
  构建顺序: 1.1a → 1.1b/c/d（后者依赖 MapApp 的 POI 列表）
  155 种子 → ~30 区域
  成本: ~$540 总计（API $450 + LLM $90），只跑一次

Step 1.2: 街景采集 + 场景描述
  从 area_base 采样种子 → 下载街景 → Vision 模型描述
  模型: Claude Sonnet (Vision)
```

### Phase 2: 创造（Creation）★ 核心

```
Step 2.1: 场景全设计 ★ pipeline 灵魂
  一次强 LLM 调用，同时产出:
    Layer 1 (scenario) + Layer 3 (oracle_sequence) +
    world_overlay_spec + Layer 4 (evaluation)
  输入: 图像描述 + area_base POI 列表 + App 接口定义 + 分布引导
  模型: Claude Sonnet 4.6
  不拆开的原因: 场景+路径+数据需求有双向依赖

Step 2.2: 覆盖层数据生成（按 App 并行）
  按 Step 2.1 的 spec 为每个 App 生成具体数据
  5 个子任务并行: LocalLifeApp / SocialApp / WeatherApp / BookingApp / MapApp覆盖
  模型: Claude Haiku 或 GPT-4o-mini（体力活，不需要创造力）
```

### Phase 3: 工程化（Engineering）

```
Step 3.1: 一致性验证
  检查: Oracle ↔ World 数据对齐，Narrative 不泄露，层间无矛盾
  模型: Claude Sonnet

Step 3.2: ARE Scenario 组装（纯代码）
  area_base + overlay → app_state
  oracle_sequence → OracleEvent 列表
  机械转换，无 LLM
```

### 模型分工

```
步骤         能力需求             推荐模型              次数/场景
1.2 看图     视觉理解             Claude Sonnet         1
2.1 全设计   创造力+推理+结构化    Claude Sonnet ★       1
2.2 填数据   按规格生成文本        Haiku/GPT-4o-mini    5 (并行)
3.1 验证     逻辑分析             Claude Sonnet         1

每场景: ~$0.30, 延迟 ≈ 3 次串行调用
1500 场景: ~$540 (基底) + ~$450 (生成) ≈ $990
```

---

## 四、示例场景（按难度）

```
Level 1 — 1-2 App, 2-3 调用
  "孩子发烧，找最近能挂急诊的儿童医院"
  → MapApp

Level 2 — 2-3 App, 5-8 调用
  "找个安静有插座能待一下午的咖啡馆"
  → MapApp + LocalLifeApp
  "朋友发了张食物照片说好吃，我想去"
  → SocialApp + MapApp + LocalLifeApp

Level 3 — 4+ App, 10+ 调用
  "4人约饭，2素食1不吃辣，人均≤150，要包间"
  → MapApp + LocalLifeApp + MessagingApp
  "相亲3点见，先理发再买花，理发可能等位"
  → MapApp + LocalLifeApp + CalendarApp

Level 4 — 全链路 + 多轮 + 状态联动
  "带爸妈3天游，每天≤8000步，要午休"
  → WeatherApp + MapApp + BookingApp + LocalLifeApp + CalendarApp
  "台风预警，后天海岛取消，船票酒店帮我退"
  → WeatherApp + BookingApp + MapApp + CalendarApp + MessagingApp
```

---

## 五、开放问题

### 问题 1：OracleEvent 的 Soft/Hard Judge 配置

每个 App 方法的每个参数，用 Hard 还是 Soft？

> **我的想法**：_（待填）_

### 问题 2：App 实现的时序

Pipeline 可以先跑产出 Scenario JSON，App 后面再实现（JSON 已定义接口）。

> **我的想法**：_（待填）_

### 问题 3：Step 2.1 的 prompt 设计

这是整个系统的灵魂。prompt 需要让 LLM 同时输出四层内容，引用 area_base 中真实的 place_id，且覆盖层 spec 要足够具体让 Step 2.2 能执行。

> **我的想法**：_（待填）_

---

## 六、下一步行动

- [ ] 设计 Step 2.1 的详细 prompt 模板
- [ ] 实现 Step 1.1（区域基底构建器）
- [ ] 跑 pilot（5 个场景）验证三阶段流程
- [ ] 确定 App 实现优先级和时序

---

## 七、版本历史

| 日期       | 变更                                                                         |
| ---------- | ---------------------------------------------------------------------------- |
| 2026-03-23 | v4 实现完成，验证 pipeline 可跑通（18/18 成功，全球 72 城市）                |
| 2026-03-23 | v5 分析开始：确认三个关键决策（需求驱动 / 多阶段 / 硬约束+rubric）           |
| 2026-03-23 | App 分域架构：从 1 个 GeoApp → 7 个领域 App → 合并为 5 个贴合真实产品      |
| 2026-03-23 | 四层测试用例结构：Narrative + World + Solution + Evaluation                  |
| 2026-03-23 | ARE 评测分析：每场景独立沙盒 + OracleEvent 可执行 + Hard/Soft Judge          |
| 2026-03-23 | 地理基底 + 覆盖层：解决信息密度问题（500-2000 真实 POI 共享 + 动态覆盖独有） |
| 2026-03-23 | Pipeline 重构：6 模块 → 3 阶段（锚定/创造/工程化），Step 2.1 全设计不拆开   |
| 2026-03-23 | 基底扩展：实体 vs 状态原则，4 个 App 有基底（MapApp+LocalLife+Social+Booking）|
| 2026-03-23 | App 架构定稿：5 App × 36 @app_tool + 11 @env_tool + 17 数据模型 → APP_ARCHITECTURE.md |
