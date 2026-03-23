# GeoAgentBench 项目上下文

## 项目定位

构建一个**高难度地理 Agent 基准测试**（benchmark），基于 GAIA2/ARE 平台。测试 AI Agent 在复杂地理场景下的综合决策能力。

## 当前状态

- **v4 pipeline 已实现并验证可跑通**（`pipeline/` 目录）
- **v5 架构已设计完成，待实现**（设计文档在 `pipeline/DESIGN_v5.md` 和 `pipeline/APP_ARCHITECTURE.md`）
- GAIA2/ARE 平台代码在 `meta-agents-research-environments/` 目录

## 关键文档（新对话必读）

1. **`pipeline/DESIGN_v5.md`** — 完整设计文档，包含：思想演变历史（v3→v4→v5 每一步的问题和解法）、当前方案全貌、Pipeline 三阶段架构、评测框架、开放问题
2. **`pipeline/APP_ARCHITECTURE.md`** — App 架构设计，包含：5 个新 App 的数据模型和全部 36 个工具接口、基底/覆盖层分层设计、场景验证

## 技术栈

- LLM: Claude Sonnet 4.6 via OpenAI 协议中转站 (yunwu.ai)
- 地图数据: Google Maps Platform API
- 目标平台: GAIA2/ARE (meta-agents-research-environments)
- 语言: Python

## API 配置

- Claude 中转站: `https://yunwu.ai/v1`, model=`claude-sonnet-4-6`
- Google Maps API key 在 `pipeline/config.yaml`（已 gitignore）

## 核心设计决策（已确认，不要推翻）

1. **App 分域架构**：5 个新 App（MapApp/LocalLifeApp/SocialApp/WeatherApp/BookingApp）+ ARE 已有 App
2. **基底 + 覆盖层**：实体（POI/评论/帖子/酒店/票价）是共享基底，状态（天气/交通/可用性）是场景覆盖层
3. **四层测试用例**：Narrative + World + Solution + Evaluation，可机械转换为 ARE Scenario
4. **三阶段 Pipeline**：锚定（API+Vision）→ 创造（一次强 LLM 全设计 + 弱 LLM 填数据）→ 工程化（验证+组装）
5. **评测**：OracleEvent 匹配（ARE 原生）+ 通用 rubric（5 维度 LLM Judge）
