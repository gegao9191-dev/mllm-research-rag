# MLLM Research RAG Assistant

## 1. 项目背景与动机

这个项目是一个**面向视觉多模态论文阅读的本地 RAG 助手**。

一开始做它，并不是为了做一个泛化的“PDF 问答机器人”，而是为了服务我自己当前的学习目标：逐步进入视觉语言模型（VLM）和视频多模态大模型（Video-MLLM）方向，提升自己阅读、整理和理解相关论文的效率。

相比普通的 PDF 问答系统，这个项目的使用场景更聚焦：它主要用于辅助阅读和分析一些有代表性的视觉多模态论文，例如：

- ViT
- CLIP
- LLaVA
- Video-LLaVA
- Qwen2.5-VL
- InternVL3.5

整个系统支持：

- PDF 解析
- 向量检索
- 中文问答
- 引用来源展示
- 基于 chunk size 和 top-k 的 RAG 消融实验与失败案例分析

简单来说，这个项目可以理解为：

> 一个用于阅读视觉多模态论文的本地 RAG 助手，支持 PDF 解析、向量检索、中文问答、引用来源展示，并通过 chunk size 和 top-k 消融实验分析 RAG 的失败模式。

---

## 2. 项目功能

目前这个项目已经具备以下功能：

- 从 `data/papers/` 目录中读取本地论文 PDF
- 使用 PyMuPDF 对 PDF 进行文本解析
- 对论文文本进行 chunk 切分，并支持自定义 chunk size 和 overlap
- 使用 embedding model 将文本转换为向量，并进行相似度检索
- 通过阿里云百炼兼容接口调用 DeepSeek / Qwen 系列模型，实现中文问答
- 展示检索来源，包括：
  - 文件名
  - 页码
  - 检索分数
  - 原文片段预览
- 提供 Gradio 图形界面
- 支持论文选择下拉框：
  - All papers
  - ViT
  - CLIP
  - LLaVA
  - Video-LLaVA
  - Qwen2.5-VL
  - InternVL3.5
- 支持检索参数控制：
  - `chunk_size`
  - `chunk_overlap`
  - `top_k`
- 支持 RAG 实验分析：
  - chunk-size 消融实验
  - top-k 消融实验
  - answer quality 人工评分
  - faithfulness 忠实性评分
  - failure case 分析

---

## 3. 技术栈

| 模块 | 工具 / 方案 |
|---|---|
| Web 界面 | Gradio |
| RAG 框架 | LlamaIndex |
| PDF 解析 | PyMuPDF |
| Embedding 模型 | BAAI/bge-small-zh-v1.5 |
| 大模型 API | 阿里云百炼 / DashScope 兼容接口 |
| 生成模型 | deepseek-v4-pro |
| 实验记录 | CSV + Markdown |
| 环境管理 | Conda / pip |
| 版本管理 | Git / GitHub |

---

## 4. 系统工作流程

整个系统遵循一个标准的 RAG（Retrieval-Augmented Generation，检索增强生成）流程：

```text
PDF 论文
↓
使用 PyMuPDF 进行文本抽取
↓
按 chunk size 和 overlap 切分文本
↓
将每个 chunk 转换成 embedding 向量
↓
建立向量索引
↓
将用户问题转换成 query 向量
↓
检索 top-k 个最相关的 chunk
↓
把检索结果和用户问题一起组织进 prompt
↓
调用 DeepSeek 生成回答
↓
返回答案并展示引用来源
```

这个项目的核心思想不是让大模型“记住”所有论文，而是：

> 先在外部论文库中检索出与问题相关的内容，再让大模型基于这些检索片段进行回答。

换句话说，这里的知识不是直接写进模型参数里的，而是通过 **外部文档库 + 向量检索 + prompt 约束** 的方式，在回答时动态提供给模型。

---

## 5. 项目结构

```text
mllm-research-rag/
├── README.md
├── app.py
├── requirements.txt
├── evaluation.md
├── data/
│   └── papers/
├── docs/
│   ├── rag_concepts.md
│   └── failure_case_analysis.md
├── experiments/
│   ├── chunk_size_experiment.py
│   ├── chunk_size_ablation.csv
│   ├── chunk_size_ablation.md
│   ├── top_k_experiment.py
│   ├── topk_ablation_results.csv
│   └── topk_ablation_summary.md
└── results/
```

说明：

- `data/papers/` 用来存放本地论文 PDF
- `docs/` 用来记录 RAG 原理、失败案例分析等文档
- `experiments/` 用来存放消融实验脚本与实验结果
- `.env` 和论文 PDF 不上传 GitHub

---

## 6. 如何运行

### 6.1 创建环境

```bash
conda create -n mllm_rag python=3.10 -y
conda activate mllm_rag
pip install -r requirements.txt
```

### 6.2 配置 API Key

在项目根目录下创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=deepseek-v4-pro
```

注意：

- `.env` 中包含密钥，**不要上传到 GitHub**
- 如果以后更换模型，比如从 Qwen 换成 DeepSeek，只需要修改 `.env` 中的模型名即可

### 6.3 准备论文 PDF

把论文放到：

```text
data/papers/
```

例如：

```text
CLIP.pdf
VIT.pdf
llava.pdf
video-llava.pdf
qwen2.5-VL.pdf
internvl.pdf
```

### 6.4 启动 Gradio Demo

```bash
python app.py
```

运行后打开终端里给出的本地 Gradio 地址即可。

---

## 7. 实验设计与评估设置

这个项目并没有停留在“能跑通一个 demo”这个层面，而是进一步做了一些基础的 RAG 评估实验，目的是更清楚地理解系统的行为边界。

### 7.1 Chunk-size 消融实验

对比设置：

```text
chunk_size = 300, 500, 800, 1200
```

固定：

```text
top_k = 4
```

实验关注的问题包括：

- 是否检索到正确论文
- 是否检索到正确章节
- 回答是否完整
- 是否出现保守拒答
- 是否出现多篇论文混杂

实验结论：

```text
chunk_size = 800
```

在当前论文库上整体表现最稳，能够较好平衡检索精度和上下文完整性。

---

### 7.2 Top-k 消融实验

对比设置：

```text
top_k = 2, 4, 6, 8
```

固定：

```text
chunk_size = 800
chunk_overlap = 120
```

实验关注的问题包括：

- top-k 太小时是否漏掉关键证据
- top-k 太大时是否混入过多无关片段
- 回答完整性是否提升
- 是否出现 mixed sources 问题

实验结论：

```text
top_k = 6
```

在当前项目中提供了较好的平衡：召回相对充分，同时噪声仍在可接受范围内。

---

### 7.3 当前推荐默认参数

根据实验结果，目前推荐的默认设置为：

```text
chunk_size = 800
chunk_overlap = 120
top_k = 6
```

---

## 8. 示例问题

### 单篇论文理解类问题

```text
ViT 为什么可以把图像当成词序列来处理？请结合 patch embedding、position embedding、class token 和 Transformer Encoder 解释。
```

```text
CLIP 的 contrastive pre-training objective 是什么？请解释 image-text pairs、cosine similarity 和 symmetric cross entropy loss 的作用。
```

```text
LLaVA 的 visual instruction tuning data 是如何由 GPT-4 生成的？请分别解释 conversation、detailed description 和 complex reasoning 三类数据。
```

```text
Video-LLaVA 的 alignment before projection 是什么意思？请结合 LanguageBind 和 unified visual representation 解释。
```

### 跨论文分析类问题

```text
请比较 CLIP 和 LLaVA 的训练目标有什么本质区别。
```

```text
请按照 ViT → CLIP → LLaVA → Video-LLaVA → Qwen2.5-VL → InternVL3.5 的顺序，梳理视觉多模态大模型的发展主线。
```

说明：

跨论文综合分析问题，比单篇论文事实型问答更难。  
对于这类问题，单一全局 top-k 检索策略往往不够理想，更适合采用按论文分别检索、再做综合分析的方式。

---

## 9. 失败案例分析

在实验过程中，这个项目暴露出了一些比较典型的 RAG 失败模式。

| 失败类型 | 含义 |
|---|---|
| Retrieval failure | 没有检索到正确论文或正确章节 |
| Retrieval incomplete | 检索到正确论文，但关键上下文不完整 |
| Mixed sources | 检索结果混入多篇无关论文 |
| Answer incomplete | 检索结果有帮助，但回答不完整 |
| Prompt over-conservatism | prompt 过于保守，导致模型即使有部分依据也倾向于拒答 |
| Query keyword mismatch | 缩写类问题（如 DvD）和全文术语匹配不稳定 |
| Cross-document synthesis failure | 跨论文综合问题下，全局 top-k 覆盖不均，导致无法进行有效归纳 |

一个比较重要的观察是：

> 单一的全局 top-k 检索策略，对于“单篇论文事实型问题”通常是有效的；但对于“跨论文技术路线梳理”“多篇论文对比分析”这类问题，全局 top-k 很容易被少数几篇论文主导，导致证据覆盖不完整。

因此，这个项目也让我意识到：

> 单篇问答型 RAG 和跨文档综合型 RAG，本质上不是完全同一种任务，它们需要不同的检索策略和 prompt 设计。

---

## 10. 这个项目带给我的主要认识

通过这个项目，我对 RAG 的理解不再停留在“做一个 PDF 问答工具”这件事上，而是更清楚地意识到，RAG 系统的表现是整个链路共同决定的。

一些比较重要的认识包括：

- PDF 解析质量会影响后面所有环节
- chunk size 影响检索粒度和上下文完整性
- top-k 影响召回率和噪声控制
- embedding model 决定了问题与 chunk 的语义匹配方式
- 检索到 top-k 不代表真的相关，只代表“相对最相似”
- prompt 会显著影响模型是更保守还是更灵活
- source citation 是判断回答是否忠于原文的重要依据
- RAG 可以降低幻觉，但不能完全消除幻觉
- 跨文档综合分析需要不同于普通 factual QA 的设计思路

---

## 11. 后续可以继续改进的方向

虽然当前版本已经完成了基础功能和 Week 2 的评估目标，但如果后续继续完善，这个项目还有不少可以继续扩展的方向：

1. 增加 metadata filtering，限制检索范围到指定论文
2. 增加 per-document retrieval，支持跨论文综合分析
3. 增加 query rewriting，提升中文问题检索英文论文的能力
4. 增加 hybrid search（BM25 + embedding retrieval）
5. 引入 reranker，提高检索质量
6. 用 Chroma 或 FAISS 做持久化索引
7. 增加 summary mode，更适合回答“这篇论文讲了什么”这类总结型问题
8. 引入 retrieval score threshold，过滤低置信度检索结果
9. 对比不同 embedding 模型的表现
10. 构建更系统的自动化 RAG 评估流程

---

## 12. 项目当前进度

这个项目属于我整个 12 周视觉多模态 / Video-MLLM 转型计划中的第一个工程项目。

当前完成情况：

```text
Week 1：最小可用 RAG demo 已完成
Week 2：RAG 评估、失败分析、Gradio v0.2 升级已完成
```

下一步计划：

```text
VLM Evaluation Lab
```

---

## 13. 补充说明：这个项目的定位

最后补充一点，这个项目的定位不是“终极形态的论文助手”，而是我进入 MLLM / RAG / VLM 方向的一个起点项目。

它的价值主要在于：

- 帮助我建立 RAG 的完整心智模型
- 帮助我阅读和整理多模态论文
- 帮助我积累一些基础的工程实现经验
- 帮助我形成实验意识，而不是只停留在 demo 层面
- 为后续的 VLM Evaluation Lab 和 Qwen-VL Fine-tuning 项目做铺垫

所以它更像是：

> 一个“小而完整”的入门型研究工程项目，而不是一个追求大而全的终极系统。