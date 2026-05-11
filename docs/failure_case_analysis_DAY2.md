# RAG Failure Case Analysis

## 0. 目的

本文件用于记录 Week 2 的第一次 RAG 失败诊断。

Week 1 的目标是让系统跑通：PDF 能读取、能检索、能问答、能显示来源。  
Week 2 的目标不是继续堆功能，而是分析系统什么时候会失败、为什么失败、如何改进。

一个 RAG 系统的失败不能简单归因于“大模型不行”。需要拆分为：

1. PDF 是否正确解析；
2. chunk 是否切分合理；
3. embedding 是否能表示问题和论文片段；
4. retriever 是否召回了正确片段；
5. top-k 是否足够；
6. prompt 是否过于保守或过于宽松；
7. generator 是否忠于上下文；
8. 用户问题是否足够明确。

---

## 1. 失败类型定义

| 失败类型 | 含义 | 典型表现 |
|---|---|---|
| A. PDF Parsing Failure               | PDF 文本没有正确解析            | retrieved sources 出现乱码、PDF 对象、endobj、URI、Annot 等 |
| B. Chunking Failure                  | 关键上下文被切断                | 检索到了相关论文，但答案缺少定义、原因或实验结果 |
| C. Retrieval Failur                  | 没有召回正确论文或正确章节       | retrieved sources 来自错误论文，或来自正确论文但不是相关页 |
| D. Cross-document Retrieval Failure  | 跨论文比较时只检索到其中一篇     | 问 CLIP vs LLaVA，但只出现 LLaVA 来源 |
| E. Prompt Over-conservatism          | prompt 太保守，导致有材料也拒答  | 检索片段部分相关，但回答 I cannot determine |
| F. Generation Failure                | 检索片段正确，但模型总结错       | sources 正确，但回答误读、遗漏或混淆 |
| G. Question Ambiguity                | 问题本身不明确                  | “这篇论文讲了什么？”但知识库里有多篇论文 |

---

## 2. Case 1：CLIP 的核心训练目标是什么？

### 原问题

CLIP 的核心训练目标是什么？为什么对比学习适合开放世界视觉理解？

### 系统回答

I cannot determine this from the provided papers.

### 期望论文

CLIP.pdf

### 期望检索内容

应该检索到 CLIP 论文中关于 contrastive pre-training objective 的段落。

核心信息应包括：

- CLIP 使用 image encoder 和 text encoder；
- 输入是一批 N 个 image-text pairs；
- 模型需要判断 N × N 个图文组合中哪些是真实配对；
- 最大化真实图文对的相似度；
- 最小化错误图文对的相似度；
- 使用 symmetric cross entropy loss；
- 这种训练方式让视觉类别不再局限于固定标签，而可以通过自然语言表达开放概念。

### 初步诊断

这不是 PDF 解析失败，因为 CLIP 的 zero-shot 问题可以被正常回答，说明 CLIP PDF 至少部分被成功解析。

更可能的失败原因是：

- 中文问题没有召回 CLIP 的 pre-training method 相关 chunk；
- top-k 太小，导致关键段落没有进入上下文；
- chunk_size 或 overlap 不合适，导致 contrastive objective 的完整解释被切散；
- prompt 太保守，在上下文不完整时直接拒答。

### 失败类型

C. Retrieval Failure  
B. Chunking Failure  
E. Prompt Over-conservatism

### 改进方式

1. 将问题改得更贴近论文原文关键词：

   CLIP 的 contrastive pre-training objective 是什么？请解释 image-text pairs、cosine similarity 和 symmetric cross entropy loss 的作用。

2. 提高 top-k：

   top_k: 4 → 6 或 8

3. 比较中英文问题：

   中文：CLIP 的核心训练目标是什么？  
   英文：What is the contrastive pre-training objective of CLIP?

4. 如果仍然失败，增大 chunk_size：

   chunk_size: 500 → 800 或 1200

---

## 3. Case 2：LLaVA 如何利用 GPT-4 生成多模态指令数据？

### 原问题

LLaVA 是如何利用 GPT-4 生成多模态指令数据的？这种方法有什么优势和风险？

### 系统回答

I cannot determine this from the provided papers.

### 期望论文

llava.pdf

### 期望检索内容

应该检索到 LLaVA 论文中关于 visual instruction data generation 的部分。

核心信息应包括：

- LLaVA 使用 language-only GPT-4 生成 multimodal language-image instruction-following data；
- 论文手动设计少量 few-shot seed examples；
- 使用 in-context learning 让 GPT-4 基于图像 caption 和 boxes 生成指令数据；
- 数据类型包括 conversation、detailed description、complex reasoning；
- 最终收集 158K 条 instruction-following samples；
- 优势是低成本构造高质量视觉指令数据；
- 风险是数据质量依赖 GPT-4，本质是合成数据，可能继承 GPT-4 的偏差或产生不可靠样本。

### 初步诊断

这不是论文内容缺失。LLaVA 论文中明确有数据构造方法。  
更可能是 retriever 只检索到了 LLaVA 摘要或模型架构部分，没有检索到数据生成细节部分。

### 失败类型

C. Retrieval Failure  
B. Chunking Failure  
E. Prompt Over-conservatism

### 改进方式

1. 改写问题，加入关键词：

   LLaVA 的 visual instruction tuning data 是如何由 GPT-4 生成的？请分别解释 conversation、detailed description 和 complex reasoning 三类数据。

2. top_k 提高到 6 或 8。

3. chunk_size 设置为 800 或 1200，避免数据构造流程被切碎。

4. 如果做 metadata filtering，指定只检索 llava.pdf。

---

## 4. Case 3：Video-LLaVA 的 alignment before projection 是什么意思？

### 原问题

Video-LLaVA 的 alignment before projection 是什么意思？为什么这对统一图像和视频表示很重要？

### 系统回答

I cannot determine this from the provided papers.

### 期望论文

video-llava.pdf

### 期望检索内容

应该检索到 Video-LLaVA 的 abstract、introduction 或 section 3.1.3。

核心信息应包括：

- 现有方法往往把图像和视频编码到不同 feature space；
- 如果图像和视频特征在 projection 之前没有对齐，LLM 需要从多个 projection layers 中学习跨模态交互，难度较大；
- Video-LLaVA 先使用 LanguageBind 将图像和视频对齐到统一特征空间；
- 然后再通过 shared projection layer 输入 LLM；
- 这样可以让 LLM 从 unified visual representation 中同时学习图像和视频。

### 初步诊断

这个失败很典型。因为 “alignment before projection” 是论文标题级概念，如果系统答不出来，说明不是大模型理解不了，而是检索没有召回关键段落。

### 失败类型

C. Retrieval Failure  
B. Chunking Failure

### 改进方式

1. 改写问题：

   在 Video-LLaVA 中，为什么作者认为 image features 和 video features 必须在 projection layer 之前对齐？请结合 unified visual representation 和 LanguageBind 解释。

2. top_k 提高到 6。

3. 如果界面支持论文选择，选择 video-llava.pdf 后再问。

4. 保留英文关键词 alignment before projection、unified visual representation、LanguageBind。

---

## 5. Case 4：InternVL3.5 的 DvD 是什么？

### 原问题

InternVL3.5 的 DvD 是什么？它为什么能提升推理效率？

### 系统回答

没有检索出来或回答不完整。

### 期望论文

internvl.pdf

### 期望检索内容

应该检索到 InternVL3.5 摘要、introduction 或 Section 2.5。

核心信息应包括：

- DvD 全称是 Decoupled Vision-Language Deployment；
- 它将 vision encoder 和 language model 分配到不同 GPU 上；
- 目的是平衡视觉计算和语言模型计算负载；
- 提高计算并行性和硬件利用率；
- 与 ViR 结合后可以进一步提升推理效率；
- 高分辨率、多图和视频输入时，视觉计算开销更高，因此 DvD 的意义更明显。

### 初步诊断

这个 case 可能和缩写有关。  
“DvD”太短，embedding 模型可能不容易把它和 “Decoupled Vision-Language Deployment” 关联起来。

### 失败类型

C. Retrieval Failure  
B. Chunking Failure  
Query Keyword Mismatch

### 改进方式

1. 提问时同时写缩写和全称：

   InternVL3.5 的 DvD（Decoupled Vision-Language Deployment）是什么？它如何把 vision encoder 和 language model 部署到不同 GPU 上来提升推理效率？

2. top_k 提高到 6 或 8。

3. 检查 retrieved sources 是否来自 internvl.pdf 的 abstract / introduction / Section 2.5。

4. 对缩写类问题，建议在 query 中加入完整英文术语。

---

## 6. 第一轮失败诊断总表

| Case | 问题 | 期望论文 | 实际回答 | 初步失败类型 | 主要原因 | 初步改进 |
|---|---|---|---|---|---|---|
| 1 | CLIP 的核心训练目标是什么？ | CLIP.pdf | I cannot determine | Retrieval Failure | 没召回 contrastive objective 段落 | top_k=6；加入英文关键词 contrastive pre-training objective |
| 2 | LLaVA 如何利用 GPT-4 生成多模态指令数据？ | llava.pdf | I cannot determine | Retrieval Failure / Prompt Over-conservatism | 没召回数据构造细节 | top_k=6/8；问题中加入 visual instruction tuning data |
| 3 | Video-LLaVA 的 alignment before projection 是什么？ | video-llava.pdf | I cannot determine | Retrieval Failure | 没召回 abstract / 3.1.3 | top_k=6；加入 LanguageBind / unified visual representation |
| 4 | InternVL3.5 的 DvD 是什么？ | internvl.pdf | 没检索出来 | Query Keyword Mismatch / Retrieval Failure | DvD 缩写太短，未匹配全称 | 同时写 DvD 和 Decoupled Vision-Language Deployment |

---

## 7. 当前结论

本轮失败诊断说明，当前 RAG 系统已经具备基本问答能力，但在以下场景下不稳定：

1. 关键概念位于论文中较局部的方法段落；
2. 用户用中文问英文论文；
3. 问题中存在英文缩写；
4. 问题需要跨多个 chunk 才能回答；
5. top-k 不足导致关键片段没有进入上下文；
6. prompt 过于保守，导致模型在上下文不完整时直接拒答。

这些问题并不代表 Week 1 demo 失败。相反，它们说明 Week 2 的实验是必要的。

---

## 8. 下一轮实验计划

下一步应该做三组实验：

### 实验 1：top-k 对比

固定 chunk_size，比较：

- top_k = 2
- top_k = 4
- top_k = 6
- top_k = 8

观察上述四个失败问题是否恢复正常。

### 实验 2：chunk_size 对比

固定 top_k，比较：

- chunk_size = 300
- chunk_size = 500
- chunk_size = 800
- chunk_size = 1200

观察答案完整性和 retrieved sources 相关性。

### 实验 3：中英文 query 对比

比较：

- 中文问题；
- 英文问题；
- 中英混合问题。

例如：

中文：CLIP 的核心训练目标是什么？  
英文：What is the contrastive pre-training objective of CLIP?  
中英混合：CLIP 的 contrastive pre-training objective 是什么？

重点观察中文问题检索英文论文是否会降低 retrieval quality。

---

## 9. Day 2 小结

Day 2 的核心收获是：

RAG 失败不是一个单点错误，而是一个链路错误。  
如果答案失败，我们需要先检查 retrieved sources，而不是直接怀疑 LLM。

当前四个失败 case 的主要问题集中在 retrieval 阶段，尤其是 top-k、chunk_size、query wording 和缩写匹配。

因此，Week 2 后续实验应优先围绕：

1. top-k；
2. chunk_size；
3. 中文 vs 英文 query；
4. metadata filtering；
5. prompt 调整。

最终目标是让这个项目从“能跑”升级为“能诊断、能解释、能改进”的 RAG 实验项目。