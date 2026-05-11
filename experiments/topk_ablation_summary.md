# RAG Top-k Ablation Summary

## 1. Experiment Setting

This experiment compares different top-k values in the MLLM paper RAG assistant.

Fixed setting:

- `chunk_size = 800`
- `chunk_overlap = 120`
- embedding model = `BAAI/bge-small-zh-v1.5`
- generator model = configured in `.env`

Tested top-k values:

- `top_k = 2`
- `top_k = 4`
- `top_k = 6`
- `top_k = 8`

The test set contains six questions covering ViT, CLIP, LLaVA, Video-LLaVA, Qwen2.5-VL and InternVL3.5. The goal is to observe how the number of retrieved chunks affects retrieval completeness, answer quality, faithfulness and cross-paper noise.

---

## 2. Why Top-k Matters

Top-k controls how many retrieved chunks are sent to the LLM.

- If `top_k` is too small, the system may miss key evidence.
- If `top_k` is too large, the system may include irrelevant chunks and cause cross-paper confusion.
- Therefore, top-k is not always better when larger.

In this experiment, `chunk_size=800` was fixed because the previous chunk-size ablation showed that it provided the best balance between context completeness and retrieval precision.

---

## 3. Manual Scoring Criteria

Answer quality score:

- 5 = 完全正确，有来源
- 4 = 基本正确，略有遗漏
- 3 = 部分正确，但不完整
- 2 = 检索到了相关内容但回答差
- 1 = 检索失败或胡编
- 0 = 完全不可用

Faithfulness score:

- 5 = 完全忠于 retrieved sources
- 4 = 基本忠实，少量概括
- 3 = 有一定推断，但无明显错误
- 2 = 混入较多外部知识
- 1 = 与来源明显不一致
- 0 = 完全幻觉

---

## 4. Quantitative Results

| top_k | Avg. Answer Quality | Avg. Faithfulness | Overall Observation |
|---:|---:|---:|---|
| 2 | 3.67 | 5.00 | 来源最集中，但上下文不足，复杂机制题容易漏召回 |
| 4 | 4.50 | 4.67 | 平衡性较好，大多数问题可稳定回答，但部分复杂问题仍不完整 |
| 6 | **4.67** | **4.83** | 整体最稳，复杂问题召回更充分，噪声可控 |
| 8 | **4.67** | **4.83** | 召回更充分，但开始明显出现 mixed sources noise |

---

## 5. Results Summary

### top_k = 2

- 优点：
  - 检索来源最集中，不容易混入无关论文。
  - ViT、Qwen2.5-VL 这类单篇事实型/机制型问题已经可以稳定回答。
  - 回答忠实性最高，模型在证据不足时倾向于保守拒答。

- 缺点：
  - 上下文数量不足，容易漏掉关键段落。
  - CLIP 问题只召回了训练目标的核心描述，但没有召回 cosine similarity 和 symmetric cross entropy loss 的完整实现细节。
  - InternVL3.5 的 DvD 问题检索失败，未找到 Decoupled Vision-Language Deployment 的关键片段。

- 典型失败 case：
  - Q2：CLIP 的 contrastive objective 回答不完整，只能说明 N×N image-text pair matching，无法解释 cosine similarity 和 symmetric cross entropy loss。
  - Q6：InternVL3.5 DvD 未被正确召回，模型拒答。

- 结论：
  - `top_k=2` 适合简单、明确、单篇论文的问题，但不适合需要多个段落支撑的机制解释题。

---

### top_k = 4

- 优点：
  - 相比 `top_k=2`，检索上下文明显更充分。
  - CLIP 的核心训练目标被完整召回，可以解释 image-text pairs、cosine similarity 和 symmetric cross entropy loss。
  - Video-LLaVA 和 Qwen2.5-VL 的问题表现稳定。
  - 整体上在检索精度和上下文完整性之间取得较好平衡。

- 缺点：
  - LLaVA 的 visual instruction tuning data 问题仍然没有完整召回 complex reasoning 的生成细节。
  - InternVL3.5 DvD 问题虽然能回答主体内容，但检索结果混入 CLIP 和 Qwen2.5-VL，出现 mixed sources noise。

- 典型失败 case：
  - Q3：LLaVA 问题仍缺少 complex reasoning 数据生成过程。
  - Q6：InternVL3.5 DvD 能回答，但来源中混入无关论文。

- 结论：
  - `top_k=4` 是一个可用的默认值，适合多数单篇论文问题，但对复杂细节题仍可能召回不足。

---

### top_k = 6

- 优点：
  - 整体表现最稳定。
  - CLIP、Video-LLaVA、Qwen2.5-VL 等问题均能完整回答。
  - InternVL3.5 DvD 的关键片段被召回，能够解释 vision server、language server、prefilling 和分离部署的效率意义。
  - 相比 `top_k=8`，噪声更可控。

- 缺点：
  - LLaVA 的 complex reasoning 数据生成细节仍然没有完全召回，说明这个问题不只是 top-k 问题，可能还需要 query rewriting 或 metadata filtering。
  - 对 InternVL3.5 DvD 问题，仍然混入 CLIP/Qwen2.5-VL 等无关来源，但没有明显污染答案。

- 典型失败 case：
  - Q3：即使 top_k 增加到 6，仍未完整回答 complex reasoning 的生成过程。
  - Q6：回答正确性较好，但 sources 中存在 mixed sources noise。

- 结论：
  - `top_k=6` 是当前最推荐的设置。它比 `top_k=4` 更适合复杂机制题，又不像 `top_k=8` 那样明显引入过多噪声。

---

### top_k = 8

- 优点：
  - 召回率最高。
  - Video-LLaVA 的问题表现最好，能召回摘要中的 misalignment before projection 和方法部分的 LanguageBind / unified visual representation。
  - Qwen2.5-VL 的回答更完整，能补充长视频秒级事件定位等细节。

- 缺点：
  - 明显更容易混入其他论文来源。
  - ViT 问题混入 Video-LLaVA；CLIP 问题混入 VIT；LLaVA 问题混入 InternVL 和 Video-LLaVA；InternVL 问题混入 CLIP/Qwen。
  - 虽然本次多数回答没有被污染，但 mixed sources noise 的风险明显上升。

- 典型失败 case：
  - Q3：召回更多来源后仍未解决 complex reasoning 细节缺失，且混入多篇无关论文。
  - Q6：能回答 DvD 主体，但仍混入 CLIP/Qwen 等无关片段。

- 结论：
  - `top_k=8` 适合跨论文比较题或长机制解释题，但不适合作为默认设置。除非有 metadata filtering 或论文选择功能，否则容易引入无关上下文。

---

## 6. Failure Case Analysis

### 6.1 Retrieval Incomplete

`top_k=2` 下最典型的问题是 retrieval incomplete。系统虽然能检索到正确论文，但上下文不足。例如 CLIP 的问题只能召回 N×N image-text matching 的核心段落，无法解释 cosine similarity 和 symmetric cross entropy loss。

### 6.2 Answer Incomplete

LLaVA 的 Q3 在所有 top-k 下都没有完整回答 complex reasoning 数据生成过程。这说明简单增大 top-k 不一定能解决所有问题。可能原因包括：

- query 没有足够贴近论文中的原始措辞；
- complex reasoning 的相关内容位于附录或较分散位置；
- 当前 embedding 对中文问题和英文论文的跨语言匹配仍不稳定；
- 需要指定只检索 `llava.pdf`，避免其他论文干扰。

### 6.3 Mixed Sources Noise

当 top-k 增大到 6 或 8 时，系统更容易混入其他论文。比如：

- ViT 问题中混入 Video-LLaVA；
- CLIP 问题中混入 VIT；
- InternVL DvD 问题中混入 CLIP 和 Qwen2.5-VL。

本次大多数 mixed sources 没有明显污染最终答案，但这仍然是一个潜在风险。后续需要引入 metadata filtering 或论文选择下拉框。

### 6.4 Prompt Conservatism Is Helpful

在 Q6 top_k=2 这种证据不足的情况下，模型选择拒答，而不是编造 DvD 的解释。这说明当前 prompt 的“只基于原文回答”约束是有效的。虽然 answer quality 低，但 faithfulness 仍然高。

---

## 7. Preliminary Conclusion

本轮 top-k 消融实验表明，`top_k` 对 RAG 系统的影响主要体现在“召回完整性”和“上下文噪声”之间的权衡。

- `top_k=2`：来源集中，忠实性高，但上下文不足，复杂问题容易失败。
- `top_k=4`：整体平衡，适合作为基础默认值。
- `top_k=6`：当前最稳，复杂问题召回更充分，噪声仍可控。
- `top_k=8`：召回率最高，但 mixed sources noise 明显增加。

因此，当前项目推荐设置为：

```text
chunk_size = 800
chunk_overlap = 120
top_k = 6
```

如果后续加入论文选择功能或 metadata filtering，则可以在“指定单篇论文”的情况下使用更大的 top-k，例如 `top_k=8`，以提高复杂问题的召回完整性。

---

## 8. Future Improvements

下一步建议从以下方向继续优化：

1. **Metadata filtering / 论文选择下拉框**  
   对于“这篇论文”或单篇论文细节问题，先选择论文，再只在对应 PDF 中检索。

2. **Query rewriting**  
   将中文问题自动改写成包含英文关键词的问题，例如：
   - “CLIP 的核心训练目标是什么？” → “What is CLIP contrastive pre-training objective, cosine similarity, symmetric cross entropy loss?”
   - “InternVL 的 DvD 是什么？” → “What is Decoupled Vision-Language Deployment in InternVL3.5?”

3. **Hybrid retrieval**  
   对缩写和术语问题，仅靠向量检索可能不稳定。后续可以考虑 keyword search + vector search 的混合检索。

4. **Reranking**  
   先召回更多候选 chunk，再用 reranker 对候选结果重新排序，减少 mixed sources noise。

5. **Question-type adaptive top-k**  
   不同问题类型使用不同 top-k：
   - 单篇事实题：`top_k=3/4`
   - 机制解释题：`top_k=6`
   - 跨论文比较题：`top_k=8`，但最好配合 metadata filtering

---

## 9. Final Takeaway

Top-k is not a simple “larger is better” parameter. It controls the trade-off between recall and noise.

For this MLLM paper RAG assistant, `top_k=6` currently provides the best balance. However, the remaining LLaVA and InternVL cases show that top-k alone is not enough. To further improve the system, the next step should be metadata filtering, query rewriting, and possibly hybrid retrieval.
