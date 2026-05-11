# RAG Chunk Size Ablation Summary

## 1. Experiment Setting

This experiment compares different chunk sizes in the MLLM paper RAG assistant.

Fixed setting:

- top_k = 4
- chunk_overlap = 15% of chunk_size
- embedding model = BAAI/bge-small-zh-v1.5
- generator model = configured in `.env`

Tested chunk sizes:

- 300
- 500
- 800
- 1200

## 2. Evaluation Questions

The questions cover ViT, CLIP, LLaVA, Video-LLaVA, Qwen2.5-VL and InternVL3.5.

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

## 4. Results Summary

请根据 `rag_ablation_results.csv` 人工填写以下观察。

### chunk_size = 300

- 优点：
- 缺点：
- 典型失败 case：

### chunk_size = 500

- 优点：
- 缺点：
- 典型失败 case：

### chunk_size = 800

- 优点：
- 缺点：
- 典型失败 case：

### chunk_size = 1200

- 优点：
- 缺点：
- 典型失败 case：

## 5. Preliminary Conclusion

请在完成评分后总结：

1. 哪个 chunk_size 整体最稳？
2. 哪些问题在小 chunk 下表现更好？
3. 哪些问题需要更大的 chunk 才能回答完整？
4. 是否出现混合多篇论文的问题？
5. 后续 top-k 实验应该重点关注什么？



# RAG Chunk Size Scoring Summary

## Average Scores

| chunk_size | avg_answer_quality | avg_faithfulness | observation |
|---|---:|---:|---|
| 300 | 2.67 | 4.50 | 两极分化明显：ViT/CLIP 好，但 LLaVA、Qwen、InternVL 容易上下文不足或拒答。 |
| 500 | 3.33 | 4.33 | 比 300 稳定一些，但 CLIP/InternVL 有混源问题，LLaVA 仍不完整。 |
| 800 | 4.50 | 4.83 | 整体最稳：Q1/Q2/Q4/Q5 表现很好，Q3 部分正确，Q6 基本可用但有噪声。 |
| 1200 | 3.50 | 4.83 | 概念型问题表现好，但 LLaVA/InternVL 检索变差，容易混入无关大块。 |

## Preliminary Conclusion

- 当前实验下，`chunk_size=800` 是最推荐的默认值。
- `chunk_size=300` 对短概念/关键词问题可用，但复杂机制题容易信息不足。
- `chunk_size=1200` 对 ViT、CLIP、Video-LLaVA 这类概念题很好，但对 LLaVA 数据构造、InternVL DvD 容易检索漂移或混源。
- InternVL 的 DvD 问题不是生成器问题，主要是检索召回问题；后续应加入全称关键词、提高 top_k 或做 metadata filtering。





# 一、最终总体结论

这次 chunk size 实验的核心结论很清楚：

| chunk_size |   平均回答质量 |    平均忠实性 | 结论                            |
| ---------- | -------: | -------: | ----------------------------- |
| 300        |     2.67 |     4.50 | 片段太短，ViT/CLIP 可以，但复杂问题容易拒答    |
| 500        |     3.33 |     4.33 | 比 300 稳，但 CLIP/InternVL 有混源问题 |
| 800        | **4.50** | **4.83** | **当前最稳，建议作为默认值**              |
| 1200       |     3.50 |     4.83 | 概念题好，但部分问题检索漂移、混入无关论文         |

所以你现在可以在 summary 里写：

> 在本轮实验中，`chunk_size=800` 表现最稳定。它在保持上下文完整性的同时，没有像 `chunk_size=1200` 那样明显引入过多无关内容，也比 `chunk_size=300/500` 更适合回答机制解释类问题。


# 三、为什么有些回答质量低，但 faithfulness 高？

这个很重要。

比如 Q3、Q5、Q6 有些回答是：

```text
根据当前检索到的论文片段无法确定。
```

这种回答虽然没解决问题，所以 `answer_quality_score` 很低；但它没有胡编，而是忠实地拒答，所以 `faithfulness_score` 可以给 5。

你要理解这两个分数的区别：

```text
answer_quality_score = 答案有没有真正解决问题
faithfulness_score = 答案有没有忠于 retrieved sources
```

所以：

```text
拒答但不胡编：answer_quality 低，faithfulness 高
编得很流畅但 sources 不支持：answer_quality 可能看起来高，faithfulness 应该低
```

---

# 四、这次最重要的失败案例

## 1. LLaVA 的 Q3 一直不稳定

Q3 是：

> LLaVA 的 visual instruction tuning data 是如何由 GPT-4 生成的？

它在不同 chunk size 下表现都一般：

```text
300：直接拒答
500：部分回答
800：部分回答
1200：又拒答
```

这说明问题不只是 chunk size，而是：

```text
retrieval 没有稳定召回 LLaVA 数据构造的核心段落
```

后面改进方式：

```text
1. top_k 提高到 6 或 8
2. 问题里加入英文关键词：
   visual instruction tuning data
   GPT-4
   conversation
   detailed description
   complex reasoning
3. 做论文选择下拉框，只检索 llava.pdf
```

---

## 2. InternVL 的 Q6 主要是缩写检索问题

Q6 是：

> InternVL3.5 的 DvD 是什么？

它经常失败或混源。

原因是：

```text
DvD 这个缩写太短，embedding 不一定能稳定匹配到
Decoupled Vision-Language Deployment
```

后面提问时应该写完整：

```text
InternVL3.5 的 DvD（Decoupled Vision-Language Deployment）是什么？
```

甚至可以继续补充：

```text
请结合 vision server、language server、prefilling 和 inference efficiency 解释。
```

---

## 3. chunk_size=800 是目前最佳默认值

`chunk_size=800` 下：

```text
Q1 ViT：5
Q2 CLIP：5
Q3 LLaVA：3
Q4 Video-LLaVA：5
Q5 Qwen2.5-VL：5
Q6 InternVL：4
```

这说明它对大多数论文问题都比较稳。

你后续 `app.py` 里可以暂时保留：

```python
chunk_size = 800
chunk_overlap = 120
top_k = 4
```


## Preliminary Conclusion

In this chunk size ablation experiment, chunk_size=800 achieved the best overall balance between retrieval precision and context completeness. Compared with chunk_size=300, it provided richer context for mechanism-level questions such as Video-LLaVA's alignment before projection and Qwen2.5-VL's dynamic resolution/time encoding. Compared with chunk_size=1200, it introduced less irrelevant cross-paper noise.

The main failure cases were concentrated on LLaVA's visual instruction data generation and InternVL3.5's DvD mechanism. These failures were mainly caused by retrieval failure, incomplete retrieval, and keyword mismatch, rather than pure generation errors. In particular, acronym-based queries such as "DvD" require the full term "Decoupled Vision-Language Deployment" to improve retrieval stability.

Therefore, the current default setting can be set as:

- chunk_size = 800
- chunk_overlap = 120
- top_k = 4

Future experiments should further compare different top_k values and introduce metadata filtering to restrict retrieval to a selected paper.
```