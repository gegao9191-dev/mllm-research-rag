# RAG Concepts Notes

## 0. 本项目中的 RAG 是在做什么？

本项目的目标不是做一个普通的聊天机器人，而是构建一个用于阅读视觉多模态论文的 RAG 助手。

普通大模型虽然具备通用知识，但它并不天然知道我本地论文库中的具体内容。RAG（Retrieval-Augmented Generation，检索增强生成）的核心思想是：先从外部文档中检索出与问题最相关的内容，再把这些内容作为上下文交给大模型生成回答。

在本项目中，RAG 的完整流程可以表示为：

```text
PDF 原文
↓
文本抽取
↓
切 chunk
↓
embedding
↓
建立向量索引
↓
问题 embedding
↓
检索 top-k 相关 chunk
↓
把 chunk + 问题放进 prompt
↓
LLM 生成回答
↓
显示答案和引用来源



## 1. Document Loading：PDF 是怎么被读进来的？

Document Loading 指的是把本地文件系统中的论文 PDF 加载到程序中。

在本项目中，论文被放在：

```text
data/papers/
```

程序会扫描这个文件夹，找到其中的 PDF 文件，然后逐个读取。这里的 PDF 还不是模型能够直接理解的形式。PDF 本质上是一种复杂的排版文件，里面包含文字、字体、坐标、图片、链接、注释、对象结构等内容。

所以 Document Loading 只是第一步：找到文件，并打开文件。真正关键的是下一步——Text Extraction，也就是从 PDF 里提取可读文本。

如果 Document Loading 出错，常见表现包括：

* 找不到 `data/papers/` 文件夹；
* 文件夹中没有 PDF；
* PDF 路径不正确；
* PDF 文件损坏或无法打开。

---

## 2. Text Extraction：PyMuPDF 为什么比默认 PDF reader 更稳？

Text Extraction 指的是从 PDF 中抽取正文文本。

一开始使用默认 PDF reader 时，本项目出现过乱码问题，例如检索结果中出现：

```text
endobj
/Annot
/URI
/Rect
```

这说明程序没有正确抽取论文正文，而是读到了 PDF 底层对象结构、链接标注或二进制内容。

后来改用 PyMuPDF 后，问题得到解决。PyMuPDF 更稳定的原因是它能够按页面解析 PDF，并通过类似：

```python
page.get_text("text", sort=True)
```

的方式提取页面上的可读文本。相比一些默认 PDF reader，PyMuPDF 对论文类 PDF、复杂排版、双栏文本、页面对象的处理通常更可靠。

在 RAG 系统中，Text Extraction 是非常关键的一步。因为如果这一步读出来的是乱码，那么后面的 chunking、embedding、retrieval 和 generation 都会跟着失败。

可以理解为：

```text
错误文本进入 RAG
↓
错误文本被切块
↓
错误文本被 embedding
↓
检索到错误内容
↓
大模型基于错误内容回答
```

所以 RAG 的质量首先取决于文档解析质量。

常见失败情况：

* PDF 是扫描版，无法直接提取文字；
* PDF 经过特殊压缩或加密；
* 双栏论文的阅读顺序混乱；
* 图表、公式、表格无法完整解析；
* 页眉、页脚、参考文献干扰正文检索。

---

## 3. Chunking：为什么要把论文切成小块？

Chunking 指的是把长文档切成多个较短的文本片段。

一篇论文可能有几十页，包含摘要、引言、方法、实验、表格、参考文献等内容。大模型无法也不应该每次都读取整篇论文。原因有三个：

第一，大模型有上下文窗口限制。即使是长上下文模型，也不适合每次塞入全部论文。

第二，用户的问题往往只和论文中的某一小部分有关。例如问 “CLIP 的 zero-shot 分类是怎么实现的”，只需要检索 CLIP 方法部分和相关图示，不需要整篇论文。

第三，检索需要更细粒度的文本单位。如果整篇论文作为一个整体，向量表示会过于粗糙，难以精确匹配具体问题。

因此，我们会把论文切成 chunk，例如：

```text
chunk_size = 800
chunk_overlap = 120
```

其中：

* `chunk_size` 表示每个片段大约多长；
* `chunk_overlap` 表示相邻片段之间保留多少重叠内容。

重叠的作用是避免重要信息刚好被切断。例如一个概念的定义在前一个 chunk 的末尾，解释在下一个 chunk 的开头，如果完全不重叠，检索时可能只拿到半截信息。

chunk 太小的问题：

* 上下文不完整；
* 容易缺少定义、原因、实验结果之间的联系；
* 大模型可能因为信息不足而拒答。

chunk 太大的问题：

* 检索不够精准；
* 一个 chunk 里混入太多无关内容；
* top-k 中无关信息增多，可能造成回答混乱。

因此，chunk size 不是越大越好，也不是越小越好。Week 2 的实验重点之一就是比较不同 chunk size 对回答质量的影响。

---

## 4. Embedding：为什么文本可以变成向量？

Embedding 指的是把文本转换成一组数字向量。

例如下面三句话：

```text
A: CLIP learns image-text representations with contrastive learning.
B: CLIP aligns images and text through a contrastive objective.
C: The weather is sunny today.
```

从字面上看，A 和 B 不完全一样，但语义非常接近；C 和它们无关。Embedding model 会把这些句子映射到向量空间中，使语义相近的文本在向量空间里距离更近。

可以粗略理解为：

```text
A → [0.12, -0.35, 0.78, ...]
B → [0.10, -0.32, 0.75, ...]
C → [-0.81, 0.22, 0.04, ...]
```

这样，系统就可以通过计算向量之间的相似度，判断用户问题和哪些论文片段最相关。

Embedding model 和 LLM 不是同一个东西。

在本项目中：

* Embedding model 负责“检索前的语义表示”；
* DeepSeek/Qwen 这类 LLM 负责“基于检索内容生成答案”。

也就是说：

```text
Embedding model：负责找资料
LLM：负责读资料并组织回答
```

本项目中一个值得注意的问题是：论文大多是英文，而用户问题是中文。因此 embedding model 是否具备跨语言检索能力，会影响中文问题检索英文论文的效果。Week 2 可以专门比较中文问题和英文问题的检索质量。

---

## 5. Vector Index：向量索引解决什么问题？

Vector Index 指的是把所有 chunk 的 embedding 向量组织起来，方便快速检索。

如果论文库中只有几十个 chunk，暴力计算问题向量和所有 chunk 向量之间的相似度也可以。但当论文数量变多，chunk 数量达到几千、几万甚至更多时，暴力搜索会变慢。

向量索引的作用类似图书馆目录：

```text
所有论文 chunk
↓
转换成 embedding
↓
建立向量索引
↓
用户提问时快速找到最相关的 chunk
```

在 Week 1 的最小版本中，可以使用内存索引。它的优点是简单，适合快速跑通流程；缺点是每次重启程序都要重新读取 PDF、重新切 chunk、重新 embedding 和建索引。

后续可以升级到 FAISS 或 Chroma 这类向量数据库/向量索引工具。

内存索引：

* 简单；
* 适合 demo；
* 程序关闭后索引消失；
* 每次启动都要重新构建。

FAISS / Chroma：

* 适合更大的文档库；
* 可以持久化保存索引；
* 适合后续扩展；
* 更像真正的知识库系统。

---

## 6. Retriever：top-k 是什么意思？

Retriever 指的是检索器。它负责根据用户问题，从向量索引中找出最相关的若干个 chunk。

`top-k` 表示取相似度最高的前 k 个片段。

例如：

```text
top_k = 4
```

表示每次用户提问时，系统会检索出最相关的 4 个 chunk，然后把它们作为上下文交给大模型。

top-k 太小的问题：

* 可能漏掉关键片段；
* 对跨论文比较问题不友好；
* 容易导致模型说“根据当前上下文无法确定”。

top-k 太大的问题：

* 可能混入无关片段；
* 多篇论文内容可能被混在一起；
* prompt 变长，增加成本；
* 大模型可能被无关上下文干扰。

因此，top-k 也不是越大越好。

通常：

* 单篇论文事实型问题：top_k=3 或 4 可能够用；
* 跨论文比较问题：top_k=6 或 8 可能更合适；
* 拒答测试问题：top-k 太大可能增加误检索风险。

Week 2 的一个核心实验就是比较不同 top-k 对检索质量和回答质量的影响。

---

## 7. Prompt Template：为什么要限制模型“只基于原文回答”？

Prompt Template 指的是把用户问题和检索到的论文片段组织成一个固定格式的提示词，再交给大模型回答。

一个严肃的论文 RAG 助手不能让大模型自由发挥。因为大模型本身有大量预训练知识，可能会把外部知识、猜测和检索内容混在一起，导致回答看似合理但无法从论文中验证。

所以需要在 prompt 中加入约束，例如：

```text
你是一个视觉多模态论文阅读助手。
请严格基于检索到的论文片段回答问题。
如果上下文不足，请明确说“根据当前检索到的论文片段无法确定”。
不要使用外部知识补全。
不要编造 benchmark、数据集、数字或实验结论。
```

这类约束的目的不是让模型变笨，而是让它更可靠。

对于科研场景来说，可靠性比流畅性更重要。一个回答如果没有原文依据，即使语言很漂亮，也不能算好回答。

Prompt Template 的作用可以概括为：

```text
控制回答范围
限制幻觉
规范回答格式
引导模型引用来源
提高答案的学术可靠性
```

---

## 8. Generator：DeepSeek/Qwen 在 RAG 中扮演什么角色？

Generator 指的是最终生成答案的大语言模型。在本项目中，DeepSeek 或 Qwen 扮演的是 Generator 的角色。

需要注意的是，在 RAG 系统中，大模型不是直接回答用户问题，而是基于检索到的论文片段回答问题。

普通 LLM 问答：

```text
用户问题
↓
LLM 根据自身知识回答
```

RAG 问答：

```text
用户问题
↓
Retriever 检索论文片段
↓
Prompt 拼接问题和片段
↓
LLM 基于片段生成回答
```

因此，DeepSeek/Qwen 的作用不是“记住论文”，而是：

```text
阅读检索片段
理解用户问题
整合相关内容
用中文组织成答案
在上下文不足时拒答
```

这意味着 RAG 的效果不只取决于大模型能力，也取决于前面的检索质量。

如果检索到的内容是错的，大模型再强也可能答错。

如果检索到的内容不完整，大模型可能回答不完整，或者拒答。

如果 prompt 没有约束好，大模型可能使用自己的外部知识补全，造成幻觉。

---

## 9. Source Citation：为什么要显示 retrieved sources？

Source Citation 指的是显示模型回答所依据的检索片段。

在本项目中，回答后面会显示：

```text
Retrieved Sources
Source 1: xxx.pdf, page x
Source 2: xxx.pdf, page y
...
```

这样做非常重要，原因有三个。

第一，可验证。用户可以看到答案是否真的来自论文，而不是模型编造。

第二，可诊断。如果回答错了，可以检查是检索错了，还是模型理解错了。

第三，可学习。对于论文阅读助手来说，source 不只是证据，也是用户进一步精读论文的入口。

没有 source 的 RAG 系统很难评估，因为你不知道模型到底是基于什么回答的。

有 source 后，就可以区分两种错误：

```text
检索失败：
source 本身就不相关，说明 retriever 没找到正确材料。

生成失败：
source 是相关的，但模型总结错了，说明 generator 或 prompt 有问题。
```

这也是 Week 2 做失败案例分析的基础。

---

## 10. Hallucination：为什么 RAG 仍然可能胡说？

Hallucination 指的是模型生成了没有依据、与原文不一致或虚构的内容。

RAG 可以降低 hallucination，但不能完全消除 hallucination。

原因包括：

### 1. 检索错误

如果 retriever 找到的是不相关 chunk，大模型可能会被错误上下文误导。

### 2. 检索不完整

如果只检索到方法部分，但没有检索到实验结果，模型可能会根据有限信息推测实验结论。

### 3. chunk 切分不合理

如果关键定义和解释被切开，模型可能只看到半截内容，导致误解。

### 4. top-k 设置不合适

top-k 太小可能漏掉关键信息；top-k 太大可能混入无关信息。

### 5. prompt 约束不够强

如果没有明确要求“只基于上下文回答”，模型可能会使用自己的外部知识补全答案。

### 6. 用户问题过于模糊

例如：

```text
这篇论文讲了什么？
```

当知识库里有多篇论文时，“这篇论文”本身不明确。系统只能猜，容易混淆。

### 7. 跨论文比较问题更难

例如：

```text
CLIP 和 LLaVA 的区别是什么？
```

这类问题需要同时检索两篇论文的相关片段。如果只检索到 LLaVA，没有检索到 CLIP，回答就可能不完整。

因此，一个好的 RAG 系统不应该承诺“永远正确”，而应该做到：

```text
尽量检索正确材料
尽量基于材料回答
显示来源
在材料不足时拒答
记录失败案例
通过实验不断改进
```

---

## 11. 本项目中如何判断 RAG 回答质量？

本项目可以从三个维度评价 RAG 回答：

### 1. Retrieval Quality：检索质量

问题：系统有没有找到正确论文、正确章节、正确页面？

评分参考：

```text
5 = 检索到完全相关的论文片段
4 = 检索到相关论文，但片段略不完整
3 = 检索到部分相关内容
2 = 检索内容弱相关
1 = 检索错误论文或错误章节
0 = 没有有效检索结果
```

### 2. Faithfulness：忠实性

问题：回答是否忠于检索到的原文？

评分参考：

```text
5 = 完全基于原文，没有编造
4 = 基本忠实，少量概括
3 = 有部分推断，但没有明显错误
2 = 混入较多外部知识或推测
1 = 明显与原文不一致
0 = 完全幻觉
```

### 3. Usefulness：有用性

问题：回答是否真正帮助用户理解论文？

评分参考：

```text
5 = 结构清晰，解释充分，有助于深入理解
4 = 回答正确，但解释略简单
3 = 基本回答问题，但不够深入
2 = 回答很浅，帮助有限
1 = 几乎没有解决问题
0 = 无效回答
```

这三个维度要分开看。

一个回答可能检索正确，但生成得不好；也可能生成得很流畅，但检索来源不对。Week 2 的目标就是把这些情况区分开。

---

## 12. 面试中如何解释这个 RAG 项目？

不能只说：

```text
我用了 LlamaIndex 做了一个 PDF 问答系统。
```

更好的表达是：

```text
我做了一个面向视觉多模态论文阅读的 RAG 助手。系统首先使用 PyMuPDF 对论文 PDF 进行页面级文本抽取，然后将文本切分为带 overlap 的 chunk，使用 embedding model 将 chunk 转换为向量并建立向量索引。用户提问后，系统将问题也编码为向量，通过 top-k 检索找到最相关的论文片段，再把检索片段和问题一起输入到 DeepSeek/Qwen 这类大语言模型中生成中文回答。为了提升科研场景下的可信度，我在 prompt 中限制模型只能基于原文回答，并显示 retrieved sources，方便检查答案是否忠于论文。后续我进一步对 chunk size、top-k、中文/英文 query 和 hallucination case 做了实验分析。
```

这段话能够体现你不是只会调 API，而是理解了 RAG 的完整链路和实验评估意识。

---

## 13. 当前版本的局限性

当前 Week 1 版本仍然有一些明显局限：

1. 索引是内存索引，程序重启后需要重新构建；
2. 对 PDF 表格、公式、图像内容的解析仍然有限；
3. 中文问题检索英文论文时可能不稳定；
4. 多文档比较问题容易出现检索不全；
5. 还没有系统比较 chunk size 和 top-k；
6. 没有自动化评估指标，主要依赖人工评分；
7. 对 “this paper” 这类模糊指代问题支持不好；
8. 还没有加入论文选择下拉框或 metadata filtering。

这些局限会成为 Week 2 的改进方向。

---

## 14. Week 2 的改进方向

Week 2 重点不是继续堆功能，而是把项目做得更像一个科研实验：

```text
1. 对比不同 chunk_size
2. 对比不同 top_k
3. 对比中文 query 和英文 query
4. 记录 retrieved sources 是否正确
5. 记录回答是否忠于原文
6. 整理 hallucination case
7. 总结失败原因
8. 更新 README
```
















你这个问题非常重要，因为它决定了你接下来是“真正掌握项目”，还是只是“跑通一个 demo”。

我先给你一个明确标准：

> 对于 `mllm-research-rag` 这个两周项目，你不需要把 LlamaIndex、PyMuPDF、embedding model 的源码级实现都吃透，但你必须能从工程链路、实验设计、失败诊断、参数影响、项目局限五个层面讲清楚。

也就是说，面试官不会期待你把 LlamaIndex 内部每个类的源码都背下来，但会期待你能回答：

```text
你为什么这样设计？
RAG 每一步在做什么？
为什么会失败？
你怎么评估？
chunk_size 和 top_k 为什么这么设？
你这个项目和普通 PDF 问答有什么区别？
```

---

# 一、这个项目你应该掌握到什么程度？

我建议你把掌握程度分成 4 层。

## 第一层：能跑通，不够

这是 Week 1 的水平：

```text
我能上传 PDF
我能问答
我能显示 sources
我能调用 DeepSeek/Qwen API
```

这个层次只能说明你会跟教程搭 demo。

如果面试官问：

> 你这个 RAG 为什么有时候答不出来？

你答不上来，就会显得比较浅。

所以这个层次不能作为最终目标。

---

## 第二层：能解释完整链路，这是最低简历标准

你必须能流畅讲出：

```text
PDF 通过 PyMuPDF 做页面级文本抽取
文本被切成 chunk
chunk 经过 embedding model 变成向量
所有向量建立 vector index
用户问题也被转成向量
retriever 根据相似度找 top-k 个相关 chunk
prompt 把 chunk 和问题拼起来
DeepSeek/Qwen 基于上下文生成中文回答
系统返回答案和 retrieved sources
```

这段你要讲到非常自然。

如果你能讲到这个程度，这个项目就可以放简历了。

---

## 第三层：能解释参数和失败，这是比较加分的水平

这就是你现在 Week 2 正在做的事。

你要能回答：

```text
chunk_size 太小会怎样？
chunk_size 太大会怎样？
top_k 太小会怎样？
top_k 太大会怎样？
为什么 Q3 LLaVA 问题有时检索不到？
为什么 InternVL 的 DvD 问题容易失败？
中文问题检索英文论文为什么不稳定？
为什么 RAG 仍然可能幻觉？
怎么判断是 retrieval failure 还是 generation failure？
```

这一层非常关键，因为它说明你不是只会调库，而是真的理解 RAG 系统的行为。

你现在做的 chunk_size 实验和 top-k 实验，本质就是为了达到这个层次。

---

## 第四层：能提出改进方向，这是优秀项目水平

你不一定现在全部实现，但要能说出后续怎么升级：

```text
1. 加 metadata filtering，让用户选择只检索某篇论文；
2. 加 query rewriting，把中文问题改写成中英混合检索 query；
3. 加 hybrid search，结合关键词 BM25 和 embedding 检索；
4. 使用 Chroma / FAISS 持久化向量索引；
5. 加 reranker，对初步检索结果重新排序；
6. 对 retrieval quality、faithfulness、answer usefulness 做更系统评估；
7. 对 embedding model 做对比，测试中英文跨语言检索效果；
8. 对常见失败 case 建立 failure taxonomy。
```

你能讲出这些，面试官会觉得你这个项目虽然不大，但你确实有工程和科研意识。

---

# 二、你需要非常详细了解代码吗？

答案是：**需要了解你自己写的核心代码，但不需要深入 LlamaIndex 内部源码。**

你需要掌握的代码范围是：

## 你必须非常清楚的部分

### 1. PDF 读取

你要知道：

```python
page.get_text("text", sort=True)
```

是在用 PyMuPDF 从 PDF 页面中抽取文本。

你也要知道为什么不用原来的默认 reader：因为之前读出了 PDF 对象、链接、乱码，说明文本抽取失败。

---

### 2. Document / metadata

你要知道每一页被包装成一个 Document，并带有：

```python
metadata={
    "file_name": pdf_path.name,
    "page": page_idx,
}
```

这个 metadata 很重要，因为后面 sources 里能显示：

```text
CLIP.pdf, page 3
```

就是靠它。

---

### 3. chunk 切分

你要知道：

```python
SentenceSplitter(chunk_size=800, chunk_overlap=120)
```

在做什么。

你不用知道 `SentenceSplitter` 内部所有算法细节，但要知道它把长文本切成可检索的小段，并且 overlap 是为了避免上下文断裂。

---

### 4. embedding model

你要知道：

```python
HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
```

不是训练模型，而是调用一个预训练 embedding model，把 chunk 和问题转成向量。

---

### 5. index 和 query_engine

你要知道：

```python
VectorStoreIndex.from_documents(...)
index.as_query_engine(similarity_top_k=...)
```

大致对应：

```text
建立向量索引
↓
检索 top-k chunk
↓
交给 LLM 回答
```

---

### 6. prompt template

你要非常清楚自己的 prompt 为什么这样写：

```text
只基于检索到的论文片段回答
上下文不足就说无法确定
不要编造 benchmark、数据集、数字或实验结论
```

这是你控制幻觉的关键。

---

### 7. 评分实验脚本

你要能解释：

```text
chunk_size_experiment.py 固定 top_k，对比不同 chunk_size
top_k_experiment.py 固定 chunk_size，对比不同 top_k
结果写入 CSV
人工从 answer_quality 和 faithfulness 两个维度打分
```

这个是你的项目从 demo 变成“实验项目”的关键。

---

## 你暂时不需要深入的部分

你现在不需要深入到：

```text
LlamaIndex 内部如何实现 node parsing；
FAISS 的 HNSW / IVF / PQ 算法；
embedding model 训练损失的数学推导；
DeepSeek API 内部推理机制；
PyMuPDF 的 PDF rendering pipeline；
```

这些当然以后可以学，但不是你这个两周项目必须掌握的。

你现在最重要的是能讲清楚“我在这个项目里做了什么、为什么这么做、实验发现了什么”。

---

# 三、这个项目放简历，应该写到什么程度？

你不要把它写成：

```text
基于 LlamaIndex 搭建 RAG 问答系统。
```

这太普通了。

你应该写成：

```text
MLLM Research RAG Assistant：构建面向视觉多模态论文阅读的本地 RAG 助手，支持 PDF 解析、向量检索、中文问答与引用来源展示；基于 ViT、CLIP、LLaVA、Video-LLaVA、Qwen2.5-VL、InternVL 等论文构建测试集，并对 chunk size 与 top-k 进行消融实验，分析 retrieval failure、mixed sources、answer incompleteness 和 hallucination case。
```

如果写成英文简历，可以这样：

```text
Built a local RAG assistant for reading vision-language and video-MLLM papers, supporting PDF parsing, vector retrieval, Chinese QA, and source citation. Conducted chunk-size and top-k ablation experiments on six representative MLLM papers, and analyzed failure cases including retrieval failure, mixed sources, incomplete answers, and hallucination.
```

这比单纯写“做了 PDF 问答”强很多。

---

# 四、面试官可能会怎么问你？

我给你列一组高频问题。你不用现在全背，但 Week 2 结束后应该都能回答。

## 基础链路类

```text
1. 你这个 RAG 系统完整流程是什么？
2. 为什么需要 chunking？
3. chunk_overlap 是干什么的？
4. embedding model 和 LLM 有什么区别？
5. vector index 是训练出来的吗？
6. top-k 是什么意思？
7. 为什么要显示 retrieved sources？
```

## 工程实现类

```text
8. 你为什么用 PyMuPDF？
9. 你怎么处理 PDF 乱码？
10. 你的 metadata 里存了什么？
11. 你怎么调用阿里云百炼 / DeepSeek API？
12. 你的 prompt 是怎么设计的？
13. .env 为什么不能上传 GitHub？
```

## 实验分析类

```text
14. chunk_size=300、500、800、1200 的效果有什么区别？
15. 为什么你最后推荐 chunk_size=800？
16. top_k=2、4、6、8 有什么差异？
17. 为什么 top_k 不是越大越好？
18. 你的 answer_quality 和 faithfulness 怎么评分？
19. 什么是 retrieval failure？
20. 什么是 generation failure？
```

## 失败案例类

```text
21. 为什么 LLaVA 的 visual instruction data 问题不稳定？
22. 为什么 InternVL 的 DvD 问题容易检索失败？
23. 为什么中文问题检索英文论文可能出问题？
24. RAG 为什么仍然可能 hallucinate？
25. 你如何判断模型是“不会答”还是“没检索到”？
```

## 改进方向类

```text
26. 如果继续优化，你会怎么做？
27. 为什么需要 metadata filtering？
28. 什么是 hybrid search？*混合搜索
29. reranker 有什么用？
30. 为什么需要持久化向量数据库？
```

如果你能把这些问题讲清楚，这个项目就已经足够放简历了。

---

# 五、你现在最容易误解的一点

你可能会觉得：

> 这个项目只做了两周，会不会太浅？

其实不是。项目深不深，不只看时间长短，而看你是否做到了这几件事：

```text
1. 有完整工程链路；
2. 有真实使用场景；
3. 有明确技术选择；
4. 有实验对比；
5. 有失败案例分析；
6. 有局限性反思；
7. 有后续改进路线。
```

很多人做 RAG 项目只做到：

```text
上传 PDF → 问答
```

你现在已经往下做了：

```text
chunk size ablation
top-k ablation
failure taxonomy
faithfulness scoring
MLLM paper reading use case
```

这就比普通 demo 深很多。

---

# 六、你应该达到的最终表达水平

Week 2 结束后，你应该能用 1 分钟讲清楚这个项目：

> 我做了一个面向视觉多模态论文阅读的 RAG 助手，用于辅助阅读 ViT、CLIP、LLaVA、Video-LLaVA、Qwen2.5-VL 和 InternVL 等代表性论文。系统使用 PyMuPDF 解析 PDF，按 chunk size 和 overlap 切分文本，使用预训练 embedding model 将文本片段转成向量，并建立向量索引。用户提问后，系统检索 top-k 相关论文片段，再将片段和问题一起输入 DeepSeek/Qwen 生成中文回答，并显示来源页码。Week 2 我进一步对 chunk size 和 top-k 做了消融实验，发现 chunk_size=800、top_k=6 在当前论文库上比较稳定，同时分析了 retrieval failure、mixed sources、answer incomplete 等失败类型。这个项目让我理解了 RAG 不只是能跑通，还需要评估检索质量、回答忠实性和失败模式。

这段话如果你能自然说出来，就说明你掌握得比较好了。

---

# 七、你接下来应该怎么把握学习深度？

我建议你分成三档。

## 必须掌握

```text
RAG 完整流程
PDF 解析
chunking
embedding
vector index
top-k
prompt
source citation
chunk_size/top-k 实验结论
failure case 分析
```

这是简历和面试必备。

## 应该了解

```text
Chroma / FAISS 持久化索引
metadata filtering
hybrid search
reranker
query rewriting
中英文 query 对 retrieval 的影响
RAG evaluation 指标
```

这是项目升级方向。

## 暂时不必深挖

```text
embedding model 训练源码
LlamaIndex 内部实现细节
向量数据库底层索引算法
大模型推理底层优化
```

这些等你后面真做大规模 RAG 或论文研究再深入。

---

# 八、我对你这个项目的定位建议

你不要把它当成最终项目。

它应该是你的：

> **第一个 MLLM 方向的工程入口项目。**

它的作用是：

```text
1. 让你熟悉 LLM 应用开发；
2. 让你读懂多模态论文；
3. 让你理解 RAG 系统；
4. 让你建立实验评估意识；
5. 为后面的 VLM Evaluation Lab 和 Qwen-VL fine-tuning 做铺垫。
```

所以你不需要在这个项目上无限深挖 2 个月，但也不能只停留在“代码能跑”。

你现在最合理的目标是：

> 用 Week 2 把它打磨成一个“能讲清楚、能展示、能回答面试追问”的小型项目，然后进入 Week 3 的 VLM Evaluation Lab。






















你这个问题问得非常关键。你现在问到的是 RAG 的核心本质：

> **RAG 里的 embedding 不是你这个项目现场训练出来的，而是用一个已经预训练好的 embedding model，把你的 chunk 做一次“特征提取”。**

所以你没有看到训练过程，是正常的。

---

# 1. 你现在的 RAG 并没有训练 embedding model

你现在代码里大概用了这一句：

```python
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)
```

这是什么意思？

意思是：

> 从 Hugging Face 下载一个已经训练好的 embedding 模型，然后用它把论文 chunk 转成向量。

所以你的流程不是：

```text
你的 6 篇 PDF
↓
训练一个 embedding 模型
↓
模型学会论文语义
```

而是：

```text
已经训练好的 embedding model
↓
读取你的 PDF chunk
↓
对每个 chunk 做一次前向计算
↓
得到每个 chunk 的 embedding 向量
```

这就像你以前做图像任务时，用一个已经在 ImageNet 上预训练好的 ResNet 提取图片特征一样。

你不是重新训练 ResNet，而是：

```text
图片
↓
预训练 ResNet
↓
图像特征向量
```

RAG 里也是类似的：

```text
文本 chunk
↓
预训练 embedding model
↓
文本语义向量
```

---

# 2. embedding model 是怎么知道语义的？

因为它在发布之前，已经被别人用大量数据训练过了。

比如 embedding 模型训练时，通常会见过大量这样的数据：

```text
query: What is contrastive learning?
positive passage: Contrastive learning is a method that learns representations by pulling similar pairs closer and pushing dissimilar pairs apart.
negative passage: A transformer is a neural network architecture based on self-attention.
```

训练目标是：

```text
让 query 和 positive passage 的向量更接近
让 query 和 negative passage 的向量更远
```

所以训练完以后，它就学会了：

```text
“contrastive learning”
“image-text matching”
“对比学习”
“图文匹配”
```

这些表达虽然字面不同，但语义接近。

所以你问：

```text
CLIP 的核心训练目标是什么？
```

它可能会把这个问题向量和论文里的：

```text
contrastive pre-training objective
image-text pairs
symmetric cross entropy loss
```

这些 chunk 拉得比较近。

这就是 embedding model 的语义能力来源。

---

# 3. 文本是怎么被转换成 embedding 的？

更底层一点，流程大概是这样：

```text
文本 chunk
↓
tokenizer 分词
↓
变成 token IDs
↓
输入 embedding model
↓
Transformer/BERT 类编码器处理
↓
得到每个 token 的 hidden states
↓
pooling 成一个句子/段落向量
↓
归一化
↓
得到最终 embedding
```

比如一个 chunk：

```text
CLIP uses contrastive learning to align images and text.
```

会先被分成 token：

```text
["CLIP", "uses", "contrastive", "learning", "to", "align", "images", "and", "text"]
```

然后变成数字 ID：

```text
[1234, 563, 7821, 2901, ...]
```

再输入 embedding model，最后输出一个向量：

```text
[0.021, -0.114, 0.356, ..., 0.087]
```

这个向量可能有几百维，比如 384 维、768 维、1024 维。

这个向量不是人手设计的，而是模型参数计算出来的。

---

# 4. 那 RAG 里的“建立向量索引”是在训练吗？

不是。

建立向量索引不是训练模型，而是把已经算好的向量存起来，方便快速搜索。

比如你的 6 篇论文切成了 1000 个 chunk。

每个 chunk 都会变成一个向量：

```text
chunk_1 → vector_1
chunk_2 → vector_2
chunk_3 → vector_3
...
chunk_1000 → vector_1000
```

然后系统建立一个向量索引：

```text
vector_1
vector_2
vector_3
...
vector_1000
```

当你提问时：

```text
CLIP 的 contrastive pre-training objective 是什么？
```

系统也会把这个问题变成一个 query vector：

```text
question → query_vector
```

然后计算：

```text
query_vector 和 chunk_1 的相似度
query_vector 和 chunk_2 的相似度
query_vector 和 chunk_3 的相似度
...
```

最后取最相似的 top-k 个 chunk。

所以这里本质是：

```text
不是训练
而是相似度搜索
```

---

# 5. 它为什么能知道 PDF chunk 的语义？

准确说，它不是“真正读懂了论文”，而是：

> 预训练 embedding model 已经学会了把语义相近的文本映射到相近的向量空间。

所以当你给它一个新 chunk，它可以根据自己已有的语言表示能力，把这个 chunk 放到语义空间里的某个位置。

比如：

```text
chunk A: CLIP uses contrastive learning for image-text alignment.
chunk B: The model predicts which image and text pairs match in a batch.
chunk C: ViT divides images into fixed-size patches.
```

如果你问：

```text
CLIP 是怎么训练图文对齐的？
```

embedding model 会认为 A、B 更相关，C 不太相关。

但注意，它不是在你的 6 篇论文上重新学习语义，而是用已有能力对新文本做语义编码。

---

# 6. embedding model 和大语言模型是一回事吗？

不是一回事，但都属于神经网络模型。

你现在的 RAG 里其实有两个模型：

## 1. Embedding model：负责找资料

比如：

```text
BAAI/bge-small-zh-v1.5
```

它的任务是：

```text
把问题和文档 chunk 转成向量
计算语义相似度
帮助检索相关内容
```

它不负责生成回答。

---

## 2. Generator LLM：负责组织答案

比如：

```text
deepseek-v4-pro
qwen-plus
qwen3.6-plus
```

它的任务是：

```text
阅读检索出来的 chunk
理解用户问题
组织成中文答案
```

所以 RAG 里通常是：

```text
embedding model：负责“找”
LLM：负责“答”
```

这两个模型可以完全不同。

---

# 7. 那 embedding 是不是借助了大模型参数？

要分情况。

你现在用的 `BAAI/bge-small-zh-v1.5` 是一个专门训练好的 embedding model，它有自己的参数和权重。

它不是直接调用 DeepSeek/Qwen 的参数。

也就是说：

```text
你的 embedding model 参数 ≠ DeepSeek/Qwen 的参数
```

当然，有些平台也提供 embedding API，比如 OpenAI embedding、DashScope embedding、智谱 embedding 等。这些也是别人训练好的 embedding 模型，你调用它们来生成向量。

所以总体可以理解成：

```text
RAG = 预训练 embedding model + 向量检索 + 大模型生成
```

---

# 8. 为什么不用训练就能用？一个类比

你学深度学习应该很容易理解这个类比。

假如你有一个预训练好的 ResNet：

```text
ResNet 在 ImageNet 上训练过
```

现在你给它一张新的故障时频图，它也能提取出一些有意义的图像特征。

你没有重新训练 ResNet，但它仍然有用。

因为它之前已经学过：

```text
边缘
纹理
形状
局部结构
高层语义
```

同理，embedding model 之前已经学过：

```text
词语关系
句子语义
中英文表达
问题和答案的匹配关系
文本相似性
检索相关性
```

所以你现在把论文 chunk 输入进去，它就能提取语义特征。

---

# 9. 但这也解释了为什么 RAG 会失败

因为 embedding model 不是为你的 6 篇论文专门训练的。

所以它可能会遇到问题：

## 1. 缩写不敏感

比如：

```text
DvD
```

它可能不知道你指的是：

```text
Decoupled Vision-Language Deployment
```

所以检索失败。

---

## 2. 中文问题检索英文论文不稳定

你问中文：

```text
CLIP 的核心训练目标是什么？
```

论文里是英文：

```text
contrastive pre-training objective
```

如果 embedding model 的中英跨语言能力不够强，就可能召回不准。

---

## 3. 专业术语太密集

比如：

```text
visual instruction tuning
alignment before projection
dynamic resolution
absolute time encoding
```

这些术语如果没有在 embedding 训练数据中充分出现，检索就可能不稳定。

---

## 4. chunk 切得不好

如果一个关键概念被切散了：

```text
chunk A：概念名称
chunk B：解释
chunk C：实验结果
```

那 query 可能只召回其中一段，导致回答不完整。

---

# 10. 如果想让它更懂你的论文，需要训练吗？

有三种升级方式，从简单到高级。

---

## 方法 1：改 query，不训练

比如原问题：

```text
InternVL3.5 的 DvD 是什么？
```

改成：

```text
InternVL3.5 的 DvD（Decoupled Vision-Language Deployment）是什么？
```

这可以显著提升召回。

这是最简单有效的方法。

---

## 方法 2：换更强 embedding model，不训练

比如换成更适合英文论文检索、跨语言检索、学术文本检索的 embedding model。

这相当于换一个更强的“特征提取器”。

---

## 方法 3：fine-tune embedding model

这是更进阶的做法。

你可以准备一批训练数据：

```text
query: CLIP 的核心训练目标是什么？
positive chunk: CLIP 论文中 contrastive objective 的原文段落
negative chunk: ViT 或 LLaVA 中无关段落
```

然后训练 embedding model，让它更适合你的论文库。

这叫：

```text
retriever fine-tuning
embedding fine-tuning
contrastive retrieval training
```

但是你现在 Week 2 不需要做这个。你现在更应该先掌握：

```text
chunk size
top-k
query rewriting
metadata filtering
hybrid search
```

这些工程方法。

---

# 11. 所以你这个 RAG 系统到底“学”了什么？

严格来说，它没有训练新模型，所以它没有“学习”你的论文参数。

它做的是：

```text
把你的论文转成向量形式存起来
```

也就是建立了一个可检索的知识库。

它的“知识”不是写进了模型参数，而是存在：

```text
chunk 文本
embedding 向量
metadata
vector index
```

里面。

大模型回答时临时读取这些内容。

所以 RAG 和 fine-tuning 最大区别是：

```text
Fine-tuning：把知识/能力部分写进模型参数
RAG：把知识放在外部数据库里，回答时检索出来用
```

这是一个非常重要的区别。

---

# 12. 一句话总结

你可以这样理解：

> 你的 RAG 系统没有自己训练出语义能力。
> 它借用了一个已经训练好的 embedding model，把 PDF chunk 转换成语义向量；然后通过向量相似度找到和问题最相关的 chunk；最后再把这些 chunk 交给 DeepSeek/Qwen 生成答案。

最简洁版流程是：

```text
预训练 embedding model 负责把文本变成语义向量
向量索引负责快速找相似 chunk
LLM 负责基于 chunk 组织答案
```

所以它不是“无中生有地懂你的 PDF”，而是：

> 用已经训练好的语义表示模型，把你的 PDF 临时组织成一个可检索知识库。
















你的理解**大方向是对的**，但有一个关键点需要修正：

> RAG 的 retriever 一定会从库里找出“相对最像”的 top-k，但“相对最像”不等于“真的相关”。
> 所以不能完全指望大模型自然判断“不相关”，更稳的做法是：**检索分数阈值 + prompt 约束 + sources 检查**。

我分开讲。

---

# 1. 更规范的 sources 是什么意思？

现在你的 RAG 回答后面会显示类似：

```text
Source 1
File: CLIP.pdf
Page: 2
Score: 0.73
Text snippet: CLIP pre-trains an image encoder and a text encoder...
```

它的意思是：

```text
Source 1：第 1 个被检索出来的片段
File：这个片段来自哪篇 PDF
Page：来自第几页
Score：这个片段和用户问题的相似度分数
Text snippet：这个片段的原文预览
```

它的作用不是装饰，而是让你检查：

```text
模型回答是否真的有依据？
检索有没有找对论文？
检索有没有找对页面？
有没有混入无关论文？
```

比如你问：

```text
CLIP 的核心训练目标是什么？
```

好的 sources 应该类似：

```text
File: CLIP.pdf
Page: 2 或 3
Text snippet: contrastive pre-training objective / image-text pairs / cosine similarity
```

如果它检索到：

```text
File: VIT.pdf
Page: 10
Text snippet: image patches / transformer encoder
```

那就说明检索错了。

---

# 2. “相关片段”是谁定义的？

不是大模型先定义的。

在 RAG 流程里，通常是 **embedding model + 相似度计算** 定义“相关”。

流程是：

```text
用户问题
↓
embedding model 转成 query vector
↓
每个论文 chunk 也已经转成 chunk vector
↓
计算 query vector 和每个 chunk vector 的相似度
↓
选出相似度最高的 top-k
```

所以“相关”在第一阶段其实是一个数学问题：

> 哪些 chunk 的向量和问题向量最接近？

---

# 3. 向量怎么判断相似？

常见方法是 **cosine similarity，余弦相似度**。

它看的是两个向量的“方向”是否接近，而不是单纯看长度。

\cos(\theta)=\frac{\mathbf{a}\cdot\mathbf{b}}{|\mathbf{a}||\mathbf{b}|}

简单说：

```text
cosine similarity 越接近 1，方向越像，语义越可能相关
越接近 0，相关性越弱
小于 0，方向相反，通常更不相关
```

---

# 4. 你举的二维例子

## 例子 1：`(1, 1)` 和 `(2, 2)`

这两个向量方向完全一样。

```text
(1,1)
(2,2)
```

虽然长度不一样，但方向一致。

所以如果用 cosine similarity：

```text
相似度 = 1
```

也就是说，它们在语义空间里可以被认为非常相似。

---

## 例子 2：`(1, 1)` 和 `(0.5, 1)`

这两个方向不完全一样，但也比较接近。

cosine similarity 大约是：

```text
0.95 左右
```

也算比较相似。

所以你可以理解为：

```text
(1,1) 和 (2,2)：方向完全一致，非常相似
(1,1) 和 (0.5,1)：方向接近，也比较相似
(1,1) 和 (-1,-1)：方向相反，不相似
```

但真实 embedding 不是二维，而是几百维，比如 384 维、768 维、1024 维。系统就是在高维空间里判断语义方向是否接近。

---

# 5. 你的想法哪里对，哪里需要修正？

你说：

> 无论向量是多少，RAG 一定能从 embedding 中抽选出最相似的 top5，即便是完全不相似，但是一定能选出相对比较近的向量。

这个完全正确。

只要你的库里有 chunk，top-k 检索通常就一定能返回几个“相对最近”的 chunk。

比如你问：

```text
GPT-5 的训练数据规模是多少？
```

你的论文库里没有 GPT-5。

但是 retriever 还是会硬找出 top-5，可能是：

```text
Qwen2.5-VL 的训练数据
CLIP 的数据集
LLaVA 的 instruction data
```

它们只是“相对最近”，但不代表真的能回答 GPT-5。

---

你又说：

> 这些检索出来的片段和问题放在一起输入给大模型的时候，大模型知道这个片段和问题毫不相关，所以自然就会回答没有依据。

这个**部分正确，但不能完全依赖它**。

更准确地说：

```text
如果 prompt 写得很强，大模型可能会判断上下文不足，然后拒答。
如果 prompt 写得弱，大模型可能会根据自己的预训练知识胡编。
如果检索片段有一点点相关但不充分，大模型可能会过度推断。
```

所以大模型不是天然就会拒答。它是因为你在 prompt 里告诉它：

```text
只能基于检索到的论文片段回答。
如果上下文不足，就说无法确定。
不要使用外部知识补全。
```

它才更可能拒答。

---

# 6. 真正限制大模型回答边界的是什么？

不是 embedding。

是这三层东西共同限制：

## 第一层：retrieval score

检索系统可以给每个 source 一个 score。

比如：

```text
Source 1: score = 0.78
Source 2: score = 0.71
Source 3: score = 0.69
```

这说明问题和这些 chunk 比较接近。

但如果是：

```text
Source 1: score = 0.31
Source 2: score = 0.28
Source 3: score = 0.25
```

可能说明这些 chunk 只是“矮子里拔高个”，其实没有可靠依据。

所以更稳的做法是设置阈值：

```text
如果最高 score 低于某个阈值：
    直接返回“没有检索到相关论文片段”
    不调用大模型
```

---

## 第二层：prompt 约束

把规则写进 prompt：

```text
只能基于 retrieved context 回答。
如果上下文不足，请回答“根据当前检索到的论文片段无法确定”。
不要使用外部知识补全。
```

这是软约束。

---

## 第三层：source citation 检查

回答出来后，你通过 sources 检查：

```text
答案是否真的来自这些 chunk？
有没有编造？
有没有混入别的论文？
```

这就是你 Week 2 做 faithfulness score 的意义。

---

# 7. “没有 retrieved sources” 更准确地说是什么？

严格来说，如果你的知识库不为空，top-k 通常都会返回 sources。

所以 “没有 retrieved sources” 这个说法不够准确。

更准确应该是：

```text
没有通过相关性阈值的 retrieved sources。
```

也就是说，不是完全没有 source，而是 source 分数太低，不可信。

比如：

```text
用户问：GPT-5 的训练数据规模是多少？

系统检索：
Source 1: CLIP.pdf, score=0.27
Source 2: LLaVA.pdf, score=0.24
Source 3: ViT.pdf, score=0.22
```

这时候虽然有 top-3，但它们都很弱。

程序就应该判断：

```text
最高分太低，不调用 LLM，直接拒答。
```

---

# 8. 一个具体例子

假设你的知识库里有这些 chunk：

```text
chunk A：CLIP uses contrastive learning to align images and texts.
chunk B：ViT splits images into fixed-size patches.
chunk C：LLaVA uses GPT-4 to generate visual instruction data.
```

用户问：

```text
CLIP 是如何训练图文对齐的？
```

embedding 后相似度可能是：

```text
chunk A: 0.82
chunk B: 0.31
chunk C: 0.45
```

top-3 都会返回，但 A 明显最相关。

这时可以交给大模型回答。

---

用户问：

```text
GPT-5 的训练数据规模是多少？
```

相似度可能是：

```text
chunk A: 0.29
chunk B: 0.22
chunk C: 0.27
```

top-3 仍然会返回，但都不高。

这时更稳的系统应该：

```text
最高 score < 阈值
↓
直接拒答
↓
不调用大模型
```

而不是把这些弱相关 chunk 交给大模型，让它自己判断。

---

# 9. 这个“边界”不是天然存在的，是你设计出来的

你问得非常关键：

> 相似的边界是什么？

答案是：

> 这个边界不是 embedding 天然告诉你的，而是你通过实验设定的。

不同 embedding model、不同向量库、不同相似度算法，score 分布都不一样。

所以你不能随便说：

```text
score < 0.5 就一定不相关
```

你需要观察自己的实验结果：

```text
成功回答的问题，source score 大概是多少？
失败检索的问题，source score 大概是多少？
混源问题的 score 大概是多少？
```

然后再设一个经验阈值。

比如你发现：

```text
有效检索通常 score > 0.55
明显不相关通常 score < 0.35
```

那你可以暂时设：

```text
MIN_RETRIEVAL_SCORE = 0.45
```

但这只是你当前系统的经验值，不是普适真理。

---

# 10. 所以你的理解可以改成这样

你原来的理解：

> top-k 一定会找出相对最像的 chunk，然后大模型知道这些片段不相关，所以会拒答。

更准确的版本是：

> top-k 一定会找出相对最像的 chunk，但这些 chunk 不一定真的相关。因此，一个可靠的 RAG 系统不能只依赖 top-k，还需要根据 similarity score 设置相关性阈值；只有通过阈值的 chunk 才送给大模型。送入大模型后，还要通过 prompt 明确要求模型只能基于 retrieved context 回答。否则，大模型仍然可能使用自身预训练知识补全甚至幻觉。

这个理解就非常专业了。

---

# 11. 你现在 v0.2 先做什么就够了？

v0.2 先做到：

```text
显示 File / Page / Score / Text snippet
```

这样你就能观察 score。

后面 v0.3 再加：

```text
MIN_RETRIEVAL_SCORE
```

例如伪代码：

```python
MIN_RETRIEVAL_SCORE = 0.45

valid_sources = [
    node for node in response.source_nodes
    if node.score is not None and node.score >= MIN_RETRIEVAL_SCORE
]

if len(valid_sources) == 0:
    return "没有检索到相关论文片段，因此无法基于当前论文库回答。"
```

但这个阈值需要你基于实验观察来定，不能拍脑袋。

---

# 12. 最后一句总结

你现在对 RAG 的理解可以升级成：

> RAG 不是保证大模型一定不胡说，而是通过“向量检索找证据 + 分数阈值判断相关性 + prompt 限制回答范围 + sources 可验证 + faithfulness 评估”来尽量降低幻觉。

这就是你这个项目从普通 demo 升级成科研工具的关键。



















你的分析是**非常对的**，而且你已经抓到了 RAG 项目里一个很核心的矛盾：

> **RAG 为了可靠性，会限制大模型只能基于检索证据回答；但如果限制太死，就会牺牲大模型的综合分析能力。**

你现在发现的不是 bug，而是 RAG 系统设计里的一个经典 trade-off：
**grounded factual QA** 和 **cross-document synthesis** 不是同一种任务。

---

# 1. 你的第一个判断：普通 top-k 不适合跨六篇论文分析

你说：

> 这个问题涉及 6 篇论文，如果只设置 top_k=6，显然不够。

完全正确。

你问的问题是：

```text
请按照 ViT → CLIP → LLaVA → Video-LLaVA → Qwen2.5-VL → InternVL3.5 的顺序，梳理视觉多模态大模型的发展主线。
```

这个问题不是单点事实检索，而是 **跨文档、跨阶段、跨技术路线的综合分析题**。

它至少需要每篇论文都召回一些代表性信息：

```text
ViT：patch embedding、Transformer encoder、大规模预训练
CLIP：image-text contrastive learning、zero-shot transfer
LLaVA：visual instruction tuning、CLIP visual encoder + LLM
Video-LLaVA：image-video unified representation、alignment before projection
Qwen2.5-VL：dynamic resolution、video temporal grounding、document/agent ability
InternVL3.5：Cascade RL、ViR、DvD、reasoning/deployment efficiency
```

如果全局 `top_k=6`，它很可能只检索到其中 1–3 篇论文。
而且正如你说的，如果粗暴设置 `top_k=30`，也不一定好。因为它可能出现：

```text
前 15 个都是 InternVL
后面 10 个都是 Qwen
ViT / CLIP / LLaVA 反而没有召回
```

所以你的结论很专业：

> 这种问题更适合 **per-document retrieval**，也就是每篇论文单独 top-k，比如每篇 top-5，而不是全局 top-30。

这就是所谓的 **document-aware retrieval** 或者 **metadata-based retrieval**。
你的 RAG v0.2 里加“选择论文”其实是往这个方向走了一步，但跨论文综合题还需要更进一步：**每篇论文分别检索，再合并给大模型分析**。

---

# 2. 你的第二个判断：prompt 写太死会压制综合能力

你说：

> 这六篇论文里面肯定没有直接写 “ViT → CLIP → LLaVA → Video-LLaVA → Qwen2.5-VL → InternVL3.5 的发展主线”，所以如果要求大模型必须只根据索引结果回答，它就会答不出来。

这个判断也非常对。

因为这类问题的答案不是某篇论文原文里直接存在的，而是需要大模型做：

```text
抽取事实
↓
跨文档归纳
↓
建立时间线
↓
总结技术演进逻辑
↓
形成分析性回答
```

但你的 prompt 现在偏向：

```text
只能基于检索片段回答
上下文不足就拒答
不要使用外部知识补全
不要发散
```

这对单篇论文事实题很好，比如：

```text
CLIP 的 contrastive objective 是什么？
LLaVA 的 visual instruction tuning data 是怎么构造的？
Qwen2.5-VL 的 dynamic resolution 是什么？
```

但对跨论文综合题就太死了。

因为跨论文综合题的答案往往不是“原文复述”，而是“基于多个原文事实做结构化推理”。

---

# 3. 这里其实有两种 RAG 模式

你现在发现的问题，可以总结为两种模式的冲突。

## 模式 A：严格证据型 RAG

适合：

```text
这篇论文用了什么数据集？
这个模型结构是什么？
这个 benchmark 结果是多少？
作者如何定义 alignment before projection？
```

这种模式要求：

```text
只能基于原文
不能发散
不能补充外部知识
没有依据就拒答
```

优点是可靠。

缺点是不灵活。

---

## 模式 B：证据增强型分析 RAG

适合：

```text
请比较 CLIP 和 LLaVA 的区别
请梳理视觉多模态模型的发展主线
请总结 ViT 到 InternVL 的技术演进
请分析为什么 MLLM 从看图问答走向视觉 agent
```

这种模式不能完全锁死大模型。更合理的要求应该是：

```text
请优先基于检索片段中的事实进行分析。
可以进行合理综合和归纳。
但需要区分：
1. 哪些是论文直接支持的事实；
2. 哪些是基于多篇论文做出的分析性总结。
不要编造具体实验数字或论文没有出现的细节。
```

这就不是“只准复述原文”，而是：

> **evidence-grounded synthesis，基于证据的综合分析。**

这比严格 RAG 更适合你这种学习型论文助手。

---

# 4. 所以不是 RAG 不适合跨论文分析，而是当前检索和 prompt 设计不适合

你现在这个系统的定位更像：

```text
单篇论文 / 局部知识点问答助手
```

而你刚刚那个问题需要的是：

```text
跨论文综述型研究助手
```

两者需要不同策略。

对于跨论文问题，理想流程应该是：

```text
识别问题涉及哪些论文
↓
对每篇论文分别检索 top-3/top-5
↓
把每篇论文的关键证据整理成小摘要
↓
再让大模型做跨论文综合
↓
输出“事实依据 + 综合分析”
```

也就是说，不是：

```text
全库 top_k=6
```

而是：

```text
ViT top-5
CLIP top-5
LLaVA top-5
Video-LLaVA top-5
Qwen2.5-VL top-5
InternVL top-5
↓
合并
↓
综合分析
```

你提出的“每篇论文 top5”就是这个思路。

---

# 5. 但是 top30 也有风险

你说 top30 是合理的，但也意识到可能前 15 个都是一个模型，这个判断很重要。

补充一点：即使真的每篇 top5，总共 30 个 chunk，也会带来三个问题：

```text
1. 上下文变长，成本更高；
2. 无关信息变多，大模型更容易回答变散；
3. 如果 chunk 很长，可能超过模型上下文或让重点稀释。
```

所以更成熟的做法不是直接把 30 个 chunk 全塞进去，而是两阶段：

```text
第一阶段：每篇论文检索 top5
第二阶段：先压缩成每篇论文的小摘要
第三阶段：再基于 6 个小摘要做综合分析
```

这个叫法可以是：

```text
map-reduce RAG
hierarchical summarization
multi-document RAG
```

你现在不做 v3 也没关系，但你理解这个思路就已经很好了。

---

# 6. 你现在对 prompt 的判断也很专业

你觉得 prompt 太死，这个很准确。

但我们不能简单说：

```text
那就放开，让大模型自由发挥
```

因为放得太开，又会回到幻觉问题。

真正好的 prompt 应该分模式。

比如：

## 严格问答模式

```text
适合问论文事实。
只能基于原文，不足就拒答。
```

## 分析综合模式

```text
适合跨论文比较和技术路线梳理。
优先基于检索证据，可以做合理归纳。
但必须标注哪些是论文事实，哪些是综合分析。
不能编造具体数字、实验结果和论文没有的结论。
```

你现在的问题属于第二种。

---

# 7. 你这个发现对简历项目反而是加分点

你可以在项目总结里写：

> 在实验中发现，单一全局 top-k 检索适合单篇论文事实型问答，但不适合跨论文综述型问题。对于涉及多篇论文的发展脉络分析，全局 top-k 容易被某一篇论文主导，导致证据覆盖不均。因此更合理的策略是基于文档 metadata 进行 per-document retrieval，并在 prompt 层面区分 factual QA 和 evidence-grounded synthesis 两种模式。

这段话很有含金量。

它说明你不是只会“搭一个 RAG”，而是在思考：

```text
不同问题类型需要不同检索策略
不同回答目标需要不同 prompt 约束
RAG 的可靠性和灵活性之间存在权衡
```

这是非常重要的工程判断。

---

# 8. 你现在可以如何理解这个项目的边界？

你现在这个 `mllm-research-rag` 两周项目，最合理的结论不是：

```text
我做了一个完美的论文阅读助手
```

而是：

```text
我做了一个基础 RAG 论文阅读系统，并通过实验理解了它在单文档问答、跨文档综合、检索粒度和 prompt 约束上的能力边界。
```

这比“完美工具”更真实，也更像科研项目。

---

# 9. 对你刚才那个问题，当前 RAG 为什么答不好？

综合来看，原因至少有三层。

第一，**retrieval coverage 不够**。
全局 top-k 不能保证六篇论文都被覆盖。

第二，**prompt 过于 strict**。
它要求答案必须直接来自检索片段，但“发展主线”本身是分析性结论，不是原文直接结论。

第三，**任务类型超出了普通 factual RAG**。
这个问题属于 multi-document synthesis，需要分文档检索和阶段性归纳。

所以它答不好是合理的，不是你理解错，也不是大模型一定不行。

---

# 10. 最后我给你一个简明结论

你的分析基本正确，可以整理成一句话：

> 当前 RAG demo 更适合“基于论文片段的事实型问答”，但对于“跨论文发展脉络梳理”这类综合分析题，单一全局 top-k 检索和过强的 grounded-only prompt 都会限制模型表现。更合适的方案是按论文分别检索、保证证据覆盖，再让大模型在明确区分“原文事实”和“综合分析”的前提下进行归纳。

这就是你这次发现的本质。
