import os
import csv
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate


load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = os.getenv("DASHSCOPE_MODEL", "deepseek-v4-pro")

PAPER_DIR = Path("data/papers")
RESULT_DIR = Path("experiments")
RESULT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = RESULT_DIR / "topk_ablation_results.csv"
OUTPUT_MD = RESULT_DIR / "topk_ablation_summary.md"


TEST_QUESTIONS = [
    {
        "question_id": "Q1",
        "question": "ViT 为什么可以把图像当成词序列来处理？请结合 patch embedding、position embedding、class token 和 Transformer Encoder 解释。",
        "expected_paper": "VIT.pdf",
    },
    {
        "question_id": "Q2",
        "question": "CLIP 的 contrastive pre-training objective 是什么？请解释 image-text pairs、cosine similarity 和 symmetric cross entropy loss 的作用。",
        "expected_paper": "CLIP.pdf",
    },
    {
        "question_id": "Q3",
        "question": "LLaVA 的 visual instruction tuning data 是如何由 GPT-4 生成的？请分别解释 conversation、detailed description 和 complex reasoning 三类数据。",
        "expected_paper": "llava.pdf",
    },
    {
        "question_id": "Q4",
        "question": "在 Video-LLaVA 中，为什么作者认为 image features 和 video features 必须在 projection layer 之前对齐？请结合 unified visual representation 和 LanguageBind 解释。",
        "expected_paper": "video-llava.pdf",
    },
    {
        "question_id": "Q5",
        "question": "Qwen2.5-VL 的动态分辨率、动态 FPS 和绝对时间编码分别解决了什么问题？",
        "expected_paper": "qwen2.5-VL.pdf",
    },
    {
        "question_id": "Q6",
        "question": "InternVL3.5 的 DvD（Decoupled Vision-Language Deployment）是什么？它如何把 vision encoder 和 language model 部署到不同 GPU 上来提升推理效率？",
        "expected_paper": "internvl.pdf",
    },
]


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K_VALUES = [2, 4, 6, 8]


def load_pdf_documents():
    documents = []

    pdf_files = sorted(PAPER_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in data/papers")

    for pdf_path in pdf_files:
        doc = pymupdf.open(pdf_path)

        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)

            if not text or len(text.strip()) < 50:
                continue

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": pdf_path.name,
                        "page": page_idx,
                    },
                )
            )

        doc.close()

    if not documents:
        raise ValueError("No readable text extracted from PDFs.")

    return documents


def setup_models():
    if not API_KEY:
        raise ValueError("DASHSCOPE_API_KEY is missing. Please check your .env file.")

    Settings.llm = OpenAILike(
        model=MODEL,
        api_base=BASE_URL,
        api_key=API_KEY,
        is_chat_model=True,
        context_window=8192,
        max_tokens=1024,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5"
    )


def build_index(documents):
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    return index, len(nodes)


def build_query_engine(index, top_k):
    qa_prompt = PromptTemplate(
        """你是一个视觉多模态论文阅读助手。请严格基于检索到的论文片段回答问题。

要求：
1. 用中文回答。
2. 如果上下文不足，请明确说“根据当前检索到的论文片段无法确定”。
3. 不要使用外部知识补全。
4. 不要编造 benchmark、数据集、数字或实验结论。
5. 如果问题涉及某篇具体论文，请优先基于对应论文内容回答。
6. 如果检索到多个论文来源，请明确区分不同论文，不要混在一起。

检索到的论文片段：
---------------------
{context_str}
---------------------

用户问题：{query_str}

回答："""
    )

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=qa_prompt,
    )

    return query_engine


def format_sources(response):
    source_items = []
    retrieved_files = []
    retrieved_pages = []

    for i, source_node in enumerate(response.source_nodes, start=1):
        node = source_node.node
        metadata = node.metadata

        file_name = metadata.get("file_name", "unknown_file")
        page = metadata.get("page", "unknown_page")
        score = source_node.score

        retrieved_files.append(file_name)
        retrieved_pages.append(str(page))

        preview = node.text[:500].replace("\n", " ").replace("\r", " ")
        source_items.append(
            f"Source {i}: {file_name}, page {page}, score={score}\n{preview}"
        )

    return {
        "retrieved_files": "; ".join(retrieved_files),
        "retrieved_pages": "; ".join(retrieved_pages),
        "retrieved_sources_preview": "\n\n".join(source_items),
    }


def write_summary_template():
    content = """# RAG Top-k Ablation Summary

## 1. Experiment Setting

This experiment compares different top-k values in the MLLM paper RAG assistant.

Fixed setting:

- chunk_size = 800
- chunk_overlap = 120
- embedding model = BAAI/bge-small-zh-v1.5
- generator model = configured in `.env`

Tested top-k values:

- 2
- 4
- 6
- 8

## 2. Why Top-k Matters

Top-k controls how many retrieved chunks are sent to the LLM.

- If top-k is too small, the system may miss key evidence.
- If top-k is too large, the system may include irrelevant chunks and cause cross-paper confusion.
- Therefore, top-k is not always better when larger.

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

请根据 `topk_ablation_results.csv` 人工填写以下观察。

### top_k = 2

- 优点：
- 缺点：
- 典型失败 case：

### top_k = 4

- 优点：
- 缺点：
- 典型失败 case：

### top_k = 6

- 优点：
- 缺点：
- 典型失败 case：

### top_k = 8

- 优点：
- 缺点：
- 典型失败 case：

## 5. Preliminary Conclusion

请在完成评分后总结：

1. 哪个 top_k 整体最稳？
2. 哪些问题在 top_k 较小时就能回答？
3. 哪些问题需要更大的 top_k？
4. top_k 变大后是否出现 mixed sources？
5. 后续是否需要 metadata filtering 或论文选择功能？
"""
    OUTPUT_MD.write_text(content, encoding="utf-8")


def main():
    setup_models()

    print("Loading PDF documents...")
    documents = load_pdf_documents()
    print(f"Loaded {len(documents)} page-level documents.")

    print("=" * 80)
    print(f"Building index with chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}...")
    index, num_nodes = build_index(documents)
    print(f"Index built. num_nodes={num_nodes}")

    fieldnames = [
        "question_id",
        "question",
        "expected_paper",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "num_nodes",
        "retrieved_files",
        "retrieved_pages",
        "answer",
        "retrieved_sources_preview",
        "answer_quality_score",
        "faithfulness_score",
        "failure_type",
        "notes",
    ]

    rows = []

    for top_k in TOP_K_VALUES:
        print("=" * 80)
        print(f"Testing top_k={top_k}...")

        query_engine = build_query_engine(index, top_k)

        for item in TEST_QUESTIONS:
            question_id = item["question_id"]
            question = item["question"]
            expected_paper = item["expected_paper"]

            print(f"Running {question_id} with top_k={top_k}...")

            try:
                response = query_engine.query(question)
                answer = str(response)
                source_info = format_sources(response)

                row = {
                    "question_id": question_id,
                    "question": question,
                    "expected_paper": expected_paper,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "top_k": top_k,
                    "num_nodes": num_nodes,
                    "retrieved_files": source_info["retrieved_files"],
                    "retrieved_pages": source_info["retrieved_pages"],
                    "answer": answer,
                    "retrieved_sources_preview": source_info["retrieved_sources_preview"],
                    "answer_quality_score": "",
                    "faithfulness_score": "",
                    "failure_type": "",
                    "notes": "",
                }

            except Exception as e:
                row = {
                    "question_id": question_id,
                    "question": question,
                    "expected_paper": expected_paper,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "top_k": top_k,
                    "num_nodes": num_nodes,
                    "retrieved_files": "",
                    "retrieved_pages": "",
                    "answer": f"ERROR: {e}",
                    "retrieved_sources_preview": "",
                    "answer_quality_score": "",
                    "faithfulness_score": "",
                    "failure_type": "runtime_error",
                    "notes": "",
                }

            rows.append(row)

            time.sleep(1)

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_summary_template()

    print("=" * 80)
    print("Top-k experiment finished.")
    print(f"CSV saved to: {OUTPUT_CSV}")
    print(f"Summary template saved to: {OUTPUT_MD}")
    print("Next step: open CSV and manually fill scores.")


if __name__ == "__main__":
    main()