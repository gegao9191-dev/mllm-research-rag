import os
from pathlib import Path

from dotenv import load_dotenv
import gradio as gr

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
MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-plus")

PAPER_DIR = Path("data/papers")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 6

PAPER_OPTIONS = {
    "All papers": None,
    "ViT": "VIT.pdf",
    "CLIP": "CLIP.pdf",
    "LLaVA": "llava.pdf",
    "Video-LLaVA": "video-llava.pdf",
    "Qwen2.5-VL": "qwen2.5-VL.pdf",
    "InternVL3.5": "internvl.pdf",
}

INDEX_CACHE = {}


def load_pdf_documents():
    """Read PDF files page by page using PyMuPDF."""
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
                    }
                )
            )

        doc.close()

    if not documents:
        raise ValueError(
            "No readable text was extracted from PDFs. "
            "The PDFs may be scanned, encrypted, or image-based."
        )

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
        model_name=EMBEDDING_MODEL_NAME
    )


ALL_DOCUMENTS = load_pdf_documents()
setup_models()


qa_prompt = PromptTemplate(
    """你是一个视觉多模态论文阅读助手。请基于检索到的论文片段回答问题。

要求：
1. 用中文回答。
2. 优先使用下方“检索到的论文片段”中的信息回答。
3. 如果片段只支持部分回答，请先回答可以确定的部分，再明确说明哪些部分无法从当前片段中确定。
4. 只有当检索片段几乎完全无关时，才回答：“根据当前检索到的论文片段无法确定。”
5. 不要编造 benchmark、数据集、数字、实验结论或论文观点。
6. 如果检索到多个论文来源，请明确区分不同论文，不要混在一起。
7. 如果用户选择了某一篇论文，请仅基于该论文的检索片段回答。

检索到的论文片段：
---------------------
{context_str}
---------------------

用户问题：{query_str}

回答："""
)


def normalize_paper_filename(selected_paper):
    target = PAPER_OPTIONS.get(selected_paper)

    if target is None:
        return None

    available_files = sorted({doc.metadata.get("file_name") for doc in ALL_DOCUMENTS})
    if target in available_files:
        return target

    target_lower = target.lower()
    for file_name in available_files:
        if file_name and file_name.lower() == target_lower:
            return file_name

    return target


def filter_documents_by_paper(selected_paper):
    target_file = normalize_paper_filename(selected_paper)

    if target_file is None:
        return ALL_DOCUMENTS

    return [
        doc for doc in ALL_DOCUMENTS
        if doc.metadata.get("file_name") == target_file
    ]


def get_index(selected_paper, chunk_size, chunk_overlap):
    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)

    cache_key = (selected_paper, chunk_size, chunk_overlap)
    if cache_key in INDEX_CACHE:
        return INDEX_CACHE[cache_key]

    documents = filter_documents_by_paper(selected_paper)

    if not documents:
        raise ValueError(
            f"No documents found for selected paper: {selected_paper}. "
            f"Please check the filename mapping in PAPER_OPTIONS."
        )

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    INDEX_CACHE[cache_key] = {
        "index": index,
        "num_documents": len(documents),
        "num_nodes": len(nodes),
    }

    return INDEX_CACHE[cache_key]


def format_sources(response):
    sources = []

    for i, source_node in enumerate(response.source_nodes, start=1):
        node = source_node.node
        metadata = node.metadata

        file_name = metadata.get("file_name", "unknown file")
        page = metadata.get("page", "unknown page")
        score = source_node.score

        score_text = "N/A" if score is None else f"{score:.4f}"
        text = node.text[:900].replace("\n", " ").replace("\r", " ")

        sources.append(
            f"### Source {i}\n"
            f"- **File:** `{file_name}`\n"
            f"- **Page:** `{page}`\n"
            f"- **Score:** `{score_text}`\n\n"
            f"**Text snippet:**\n\n"
            f"> {text}..."
        )

    return "\n\n".join(sources) if sources else "No retrieved sources."


def ask_rag(question, selected_paper, chunk_size, chunk_overlap, top_k):
    if not question or not question.strip():
        return "请输入问题。"

    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)
    top_k = int(top_k)

    if chunk_overlap >= chunk_size:
        return "参数错误：chunk_overlap 必须小于 chunk_size。"

    try:
        index_info = get_index(
            selected_paper=selected_paper,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        query_engine = index_info["index"].as_query_engine(
            similarity_top_k=top_k,
            text_qa_template=qa_prompt,
        )

        response = query_engine.query(question)

        answer = str(response)
        source_text = format_sources(response)

        params_text = (
            f"## Retrieval Settings\n\n"
            f"- **Selected paper:** `{selected_paper}`\n"
            f"- **Model:** `{MODEL}`\n"
            f"- **Embedding model:** `{EMBEDDING_MODEL_NAME}`\n"
            f"- **chunk_size:** `{chunk_size}`\n"
            f"- **chunk_overlap:** `{chunk_overlap}`\n"
            f"- **top_k:** `{top_k}`\n"
            f"- **Page-level documents used:** `{index_info['num_documents']}`\n"
            f"- **Chunks / nodes in current index:** `{index_info['num_nodes']}`\n"
        )

        return (
            f"## Answer\n\n{answer}\n\n"
            f"---\n\n"
            f"{params_text}\n\n"
            f"---\n\n"
            f"## Retrieved Sources\n\n{source_text}"
        )

    except Exception as e:
        return f"运行出错：{e}"


with gr.Blocks(title="MLLM Research RAG Assistant v0.2") as demo:
    gr.Markdown(
        """
# MLLM Research RAG Assistant v0.2

一个用于阅读视觉多模态论文的本地 RAG 助手。  
支持选择论文、调整检索参数，并显示更规范的 retrieved sources。
"""
    )

    with gr.Row():
        selected_paper = gr.Dropdown(
            choices=list(PAPER_OPTIONS.keys()),
            value="All papers",
            label="选择论文"
        )

        chunk_size = gr.Dropdown(
            choices=[300, 500, 800, 1200],
            value=DEFAULT_CHUNK_SIZE,
            label="chunk_size"
        )

        chunk_overlap = gr.Dropdown(
            choices=[45, 75, 120, 180],
            value=DEFAULT_CHUNK_OVERLAP,
            label="chunk_overlap"
        )

        top_k = gr.Dropdown(
            choices=[2, 4, 6, 8],
            value=DEFAULT_TOP_K,
            label="top_k"
        )

    question = gr.Textbox(
        label="Ask a question about your MLLM papers",
        placeholder="例如：这篇论文的核心贡献是什么？或者：CLIP 的 contrastive pre-training objective 是什么？",
        lines=3,
    )

    run_button = gr.Button("Run RAG")
    output = gr.Markdown(label="Answer with retrieval settings and sources")

    run_button.click(
        fn=ask_rag,
        inputs=[question, selected_paper, chunk_size, chunk_overlap, top_k],
        outputs=output,
    )

    gr.Markdown(
        f"""
## Default Settings

- `chunk_size = {DEFAULT_CHUNK_SIZE}`
- `chunk_overlap = {DEFAULT_CHUNK_OVERLAP}`
- `top_k = {DEFAULT_TOP_K}`
- `generator = {MODEL}`
- `embedding_model = {EMBEDDING_MODEL_NAME}`

建议默认使用 `chunk_size=800, chunk_overlap=120, top_k=6`，这是消融实验中较稳定的一组参数。
"""
    )


if __name__ == "__main__":
    demo.launch()
