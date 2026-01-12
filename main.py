import os
import gc
import re
import json
import pickle
import threading
import requests
import fitz
import torch
import numpy as np
from glob import glob
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document as LangchainDocument
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rapidocr_onnxruntime import RapidOCR
from openai import OpenAI

DATA_PATH = "data/pdfs"
QUESTIONS_PATH = "data/questions.json"
SUBMISSION_NAME = "Shcherbakov_v4"
OUTPUT_FILENAME = f"submission_{SUBMISSION_NAME}.json"
LOGS_FILENAME = f"logs_processing.json"
YOUR_EMAIL = "maxim.shcherbakov@spbu.ru"
RAW_DOCS_FILENAME = "raw_knowledge_base.pkl"
CHROMA_PATH = "chroma_db"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

device = "cuda" if torch.cuda.is_available() else "cpu"

ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
ocr_lock = threading.Lock()

def get_images_area_ratio(page) -> float:
    total_page_area = page.rect.width * page.rect.height
    if total_page_area == 0: return 0.0
    images_area = 0.0
    for img in page.get_image_info(xrefs=True):
        w = img['bbox'][2] - img['bbox'][0]
        h = img['bbox'][3] - img['bbox'][1]
        images_area += w * h
    return images_area / total_page_area

def needs_ocr(page, text: str) -> bool:
    if len(text.strip()) < 100: return True
    if get_images_area_ratio(page) > 0.25: return True
    return False

def process_single_pdf(pdf_path: str) -> dict:
    local_docs = []
    local_stats = {"text": 0, "ocr": 0}
    try:
        file_name = os.path.basename(pdf_path)
        file_sha1 = os.path.splitext(file_name)[0]
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if needs_ocr(page, text):
                local_stats["ocr"] += 1
                pix = page.get_pixmap(dpi=150, colorspace=fitz.csRGB)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                with ocr_lock:
                    ocr_result, _ = ocr_engine(img_array)
                if ocr_result:
                    text = "\n".join([item[1] for item in ocr_result])
            else:
                local_stats["text"] += 1
            if not text.strip():
                continue
            metadata = {
                "source": file_name,
                "pdf_sha1": file_sha1,
                "page_index": page_num,
            }
            local_docs.append(
                LangchainDocument(page_content=text, metadata=metadata)
            )
        doc.close()
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return local_docs, local_stats

def load_pdfs_and_process(path: str) -> List[LangchainDocument]:
    if not os.path.exists(path):
        return []
    pdf_files = glob(os.path.join(path, "*.pdf"))
    all_documents = []
    num_threads = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_single_pdf, pdf): pdf for pdf in pdf_files}
        for future in as_completed(futures):
            docs, stats = future.result()
            all_documents.extend(docs)
    return all_documents

class OpenAIPipelineWrapper:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    def chat(self, messages: List[dict], json_mode: bool = False, temperature: float = 0.0):
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 100
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def __call__(self, prompt: str, kind: str):
        system_instruction = (
            "You are a precise financial analyst. Answer strictly using ONLY the Context provided.\n\n"
            "Answer Rules by 'kind':\n"
            "- number: only digits, no commas, spaces, or abbreviations. Example: 122333. N/A if unknown.\n"
            "- name: only a single name. N/A if unknown.\n"
            "- names: multiple names separated by commas, no extra text. N/A if unknown.\n"
            "- boolean: only Yes, No, True, or False (case-insensitive). N/A if unknown.\n\n"
            "Additional rules:\n"
            "1. If the information is missing, respond exactly 'N/A' or 'n/a'.\n"
            "2. Always cite the Source ID(s) at the end in brackets, e.g., '25 million [0]' or 'Yes [1,2]'.\n"
            "3. Do NOT explain your answer. Only provide the value + [Source ID].\n"
            "4. If multiple values apply, list them clearly, each followed by their source ID.\n"
            "5. Ensure formatting exactly matches the rules above for the given 'kind'.\n\n"
            f"Your expected answer type is '{kind}'. Follow rules strictly."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        answer = self.chat(messages)
        return [{"generated_text": answer if answer else "N/A"}]

def decompose_question(question: str) -> List[str]:
    reader = OpenAIPipelineWrapper()
    system_prompt = (
        "You are an expert at decomposing financial questions.\n"
        "Decide whether the input question is complex or simple.\n"
        "If and only if the question contains multiple intents, comparisons, or requires multiple pieces of information, "
        "decompose it into multiple atomic sub-questions.\n"
        "Otherwise, return the original question unchanged as the only item in the list.\n"
        "Do not paraphrase unless decomposition is required.\n"
        "Output ONLY valid JSON in exactly this format:\n"
        "{ \"sub_questions\": [\"string\", ...] }"
    )
    response = reader.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        json_mode=True
    )
    try:
        if response:
            return json.loads(response).get("sub_questions", [question])
    except:
        pass
    return [question]

def generate_search_queries(question: str) -> List[str]:
    reader = OpenAIPipelineWrapper()
    system_prompt = (
        "You are an AI search optimizer specialized in financial document retrieval.\n"
        "Rewrite the user's question into exactly 3 alternative search queries to maximize recall in financial reports.\n"
        "Use financial synonyms and equivalent terminology (e.g., 'sales' -> 'revenue', 'profit' -> 'net income').\n"
        "Preserve company names, ticker symbols, and key entities exactly as given.\n"
        "Do not answer the question and do not add new facts.\n"
        "Output ONLY valid JSON in exactly this format:\n"
        "{ \"queries\": [\"string\", \"string\", \"string\"] }"
    )
    response = reader.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        json_mode=True
    )
    try:
        if response:
            queries = json.loads(response).get("queries", [])
            if question not in queries:
                queries.insert(0, question)
            return queries
    except:
        pass
    return [question]

def solve_challenge_question(question: str, kind: str, db, reranker, llm_wrapper) -> dict:
    sub_questions = decompose_question(question)
    retrieved_docs = []
    
    for sub_q in sub_questions:
        search_queries = generate_search_queries(sub_q)
        k_docs = 20 // len(search_queries)
        for query in search_queries:
            docs = db.similarity_search(query, k=k_docs)
            retrieved_docs.extend(docs)
            
    unique_docs_map = {}
    for doc in retrieved_docs:
        key = (doc.metadata.get("pdf_sha1"), doc.metadata.get("page_index"))
        if key not in unique_docs_map:
            unique_docs_map[key] = doc
            
    retrieved_docs = list(unique_docs_map.values())
    
    pairs = [[question, doc.page_content] for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:5]]
    
    context_text = ""
    for i, doc in enumerate(top_docs):
        content = doc.page_content.replace("\n", " ")
        context_text += f"\n--- Source ID: {i} ---\n{content}\n"
        
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
    raw_answer = llm_wrapper(prompt, kind)[0]["generated_text"].strip()
    
    cited_indices = set()
    matches = re.findall(r'\[([\d,\s]+)\]', raw_answer)
    for cur_match in matches:
        nums = [int(n.strip()) for n in cur_match.split(',') if n.strip().isdigit()]
        for n in nums:
            if 0 <= n < len(top_docs):
                cited_indices.add(n)
                
    clean_text = re.sub(r'\[[\d,\s]+\]', '', raw_answer).strip()
    
    unique_references_tuples = set()
    for idx in cited_indices:
        doc = top_docs[idx]
        sha = doc.metadata.get("pdf_sha1")
        page = doc.metadata.get("page_index")
        if sha and page is not None:
             unique_references_tuples.add((sha, page))
             
    references = [{"pdf_sha1": r[0], "page_index": r[1]} for r in unique_references_tuples]
    
    logs = {
        "prompt": prompt,
        "answer": clean_text,
        "references": references,
        "kind": kind,
        "question": question
    }
    return clean_text, references, logs

def clean_answer(raw_text, kind):
    text = raw_text.strip()
    lower_text = text.lower()
    if any(phrase in lower_text for phrase in ["not available", "n/a", "no mention", "data is missing"]):
        return "N/A"
    if kind == "boolean":
        if "yes" in lower_text or "true" in lower_text:
            return "Yes"
        elif "no" in lower_text or "false" in lower_text:
            return "No"
        return "N/A"
    elif kind == "number":
        text = text.replace(",", "").replace(" ", "")
        if re.search(r'[kKmM]', text):
            return "N/A"
        match = re.search(r'-?\d+(\.\d+)?', text)
        if match:
            val = match.group(0)
            try:
                return float(val) if '.' in val else int(val)
            except:
                return val
        return "N/A"
    elif kind == "name":
        return text.strip(".'\"")
    elif kind == "names":
        names = [n.strip(" .'\"") for n in text.split(',') if n.strip()]
        if not names:
            return "N/A"
        return names
    return text.strip(".'\"")

def main():
    if os.path.exists(RAW_DOCS_FILENAME):
        with open(RAW_DOCS_FILENAME, "rb") as f:
            raw_knowledge_base = pickle.load(f)
    else:
        raw_knowledge_base = load_pdfs_and_process(DATA_PATH)
        with open(RAW_DOCS_FILENAME, "wb") as f:
            pickle.dump(raw_knowledge_base, f)
            
    global ocr_engine
    del ocr_engine
    gc.collect()
    torch.cuda.empty_cache()
    
    splitter_embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    text_splitter = SemanticChunker(
        splitter_embedding,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    
    semantic_docs_processed = []
    for doc in raw_knowledge_base:
        try:
            chunks = text_splitter.split_documents([doc])
            valid_chunks = [c for c in chunks if len(c.page_content) > 50]
            semantic_docs_processed.extend(valid_chunks)
        except Exception:
            continue
            
    del splitter_embedding
    del text_splitter
    gc.collect()
    torch.cuda.empty_cache()
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        knowledge_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model
        )
    else:
        knowledge_db = Chroma.from_documents(
            documents=semantic_docs_processed,
            embedding=embedding_model,
            persist_directory=CHROMA_PATH
        )
        
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", automodel_args={"torch_dtype": torch.float16})
    llm_wrapper = OpenAIPipelineWrapper()
    
    with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
        
    submission_answers = []
    logs_data = []
    
    for item in questions_data:
        torch.cuda.empty_cache()
        q_text = item.get("text", "")
        q_kind = item.get("kind", "")
        try:
            raw_val, refs, logs = solve_challenge_question(q_text, q_kind, knowledge_db, reranker, llm_wrapper)
            final_val = clean_answer(raw_val, q_kind)
            
            if final_val == "N/A":
                refs = []
                
            answer_entry = {
                "value": final_val,
                "references": refs,
                "question_text": q_text,
                "kind": q_kind,
            }
            submission_answers.append(answer_entry)
            logs_data.append(logs)
        except Exception as e:
            print(f"Error processing question: {e}")
            torch.cuda.empty_cache()
            submission_answers.append({
                "value": "N/A",
                "references": []
            })
            
    final_submission = {
        "team_email": YOUR_EMAIL,
        "submission_name": SUBMISSION_NAME,
        "answers": submission_answers
    }
    
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(final_submission, f, ensure_ascii=False, indent=2)
        
    with open(LOGS_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(logs_data, f, ensure_ascii=False, indent=2)
        
    url = "http://5.35.3.130:800/submit"
    with open(OUTPUT_FILENAME, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    print("Status code:", response.status_code)
    print("Response:", response.json())

if __name__ == "__main__":
    main()