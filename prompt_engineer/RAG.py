import os
import faiss
import pickle
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource # 关键：让模型和索引在全局只加载一次
def get_retriever(index_path="algo_repo/rag_index.index", pkl_path="algo_repo/rag_index.pkl"):
    return VectorRetriever(index_path, pkl_path)

class VectorRetriever:
    """
    专门负责向量检索的类
    """
    def __init__(
        self, 
        index_path: str = "algo_repo/rag_index.index", 
        pkl_path: str = "algo_repo/rag_index.pkl", 
        model_name: str = 'BAAI/bge-small-zh-v1.5',
        device: str = "cpu"
    ):
        """
        初始化检索器
        :param index_path: FAISS 索引文件路径
        :param pkl_path: 元数据 pickle 文件路径
        :param model_name: Embedding 模型名称或本地路径
        :param device: 运行设备 ('cpu' 或 'cuda')
        """
        self.index_path = index_path
        self.pkl_path = pkl_path
        
        # 1. 加载 Embedding 模型
        print(f"Loading Embedding Model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        
        # 2. 加载 FAISS 索引
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        self.index = faiss.read_index(index_path)
        
        # 3. 加载元数据
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Metadata file not found: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.docs_list = pickle.load(f)
            
        print(f"Retriever ready. Database size: {len(self.docs_list)} documents.")

    def search(self, query: str, top_k: int = 3) -> str:
        """
        实现你 call 方法中需要的 .search 接口
        """
        if not query or not query.strip():
            return ""

        # 1. 向量化查询词
        # convert_to_numpy=True 配合 .astype('float32') 保证 FAISS 兼容性
        query_vec = self.model.encode([query], convert_to_numpy=True).astype('float32')

        # 2. 检索
        distances, indices = self.index.search(query_vec, top_k)

        # 3. 解析结果并构建 context 字符串
        context_parts = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue # 没搜够 k 条时的处理
            
            doc_data = self.docs_list[idx]
            title = doc_data.get('title', f"Doc_{idx}")
            content = doc_data.get('content', '')
            
            # 格式化输出，方便 LLM 理解
            part = f"--- 来源: {title} ---\n{content}"
            context_parts.append(part)

        return "\n\n".join(context_parts)

# 这样写可以方便你单独测试 RAG 逻辑是否正常
if __name__ == "__main__":
    # 假设你已经有了索引文件
    try:
        retriever = VectorRetriever()
        test_res = retriever.search("测试问题")
        print("测试搜索结果：\n", test_res)
    except Exception as e:
        print(f"初始化失败，请检查索引文件是否存在: {e}")