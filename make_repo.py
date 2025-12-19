import os
from pathlib import Path
from docx import Document

def load_all_docx(root_dir, blacklist=None):
    """
    root_dir: 根文件夹路径
    blacklist: 列表，包含不想要的【文件名】(例如 ['temp.docx', 'test.docx'])
    """
    if blacklist is None:
        blacklist = set()
    else:
        blacklist = set(blacklist)

    all_data = []      # 存储最终结果: [{'title': '...', 'content': '...'}, ...]
    seen_names = set() # 用于过滤重名文件

    # 使用 pathlib 递归查找所有 .docx 文件
    base_path = Path(root_dir)
    
    # rglob("*.[dD][oO][cC][xX]") 可以匹配 docx, DOCX 等
    for file_path in base_path.rglob("*.docx"):
        file_name = file_path.name
        
        # 1. 过滤黑名单
        if file_name in blacklist:
            print(f"跳过(黑名单): {file_name}")
            continue
            
        # 2. 过滤重名文件
        if file_name in seen_names:
            print(f"跳过(重名): {file_path}")
            continue
            
        # 3. 过滤临时文件 (Word生成的临时文件通常以 ~$ 开头)
        if file_name.startswith("~$"):
            continue

        try:
            # 读取 docx 内容
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip(): # 只保留非空行
                    full_text.append(para.text.strip())
            
            content = "\n".join(full_text)
            
            # 只有内容不为空才记录
            if content.strip():
                all_data.append({
                    "title": file_name,
                    "path": str(file_path),
                    "content": content
                })
                seen_names.add(file_name)
                print(f"成功读取: {file_name}")
                
        except Exception as e:
            print(f"读取失败 {file_path}: {e}")

    return all_data

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def build_and_save_index(docs_list, model_name='BAAI/bge-small-zh-v1.5', save_path='algo_repo/rag_index'):
    """
    docs_list: 上一步得到的文档列表
    model_name: Embedding 模型名称
    save_path: 保存的文件名前缀
    """
    if not docs_list:
        print("没有文档可以处理。")
        return

    # 1. 加载 Embedding 模型
    print(f"正在加载模型: {model_name}...")
    model = SentenceTransformer(model_name)

    # 2. 准备文本数据
    # 我们只取内容进行向量化
    contents = [doc['content'] for doc in docs_list]
    
    print(f"正在进行向量化，共 {len(contents)} 条文档...")
    # 向量化结果转为 float32 的 numpy 数组 (FAISS 要求)
    embeddings = model.encode(contents, convert_to_numpy=True).astype('float32')

    # 3. 创建 FAISS 索引
    dimension = embeddings.shape[1]  # 获取向量维度 (BGE-small 通常是 512)
    # IndexFlatL2 是最基础的暴力搜索索引，精度最高，适合小规模数据
    index = faiss.IndexFlatL2(dimension)
    
    print("正在构建 FAISS 索引...")
    index.add(embeddings)

    # 4. 保存 FAISS 索引
    faiss.write_index(index, f"{save_path}.index")

    # 5. 保存元数据 (docs_list)
    # 这一步至关重要，因为 FAISS 检索只返回索引(0,1,2...)，我们需要这个 list 找回原话
    with open(f"{save_path}.pkl", "wb") as f:
        pickle.dump(docs_list, f)

    print(f"保存成功！\n索引文件: {save_path}.index\n元数据文件: {save_path}.pkl")

# --- 使用示例 ---
if __name__ == "__main__":
    my_folder = "algo_repo"       # 你的文档存放路径
    my_blacklist = ["目录分级四版.docx", "算法黄页完成目录第三版.docx"] # 排除名单

    docs_list = load_all_docx(my_folder, my_blacklist)

    print(f"\n全部扫描完成，共加载 {len(docs_list)} 个有效文档。")

    # 打印第一个文档的前100个字符看看
    if docs_list:
        print(f"示例内容来自 [{docs_list[0]['title']}]:\n{docs_list[0]['content'][:100]}...")
    
    build_and_save_index(docs_list)