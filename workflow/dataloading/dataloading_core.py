import csv
import io
import os
from typing import List, Optional

import chardet
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat, arff
import streamlit as st
import streamlit_antd_components as sac


def read_data_from_file(
    uploaded_data_file,
    col_names: Optional[List[str]] = None,
    sep: Optional[str] = None,
    na_values: List[str] = ['?'],
    encoding: Optional[str] = None
) -> pd.DataFrame:
    """
    从上传的数据文件读取 DataFrame。
    - 支持 .csv/.data/.txt/.xlsx/.xls/.mat
    - col_names=None 时使用 header=0（文件首行做列名）
    - col_names 不为 None 时使用 header=None 并指定 names=col_names
    - 文本文件：自动探测编码、嗅探分隔符，跳过坏行
    - Excel 文件：直接使用 pandas.read_excel
    - MAT 文件：使用 scipy.loadmat，提取第一个主要变量，转为 DataFrame，并保证一维列
    """
    # 读取所有字节
    data_bytes = uploaded_data_file.read()
    # 重置流位置
    try:
        uploaded_data_file.seek(0)
    except Exception:
        pass

    name = uploaded_data_file.name
    ext = os.path.splitext(name)[1].lower()

    # Excel 文件处理
    if ext in ('.xlsx', '.xls'):
        excel_kwargs = {}
        if col_names is None:
            excel_kwargs['header'] = 0
        else:
            excel_kwargs['header'] = None
            excel_kwargs['names'] = col_names
        return pd.read_excel(io.BytesIO(data_bytes), **excel_kwargs)

    # ARFF 文件特殊处理
    if ext == '.arff':
        text = data_bytes.decode(encoding or 'utf-8', errors='ignore')
        raw_data, meta = arff.loadarff(io.StringIO(text))
        df = pd.DataFrame(raw_data)
        for col in df.select_dtypes([object]).columns:
            if isinstance(df[col].iloc[0], bytes):
                df[col] = df[col].str.decode('utf-8', errors='ignore')
        if col_names is not None and df.shape[1] == len(col_names):
            df.columns = col_names
        return df
        
    # —— MAT 文件特殊处理 —— #
    if ext == '.mat':
        mat = loadmat(io.BytesIO(data_bytes))
        data_keys = [k for k in mat.keys() if not k.startswith('__')]
        if not data_keys:
            raise ValueError('MAT 文件中未发现有效数据变量')
        arr = mat[data_keys[0]]

        # —— 先处理稀疏矩阵 —— #
        if sparse.issparse(arr):
            arr = arr.toarray()

        arr = np.array(arr)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        df = pd.DataFrame(arr)

        if col_names is not None and df.shape[1] == len(col_names):
            df.columns = col_names

        return df

    if encoding is None:
        det = chardet.detect(data_bytes)
        encoding = det.get("encoding", "utf-8")

    if encoding.lower() in ("utf-16", "utf-16le", "utf-16be", 
                            "utf-32", "utf-32le", "utf-32be"):
        text = data_bytes.decode(encoding, errors="ignore")
        data_bytes = text.encode("utf-8")
        encoding = "utf-8"

    sample = data_bytes[:10000].decode(encoding, errors="ignore")

    first_line = sample.splitlines()[0].strip()

    if sep is not None:
        detected_sep = sep
        use_whitespace = False

    elif "," in first_line:
        detected_sep = ","
        use_whitespace = False

    else:
        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=[",", ";", "\t", "|"]
            )
            detected_sep = dialect.delimiter
            use_whitespace = False
        except csv.Error:
            detected_sep = None
            use_whitespace = True  # fallback

    read_kwargs = {
        "engine": "python",
        "encoding": encoding,
        "na_values": na_values,
        "skipinitialspace": True,
        "on_bad_lines": "skip",
    }

    if col_names is None:
        read_kwargs["header"] = 0
    else:
        read_kwargs["header"] = None
        read_kwargs["names"] = col_names

    if use_whitespace:
        read_kwargs["delim_whitespace"] = True
    else:
        read_kwargs["sep"] = detected_sep

    return pd.read_csv(io.BytesIO(data_bytes), **read_kwargs)


def process_complex_data(uploaded_files, dataloadingagent):
    """
    上传处理逻辑：
    - 单文件：当作普通表格或 MAT 文件读（第一行当表头）
    - 多文件：分别读取每个文件，保持各自的列名和格式
      不强制拼接，由用户在界面上选择处理方式（拼接或分别处理）
    """
    if not uploaded_files:
        st.error("请先上传文件")
        return None, None, None

    names_exts = ('.names', '.arff', '.doc')
    data_exts = ('.data', '.csv', '.txt', '.xlsx', '.xls', '.mat', '.arff', '.tsv', '.dat', '.tst')

    names_files = [f for f in uploaded_files
                   if os.path.splitext(f.name)[1].lower() in names_exts]
    data_files = [f for f in uploaded_files
                  if os.path.splitext(f.name)[1].lower() in data_exts]

    # 单文件直接读取
    if len(uploaded_files) == 1 and uploaded_files[0] in data_files:
        df = read_data_from_file(uploaded_files[0], col_names=None)
        return df, [df], [uploaded_files[0].name]

    if not data_files:
        raise ValueError(
            "未检测到任何数据文件，请上传支持的格式：.csv/.data/.txt/.xlsx/.xls/.mat/.arff/.tsv/.dat/.tst"
        )

    # 读取所有数据文件，每个文件使用自己的列名
    # 如果存在表头文件，只对第一个数据文件应用表头
    dfs = []
    file_names = []
    
    for idx, data_file in enumerate(data_files):
        # 如果存在表头文件且是第一个数据文件，使用表头文件的列名
        if names_files and idx == 0:
            sample_df = read_data_from_file(data_file, col_names=None)
            col_names = dataloadingagent.read_names_from_file(names_files[0], sample_df.head())
            df = read_data_from_file(data_file, col_names=col_names)
        else:
            # 其他文件使用自己的列名
            df = read_data_from_file(data_file, col_names=None)
        
        dfs.append(df)
        file_names.append(data_file.name)

    # 返回第一个 DataFrame 作为默认显示，所有 DataFrame 列表，以及文件名称列表
    # 不进行自动拼接，由用户在界面上选择处理方式
    if len(dfs) == 1:
        return dfs[0], dfs, file_names
    else:
        # 多个文件时，返回第一个文件作为默认，但保留所有文件供用户选择
        return dfs[0], dfs, file_names


def load_from_path(local_path):

    ext = os.path.splitext(local_path)[1].lower()
    if ext in (".csv", ".txt", ".data"):
        df_local = pd.read_csv(local_path)
    elif ext in (".xls", ".xlsx"):
        df_local = pd.read_excel(local_path)
    elif ext == ".json":
        df_local = pd.read_json(local_path)
    elif ext == ".jsonl":
        df_local = pd.read_json(local_path, lines=True)
    elif ext == ".parquet":
        df_local = pd.read_parquet(local_path)
    elif ext in (".pkl", ".pickle"):
        df_local = pd.read_pickle(local_path)
    elif ext == ".feather":
        df_local = pd.read_feather(local_path)
    elif ext == ".arff":
        data, meta = arff.loadarff(local_path)
        df_local = pd.DataFrame(data)
        for col in df_local.select_dtypes([object]).columns:
            if isinstance(df_local[col].iloc[0], bytes):
                df_local[col] = df_local[col].str.decode('utf-8')
    else:
        st.error(f"不支持的文件类型：{ext}")
        df_local = None

    return df_local


def load_concat_file(dfs, agent, file_names=None):
    """
    处理多个数据文件的选择界面
    - 拼接：横向或纵向拼接
    - 分别处理：选择使用哪个文件
    """
    if file_names is None:
        file_names = [f"文件 {i+1}" for i in range(len(dfs))]
    
    st.info(f"检测到 {len(dfs)} 个数据文件，请选择处理方式：")
    
    # 显示文件信息
    with st.expander("📁 查看文件信息", expanded=False):
        for idx, (df, name) in enumerate(zip(dfs, file_names)):
            st.write(f"**{name}**: {df.shape[0]} 行 × {df.shape[1]} 列")
            # 安全地处理列名显示，防止类型错误
            try:
                columns_list = df.columns.tolist() if hasattr(df.columns, 'tolist') else list(df.columns)
                displayed_columns = ', '.join(columns_list[:5]) if columns_list else ''
                st.write(f"列名: {displayed_columns}{'...' if len(columns_list) > 5 else ''}")
            except Exception as e:
                st.write(f"列名显示错误: {str(e)}")
            if idx < len(dfs) - 1:
                st.divider()
    
    # 使用 session_state 来跟踪处理模式
    mode_key = f"concat_mode_{id(dfs)}"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = 0  # 默认选择"分别处理"
    
    mode = sac.segmented(
        items=[
            sac.SegmentedItem(label='分别处理'),
            sac.SegmentedItem(label='纵向拼接'),
            sac.SegmentedItem(label='横向拼接'),
        ], 
        label='选择处理方式', 
        size='sm', 
        radius='sm', 
        index=st.session_state[mode_key]
    )
    
    # 更新 session_state 以跟踪当前选择
    if mode == '分别处理':
        st.session_state[mode_key] = 0
    elif mode == '纵向拼接':
        st.session_state[mode_key] = 1
    elif mode == '横向拼接':
        st.session_state[mode_key] = 2

    if mode == '分别处理' or (isinstance(mode, str) and mode.startswith("分别处理")):
        # 让用户选择使用哪个文件
        select_key = f"select_file_idx_{id(dfs)}"
        
        # 使用 key 参数时，Streamlit 会自动管理 session_state，不需要手动更新
        selected_idx = st.selectbox(
            "选择要使用的数据文件",
            options=range(len(dfs)),
            format_func=lambda x: f"{file_names[x]} ({dfs[x].shape[0]} 行 × {dfs[x].shape[1]} 列)",
            index=0,  # 默认选择第一个文件
            key=select_key
        )
        
        selected_df = dfs[selected_idx]
        agent.add_df(selected_df)
        st.success(f"已选择使用：{file_names[selected_idx]}")
        
    elif mode == '横向拼接' or (isinstance(mode, str) and mode.startswith("横向拼接")):
        # 横向拼接：要求行数相同
        try:
            dfs_pos = [df.reset_index(drop=True) for df in dfs]
            big_df = pd.concat(dfs_pos, axis=1)
            
            # 处理重复列名
            cols = []
            seen = {}
            for c in big_df.columns:
                if c in seen:
                    seen[c] += 1
                    cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    cols.append(c)
            big_df.columns = cols
            agent.add_df(big_df)
            st.success(f"横向拼接完成：{big_df.shape[0]} 行 × {big_df.shape[1]} 列")
        except Exception as e:
            st.error(f"横向拼接失败：{str(e)}。请确保所有文件的行数相同。")
            return
            
    else:  # 纵向拼接
        # 纵向拼接：要求列名相同或兼容
        try:
            big_df = pd.concat(dfs, axis=0, ignore_index=True)
            agent.add_df(big_df)
            st.success(f"纵向拼接完成：{big_df.shape[0]} 行 × {big_df.shape[1]} 列")
        except Exception as e:
            st.warning(f"纵向拼接时出现警告：{str(e)}。尝试统一列名后拼接...")
            # 尝试统一列名后拼接
            try:
                # 获取所有列名的并集
                all_cols = set()
                for df in dfs:
                    all_cols.update(df.columns)
                all_cols = sorted(list(all_cols))
                
                # 为每个 DataFrame 添加缺失的列
                dfs_aligned = []
                for df in dfs:
                    df_aligned = df.copy()
                    for col in all_cols:
                        if col not in df_aligned.columns:
                            df_aligned[col] = None
                    dfs_aligned.append(df_aligned[all_cols])
                
                big_df = pd.concat(dfs_aligned, axis=0, ignore_index=True)
                agent.add_df(big_df)
                st.success(f"纵向拼接完成（已统一列名）：{big_df.shape[0]} 行 × {big_df.shape[1]} 列")
            except Exception as e2:
                st.error(f"纵向拼接失败：{str(e2)}。建议使用「分别处理」选项。")
                return

    # 下载按钮
    current_df = agent.load_df()
    if current_df is not None:
        csv_bytes = current_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下载处理后的文件",
            data=csv_bytes,
            file_name="processed_data.csv",
            mime="text/csv"
        )


class PathFileWrapper:
    """A wrapper to treat a local file path as a Streamlit UploadedFile."""
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self._file = None

    def read(self, *args, **kwargs):
        with open(self.path, 'rb') as f:
            return f.read()

    def seek(self, offset, whence=0):

        pass

    def __repr__(self):
        return f"PathFileWrapper(path='{self.path}')"