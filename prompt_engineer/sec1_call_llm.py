import re

import streamlit as st
from typing import IO, List

from prompt_engineer.call_llm import LLMClient


class DataLoadingAgent(LLMClient):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.file_name = []
        self.user_input = None
        self.par_content = ""
        self.dfs = None
        self.abstract=None
        self.full = None
        self.finish_auto_task = False
        self.data_metadata = None  # 数据元数据（表结构、数据规范）
        self.business_context = None  # 业务背景信息
        self.mining_scenarios = None  # 可挖掘场景
        self.analysis_suggestions = None  # 分析挖掘建议


    def finish_auto(self):

        self.finish_auto_task = True


    def save_file_name(self, file_name):

        self.file_name.append(file_name)


    def load_file_name(self):

        return self.file_name


    def save_dfs(self, dfs):

        self.dfs = (dfs)


    def load_dfs(self):

        return self.dfs


    def clear_file_name(self):
        
        self.file_name = []


    def read_names_from_file(self, uploaded_names_file, df_head):

        raw = uploaded_names_file.read().decode('utf-8', errors='ignore')
        try:
            uploaded_names_file.seek(0)
        except Exception:
            pass

        prompt = (
            "下面是上传的 names 和 df_head 文件内容，请仅以 Python 列表格式返回与df_head一一对应的所有属性(attribute)名称，"
            "并保持顺序，不要添加多余文字，请注意，你只需要返回一个列表，不要出现任何markdown语法：\n```\n"
            f"name文件：{raw}\n```"
            f"df_head：{df_head}\n```"
        )
        try:
            response = self.call(prompt)
            names_list = eval(response.strip())
            if isinstance(names_list, list) and all(isinstance(n, str) for n in names_list):
                col_names = names_list
            else:
                raise ValueError("LLM 输出格式不正确")
        except Exception:

            col_names = []
            attr_re = re.compile(
                r"""^@attribute\s+ 
                    ['"]?([^'"\s]+)['"]?
                    \s+.+
                """,
                re.IGNORECASE | re.VERBOSE
            )
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith('@data'):
                    break
                m = attr_re.match(line)
                if m:
                    col_names.append(m.group(1))

        counts: dict[str, int] = {}
        unique_names: List[str] = []
        for name in col_names:
            if name in counts:
                counts[name] += 1
                unique_names.append(f"{name}_{counts[name]}")
            else:
                counts[name] = 0
                unique_names.append(name)

        return unique_names


    def do_data_description(self, df, user_input=None, memory_limit=6):

        recent_memory = self.memory[-memory_limit:] if self.memory else []
        if recent_memory:
            formatted_memory = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent_memory
            )
            memory_block = f"{formatted_memory}"
        else:
            memory_block = ""

        prompt = (
            "你是一名专业的数据分析助手，负责解释数据结构与业务含义。\n"
            f"- 数据维度：{df.shape[0]} 行 × {df.shape[1]} 列\n"
            f"- 列名和数据类型：{dict(zip(df.columns.tolist(), df.dtypes.astype(str).tolist()))}\n"
            f"- 前 5 行样本：\n{df.head().to_dict(orient='list')}\n\n"
            f"""- 数据解释聊天对话：
            --- 开始聊天记录 ---
            {memory_block}
            --- 结束聊天记录 ---"""
        )

        if user_input is not None:
            prompt += f"""
            请严格依据用户需求“{user_input}”，对当前数据进行深入、系统的分析。
            要求：
            1. 分析内容必须与该需求完全对应，不能添加无关推断。
            2. 结论要具体、清晰，可直接支持后续报告撰写或建模步骤。
            3. 分析语言应专业、简洁，不使用模糊或情绪化表述。
            """
        else:
            prompt += """
            以下是一个数据集的基本概览。请帮助我分析它的性质和结构，并回答以下问题：

            1. 该数据集可能来源于什么业务或研究场景？
            2. 各主要字段分别代表什么含义？若能判断，请说明其单位或数值含义。
            3. 数据中是否存在明显异常、异常分布或需要注意的特征？

            输出要求：
            - 使用自然、流畅的中文描述；
            - 采用清晰的分条结构（1、2、3）；
            - 语言客观简洁，不使用“可能”“也许”“似乎”等模糊词；
            - 重点突出数据结构、含义与潜在问题。
            """

        if st.session_state.preference_select:
            prompt += f"以下是用户的分析偏好设置：{st.session_state.preference_select}”。\n\n"
        if st.session_state.additional_preference:
            prompt += f"用户提供了以下建模目的与特殊需求：{st.session_state.additional_preference}”。\n\n"

        desc = self.call(prompt)

        return desc
    
    
    def summary_html(self):

        df = self.load_df()
        df_head = df.head()
        dtype_info = df.dtypes.astype(str)

        prompt = f"""
        你正在撰写一份数据分析报告的第一章——《数据概览与数据含义分析》。
        请根据以下输入内容，整理关键信息并进行分析说明：
        数据格式：
        {dtype_info}

        前五行数据：
        {df_head}
        
        数据解释聊天对话：
        --- 开始聊天记录 ---
        {self.memory}
        --- 结束聊天记录 ---

        额外要求：
        1. 要用流畅的自然语言
        2. 不要滥用形容词和副词，尽量用简单的动词和名词表达意思
        3. 不用"可能""也许""似乎""微妙"等模糊表述
        """.strip()

        desc = self.call(prompt)

        summary = {
                    "title": "数据导入",
                    "df": df_head,
                   "desc": desc,
                }

        return summary


    def summary_word(self):

        return self.summary_html()


    def check_abstract(self):

        if self.abstract is None:
            df = self.load_df()
            df_head = df.head()
            dtype_info = df.dtypes.astype(str)

            prompt = f"""
            这是数据分析的数据导入阶段
            数据格式：
            {dtype_info}

            前五行数据：
            {df_head}

            数据解释聊天对话：
            --- 开始聊天记录 ---
            {self.memory}
            --- 结束聊天记录 ---

            要求：
            请基于上述数据与对话内容，生成一段简洁、准确的综合摘要。
            摘要需完整呈现核心信息，便于后续自动判断该内容在报告撰写中是否需要被引用。
            """.strip()

            desc = self.call(prompt)
            self.abstract = desc

        return self.abstract


    def check_full(self):

        if self.full is None:
            df = self.load_df()
            df_head = df.head()
            dtype_info = df.dtypes.astype(str)

            self.full = (
                f"【阶段说明】这是数据分析流程中的数据导入阶段。\n"
                f"【数据格式】{dtype_info}\n"
                f"【样本预览】\n{df_head}\n"
                f"【分析对话记录】\n{self.memory}"
            )

        return self.full

    def save_data_metadata(self, metadata):
        """保存数据元数据（表结构、数据规范）"""
        self.data_metadata = metadata

    def load_data_metadata(self):
        """加载数据元数据"""
        return self.data_metadata

    def save_business_context(self, context):
        """保存业务背景信息（业务范围、形成数据的条件）"""
        self.business_context = context

    def load_business_context(self):
        """加载业务背景信息"""
        return self.business_context

    def generate_mining_scenarios(self, df, data_metadata=None, business_context=None):
        """
        基于数据表结构、数据规范和业务背景，生成可挖掘场景
        
        Args:
            df: 数据DataFrame
            data_metadata: 数据元数据（表结构、数据规范）
            business_context: 业务背景信息（业务范围、形成数据的条件）
        """
        # 构建数据结构信息
        schema_info = {
            "dimensions": f"{df.shape[0]} 行 × {df.shape[1]} 列",
            "columns": dict(zip(df.columns.tolist(), df.dtypes.astype(str).tolist())),
            "sample_data": df.head().to_dict(orient='list'),
            "statistics": {
                col: {
                    "null_count": int(df[col].isnull().sum()),
                    "null_percentage": float(df[col].isnull().mean() * 100),
                    "unique_count": int(df[col].nunique()),
                    "dtype": str(df[col].dtype)
                }
                for col in df.columns
            }
        }

        prompt = (
            "你是一名资深的数据挖掘专家，擅长基于数据特征和业务背景识别潜在的分析场景。\n\n"
            f"【数据表结构信息】\n"
            f"- 数据维度：{schema_info['dimensions']}\n"
            f"- 列名和数据类型：{schema_info['columns']}\n"
            f"- 数据统计信息：{schema_info['statistics']}\n"
            f"- 前5行样本数据：{schema_info['sample_data']}\n\n"
        )

        if data_metadata:
            prompt += f"【数据规范与元数据】\n{data_metadata}\n\n"

        if business_context:
            prompt += f"【业务背景信息】\n"
            if isinstance(business_context, dict):
                if business_context.get('business_scope'):
                    prompt += f"- 业务范围：{business_context['business_scope']}\n"
                if business_context.get('data_conditions'):
                    prompt += f"- 数据形成条件：{business_context['data_conditions']}\n"
                if business_context.get('business_domain'):
                    prompt += f"- 业务领域：{business_context['business_domain']}\n"
                if business_context.get('additional_info'):
                    prompt += f"- 其他背景信息：{business_context['additional_info']}\n"
            else:
                prompt += f"{business_context}\n"
            prompt += "\n"

        prompt += """
        请基于以上信息，系统性地识别和生成该数据集的可挖掘场景。要求：

        1. **场景识别**：识别3-8个具有实际价值的分析挖掘场景
           - 每个场景应明确说明分析目标
           - 说明该场景的业务价值或研究意义
           - 指出实现该场景所需的关键字段和分析方法

        2. **场景分类**：将场景按以下维度分类：
           - 描述性分析场景（如分布分析、趋势分析）
           - 关联性分析场景（如相关性分析、关联规则挖掘）
           - 预测性分析场景（如分类、回归、时间序列预测）
           - 异常检测场景（如离群点检测、异常模式识别）
           - 聚类分析场景（如客户分群、行为模式识别）

        3. **优先级评估**：为每个场景标注优先级（高/中/低），并说明理由

        4. **可行性分析**：评估每个场景的数据充分性和技术可行性

        输出格式要求：
        - 使用清晰的分级结构（一、二、三级标题）
        - 每个场景独立成段，包含：场景名称、分析目标、业务价值、关键字段、分析方法、优先级、可行性
        - 使用专业但易懂的语言
        - 避免模糊表述，给出具体建议
        """

        scenarios = self.call(prompt)
        self.mining_scenarios = scenarios
        return scenarios

    def generate_analysis_suggestions(self, df, data_metadata=None, business_context=None, mining_scenarios=None):
        """
        基于数据表结构、业务背景和挖掘场景，生成分析挖掘建议
        
        Args:
            df: 数据DataFrame
            data_metadata: 数据元数据
            business_context: 业务背景信息
            mining_scenarios: 已生成的挖掘场景（可选）
        """
        # 构建数据结构信息
        schema_info = {
            "dimensions": f"{df.shape[0]} 行 × {df.shape[1]} 列",
            "columns": dict(zip(df.columns.tolist(), df.dtypes.astype(str).tolist())),
            "sample_data": df.head().to_dict(orient='list'),
        }

        prompt = (
            "你是一名资深的数据分析顾问，擅长为数据分析项目提供系统性的分析挖掘建议。\n\n"
            f"【数据表结构信息】\n"
            f"- 数据维度：{schema_info['dimensions']}\n"
            f"- 列名和数据类型：{schema_info['columns']}\n"
            f"- 前5行样本数据：{schema_info['sample_data']}\n\n"
        )

        if data_metadata:
            prompt += f"【数据规范与元数据】\n{data_metadata}\n\n"

        if business_context:
            prompt += f"【业务背景信息】\n"
            if isinstance(business_context, dict):
                if business_context.get('business_scope'):
                    prompt += f"- 业务范围：{business_context['business_scope']}\n"
                if business_context.get('data_conditions'):
                    prompt += f"- 数据形成条件：{business_context['data_conditions']}\n"
                if business_context.get('business_domain'):
                    prompt += f"- 业务领域：{business_context['business_domain']}\n"
            else:
                prompt += f"{business_context}\n"
            prompt += "\n"

        if mining_scenarios:
            prompt += f"【已识别的挖掘场景】\n{mining_scenarios}\n\n"

        prompt += """
        请基于以上信息，提供系统性的分析挖掘建议。要求：

        1. **分析路径建议**：
           - 建议的分析流程和步骤顺序
           - 每个步骤的目标和预期产出
           - 步骤之间的依赖关系

        2. **技术方法建议**：
           - 推荐使用的统计方法、机器学习算法或数据挖掘技术
           - 说明选择该方法的原因和适用场景
           - 指出方法实施的技术要求和注意事项

        3. **数据预处理建议**：
           - 针对当前数据特点的预处理需求
           - 缺失值、异常值、重复值的处理策略
           - 特征工程建议（如特征选择、特征变换、特征构造）

        4. **分析深度建议**：
           - 建议的分析深度和详细程度
           - 关键指标和评估标准
           - 结果解释和可视化建议

        5. **风险与注意事项**：
           - 可能遇到的数据质量问题
           - 分析方法选择的限制和风险
           - 结果解释的注意事项

        6. **实施优先级**：
           - 建议优先实施的分析任务
           - 各任务的预期价值和投入产出比
           - 分阶段实施建议

        输出格式要求：
        - 使用清晰的分级结构
        - 每个建议独立成段，包含：建议内容、理由说明、实施要点、预期效果
        - 使用专业但易懂的语言
        - 给出具体可操作的建议，避免空泛描述
        """

        suggestions = self.call(prompt)
        self.analysis_suggestions = suggestions
        return suggestions
