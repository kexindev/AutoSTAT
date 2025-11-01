import sys, os
import tempfile
import streamlit as st

from config import MODEL_CONFIGS, CUSTOM_MODEL_KEY
from utils.save_secrets import *
from prompt_engineer.sec1_call_llm import DataLoadingAgent
from prompt_engineer.sec2_call_llm import DataPreprocessAgent
from prompt_engineer.sec3_call_llm import VisualizationAgent
from prompt_engineer.sec4_call_llm import ModelingCodingAgent
from prompt_engineer.sec5_call_llm import ReportAgent
from prompt_engineer.planner import PlannerAgent

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

import numpy as np
np.set_printoptions(edgeitems=250, threshold=501)

sys.path.append(os.path.dirname(__file__))


st.set_page_config(
    page_title="Autostat",
    page_icon="🤖",
    layout="wide"
)


def init_session_state():

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "DeepSeek"
    
    if 'model_configs_runtime' not in st.session_state:
        # 运行时模型配置，包含预设和自定义模型
        st.session_state.model_configs_runtime = MODEL_CONFIGS.copy()
        # 加载用户配置（包括 API 密钥和自定义模型）
        user_configs = load_local_model_configs()
        for model_name, config in user_configs.items():
            if model_name in MODEL_CONFIGS:
                # 预设模型：只更新 API 密钥
                st.session_state.model_configs_runtime[model_name]["api_key"] = config.get("api_key", "")
            else:
                # 自定义模型：添加完整配置
                st.session_state.model_configs_runtime[model_name] = {
                    "api_base": config.get("api_base", ""),
                    "model_name": config.get("model_name", ""),
                    "api_key": config.get("api_key", ""),
                    "api_type": "openai",
                    "is_preset": False,
                }
    
    # 从 model_configs_runtime 提取 api_keys（用于传递给 Agent）
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            name: config.get("api_key", "")
            for name, config in st.session_state.model_configs_runtime.items()
        }
    
    if 'auto_mode' not in st.session_state:
        st.session_state.auto_mode = False

    if 'preference_select' not in st.session_state:
        st.session_state.preference_select = None
    if 'additional_preference' not in st.session_state:
        st.session_state.additional_preference = None
    if "from_auto" not in st.session_state:
        st.session_state.from_auto = False

    if 'data_loading_agent' not in st.session_state:
        st.session_state.data_loading_agent = DataLoadingAgent(
            api_keys=st.session_state.api_keys,
            model_configs=st.session_state.model_configs_runtime,
            model=st.session_state.selected_model
        )
    if 'data_preprocess_agent' not in st.session_state:
        st.session_state.data_preprocess_agent = DataPreprocessAgent(
            api_keys=st.session_state.api_keys,
            model_configs=st.session_state.model_configs_runtime,
            model=st.session_state.selected_model
        )
    if 'visualization_agent' not in st.session_state:
        st.session_state.visualization_agent = VisualizationAgent(
            api_keys=st.session_state.api_keys,
            model_configs=st.session_state.model_configs_runtime,
            model=st.session_state.selected_model
        )
    if 'modeling_coding_agent' not in st.session_state:
        st.session_state.modeling_coding_agent = ModelingCodingAgent(
            api_keys=st.session_state.api_keys,
            model_configs=st.session_state.model_configs_runtime,
            model=st.session_state.selected_model
        )
    if 'report_agent' not in st.session_state:
        st.session_state.report_agent = ReportAgent(
            api_keys=st.session_state.api_keys,
            model_configs=st.session_state.model_configs_runtime,
            model=st.session_state.selected_model
        )
    if 'planner_agent' not in st.session_state:
        st.session_state.planner_agent = PlannerAgent(
            api_keys=st.session_state.api_keys,
            model_configs=st.session_state.model_configs_runtime,
            model=st.session_state.selected_model
        )


def on_model_selector_change():
    """
    Callback when the model selector in the sidebar changes.
    """
    st.session_state.selected_model = st.session_state.model_selector
    

def run_app():
    """
    Main entry point to render the Streamlit app.
    """
    init_session_state()
    with st.sidebar:
        st.subheader("选择大模型")
        
        # 获取所有可用的模型（预设模型 + OpenAI API 兼容模型）
        models = list(MODEL_CONFIGS.keys()) + [CUSTOM_MODEL_KEY]
        
        # 确保选择的索引有效
        try:
            current_index = models.index(st.session_state.selected_model)
        except ValueError:
            current_index = 0
            st.session_state.selected_model = models[0]
        
        st.selectbox(
            "选择要使用的大模型",
            models,
            index=current_index,
            key="model_selector",
            on_change=on_model_selector_change,
        )

        st.subheader("API 密钥设置")
        selected = st.session_state.selected_model

        # 判断是否为 OpenAI API 兼容模型
        is_custom_model = (selected == CUSTOM_MODEL_KEY)
        
        if is_custom_model:
            # 显示 OpenAI API 兼容模型的配置界面
            existing_config = st.session_state.model_configs_runtime.get(CUSTOM_MODEL_KEY, {})
            
            if existing_config and existing_config.get("api_base"):
                st.info(f"当前配置: {existing_config.get('model_name', 'N/A')}")
            else:
                st.info("配置 OpenAI API 兼容模型")
            
            base_url_input = st.text_input(
                "Base URL",
                value=existing_config.get("api_base", ""),
                key="base_url_input",
                placeholder="例如: https://api.siliconflow.cn/v1"
            )
            
            model_name_input = st.text_input(
                "模型 ID",
                value=existing_config.get("model_name", ""),
                key="model_name_input",
                placeholder="例如: Qwen/Qwen3-8B"
            )
            
            api_key_input = st.text_input(
                "API 密钥",
                value=st.session_state.api_keys.get(CUSTOM_MODEL_KEY, ""),
                type="password",
                key="api_key_input",
            )
            
            if st.button("💾 保存配置", use_container_width=True, key="save_key"):
                if not base_url_input or not model_name_input or not api_key_input:
                    st.error("请填写所有必需字段")
                else:
                    # 保存到配置文件
                    update_local_model_config(
                        display_name=CUSTOM_MODEL_KEY,
                        api_key=api_key_input,
                        base_url=base_url_input,
                        model_name=model_name_input
                    )
                    
                    # 更新运行时配置
                    st.session_state.model_configs_runtime[CUSTOM_MODEL_KEY] = {
                        "api_base": base_url_input,
                        "model_name": model_name_input,
                        "api_key": api_key_input,  # 也保存 api_key
                        "api_type": "openai",
                        "is_preset": False,
                    }
                    # 同步到 api_keys
                    st.session_state.api_keys[CUSTOM_MODEL_KEY] = api_key_input
                    st.session_state.selected_model = CUSTOM_MODEL_KEY
                    
                    st.success("已保存配置")
                    st.rerun()
        else:
            # 预设模型或已保存的自定义模型
            api_key_input = st.text_input(
                f"{selected} API 密钥",
                value=st.session_state.api_keys.get(selected, ""),
                type="password",
                key="api_key_input",
            )
            
            # 如果是自定义模型，显示其配置信息
            if selected in st.session_state.model_configs_runtime:
                config = st.session_state.model_configs_runtime[selected]
                if not config.get("is_preset", False):
                    st.caption(f"Base URL: {config.get('api_base', 'N/A')}")
                    st.caption(f"Model: {config.get('model_name', 'N/A')}")

            if st.button("💾 保存密钥", use_container_width=True, key="save_key"):
                # 保存到配置文件
                config = st.session_state.model_configs_runtime.get(selected, {})
                if config.get("is_preset", False):
                    # 预设模型，只保存 API key
                    update_local_model_config(display_name=selected, api_key=api_key_input)
                else:
                    # 自定义模型，保存完整配置
                    update_local_model_config(
                        display_name=selected,
                        api_key=api_key_input,
                        base_url=config.get("api_base"),
                        model_name=config.get("model_name")
                    )

                # 同步更新运行时配置和 api_keys
                st.session_state.model_configs_runtime[selected]["api_key"] = api_key_input
                st.session_state.api_keys[selected] = api_key_input
                st.success("已保存")
                st.rerun()

        if st.button("🧹 清空数据", use_container_width=True, key="clear_data"):

            st.session_state.data_loading_agent = DataLoadingAgent(
                api_keys=st.session_state.api_keys,
                model_configs=st.session_state.model_configs_runtime,
                model=st.session_state.selected_model
            )
            st.session_state.data_preprocess_agent = DataPreprocessAgent(
                api_keys=st.session_state.api_keys,
                model_configs=st.session_state.model_configs_runtime,
                model=st.session_state.selected_model
            )
            st.session_state.visualization_agent = VisualizationAgent(
                api_keys=st.session_state.api_keys,
                model_configs=st.session_state.model_configs_runtime,
                model=st.session_state.selected_model
            )
            st.session_state.modeling_coding_agent = ModelingCodingAgent(
                api_keys=st.session_state.api_keys,
                model_configs=st.session_state.model_configs_runtime,
                model=st.session_state.selected_model
            )
            st.session_state.report_agent = ReportAgent(
                api_keys=st.session_state.api_keys,
                model_configs=st.session_state.model_configs_runtime,
                model=st.session_state.selected_model
            )
            st.session_state.planner_agent = PlannerAgent(
                api_keys=st.session_state.api_keys,
                model_configs=st.session_state.model_configs_runtime,
                model=st.session_state.selected_model
            )
            st.session_state.auto_mode = False
            st.rerun()

        if st.session_state.data_loading_agent.load_df() is not None:
            planner = st.session_state.planner_agent

            if st.session_state.auto_mode is False:
                if st.button("🚗 自动模式", use_container_width=True):
                    st.session_state.auto_mode = True
                    planner.self_driving(st.session_state.data_loading_agent.load_df())
                    st.switch_page("workflow/dataloading/dataloading_render.py")
                    st.rerun()
            else:
                if st.button("❌ 结束自动模式", use_container_width=True):
                    st.session_state.auto_mode = False
                    st.session_state.planner_agent = PlannerAgent(
                    api_keys=st.session_state.api_keys,
                    model_configs=st.session_state.model_configs_runtime,
                    model=st.session_state.selected_model
                    )
                    st.rerun()

        st.image(
            "logo/logo_big.png",
            use_container_width=True
        )

    # Define pages
    preference = st.Page(
        "workflow/preference/pref_render.py",
        title="⚙️ 偏好设置",
    )
    data_loading = st.Page(
        "workflow/dataloading/dataloading_render.py",
        title="📥 数据导入",
    )
    preprocessing = st.Page(
        "workflow/preprocessing/preprocessing_render.py",
        title="🛠️ 数据预处理",
    )
    visualization = st.Page(
        "workflow/visualization/viz_render.py",
        title="📊 数据可视化",
    )
    report = st.Page(
        "workflow/report/report_render.py",
        title="📝 报告生成",
    )
    coding_modeling = st.Page(
        "workflow/modeling/modeling_render.py",
        title="🧠 建模分析",
    )
    # Navigation
    pg = st.navigation(
        {
            "功能": [data_loading, preprocessing, visualization, coding_modeling, report],
            "设置": [preference]
        }
    )
    pg.run()
    
if __name__ == "__main__":
    run_app()
