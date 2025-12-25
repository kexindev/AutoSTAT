import os
from typing import List, Optional

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac

from workflow.dataloading.dataloading_core import process_complex_data, load_from_path, load_concat_file, PathFileWrapper


def loading_data_file(agent):

    st.info(
        "💡 提示：\n"
        "1. 支持一次上传多个数据文件\n"
        "2. 自动使用大模型分析并处理数据\n"
        "3. 支持多种格式的文件类型上传\n"
    )

    selected_index = sac.tabs([
        sac.TabsItem(label='本地上传'),
        # sac.TabsItem(label='路径导入'),
    ], color='#5980AE',)

    if selected_index == "本地上传":
        # 点击上传文件
        uploaded_files = st.file_uploader(
            "选择新文件",
            accept_multiple_files=True,
            help="拖拽或点击上传多个文件。如果上传多个格式不同的文件，可以选择分别处理。",
        )

        if uploaded_files:
            current_memory_file_name = agent.load_file_name()
            new_files = [f for f in uploaded_files if f.name not in current_memory_file_name]
            if new_files:
                try:
                    with st.spinner("正在处理数据..."):
                        df, dfs, file_names = process_complex_data(new_files, agent)
                    if df is not None:
                        # 保存文件信息
                        agent.save_dfs(dfs)
                        # 保存文件名称列表
                        if not hasattr(agent, 'file_names_list'):
                            agent.file_names_list = []
                        agent.file_names_list = file_names
                        
                        # 如果是单个文件，直接设置为主数据
                        if len(dfs) == 1:
                            agent.add_df(df)
                            for f in new_files:
                                agent.save_file_name(f.name)
                            st.rerun()
                        else:
                            # 多个文件时，先设置第一个为默认，但会在下面让用户选择
                            agent.add_df(df)
                            for f in new_files:
                                agent.save_file_name(f.name)
                            st.rerun()
                except Exception as err:
                    st.error(f"导入失败：{err}")

    elif selected_index == "路径导入":
        # 路径上传文件
        raw_paths = st.text_area(
            "从路径导入数据 (每行一个文件路径)",
            placeholder=    "C:\\data\\iris.names\nC:\\data\\iris.data",
            height=100
        )

        if st.button("从路径加载文件", use_container_width=True):
            if raw_paths:

                path_list = [p.strip().strip("'\"") for p in raw_paths.strip().split('\n') if p.strip()]
                
                valid_paths = [p for p in path_list if os.path.exists(p)]
                invalid_paths = [p for p in path_list if not os.path.exists(p)]

                if invalid_paths:
                    st.warning(f"路径不存在，已跳过：\n- " + "\n- ".join(invalid_paths))

                if not valid_paths:
                    st.error("未找到任何有效的本地文件路径。")
                else:
                    current_memory_file_name = agent.load_file_name()
                    new_paths = [p for p in valid_paths if p not in current_memory_file_name]

                    if not new_paths:
                        st.info("所有指定的路径文件均已加载。")
                    else:
                        files_to_process = [PathFileWrapper(p) for p in new_paths]
                        try:
                            with st.spinner("正在处理数据..."):
                                df, dfs, file_names = process_complex_data(files_to_process, agent)
                            if df is not None:
                                # 保存文件信息
                                agent.save_dfs(dfs)
                                # 保存文件名称列表
                                if not hasattr(agent, 'file_names_list'):
                                    agent.file_names_list = []
                                agent.file_names_list = file_names
                                
                                # 如果是单个文件，直接设置为主数据
                                if len(dfs) == 1:
                                    agent.add_df(df)
                                    for p in new_paths:
                                        agent.save_file_name(p)
                                    st.rerun()
                                else:
                                    # 多个文件时，先设置第一个为默认，但会在下面让用户选择
                                    agent.add_df(df)
                                    for p in new_paths:
                                        agent.save_file_name(p)
                                    st.rerun()
                        except Exception as err:
                            st.error(f"本地文件读取失败：{err}")
    
    # 如果有多个文件，显示选择界面
    dfs = agent.load_dfs()
    if dfs is not None and len(dfs) >= 2:
        # 获取文件名称
        file_names = None
        if hasattr(agent, 'file_names_list') and agent.file_names_list:
            file_names = agent.file_names_list
        load_concat_file(dfs, agent, file_names)


def loading_basic_info(agent):
    
    df = agent.load_df()
    if df is not None:
        r, c = df.shape
        missing = int(df.isnull().sum().sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("行数", r)
        col2.metric("列数", c)
        col3.metric("缺失值总数", missing)

        dtype_info = pd.DataFrame({
            "列名": df.columns,
            "类型": df.dtypes.astype(str),
            "非空": df.count().values,
            "缺失%": (df.isnull().mean() * 100).round(2).values,
        }).reset_index(drop=True)

        selected_index = sac.tabs([
            sac.TabsItem(label='数据类型概览'),
            sac.TabsItem(label='数据预览'),
        ],color='#5980AE',)

        if selected_index == "数据类型概览":
            st.dataframe(dtype_info, use_container_width=True)
        elif selected_index == "数据预览":
            if st.button("🎲 随机抽样"):
                # 添加保护措施，防止数据行数不足10行的情况
                sample_size = min(10, len(df))
                if sample_size == 0:
                    st.warning("数据为空，无法进行抽样")
                    display_df = df
                else:
                    display_df = df.sample(sample_size)
                st.dataframe(display_df, use_container_width=True)
            else:
                st.dataframe(df.head(10), use_container_width=True)


def loading_business_context(agent):
    """业务背景信息收集界面"""
    df = agent.load_df()
    if df is None:
        st.info("请先上传数据文件")
        return

    st.subheader("📋 业务背景信息")
    st.caption("填写业务背景信息有助于生成更精准的分析场景和建议")

    with st.expander("💼 业务背景信息", expanded=True):
        context = agent.load_business_context() or {}
        
        business_scope = st.text_area(
            "业务范围",
            value=context.get('business_scope', ''),
            help="描述该数据覆盖的业务范围，例如：客户交易数据、产品销售数据、用户行为数据等",
            height=100,
            key="business_scope_input"
        )

        data_conditions = st.text_area(
            "数据形成条件",
            value=context.get('data_conditions', ''),
            help="描述数据是如何形成的，包括数据采集方式、时间范围、筛选条件等",
            height=100,
            key="data_conditions_input"
        )

        business_domain = st.text_input(
            "业务领域",
            value=context.get('business_domain', ''),
            help="例如：电商、金融、医疗、教育等",
            key="business_domain_input"
        )

        additional_info = st.text_area(
            "其他背景信息（可选）",
            value=context.get('additional_info', ''),
            help="补充其他有助于理解数据的背景信息",
            height=80,
            key="additional_info_input"
        )

        if st.button("💾 保存业务背景", use_container_width=True):
            context = {
                'business_scope': business_scope,
                'data_conditions': data_conditions,
                'business_domain': business_domain,
                'additional_info': additional_info
            }
            agent.save_business_context(context)
            st.success("业务背景信息已保存！")

    with st.expander("📊 数据规范与元数据（可选）", expanded=False):
        data_metadata = st.text_area(
            "数据规范说明",
            value=agent.load_data_metadata() or '',
            help="描述数据表结构规范、字段含义、数据质量标准等",
            height=150,
            key="data_metadata_input"
        )

        if st.button("💾 保存数据规范", use_container_width=True):
            agent.save_data_metadata(data_metadata)
            st.success("数据规范已保存！")


def loading_scenario_mining(agent):
    """场景挖掘功能界面"""
    df = agent.load_df()
    if df is None:
        st.info("请先上传数据文件")
        return

    st.caption("基于数据结构和业务背景，生成可挖掘的分析场景")

    col1, col2 = st.columns(2)
    show_existing = False
    
    with col1:
        generate_btn = st.button("🚀 生成挖掘场景", use_container_width=True, type="primary")
    
    with col2:
        if agent.mining_scenarios:
            show_existing = st.button("📄 查看已有场景", use_container_width=True)

    if generate_btn:
        with st.spinner("正在分析数据结构和业务背景，生成挖掘场景..."):
            data_metadata = agent.load_data_metadata()
            business_context = agent.load_business_context()
            scenarios = agent.generate_mining_scenarios(df, data_metadata, business_context)
            agent.mining_scenarios = scenarios

        st.success("挖掘场景生成完成！")
        st.markdown("---")
        st.markdown(scenarios)

    if show_existing and agent.mining_scenarios:
        st.markdown("---")
        st.markdown(agent.mining_scenarios)


def loading_analysis_suggestions(agent):
    """分析挖掘建议功能界面"""
    df = agent.load_df()
    if df is None:
        st.info("请先上传数据文件")
        return

    st.caption("基于数据特征、业务背景和挖掘场景，生成系统性的分析建议")

    col1, col2 = st.columns(2)
    show_existing = False
    
    with col1:
        generate_btn = st.button("🎯 生成分析建议", use_container_width=True, type="primary")
    
    with col2:
        if agent.analysis_suggestions:
            show_existing = st.button("📋 查看已有建议", use_container_width=True)

    if generate_btn:
        with st.spinner("正在生成分析挖掘建议..."):
            data_metadata = agent.load_data_metadata()
            business_context = agent.load_business_context()
            mining_scenarios = agent.mining_scenarios
            suggestions = agent.generate_analysis_suggestions(
                df, data_metadata, business_context, mining_scenarios
            )
            agent.analysis_suggestions = suggestions

        st.success("分析建议生成完成！")
        st.markdown("---")
        st.markdown(suggestions)

    if show_existing and agent.analysis_suggestions:
        st.markdown("---")
        st.markdown(agent.analysis_suggestions)


def loading_chat(agent, auto=False) -> None:

    df = agent.load_df()
    if df is None:
        return

    with st.chat_message("assistant"):
        st.write(
            "我是您的数据分析助手，很高兴为您服务！\n\n"
            "请先上传您的数据文件，上传完成后，您可以在下方和我对话，也可以直接点击按钮解析数据含义。"
        )
        analyze_btn = st.button("🔍 解析含义")
        result_placeholder = st.empty()
        
    # 渲染历史对话
    chat_history = agent.load_memory()

    for idx, entry in enumerate(chat_history):
        bubble = st.chat_message(entry["role"])
        content = entry["content"]
        if isinstance(content, str):
            bubble.write(content)

    already_generated = any(
        entry["role"] == "assistant" and "含义" in str(entry["content"])
        for entry in chat_history
    )

    if analyze_btn or (auto and not already_generated):
        st.chat_message("user").write("请帮我解析数据含义")
        agent.add_memory({"role": "user", "content": "请帮我解析数据含义"})
        with st.spinner("分析中..."):
            desc = agent.do_data_description(df)

        agent.finish_auto()
        st.chat_message("assistant").write(desc)
        agent.add_memory({"role": "assistant", "content": desc})
        st.rerun()

    # 用户自定义输入
    user_input = st.chat_input("请输入需求，例如「帮我分析xx列」")
    if user_input:
        st.chat_message("user").write(user_input)
        agent.add_memory({"role": "user", "content": user_input})
        with st.spinner("处理中…"):
            reply = agent.do_data_description(df, user_input)

        st.chat_message("assistant").write(reply)
        agent.add_memory({"role": "assistant", "content": reply})
        st.rerun()


if __name__ == "__main__":

    agent = st.session_state.data_loading_agent
    planner = st.session_state.planner_agent
    auto = planner.loading_auto

    if st.session_state.auto_mode == True:
        if (agent.finish_auto_task == True and planner.switched_prep == False) or planner.loading_auto == False:
            planner.finish_loading_auto()
            st.switch_page("workflow/preprocessing/preprocessing_render.py")

    c1,c2 = st.columns(2)
    with c1:
        st.title("数据导入")
    with c2:
        st.write("")  
        st.write("")  
        # sac.buttons([
        #     sac.ButtonsItem(label='Github', icon='github', href='https://github.com/Automated-Statistician/AutoSTAT'),
        #     sac.ButtonsItem(label='Doc', icon=sac.BsIcon(name='bi bi-file-earmark-post-fill', size=16), href='https://automated-statistician.github.io/autostatdoc.github.io/'),
        # ], align='end', color='dark', variant='filled', index=None)
    st.markdown("---")

    c = st.columns(2)
    with c[0].expander('数据上传', True):
        loading_data_file(agent)
    with c[1].expander('数据建议', True):
        loading_chat(agent, auto)
    with c[0].expander('数据展示', True):
        loading_basic_info(agent)
    
    # 新增功能区域
    st.markdown("---")
    st.markdown("### 🎯 智能分析规划")
    
    c2 = st.columns(2)
    with c2[0].expander('业务背景信息', True):
        loading_business_context(agent)
    with c2[1].expander('场景挖掘', True):
        loading_scenario_mining(agent)
    
    with st.expander('分析挖掘建议', True):
        loading_analysis_suggestions(agent)

