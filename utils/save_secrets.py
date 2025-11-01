import toml
from pathlib import Path
from typing import Dict, Optional

# 我们把 secrets 放在项目 utils目录下的 .streamlit 文件夹
BASE = Path(__file__).parent
SECRETS_DIR  = BASE / ".streamlit"
SECRETS_FILE = SECRETS_DIR / "secrets.toml"


def load_local_model_configs() -> Dict[str, Dict[str, str]]:
    """
    从本地 secrets.toml 中读取完整的模型配置。
    返回格式：{model_name: {api_base: str, model_name: str, api_key: str}}
    """
    if not SECRETS_FILE.exists():
        return {}
    
    data = toml.load(SECRETS_FILE)
    return data.get("models", {})


def update_local_model_config(display_name: str, api_key: str, 
                              base_url: Optional[str] = None, 
                              model_name: Optional[str] = None) -> None:
    """
    更新本地模型配置到 secrets.toml。
    
    参数：
        display_name: 模型显示名称（菜单项名称，如 DeepSeek、Claude、OpenAI API 兼容模型等）
        api_key: API 密钥
        base_url: API base URL（自定义模型必需）
        model_name: 模型 ID，API 调用时使用（自定义模型必需）
    """
    SECRETS_DIR.mkdir(exist_ok=True)
    
    if SECRETS_FILE.exists():
        data = toml.load(SECRETS_FILE)
    else:
        data = {"models": {}}
    
    if "models" not in data:
        data["models"] = {}
    
    # 保存模型配置
    model_config = {"api_key": api_key}
    if base_url:
        model_config["api_base"] = base_url
    if model_name:
        model_config["model_name"] = model_name
    
    data["models"][display_name] = model_config
    
    with SECRETS_FILE.open("w", encoding="utf-8") as f:
        toml.dump(data, f)
