# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ================= 1. 页面基本配置 =================
st.set_page_config(
    page_title="云南珍稀保护植物智能识别",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 2. 左侧边栏与主题切换 =================
with st.sidebar:
    st.markdown("### ⚙️ 界面控制")
    dark_mode = st.toggle("🌙 开启深色模式 (Dark Mode)", value=False)

    st.write("---")
    st.markdown("### 🌿 云南保护植物名录")
    st.caption("系统当前支持鉴定以下 10 种珍稀植物：")

    plant_levels = {
        "云南蓝果树": "一级", "五针白皮松": "一级", "华盖木": "一级", "绿春苏铁": "一级",
        "景东翅子树": "二级", "朱红大杜鹃": "二级", "漾濞槭": "二级", "红河橙、宜昌橙": "二级",
        "希陶木": "省级", "滇重楼": "省级"
    }

    CLASS_JSON_PATH = r"class_indices.json"
    try:
        with open(CLASS_JSON_PATH, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
        plant_list = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

        # 提取标签颜色
        badge_color = "#4CAF50" if dark_mode else "#8B4513"

        for plant in plant_list:
            level_text = plant_levels.get(plant, "保护")
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px dashed rgba(150, 150, 150, 0.3);">
                    <span style="font-size: 15px;">🌱 {plant}</span>
                    <span class="plant-badge" style="font-size: 12px; color: {badge_color} !important; font-weight: bold; padding: 2px 8px; border: 1px solid {badge_color} !important; border-radius: 6px;">{level_text}</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
    except Exception as e:
        st.warning(f"未能加载植物列表: {e}")

# ================= 3. 全局强制主题覆盖 =================
# 提取颜色变量，避免 f-string 内部嵌套判断导致语法错误
if dark_mode:
    bg_app = "#0f1712"
    text_app = "#e8f5e9"
    sidebar_bg = "#152019"
    title_color = "#4CAF50"
    sub_color = "#a3b8a6"
    card_bg = "#1a261f"
    card_border = "#2a3d31"
    card_text = "#e8f5e9"
    info_bg = "#1e2e24"
    upload_info_text = "#a5d6a7"
    btn_bg = "#4CAF50"
else:
    bg_app = "#F4F7F4"
    text_app = "#1B4332"
    sidebar_bg = "#FFFFFF"
    title_color = "#2D6A4F"
    sub_color = "#64748b"
    card_bg = "#FFFFFF"
    card_border = "#E2E8F0"
    card_text = "#1B4332"
    info_bg = "#E8F3EC"
    upload_info_text = "#1B4332"
    btn_bg = "#2D6A4F"

theme_css = f"""
<style>
    /* 顶部透明 */
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    #MainMenu {{ visibility: hidden; }}
    .block-container {{ padding-top: 3rem; }}
    [data-testid="stToggle"] div[data-baseweb="checkbox"] div {{ background-color: {btn_bg} !important; }}

    /* 强制主界面和侧边栏颜色 */
    .stApp {{ background-color: {bg_app} !important; color: {text_app} !important; }}
    [data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; border-right: 1px solid {card_border} !important; }}

    /* 强制侧边栏文字变深/变浅 */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span {{ 
        color: {text_app} !important; 
    }}
    /* 还原标签颜色 */
    [data-testid="stSidebar"] span.plant-badge {{ color: {badge_color} !important; }}

    .hero-title {{ color: {title_color} !important; }}
    .hero-subtitle {{ color: {sub_color} !important; }}

    .custom-card {{ background-color: {card_bg} !important; border: 1px solid {card_border} !important; border-left: 5px solid {title_color} !important; color: {card_text} !important; }}
    .upload-info {{ background-color: {info_bg} !important; border-color: {title_color} !important; color: {upload_info_text} !important; }}
    [data-testid="stFileUploader"] {{ background-color: {card_bg} !important; border-color: {card_border} !important; }}
    .stButton>button {{ background-color: {btn_bg} !important; color: white !important; }}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# 全局通用样式
st.markdown("""
<style>
    .hero-title { font-size: 2.8rem; font-weight: 800; margin-bottom: 0px; letter-spacing: -1px; }
    .hero-subtitle { font-size: 1.1rem; margin-bottom: 2rem; font-weight: 400; }
    .custom-card { padding: 24px; border-radius: 12px; }
    .upload-info { padding: 20px; border-radius: 12px; text-align: center; border: 1px dashed; }
    [data-testid="stFileUploader"] { border-radius: 12px; padding: 15px; border: 1px solid; }
    .stButton>button {
        border: none !important; border-radius: 8px !important;
        padding: 12px 28px !important; font-weight: 600 !important; font-size: 1.1rem !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important; transition: all 0.2s !important;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(0,0,0,0.15) !important; }
</style>
""", unsafe_allow_html=True)

# ================= 4. 加载模型 =================
MODEL_PATH = r"plant_model.h5"
IMG_SIZE = (224, 224)
IPLANT_NAME_MAPPING = {"红河橙、宜昌橙": "宜昌橙"}

@st.cache_resource
def load_plant_model():
    with open(CLASS_JSON_PATH, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    model = load_model(MODEL_PATH)
    return model, idx_to_class

try:
    model, idx_to_class = load_plant_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ 模型加载失败: {e}")

# ================= 5. 主界面布局 =================
st.markdown('<div class="hero-title">🌿 云南珍稀保护植物智能识别</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">基于 EfficientNet 深度学习模型 · 致力于云南生物多样性保护科普</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("#### 📸 图片采集区")
    uploaded_file = st.file_uploader("点击或拖拽上传植物照片", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="待识别图像", use_column_width=True)
        btn_predict = st.button("🔍 启动 AI 识别", use_container_width=True)
    else:
        st.markdown('<div class="upload-info">👆 请上传一张清晰的植物照片</div>', unsafe_allow_html=True)
        btn_predict = False

with col2:
    st.markdown("#### 📊 物种鉴定结果")

    if uploaded_file is not None and btn_predict and model_loaded:
        with st.spinner("AI 正在提取特征比对中..."):
            img = image.resize(IMG_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            predictions = model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx] * 100
            predicted_class = idx_to_class.get(predicted_idx, "未知物种")

            st.session_state['predicted_class'] = predicted_class

        level = plant_levels.get(predicted_class, "保护物种")

        # 提取结果卡片的颜色
        res_color = "#4CAF50" if dark_mode else "#2D6A4F"
        res_sub_color = "#a3b8a6" if dark_mode else "#64748b"
        res_badge_color = "#4CAF50" if dark_mode else "#8B4513"
        res_conf_color = "#FFFFFF" if dark_mode else "#1B4332"

        st.markdown(
            f"""
            <div class="custom-card">
                <h2 style="margin-top:0; color: {res_color}; font-size: 2rem;">🌱 {predicted_class}</h2>
                <div style="margin-top: 15px;">
                    <span style="font-size: 1.1rem; color: {res_sub_color};">保护级别：</span>
                    <span style="color: {res_badge_color}; font-weight: bold; font-size: 1.1rem; border: 1px solid {res_badge_color}; padding: 3px 10px; border-radius: 20px; margin-left: 10px;">{level}</span>
                </div>
                <p style="font-size: 1.1rem; margin-top: 15px; color: {res_sub_color};">
                    模型置信度：<b style="color: {res_conf_color};">{confidence:.2f}%</b>
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.progress(int(confidence))

        if confidence >= 85:
            st.success("✅ 鉴定结果可靠，置信度较高。")
        elif confidence >= 60:
            st.warning("⚠️ 鉴定结果存疑，建议结合多角度图片综合判断。")
        else:
            st.error("❌ 置信度较低，请确认是否属于本系统收录的物种。")

    elif uploaded_file is None:
        st.markdown('<div class="custom-card" style="height: 200px; display: flex; align-items: center; justify-content: center; color: #64748b; border-left-color: transparent;">等待图像输入...</div>', unsafe_allow_html=True)
    elif not btn_predict:
        st.markdown('<div class="custom-card" style="height: 200px; display: flex; align-items: center; justify-content: center; color: #64748b; border-left-color: transparent;">图像已加载，请点击左侧按钮启动识别</div>', unsafe_allow_html=True)

# ================= 6. 底部权威详情 =================
if 'predicted_class' in st.session_state:
    plant_name = st.session_state['predicted_class']
    query_name = IPLANT_NAME_MAPPING.get(plant_name, plant_name)
    iplant_url = f"https://www.iplant.cn/info/{query_name}"

    st.write("---")
    st.markdown(f"### 📖 中国植物智 (iPlant) 权威资料：{plant_name}")

    link_bg = "#1a261f" if dark_mode else "#FFFFFF"
    link_color = "#4CAF50" if dark_mode else "#2D6A4F"

    st.markdown(
        f'''
        <a href="{iplant_url}" target="_blank" style="text-decoration:none;">
            <div style="
                background-color: {link_bg}; color: {link_color};
                padding: 14px 20px; border-radius: 8px; 
                border: 1px solid {link_color}; display: inline-block;
                margin-bottom: 20px; font-weight: 600;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05); transition: all 0.3s;">
                🔗 在官网新窗口打开完整形态描述与标本记录 ↗
            </div>
        </a>
        ''', unsafe_allow_html=True
    )

    st.caption("以下网页由中国科学院植物研究所提供（若无法显示请点击上方按钮）：")
    components.iframe(iplant_url, height=800, scrolling=True)
