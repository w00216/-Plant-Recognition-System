# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import datetime
import streamlit as st
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

# 兼容 Jupyter 环境和云端环境
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

HISTORY_FILE = os.path.join(BASE_DIR, "history.csv")

# ================= 2. 相似植物数据库 =================
SIMILAR_PLANTS = {
    "云南蓝果树": [{"name": "华南蓝果树", "diff": "本种比较纤细，叶片薄纸质且较小，近椭圆形，叶柄短而纤细；果实近卵圆形或倒卵圆形。"}],
    "五针白皮松": [{"name": "白皮松", "diff": "白皮松针叶为3针一束，老干树皮呈不规则鳞状块片脱落；五针白皮松为5针一束。"}],
    "华盖木": [{"name": "木莲属 / 木兰属", "diff": "华盖木属与木莲属等特征相似，但华盖木为单种属，可通过花被片9片、外轮最大及聚合果形态区分。"}],
    "希陶木": [{"name": "白大凤属 / 蝴蝶果属", "diff": "希陶木属为雌雄异株，雌雄花着生于退化的短枝上，雌花存在花盘，与近缘属有显著区别。"}],
    "景东翅子树": [{"name": "截裂翅子树", "diff": "景东翅子树叶顶端不裂，而截裂翅子树叶顶端有浅裂；本种花瓣倒卵形，宽约2.8厘米。"}],
    "朱红大杜鹃": [{"name": "两色杜鹃 / 云锦杜鹃", "diff": "花冠为漏斗状钟形，深蔷薇色至深红色，管状部外面密生绒毛；叶下面密被白色或黄褐色绒毛。"}],
    "滇重楼": [{"name": "七叶一枝花 (华重楼)", "diff": "七叶一枝花的花瓣较短，长度通常只有萼片的1/2至1/3，且常反折向下；滇重楼花瓣与萼片近等长或更长。"}],
    "漾濞槭": [{"name": "贡山槭 / 雷波槭", "diff": "漾濞槭叶片为掌状5裂，下表面有浓密白色绒毛；贡山槭通常为3裂。"}],
    "绿春苏铁": [{"name": "单羽苏铁", "diff": "绿春苏铁的大孢子叶顶片近圆形，顶裂片不明显，胚珠仅2枚，种子较大。"}],
    "红河橙、宜昌橙": [{"name": "马蜂柑", "diff": "红河橙的叶柄翅窄长圆形，总状花序具5-9朵花；宜昌橙翼叶与身叶近等大，多为单花。"}]
}

# ================= 3. 左侧边栏与主题切换 =================
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
    
    CLASS_JSON_PATH = os.path.join(BASE_DIR, "class_indices.json")
    
    try:
        with open(CLASS_JSON_PATH, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
        plant_list = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
        
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

# ================= 4. 全局强制主题覆盖 =================
if dark_mode:
    bg_app, text_app, sidebar_bg = "#0f1712", "#e8f5e9", "#152019"
    title_color, sub_color = "#4CAF50", "#a3b8a6"
    card_bg, card_border, card_text = "#1a261f", "#2a3d31", "#e8f5e9"
    info_bg, upload_info_text, btn_bg = "#1e2e24", "#a5d6a7", "#4CAF50"
    warn_bg, warn_border, warn_text = "#2a1f1f", "#ef5350", "#ffcdd2"
else:
    bg_app, text_app, sidebar_bg = "#F4F7F4", "#1B4332", "#FFFFFF"
    title_color, sub_color = "#2D6A4F", "#64748b"
    card_bg, card_border, card_text = "#FFFFFF", "#E2E8F0", "#1B4332"
    info_bg, upload_info_text, btn_bg = "#E8F3EC", "#1B4332", "#2D6A4F"
    warn_bg, warn_border, warn_text = "#FFF3E0", "#F57C00", "#E65100"

theme_css = f"""
<style>
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    #MainMenu {{ visibility: hidden; }}
    .block-container {{ padding-top: 3rem; }}
    [data-testid="stToggle"] div[data-baseweb="checkbox"] div {{ background-color: {btn_bg} !important; }}
    .stApp {{ background-color: {bg_app} !important; color: {text_app} !important; }}
    [data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; border-right: 1px solid {card_border} !important; }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span {{ color: {text_app} !important; }}
    [data-testid="stSidebar"] span.plant-badge {{ color: {badge_color} !important; }}
    .hero-title {{ color: {title_color} !important; }}
    .hero-subtitle {{ color: {sub_color} !important; }}
    .custom-card {{ background-color: {card_bg} !important; border: 1px solid {card_border} !important; border-left: 5px solid {title_color} !important; color: {card_text} !important; }}
    .upload-info {{ background-color: {info_bg} !important; border-color: {title_color} !important; color: {upload_info_text} !important; }}
    [data-testid="stFileUploader"] {{ background-color: {card_bg} !important; border-color: {card_border} !important; }}
    .stButton>button {{ background-color: {btn_bg} !important; color: white !important; }}
    .similar-warn {{ background-color: {warn_bg} !important; border-left: 5px solid {warn_border} !important; padding: 16px 20px !important; border-radius: 8px !important; margin-top: 15px; }}
    .similar-title {{ color: {warn_text} !important; font-weight: bold; font-size: 1.1rem; }}
    .similar-text {{ color: {warn_text} !important; font-size: 0.95rem; line-height: 1.5; }}
    .similar-link {{ color: {warn_border} !important; text-decoration: underline; font-weight: bold; }}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

st.markdown("""
<style>
    .hero-title { font-size: 2.8rem; font-weight: 800; margin-bottom: 0px; letter-spacing: -1px; }
    .hero-subtitle { font-size: 1.1rem; margin-bottom: 2rem; font-weight: 400; }
    .custom-card { padding: 24px; border-radius: 12px; }
    .upload-info { padding: 20px; border-radius: 12px; text-align: center; border: 1px dashed; }
    [data-testid="stFileUploader"] { border-radius: 12px; padding: 15px; border: 1px solid; }
    .stButton>button { border: none !important; border-radius: 8px !important; padding: 12px 28px !important; font-weight: 600 !important; font-size: 1.1rem !important; box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important; transition: all 0.2s !important; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(0,0,0,0.15) !important; }
</style>
""", unsafe_allow_html=True)

# ================= 5. 加载模型 =================
MODEL_PATH = os.path.join(BASE_DIR, "plant_model.h5")
IMG_SIZE = (224, 224)
IPLANT_NAME_MAPPING = {"红河橙、宜昌橙": "宜昌橙"}

@st.cache_resource
def load_plant_model():
    model = load_model(MODEL_PATH)
    return model

try:
    model = load_plant_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ 模型加载失败: {e}")

# ================= 6. 主界面布局（使用 Tab 区分识别与历史） =================
st.markdown('<div class="hero-title">🌿 云南珍稀保护植物智能识别</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">基于 EfficientNet 深度学习模型 · 致力于云南生物多样性保护科普</div>', unsafe_allow_html=True)

# 创建两个标签页
tab1, tab2 = st.tabs(["🔍 智能识别", "📜 历史识别记录"])

# ---------- Tab 1: 识别系统 ----------
with tab1:
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
                img = image.convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                img_array = img_array.astype('float32')
                
                predictions = model.predict(img_array, verbose=0)
                predicted_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_idx] * 100
                
                idx_to_class = {v: k for k, v in class_indices.items()}
                predicted_class = idx_to_class.get(predicted_idx, "未知物种")
                
                if confidence < 80.0:
                    st.error(f"❌ 拒绝识别：该图片可能不是植物，或者不属于本系统支持的 10 种植物（最高疑似度仅为 {confidence:.2f}%）。")
                    st.session_state['predicted_class'] = None  
                else:
                    st.session_state['predicted_class'] = predicted_class
                    
                    # 🌟 记录到历史文件
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    record = {
                        "识别时间": now_str,
                        "上传图片名": uploaded_file.name,
                        "预测品种": predicted_class,
                        "置信度": f"{confidence:.2f}%"
                    }
                    try:
                        # 如果文件不存在，写入表头；存在则追加
                        df_record = pd.DataFrame([record])
                        df_record.to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False, encoding='utf-8-sig')
                    except Exception as e:
                        pass # 忽略写入报错，不影响主流程

                    level = plant_levels.get(predicted_class, "保护物种")
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
                    st.success("✅ 鉴定结果可靠，置信度较高。")
                    
                    if predicted_class in SIMILAR_PLANTS:
                        st.markdown("---")
                        st.markdown("#### 🔍 相似植物鉴别")
                        for sim in SIMILAR_PLANTS[predicted_class]:
                            sim_query = sim['name'].split(' ')[0] if ' ' in sim['name'] else sim['name']
                            sim_url = f"https://www.iplant.cn/info/{sim_query}"
                            st.markdown(
                                f"""
                                <div class="similar-warn">
                                    <div class="similar-title">⚠️ 易混淆物种：{sim['name']}</div>
                                    <div class="similar-text">
                                        <b>鉴别要点：</b>{sim['diff']}<br>
                                        🔗 <a href="{sim_url}" target="_blank" class="similar-link">点击查看该相似植物的详细资料</a>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                
        elif uploaded_file is None:
            st.markdown('<div class="custom-card" style="height: 200px; display: flex; align-items: center; justify-content: center; color: #64748b; border-left-color: transparent;">等待图像输入...</div>', unsafe_allow_html=True)
        elif not btn_predict:
            st.markdown('<div class="custom-card" style="height: 200px; display: flex; align-items: center; justify-content: center; color: #64748b; border-left-color: transparent;">图像已加载，请点击左侧按钮启动识别</div>', unsafe_allow_html=True)

    # 底部权威详情
    if st.session_state.get('predicted_class') is not None:
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
        
        st.info("💡 提示：由于中国植物智官网的安全策略限制，无法直接在系统内嵌显示。点击上方按钮即可在新窗口中打开官方页面查看完整内容。")

# ---------- Tab 2: 历史记录查看 ----------
with tab2:
    st.markdown("### 📜 识别历史记录")
    st.caption("以下是系统近期成功识别的记录。你可以随时查看并下载这些数据。")
    
    if os.path.exists(HISTORY_FILE):
        try:
            history_df = pd.read_csv(HISTORY_FILE)
            
            # 倒序显示，最新的在最上面
            history_df = history_df.iloc[::-1].reset_index(drop=True)
            
            # 使用 Streamlit 的 dataframe 展示
            st.dataframe(history_df, use_container_width=True, height=500)
            
            # 提供下载按钮
            csv_data = history_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="📥 下载历史记录 (CSV)",
                data=csv_data,
                file_name="plant_recognition_history.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.info("暂无历史记录或数据读取失败。")
    else:
        st.info("暂无历史记录。快去「智能识别」页面上传第一张照片吧！")
