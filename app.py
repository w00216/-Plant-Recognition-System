# -*- coding: utf-8 -*-
import os
import json
import hashlib
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ================= 1. 页面基本配置 =================
st.set_page_config(
    page_title="云南珍稀保护植物智能识别系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

USERS_FILE = os.path.join(BASE_DIR, "users.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")
MESSAGES_FILE = os.path.join(BASE_DIR, "messages.json") # 存储管理员公告
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 硬编码管理员账号
ADMIN_USER = "云晓植"
ADMIN_PASS = "Wxq@0216"
# 专家注册邀请码
EXPERT_CODE = "EXPERT2024"

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

# ================= 3. 工具函数 =================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        default_users = {ADMIN_USER: {"password": hash_password(ADMIN_PASS), "role": "admin"}}
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_users, f, ensure_ascii=False, indent=4)
        return default_users
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_user(username, password, role):
    users = load_users()
    users[username] = {"password": hash_password(password), "role": role}
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

def load_messages():
    if os.path.exists(MESSAGES_FILE):
        with open(MESSAGES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_message(msg):
    messages = load_messages()
    messages.append({"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "content": msg})
    with open(MESSAGES_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

# 记录历史记录
def log_history(username, predicted_class, confidence, image_path):
    record = {
        "username": username,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path,
        "predicted_class": predicted_class,
        "confidence": float(confidence)
    }
    history_data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try: history_data = json.load(f)
            except: pass
    history_data.append(record)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=4)

# ================= 4. 登录与注册 =================
if 'logged_in' not in st.session_state:
    st.session_state.update({'logged_in': False, 'username': "", 'role': ""})

if not st.session_state['logged_in']:
    st.markdown("<h1 style='text-align: center; color: #2D6A4F; margin-top: 50px;'>🌿 云南珍稀保护植物智能识别系统</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>请登录或注册以使用系统</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auth_tab1, auth_tab2 = st.tabs(["🔑 登录", "📝 注册"])
        with auth_tab1:
            login_user = st.text_input("账号", key="login_user")
            login_pass = st.text_input("密码", type="password", key="login_pass")
            if st.button("登录", use_container_width=True):
                users = load_users()
                if login_user in users and users[login_user]["password"] == hash_password(login_pass):
                    st.session_state.update({'logged_in': True, 'username': login_user, 'role': users[login_user]["role"]})
                    st.rerun()
                else:
                    st.error("账号或密码错误！")
        with auth_tab2:
            reg_user = st.text_input("设置账号", key="reg_user")
            reg_pass = st.text_input("设置密码", type="password", key="reg_pass")
            reg_role = st.selectbox("选择身份", ["普通用户", "专家"])
            expert_code_input = st.text_input("专家邀请码（测试码：EXPERT2024）", type="password") if reg_role == "专家" else ""
            if st.button("注册", use_container_width=True):
                users = load_users()
                if reg_user in users or reg_user == ADMIN_USER: st.error("该账号已存在或不可用！")
                elif not reg_user or not reg_pass: st.error("账号和密码不能为空！")
                else:
                    final_role = "expert" if (reg_role == "专家" and expert_code_input == EXPERT_CODE) else "user"
                    if reg_role == "专家" and expert_code_input != EXPERT_CODE: st.error("专家邀请码错误，已自动注册为普通用户。")
                    save_user(reg_user, reg_pass, final_role)
                    st.success("注册成功，请登录！")
    st.stop()

# ================= 5. Grad-CAM 算法 =================
def make_gradcam_heatmap(img_array, model, pred_index=None):
    base_model = model.layers[0]
    last_conv_layer_name = 'top_activation' 
    try:
        grad_model = Model([base_model.inputs], [base_model.get_layer(last_conv_layer_name).output, model.output])
    except:
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        grad_model = Model([base_model.inputs], [base_model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None: pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

# ================= 6. 全局主题配置 =================
if 'dark_mode' not in st.session_state: st.session_state['dark_mode'] = False
dark_mode = st.session_state['dark_mode']

if dark_mode:
    bg_app, text_app, sidebar_bg = "#0f1712", "#e8f5e9", "#152019"
    title_color, sub_color = "#4CAF50", "#a3b8a6"
    card_bg, card_border, card_text = "#1a261f", "#2a3d31", "#e8f5e9"
    warn_bg, warn_border, warn_text = "#2a1f1f", "#ef5350", "#ffcdd2"
    btn_bg = "#4CAF50"
else:
    bg_app, text_app, sidebar_bg = "#F4F7F4", "#1B4332", "#FFFFFF"
    title_color, sub_color = "#2D6A4F", "#64748b"
    card_bg, card_border, card_text = "#FFFFFF", "#E2E8F0", "#1B4332"
    warn_bg, warn_border, warn_text = "#FFF3E0", "#F57C00", "#E65100"
    btn_bg = "#2D6A4F"

theme_css = f"""
<style>
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    #MainMenu {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; }}
    .stApp {{ background-color: {bg_app} !important; color: {text_app} !important; }}
    [data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; border-right: 1px solid {card_border} !important; }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span {{ color: {text_app} !important; }}
    .hero-title {{ color: {title_color} !important; font-size: 2.8rem; font-weight: 800; margin-bottom: 0px; }}
    .hero-subtitle {{ color: {sub_color} !important; font-size: 1.1rem; margin-bottom: 2rem; }}
    .custom-card {{ background-color: {card_bg} !important; border: 1px solid {card_border} !important; border-left: 5px solid {title_color} !important; padding: 24px; border-radius: 12px; color: {card_text} !important; }}
    .upload-info {{ background-color: {sidebar_bg} !important; border: 1px dashed {title_color} !important; color: {text_app} !important; padding: 20px; border-radius: 12px; text-align: center; }}
    .similar-warn {{ background-color: {warn_bg} !important; border-left: 5px solid {warn_border} !important; padding: 16px 20px !important; border-radius: 8px !important; margin-top: 15px; }}
    .similar-title {{ color: {warn_text} !important; font-weight: bold; font-size: 1.1rem; }}
    .similar-text {{ color: {warn_text} !important; font-size: 0.95rem; line-height: 1.5; }}
    .stButton>button {{ background-color: {btn_bg} !important; color: white !important; border-radius: 8px !important; }}
    .stButton>button:hover {{ transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important; }}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# ================= 7. 侧边栏 =================
with st.sidebar:
    role_name = "👑 管理员" if st.session_state['role']=='admin' else "🔬 植物专家" if st.session_state['role']=='expert' else "🌱 普通用户"
    st.markdown(f"### 👤 欢迎，**{st.session_state['username']}**")
    st.info(f"当前身份：{role_name}")
    if st.button("🚪 退出登录", use_container_width=True):
        st.session_state.update({'logged_in': False, 'username': "", 'role': ""})
        st.rerun()
    st.write("---")
    if st.toggle("🌙 开启深色模式", value=dark_mode):
        st.session_state['dark_mode'] = True
    else:
        st.session_state['dark_mode'] = False
    if st.session_state['dark_mode'] != dark_mode: st.rerun()

# ================= 8. 加载模型 =================
MODEL_PATH = os.path.join(BASE_DIR, "plant_model.h5")
IMG_SIZE = (224, 224)
IPLANT_NAME_MAPPING = {"红河橙、宜昌橙": "宜昌橙"}
@st.cache_resource
def load_plant_model(): return load_model(MODEL_PATH)
try: model = load_plant_model(); model_loaded = True
except Exception as e: model_loaded = False; st.error(f"❌ 模型加载失败: {e}")

CLASS_JSON_PATH = os.path.join(BASE_DIR, "class_indices.json")
with open(CLASS_JSON_PATH, 'r', encoding='utf-8') as f: class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}
plant_levels = {
    "云南蓝果树": "一级", "五针白皮松": "一级", "华盖木": "一级", "绿春苏铁": "一级",
    "景东翅子树": "二级", "朱红大杜鹃": "二级", "漾濞槭": "二级", "红河橙、宜昌橙": "二级",
    "希陶木": "省级", "滇重楼": "省级"
}

# 显示管理员公告（所有用户可见）
messages = load_messages()
if messages:
    latest_msg = messages[-1]
    st.toast(f"📢 系统公告 ({latest_msg['time']}): {latest_msg['content']}", icon="📢")

# ================= 9. 渲染各角色界面 =================

# --------- 9.1 管理员界面 (后台控制台 + 识别功能) ---------
if st.session_state['role'] == "admin":
    st.markdown('<div class="hero-title">👑 管理员控制台</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">用户数据管理、公告发布与全功能测试</div>', unsafe_allow_html=True)
    
    admin_tab1, admin_tab2, admin_tab3, admin_tab4 = st.tabs(["📊 数据管理", "👥 用户管理", "📢 发布公告", "🔬 专家级识别"])
    
    with admin_tab1:
        st.markdown("### 全平台识别记录")
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history_records = json.load(f)
            if len(history_records) > 0:
                df_hist = pd.DataFrame(history_records)
                if 'username' not in df_hist.columns: df_hist['username'] = '未知'
                st.dataframe(df_hist[["username", "timestamp", "predicted_class", "confidence"]].iloc[::-1], use_container_width=True)
            else: st.info("暂无记录。")
            
    with admin_tab2:
        st.markdown("### 注册用户列表")
        users = load_users()
        df_users = pd.DataFrame([(k, v['role']) for k, v in users.items()], columns=["账号名称", "身份"])
        df_users["身份"] = df_users["身份"].map({"admin": "管理员", "expert": "专家", "user": "普通用户"})
        st.dataframe(df_users, use_container_width=True, hide_index=True)
        
    with admin_tab3:
        st.markdown("### 发布全局系统公告")
        new_msg = st.text_area("输入公告内容（所有用户登录时都会在右上角弹出提示）")
        if st.button("🚀 发布公告"):
            if new_msg.strip():
                save_message(new_msg.strip())
                st.success("公告已发布！刷新页面即可看到。")
            else: st.warning("公告内容不能为空。")
            
    with admin_tab4:
        st.markdown("### 管理员测试通道 (与专家功能一致)")
        # 复用专家界面的渲染逻辑，直接调用后面的专家面板（此处通过占位符提醒）
        st.info("请使用下方的专家面板进行全功能测试。")
        # 为了简化代码，这里管理员界面包含和专家完全一样的识别区
        # 下一部分复制专家界面代码

# --------- 9.2 专家界面 (专业分析仪表盘风格) ---------
if st.session_state['role'] in ["expert", "admin"]:
    with st.sidebar:
        st.write("---")
        st.markdown("### 🔬 专家分析工具")
        show_gradcam = st.toggle("🔥 生成 Grad-CAM 热力图", value=False)
        st.caption("提示：热力图能解释 AI 的注意力焦点")

    col1, col2 = st.columns([1, 1.5], gap="large")
    with col1:
        st.markdown("#### 📸 图像输入")
        uploaded_file = st.file_uploader("上传图片进行专业分析", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="待分析图像", use_column_width=True)
            btn_predict = st.button("🔬 启动专业分析", use_container_width=True)
        else:
            st.markdown('<div class="upload-info">👆 请上传照片</div>', unsafe_allow_html=True)
            btn_predict = False

    with col2:
        st.markdown("#### 📊 深度分析报告")
        if uploaded_file and btn_predict and model_loaded:
            with st.spinner("AI 正在提取深度特征..."):
                img = image.convert('RGB').resize(IMG_SIZE)
                img_array = preprocess_input(np.expand_dims(img_to_array(img), axis=0)).astype('float32')
                predictions = model.predict(img_array, verbose=0)
                
                probs = predictions[0]
                top5_indices = np.argsort(probs)[::-1][:5]
                top5_results = [(idx_to_class.get(i, "未知"), probs[i] * 100) for i in top5_indices]
                predicted_class, confidence = top5_results[0]
                
                if confidence < 80.0:
                    st.error(f"❌ 置信度较低 ({confidence:.2f}%)，可能为未知或非目标物种。")
                else:
                    # 写入历史
                    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    safe_filename = f"{now_str}_{uploaded_file.name}"
                    saved_img_path = os.path.join(UPLOAD_DIR, safe_filename)
                    image.convert('RGB').save(saved_img_path)
                    log_history(st.session_state['username'], predicted_class, confidence, saved_img_path)
                    
                    level = plant_levels.get(predicted_class, "未知")
                    st.markdown(f"""
                        <div class="custom-card">
                            <h2 style="margin-top:0; color: {title_color}; font-size: 2rem;">🌱 {predicted_class}</h2>
                            <div style="display: flex; gap: 30px; margin-top: 15px;">
                                <p style="font-size: 1.1rem; margin:0;">保护级别：<span style="color: {warn_border}; font-weight: bold;">{level}</span></p>
                                <p style="font-size: 1.1rem; margin:0;">置信度：<b>{confidence:.2f}%</b></p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(confidence))
                    
                    # 专家专属：Top-5 与 相似物种预警排布
                    st.markdown("---")
                    c2_1, c2_2 = st.columns([1, 1])
                    with c2_1:
                        st.markdown("##### 📈 模型内部 Top-5 分布")
                        df_top5 = pd.DataFrame(top5_results, columns=["候选", "概率(%)"])
                        df_top5["概率(%)"] = df_top5["概率(%)"].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(df_top5, use_container_width=True, hide_index=True)
                    
                    with c2_2:
                        if predicted_class in SIMILAR_PLANTS:
                            st.markdown("##### ⚠️ 相似物种鉴别")
                            for sim in SIMILAR_PLANTS[predicted_class]:
                                st.markdown(f"""
                                    <div class="similar-warn" style="margin-top:5px;">
                                        <div class="similar-title">🚨 易混淆：{sim['name']}</div>
                                        <div class="similar-text" style="font-size:0.85rem;">{sim['diff']}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                    
                    if show_gradcam:
                        st.markdown("---")
                        st.markdown("##### 🔥 AI 注意力热力图")
                        try:
                            heatmap = make_gradcam_heatmap(img_array, model, pred_index=top5_indices[0])
                            superimposed_img = display_gradcam(saved_img_path, heatmap)
                            gc1, gc2 = st.columns(2)
                            gc1.image(image.resize(IMG_SIZE), caption="原图", use_column_width=True)
                            gc2.image(superimposed_img, caption="AI 关注区域", use_column_width=True)
                        except Exception as e:
                            st.warning("热力图生成失败")

# --------- 9.3 普通用户界面 (极简科普风，恢复原有排版) ---------
elif st.session_state['role'] == "user":
    st.markdown('<div class="hero-title">🌿 云南珍稀植物科普识别</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">上传照片，获取植物百科与保护级别</div>', unsafe_allow_html=True)
    
    # 使用原本的 1:1.2 分栏
    col1, col2 = st.columns([1, 1.2], gap="large")
    with col1:
        st.markdown("#### 📸 图片采集区")
        uploaded_file = st.file_uploader("点击或拖拽上传植物照片", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="待识别图像", use_column_width=True)
            btn_predict = st.button("🔍 启动 AI 识别", use_container_width=True)
        else:
            st.markdown('<div class="upload-info">👆 请上传一张清晰的植物照片</div>', unsafe_allow_html=True)
            btn_predict = False

    with col2:
        st.markdown("#### 📊 物种鉴定结果")
        if uploaded_file and btn_predict and model_loaded:
            with st.spinner("AI 正在提取特征比对中..."):
                img = image.convert('RGB').resize(IMG_SIZE)
                img_array = preprocess_input(np.expand_dims(img_to_array(img), axis=0)).astype('float32')
                predictions = model.predict(img_array, verbose=0)
                
                pred_idx = np.argmax(predictions[0])
                confidence = predictions[0][pred_idx] * 100
                predicted_class = idx_to_class.get(pred_idx, "未知物种")
                
                if confidence < 80.0:
                    st.error("❌ 拒绝识别：该图片可能不是植物，或者不属于本系统支持的 10 种植物。")
                else:
                    # 写入历史
                    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    safe_filename = f"{now_str}_{uploaded_file.name}"
                    saved_img_path = os.path.join(UPLOAD_DIR, safe_filename)
                    image.convert('RGB').save(saved_img_path)
                    log_history(st.session_state['username'], predicted_class, confidence, saved_img_path)

                    level = plant_levels.get(predicted_class, "保护物种")
                    st.markdown(f"""
                        <div class="custom-card">
                            <h2 style="margin-top:0; color: {title_color}; font-size: 2rem;">🌱 {predicted_class}</h2>
                            <div style="margin-top: 15px;">
                                <span style="font-size: 1.1rem; color: {sub_color};">保护级别：</span>
                                <span style="color: {warn_border}; font-weight: bold; font-size: 1.1rem; border: 1px solid {warn_border}; padding: 3px 10px; border-radius: 20px; margin-left: 10px;">{level}</span>
                            </div>
                            <p style="font-size: 1.1rem; margin-top: 15px; color: {sub_color};">
                                模型置信度：<b style="color: {text_app};">{confidence:.2f}%</b>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(confidence))
                    st.success("✅ 鉴定结果可靠，点击下方链接查看权威科普。")
                    
                    query_name = IPLANT_NAME_MAPPING.get(predicted_class, predicted_class)
                    st.markdown(f"""
                        <a href="https://www.iplant.cn/info/{query_name}" target="_blank" style="text-decoration:none;">
                            <div style="background-color: {card_bg}; color: {title_color}; padding: 14px 20px; border-radius: 8px; border: 1px solid {title_color}; font-weight: 600; text-align: center; margin-top: 20px;">
                                🔗 查看「{predicted_class}」官方完整形态描述与标本记录 ↗
                            </div>
                        </a>
                    """, unsafe_allow_html=True)
