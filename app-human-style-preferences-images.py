import streamlit as st
from datasets import load_dataset, Image
import math
import io
import zipfile
import pandas as pd
import os

@st.cache_resource
def load_style_preferences():
    # 同样，部署到云端时建议注释掉镜像设置
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    st.info("正在加载 Human Style Preferences 数据集...")
    
    # 加载数据集 (对应你 demo.py 里的有效代码)
    ds = load_dataset("Rapidata/human-style-preferences-images")
    
    return ds

try:
    dataset = load_style_preferences()
    st.success("Style 数据加载成功！")
    
    # 示例：显示第一张图片测试一下 (你可以删掉这行)
    # if 'train' in dataset:
    #     st.image(dataset['train'][0]['image'], caption="示例图片")

except Exception as e:
    st.error(f"加载失败: {e}")

# 1. 页面基础设置
st.set_page_config(page_title="Human Style Preferences 浏览器", layout="wide", page_icon="🎨")
st.title("🎨 Human Style Preferences 数据集浏览器")
st.markdown("""
**数据集**: `Rapidata/human-style-preferences-images`  
**用途**: 展示人类对 AI 生成图像在 **风格 (Style)**、**连贯性 (Coherence)** 等维度的偏好评分。
""")

# 2. 核心加载函数
@st.cache_resource
def load_data_structure():
    # 这里使用 streaming=False 确保数据下载到本地缓存，保证演示流畅
    # 如果网速慢，第一次加载可能需要几分钟
    dataset_name = "Rapidata/human-style-preferences-images"
    try:
        ds = load_dataset(dataset_name)
        return ds
    except Exception as e:
        return None, str(e)

# 加载数据
with st.spinner('正在连接 Hugging Face 加载元数据，请稍候...'):
    data_result = load_data_structure()

if isinstance(data_result, tuple): # 报错了
    st.error(f"❌ 数据加载失败: {data_result[1]}")
    st.stop()
else:
    full_dataset = data_result

# ==================== 智能字段分析 ====================
# 不同的数据集列名不一样，我们写一个自动检测逻辑
sample_split = list(full_dataset.keys())[0]
features = full_dataset[sample_split].features

# 1. 找图片列 (通常叫 image, img, jpg)
image_col = next((col for col in features.keys() if isinstance(features[col], Image) or col in ['image', 'img', 'jpg']), None)

# 2. 找提示词列 (通常叫 prompt, caption, text)
text_col = next((col for col in features.keys() if col in ['prompt', 'caption', 'text']), None)

# 3. 找评分列 (这个数据集特有的)
# 根据数据集介绍，可能包含 'style_score', 'coherence_score' 等，或者只是 'score'
score_cols = [col for col in features.keys() if 'score' in col or 'rating' in col or 'preference' in col]

if not image_col:
    st.error("⚠️ 未在数据集中找到图片列！请检查数据集结构。")
    st.stop()

# ==================== 侧边栏控制区 ====================
with st.sidebar:
    st.header("🎛️ 控制面板")
    
    # [功能 1] 数据集划分选择
    available_splits = list(full_dataset.keys())
    selected_split = st.selectbox(
        "📂 选择数据划分 (Split)", 
        available_splits,
        format_func=lambda x: f"{x.capitalize()} ({len(full_dataset[x])} 张)"
    )
    
    current_dataset = full_dataset[selected_split]
    total_images = len(current_dataset)
    
    st.divider()

    # [功能 2] 分页控制
    batch_size = st.slider("每页显示数量", 10, 100, 50)
    total_pages = math.ceil(total_images / batch_size)
    page = st.number_input("📄 跳转到页码", min_value=1, max_value=total_pages, value=1)
    
    start_idx = (page - 1) * batch_size
    end_idx = min(start_idx + batch_size, total_images)
    
    st.info(f"当前显示: {start_idx + 1} - {end_idx} / {total_images}")

# ==================== 批量下载逻辑 ====================
def create_zip_buffer(batch_data, start_index):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, item in enumerate(batch_data):
            img = item[image_col]
            # 尝试获取文件名友好的标签
            label_str = f"id_{start_index + i}"
            
            # 将图片转为字节
            img_byte_arr = io.BytesIO()
            # 某些数据集图片可能是 RGBA，转为 RGB 兼容 JPEG
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(img_byte_arr, format='JPEG')
            
            zf.writestr(f"{label_str}.jpg", img_byte_arr.getvalue())
            
            # 如果有提示词，也把提示词存成文本文件
            if text_col:
                prompt_text = item.get(text_col, "")
                zf.writestr(f"{label_str}.txt", prompt_text)
                
    return zip_buffer.getvalue()

# ==================== 主界面展示区 ====================
st.subheader(f"🧩 {selected_split} Set - 第 {page} 页")

# 获取切片数据
current_batch = current_dataset[start_idx:end_idx]

# 下载按钮区
col_info, col_btn = st.columns([3, 1])
with col_btn:
    # 注意：这里需要把 batch 转为 list 才能遍历，HuggingFace Dataset 切片返回的是字典的列表 {col: [vals]}
    # 所以我们需要重新组装一下方便处理
    batch_list = [current_dataset[i] for i in range(start_idx, end_idx)]
    
    zip_data = create_zip_buffer(batch_list, start_idx)
    st.download_button(
        label=f"📥 下载本页图片与Prompt (.zip)",
        data=zip_data,
        file_name=f"human_prefs_{selected_split}_p{page}.zip",
        mime="application/zip"
    )

st.divider()

# 网格显示
cols = st.columns(4) # 4列布局，因为这次文字可能比较多，宽一点好
for i, item in enumerate(batch_list):
    with cols[i % 4]:
        # 1. 显示图片
        st.image(item[image_col], use_container_width=True)
        
        # 2. 显示 Prompt (可折叠，防止太长)
        if text_col:
            with st.expander("查看 Prompt (提示词)"):
                st.caption(item[text_col])
        
        # 3. 显示评分信息 (如有)
        if score_cols:
            score_text = " | ".join([f"**{c.split('_')[0]}**: {item[c]:.2f}" for c in score_cols if item[c] is not None])
            st.markdown(f"<small>{score_text}</small>", unsafe_allow_html=True)
        
        st.caption(f"ID: {start_idx + i}")

# 底部
st.markdown("---")
st.caption("注：如果某些图片加载较慢，是因为该数据集包含高分辨率生成图。")