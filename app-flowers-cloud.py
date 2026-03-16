import streamlit as st
import math
import io
import zipfile
# 注意这里引入了 concatenate_datasets
from datasets import load_dataset, concatenate_datasets, Image as HfImage

st.set_page_config(page_title="Flowers 102 云端浏览器", layout="wide", page_icon="🌸")
st.title("🌸 Oxford Flowers 102 (云端动态 7:1.5:1.5 划分版)")
st.markdown("""
数据来源: Hugging Face `dpdl-benchmark/oxford_flowers102`  
处理状态: 已在云端内存中动态合并原版数据，并重新按照 70% 训练集、15% 验证集、15% 测试集进行随机重组。
""")

# ==================== 1. 云端拉取 + 动态重组 ====================
@st.cache_resource
def load_and_split_dataset():
    st.info("正在从云端拉取数据并动态重新划分比例，请稍候...")
    try:
        # 1. 从 Hugging Face 拉取原始数据 (部署在海外，速度极快)
        ds = load_dataset("dpdl-benchmark/oxford_flowers102")
        
        # 2. 合并所有数据 (共 8189 张)
        full_ds = concatenate_datasets([ds['train'], ds['validation'], ds['test']])
        
        # 3. 按照 7 : 1.5 : 1.5 重新划分
        train_testval = full_ds.train_test_split(test_size=0.3, seed=42)
        test_val = train_testval['test'].train_test_split(test_size=0.5, seed=42)
        
        # 4. 组装成新的字典返回
        custom_splits = {
            'train': train_testval['train'],
            'val': test_val['train'],
            'test': test_val['test']
        }
        
        # 为了后续能拿到类别名称，我们把原始的 features 也带上
        features = ds['train'].features
        
        return custom_splits, features
    except Exception as e:
        return None, str(e)

with st.spinner('正在云端处理数据...'):
    data_result = load_and_split_dataset()

if isinstance(data_result, tuple) and data_result[0] is None:
    st.error(f"❌ 数据加载失败: {data_result[1]}")
    st.stop()
else:
    full_dataset, features = data_result
    st.success("✅ 云端花卉数据动态重组成功！")

# ==================== 2. 字段分析与类别映射 ====================
image_col = next((col for col, f_type in features.items() if isinstance(f_type, HfImage) or col in ['image', 'img']), 'image')
label_col = next((col for col in features.keys() if 'label' in col or 'class' in col), 'label')

class_names = []
if hasattr(features[label_col], 'names'):
    class_names = features[label_col].names

# ==================== 3. 侧边栏控制 ====================
with st.sidebar:
    st.header("🎛️ 控制面板")
    
    selected_split = st.selectbox(
        "📂 选择数据划分 (Split)", 
        list(full_dataset.keys()),
        format_func=lambda x: f"{x.capitalize()} ({len(full_dataset[x])} 张)"
    )
    
    current_dataset = full_dataset[selected_split]
    total_images = len(current_dataset)
    st.divider()

    batch_size = st.slider("每页显示数量", 8, 48, 16)
    total_pages = math.ceil(total_images / batch_size)
    page = st.number_input("📄 跳转到页码", min_value=1, max_value=total_pages, value=1)
    
    start_idx = (page - 1) * batch_size
    end_idx = min(start_idx + batch_size, total_images)

# ==================== 4. 分类友好的下载逻辑 ====================
def create_cls_zip_buffer(batch_data, start_index):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, item in enumerate(batch_data):
            img = item[image_col]
            label_id = item[label_col]
            
            folder_name = class_names[label_id] if class_names else f"class_{label_id}"
            folder_name = str(folder_name).replace("/", "_").replace("\\", "_")
            file_name = f"{folder_name}/img_{start_index + i}.jpg"
            
            img_byte_arr = io.BytesIO()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(img_byte_arr, format='JPEG')
            zf.writestr(file_name, img_byte_arr.getvalue())
                
    return zip_buffer.getvalue()

# ==================== 5. 主界面展示 ====================
st.subheader(f"🧩 {selected_split} Set - 第 {page} 页")

batch_list = [current_dataset[i] for i in range(start_idx, end_idx)]

col_info, col_btn = st.columns([3, 1])
with col_btn:
    zip_data = create_cls_zip_buffer(batch_list, start_idx)
    st.download_button(
        label=f"📥 按类别打包下载本页 (.zip)",
        data=zip_data,
        file_name=f"flowers_{selected_split}_p{page}.zip",
        mime="application/zip"
    )

st.divider()

cols = st.columns(4) 
for i, item in enumerate(batch_list):
    with cols[i % 4]:
        st.image(item[image_col], width="stretch")
        
        label_id = item[label_col]
        display_name = class_names[label_id] if class_names else f"Class {label_id}"
        
        st.markdown(f"**类别:** `{display_name}`")
        st.caption(f"Image ID: {start_idx + i} | Label ID: {label_id}")

st.markdown("---")