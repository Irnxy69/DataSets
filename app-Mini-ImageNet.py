import streamlit as st
from datasets import load_dataset
import math
import io
import zipfile
import os

# 1. 设置缓存，防止每次刷新网页都重新下载数据
@st.cache_resource
def load_mini_imagenet():
    # 注意：如果部署到 Streamlit Cloud (服务器在美国)，不需要下面这行镜像设置
    # 如果你在国内本地运行，可以取消注释
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    st.info("正在加载 Mini-ImageNet 数据集，首次运行可能需要几分钟...")
    
    # 加载数据集 (对应你 demo.py 里注释掉的那行)
    ds = load_dataset("timm/mini-imagenet")
    
    return ds

# 2. 调用函数获取数据
try:
    dataset = load_mini_imagenet()
    st.success("数据集加载成功！")
    
    # 这里的 dataset 就是你后面代码要用的变量
    # 例如：train_data = dataset['train']
    
except Exception as e:
    st.error(f"数据集加载失败: {e}")

# 1. 页面基础设置
st.set_page_config(page_title="Mini-ImageNet 数据集浏览器 Pro", layout="wide", page_icon="🖼️")
st.title("📚 Mini-ImageNet 全能浏览器")
st.markdown("支持 **Train/Val/Test 切换** 与 **批量导出** 功能")

# 2. 加载数据 (利用缓存)
@st.cache_resource
def get_dataset_dict():
    # 加载整个 DatasetDict (包含所有划分)
    return load_dataset("timm/mini-imagenet")

try:
    # 获取整个数据集字典
    full_dataset = get_dataset_dict()
    
    # 获取所有可用的划分名称 (通常是 train, validation, test)
    available_splits = list(full_dataset.keys())
    
except Exception as e:
    st.error(f"❌ 数据加载失败: {e}")
    st.stop()

# ==================== 侧边栏控制区 ====================
with st.sidebar:
    st.header("🎛️ 控制面板")
    
    # [功能 1] 选择数据集划分
    selected_split = st.selectbox(
        "📂 选择数据集划分 (Split)", 
        available_splits,
        index=0,
        format_func=lambda x: f"{x.capitalize()} Set (集)" # 让显示更漂亮
    )
    
    # 获取当前选中的数据集
    current_dataset = full_dataset[selected_split]
    total_images = len(current_dataset)
    
    st.success(f"当前加载: {selected_split} | 共 {total_images} 张")
    st.divider()

    # [功能 2] 分页控制
    batch_size = st.slider("每页显示数量", 10, 100, 50)
    total_pages = math.ceil(total_images / batch_size)
    page = st.number_input("📄 跳转到页码", min_value=1, max_value=total_pages, value=1)
    
    # 计算切片索引
    start_idx = (page - 1) * batch_size
    end_idx = min(start_idx + batch_size, total_images)
    
    st.info(f"正在浏览: 第 {start_idx + 1} - {end_idx} 张")

# ==================== 主界面展示区 ====================

st.subheader(f"🧩 {selected_split.capitalize()} Set - 第 {page} 页")

# 获取当前页的数据
current_batch = current_dataset[start_idx:end_idx]
images = current_batch['image']
labels = current_batch['label']

# --- [功能 3] 批量下载功能 (核心升级) ---
def create_zip_buffer(images, labels, start_index):
    """在内存中创建一个 ZIP 文件"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(images):
            # 将图片转换为字节流
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            # 定义压缩包内的文件名 (格式: img_索引_类别.jpg)
            file_name = f"img_{start_index + i}_class_{labels[i]}.jpg"
            zf.writestr(file_name, img_byte_arr.getvalue())
    return zip_buffer.getvalue()

# 在主界面右侧放一个下载按钮
col_info, col_btn = st.columns([3, 1])
with col_btn:
    # 只有当点击按钮时才开始打包（节省资源）
    zip_data = create_zip_buffer(images, labels, start_idx)
    st.download_button(
        label=f"📥 下载本页 {len(images)} 张图片 (.zip)",
        data=zip_data,
        file_name=f"mini_imagenet_{selected_split}_page_{page}.zip",
        mime="application/zip",
        help="点击将当前显示的图片打包下载"
    )

st.divider()

# --- 图片网格显示 ---
cols = st.columns(5) # 5列布局
for i, image in enumerate(images):
    with cols[i % 5]:
        # 使用 width="stretch" 自适应宽度 (修复之前的 Warning)
        st.image(image, width="stretch") 
        st.caption(f"**ID**: {start_idx + i} | **Class**: {labels[i]}")

# 底部状态栏
st.markdown("---")
st.caption(f"🚀 Powered by Streamlit & Hugging Face Datasets | 当前模式: {selected_split}")