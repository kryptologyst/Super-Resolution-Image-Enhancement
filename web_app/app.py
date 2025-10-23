"""
Streamlit web application for Super-Resolution image enhancement.

Provides an interactive interface for uploading and upscaling images
using various super-resolution models.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import time

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from super_resolution import SuperResolutionModel, ImageQualityMetrics, create_sample_dataset
from config_manager import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Super-Resolution Image Enhancement",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .image-container {
        text-align: center;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []


def load_model_config() -> Tuple[SuperResolutionModel, dict]:
    """Load model and configuration."""
    try:
        config = load_config()
        model = SuperResolutionModel(
            model_type=config.model.type,
            scale_factor=config.model.scale_factor,
            device=config.model.device if config.model.device != "auto" else None
        )
        return model, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def calculate_metrics(img1: np.ndarray, img2: np.ndarray) -> dict:
    """Calculate image quality metrics."""
    try:
        psnr = ImageQualityMetrics.calculate_psnr(img1, img2)
        ssim = ImageQualityMetrics.calculate_ssim(img1, img2)
        return {"PSNR": psnr, "SSIM": ssim}
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"PSNR": 0, "SSIM": 0}


def display_image_comparison(original: Image.Image, upscaled: Image.Image, metrics: dict = None):
    """Display side-by-side image comparison."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Original Image")
        st.image(original, use_column_width=True)
        st.caption(f"Size: {original.size[0]} × {original.size[1]}")
    
    with col2:
        st.markdown("### Enhanced Image")
        st.image(upscaled, use_column_width=True)
        st.caption(f"Size: {upscaled.size[0]} × {upscaled.size[1]}")
    
    if metrics:
        st.markdown("### Quality Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("PSNR", f"{metrics['PSNR']:.2f} dB")
        with col2:
            st.metric("SSIM", f"{metrics['SSIM']:.4f}")


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Super-Resolution Image Enhancement</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Enhance your low-resolution images using state-of-the-art deep learning models. 
    Upload an image and watch it transform into a high-resolution version with improved detail and clarity.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            ["espcn", "real_esrgan", "swinir"],
            index=0,
            help="Select the super-resolution model to use"
        )
        
        scale_factor = st.selectbox(
            "Scale Factor",
            [2, 3, 4, 8],
            index=1,
            help="Upscaling factor (2x, 3x, 4x, or 8x)"
        )
        
        device = st.selectbox(
            "Device",
            ["auto", "cpu", "cuda"],
            index=0,
            help="Processing device (auto-detect recommended)"
        )
        
        # Load model button
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                model, config = load_model_config()
                if model:
                    st.session_state.model = model
                    st.session_state.config = config
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Image", "🎲 Sample Images", "📊 History", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Your Image")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Supported formats: JPG, PNG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🚀 Enhance Image", type="primary"):
                if st.session_state.model is None:
                    st.warning("Please load a model first using the sidebar configuration.")
                else:
                    with st.spinner("Processing image..."):
                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            # Process image
                            start_time = time.time()
                            original, upscaled = st.session_state.model.upscale_image(tmp_path)
                            processing_time = time.time() - start_time
                            
                            # Calculate metrics
                            original_array = np.array(original.convert('L'))
                            upscaled_array = np.array(upscaled.convert('L'))
                            metrics = calculate_metrics(original_array, upscaled_array)
                            
                            # Store in history
                            st.session_state.processing_history.append({
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'filename': uploaded_file.name,
                                'original_size': original.size,
                                'upscaled_size': upscaled.size,
                                'processing_time': processing_time,
                                'metrics': metrics
                            })
                            
                            # Display results
                            st.success(f"Image enhanced successfully! Processing time: {processing_time:.2f}s")
                            display_image_comparison(original, upscaled, metrics)
                            
                            # Download button
                            img_buffer = io.BytesIO()
                            upscaled.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Download Enhanced Image",
                                data=img_buffer.getvalue(),
                                file_name=f"enhanced_{uploaded_file.name}",
                                mime="image/png"
                            )
                            
                            # Clean up temp file
                            Path(tmp_path).unlink()
                            
                        except Exception as e:
                            st.error(f"Error processing image: {e}")
                            logger.error(f"Processing error: {e}")
    
    with tab2:
        st.header("Sample Images")
        st.markdown("Generate and process sample images for testing.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.slider("Number of samples", 1, 10, 5)
            if st.button("🎲 Generate Sample Images"):
                with st.spinner("Generating sample images..."):
                    try:
                        sample_dir = Path("data/samples")
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        sample_images = create_sample_dataset(sample_dir, num_samples)
                        st.success(f"Generated {len(sample_images)} sample images!")
                        
                        # Display first sample
                        if sample_images:
                            sample_img = Image.open(sample_images[0])
                            st.image(sample_img, caption="Sample Image", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error generating samples: {e}")
        
        with col2:
            if st.button("🚀 Process All Samples"):
                if st.session_state.model is None:
                    st.warning("Please load a model first.")
                else:
                    sample_dir = Path("data/samples")
                    if sample_dir.exists():
                        sample_files = list(sample_dir.glob("*.jpg"))
                        if sample_files:
                            with st.spinner("Processing sample images..."):
                                try:
                                    results = st.session_state.model.batch_upscale(sample_files)
                                    successful = sum(1 for r in results if r is not None)
                                    st.success(f"Processed {successful}/{len(sample_files)} images successfully!")
                                    
                                    # Display first result
                                    if results and results[0]:
                                        original, upscaled = results[0]
                                        display_image_comparison(original, upscaled)
                                except Exception as e:
                                    st.error(f"Error processing samples: {e}")
                        else:
                            st.warning("No sample images found. Generate some first!")
                    else:
                        st.warning("Sample directory not found. Generate samples first!")
    
    with tab3:
        st.header("Processing History")
        
        if st.session_state.processing_history:
            for i, entry in enumerate(reversed(st.session_state.processing_history)):
                with st.expander(f"📸 {entry['filename']} - {entry['timestamp']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Size", f"{entry['original_size'][0]} × {entry['original_size'][1]}")
                    with col2:
                        st.metric("Enhanced Size", f"{entry['upscaled_size'][0]} × {entry['upscaled_size'][1]}")
                    with col3:
                        st.metric("Processing Time", f"{entry['processing_time']:.2f}s")
                    
                    if 'metrics' in entry:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("PSNR", f"{entry['metrics']['PSNR']:.2f} dB")
                        with col2:
                            st.metric("SSIM", f"{entry['metrics']['SSIM']:.4f}")
        else:
            st.info("No processing history yet. Upload and process some images!")
    
    with tab4:
        st.header("About Super-Resolution")
        
        st.markdown("""
        ### What is Super-Resolution?
        
        Super-resolution is a computer vision technique that enhances the resolution and quality 
        of low-resolution images using deep learning models. It's particularly useful for:
        
        - **Medical Imaging**: Enhancing X-rays, MRIs, and other medical scans
        - **Satellite Imagery**: Improving satellite and aerial photography
        - **Video Enhancement**: Upscaling video content for better viewing
        - **Forensic Analysis**: Enhancing surveillance footage and evidence
        - **Art Restoration**: Digitally restoring old or damaged images
        
        ### Models Available
        
        - **ESPCN**: Efficient Sub-Pixel Convolutional Network - Fast and lightweight
        - **Real-ESRGAN**: Real-world Super-Resolution using Generative Adversarial Networks
        - **SwinIR**: Swin Transformer for Image Restoration
        
        ### Technical Details
        
        The application uses PyTorch-based models and processes images in the YCbCr color space 
        for optimal results. Quality metrics (PSNR and SSIM) are calculated to evaluate the 
        enhancement quality.
        
        ### Performance Tips
        
        - Use GPU acceleration when available for faster processing
        - Smaller images process faster than larger ones
        - Higher scale factors require more processing time
        - Supported formats: JPG, PNG, BMP, TIFF
        """)
        
        # System information
        st.markdown("### System Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**PyTorch Version**: {torch.__version__}")
            st.info(f"**CUDA Available**: {'Yes' if torch.cuda.is_available() else 'No'}")
        with col2:
            if torch.cuda.is_available():
                st.info(f"**GPU**: {torch.cuda.get_device_name(0)}")
                st.info(f"**GPU Memory**: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
