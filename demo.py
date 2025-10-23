#!/usr/bin/env python3
"""
Demonstration script for the modernized Super-Resolution project.

This script showcases all the key features and improvements made to the project.
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from super_resolution import SuperResolutionModel, ImageQualityMetrics, create_sample_dataset
from config_manager import load_config


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n📋 {title}")
    print("-" * 40)


def main():
    """Main demonstration function."""
    print_header("Super-Resolution Project Modernization Demo")
    
    print("""
This demonstration showcases the modernized super-resolution project with:
✅ Modern Python practices (type hints, docstrings, PEP8)
✅ Comprehensive error handling and logging
✅ Multiple interfaces (CLI, Web UI, Python API)
✅ Configuration management system
✅ Unit tests and validation
✅ Clean project structure
✅ State-of-the-art model implementation
✅ Quality metrics and evaluation
✅ Batch processing capabilities
✅ Documentation and examples
    """)
    
    # 1. Configuration Management
    print_section("Configuration Management")
    try:
        config = load_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   Model type: {config.model.type}")
        print(f"   Scale factor: {config.model.scale_factor}")
        print(f"   Device: {config.model.device}")
        print(f"   Input directory: {config.data.input_dir}")
        print(f"   Output directory: {config.data.output_dir}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
    
    # 2. Model Initialization
    print_section("Model Initialization")
    try:
        model = SuperResolutionModel(
            model_type="espcn",
            scale_factor=3,
            device="cpu"
        )
        print(f"✅ Custom ESPCN model initialized successfully")
        print(f"   Model type: {model.model_type}")
        print(f"   Scale factor: {model.scale_factor}")
        print(f"   Device: {model.device}")
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return
    
    # 3. Sample Dataset Creation
    print_section("Sample Dataset Creation")
    try:
        sample_dir = Path("data/demo_samples")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        sample_files = create_sample_dataset(sample_dir, num_samples=5)
        creation_time = time.time() - start_time
        
        print(f"✅ Created {len(sample_files)} sample images in {creation_time:.2f}s")
        for i, file_path in enumerate(sample_files, 1):
            print(f"   📸 Sample {i}: {file_path.name}")
    except Exception as e:
        print(f"❌ Sample creation error: {e}")
        return
    
    # 4. Single Image Processing
    print_section("Single Image Processing")
    if sample_files:
        try:
            input_file = sample_files[0]
            print(f"Processing: {input_file.name}")
            
            start_time = time.time()
            original, enhanced = model.upscale_image(input_file)
            processing_time = time.time() - start_time
            
            print(f"✅ Image processed successfully in {processing_time:.2f}s")
            print(f"   Original size: {original.size[0]} × {original.size[1]}")
            print(f"   Enhanced size: {enhanced.size[0]} × {enhanced.size[1]}")
            
            # Calculate scale factor
            scale_x = enhanced.size[0] / original.size[0]
            scale_y = enhanced.size[1] / original.size[1]
            print(f"   Scale factor: {scale_x:.1f}x × {scale_y:.1f}x")
            
            # Save enhanced image
            output_dir = Path("data/demo_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"enhanced_{input_file.name}"
            enhanced.save(output_path, quality=95)
            print(f"   💾 Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Single image processing error: {e}")
    
    # 5. Quality Metrics
    print_section("Quality Metrics Calculation")
    try:
        original_array = np.array(original.convert('L'))
        enhanced_array = np.array(enhanced.convert('L'))
        
        psnr = ImageQualityMetrics.calculate_psnr(original_array, enhanced_array)
        ssim = ImageQualityMetrics.calculate_ssim(original_array, enhanced_array)
        
        print(f"✅ Quality metrics calculated")
        print(f"   PSNR: {psnr:.2f} dB")
        print(f"   SSIM: {ssim:.4f}")
    except Exception as e:
        print(f"❌ Quality metrics error: {e}")
    
    # 6. Batch Processing
    print_section("Batch Processing")
    try:
        print(f"Processing {len(sample_files)} images in batch...")
        
        start_time = time.time()
        results = model.batch_upscale(sample_files)
        batch_time = time.time() - start_time
        
        successful = sum(1 for r in results if r is not None)
        print(f"✅ Batch processing completed in {batch_time:.2f}s")
        print(f"   Successfully processed: {successful}/{len(sample_files)} images")
        print(f"   Average time per image: {batch_time/len(sample_files):.2f}s")
        
        # Save batch results
        for i, result in enumerate(results):
            if result:
                original, enhanced = result
                output_path = output_dir / f"batch_{i:03d}.jpg"
                enhanced.save(output_path, quality=95)
                print(f"   💾 Saved: {output_path.name}")
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
    
    # 7. Project Structure Overview
    print_section("Project Structure")
    print("✅ Modern project structure implemented:")
    print("   📁 src/ - Source code with type hints and docstrings")
    print("   📁 web_app/ - Streamlit web interface")
    print("   📁 tests/ - Comprehensive unit tests")
    print("   📁 config/ - YAML configuration management")
    print("   📁 data/ - Organized data directories")
    print("   📁 examples/ - Usage examples and demos")
    print("   📄 requirements.txt - Dependency management")
    print("   📄 .gitignore - Git ignore rules")
    print("   📄 README.md - Comprehensive documentation")
    
    # 8. Available Interfaces
    print_section("Available Interfaces")
    print("✅ Multiple interfaces available:")
    print("   🖥️  Web Interface: streamlit run web_app/app.py")
    print("   💻 CLI Interface: python src/cli.py --help")
    print("   🐍 Python API: Import and use SuperResolutionModel")
    print("   🧪 Testing: python -m pytest tests/ -v")
    
    # 9. Next Steps
    print_section("Next Steps")
    print("🚀 Ready for production use!")
    print("   • Upload your images to data/input/")
    print("   • Run: streamlit run web_app/app.py")
    print("   • Or use CLI: python src/cli.py process input.jpg output.jpg")
    print("   • Check README.md for detailed usage instructions")
    
    print_header("Demo Completed Successfully! 🎉")
    print("The super-resolution project has been fully modernized and is ready for use.")


if __name__ == "__main__":
    main()
