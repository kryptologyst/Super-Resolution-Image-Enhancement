#!/usr/bin/env python3
"""
Basic usage example for the Super-Resolution system.

This script demonstrates how to use the super-resolution functionality
for enhancing low-resolution images.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from super_resolution import SuperResolutionModel, create_sample_dataset
from config_manager import load_config
import matplotlib.pyplot as plt


def main():
    """Main example function."""
    print("🔍 Super-Resolution Image Enhancement Example")
    print("=" * 50)
    
    # Load configuration
    print("📋 Loading configuration...")
    try:
        config = load_config()
        print(f"   Model type: {config.model.type}")
        print(f"   Scale factor: {config.model.scale_factor}")
        print(f"   Device: {config.model.device}")
    except Exception as e:
        print(f"   Using default configuration: {e}")
        config = None
    
    # Initialize model
    print("\n🚀 Initializing super-resolution model...")
    try:
        model = SuperResolutionModel(
            model_type="espcn",
            scale_factor=3,
            device="cpu"  # Use CPU for compatibility
        )
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    # Create sample dataset
    print("\n🎲 Creating sample images...")
    try:
        sample_dir = Path("data/samples")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        sample_files = create_sample_dataset(sample_dir, num_samples=3)
        print(f"   ✅ Created {len(sample_files)} sample images")
        
        # List created files
        for i, file_path in enumerate(sample_files, 1):
            print(f"   📸 Sample {i}: {file_path.name}")
            
    except Exception as e:
        print(f"   ❌ Error creating samples: {e}")
        return
    
    # Process first sample image
    if sample_files:
        print(f"\n🔧 Processing sample image: {sample_files[0].name}")
        try:
            original, enhanced = model.upscale_image(sample_files[0])
            
            print(f"   📏 Original size: {original.size[0]} × {original.size[1]}")
            print(f"   📏 Enhanced size: {enhanced.size[0]} × {enhanced.size[1]}")
            
            # Calculate scale factor
            scale_x = enhanced.size[0] / original.size[0]
            scale_y = enhanced.size[1] / original.size[1]
            print(f"   📈 Scale factor: {scale_x:.1f}x × {scale_y:.1f}x")
            
            # Save enhanced image
            output_dir = Path("data/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"enhanced_{sample_files[0].name}"
            enhanced.save(output_path, quality=95)
            print(f"   💾 Saved enhanced image: {output_path}")
            
            # Display images (if matplotlib is available)
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                axes[0].imshow(original)
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                axes[1].imshow(enhanced)
                axes[1].set_title("Enhanced Image")
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches='tight')
                print(f"   📊 Saved comparison image: {output_dir / 'comparison.png'}")
                
            except Exception as e:
                print(f"   ⚠️  Could not create comparison plot: {e}")
            
        except Exception as e:
            print(f"   ❌ Error processing image: {e}")
            return
    
    # Batch processing example
    print(f"\n📦 Batch processing all samples...")
    try:
        results = model.batch_upscale(sample_files)
        successful = sum(1 for r in results if r is not None)
        print(f"   ✅ Successfully processed {successful}/{len(sample_files)} images")
        
        # Save all enhanced images
        for i, result in enumerate(results):
            if result:
                original, enhanced = result
                output_path = output_dir / f"batch_enhanced_{i:03d}.jpg"
                enhanced.save(output_path, quality=95)
                print(f"   💾 Saved: {output_path.name}")
        
    except Exception as e:
        print(f"   ❌ Error in batch processing: {e}")
    
    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("   • Try the web interface: streamlit run web_app/app.py")
    print("   • Use the CLI: python src/cli.py --help")
    print("   • Process your own images by placing them in data/input/")


if __name__ == "__main__":
    main()
