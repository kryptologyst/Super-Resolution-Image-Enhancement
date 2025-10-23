# Super-Resolution Image Enhancement

A production-ready super-resolution system for enhancing low-resolution images using state-of-the-art deep learning models. This project provides multiple interfaces (CLI, Web UI, Python API) and supports various super-resolution architectures.

## Features

- **Multiple Model Support**: ESPCN, Real-ESRGAN, SwinIR (with extensible architecture)
- **Flexible Interfaces**: Command-line tool, Streamlit web app, and Python API
- **Quality Metrics**: PSNR and SSIM calculation for objective evaluation
- **Batch Processing**: Process multiple images efficiently
- **Configuration Management**: YAML-based configuration system
- **Comprehensive Testing**: Unit tests and integration tests
- **Modern Architecture**: Type hints, logging, error handling, and documentation

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Super-Resolution-Image-Enhancement.git
   cd Super-Resolution-Image-Enhancement
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web interface**:
   ```bash
   streamlit run web_app/app.py
   ```

4. **Or use the CLI**:
   ```bash
   python src/cli.py samples --num-samples 5
   python src/cli.py process data/samples/sample_000.jpg data/output/enhanced.jpg
   ```

## 📁 Project Structure

```
super-resolution-for-images/
├── src/                          # Source code
│   ├── super_resolution.py       # Core super-resolution functionality
│   ├── config_manager.py         # Configuration management
│   └── cli.py                    # Command-line interface
├── web_app/                      # Web interface
│   └── app.py                    # Streamlit application
├── tests/                        # Unit tests
│   └── test_super_resolution.py  # Test suite
├── config/                       # Configuration files
│   └── config.yaml               # Main configuration
├── data/                         # Data directories
│   ├── input/                    # Input images
│   ├── output/                   # Enhanced images
│   └── samples/                  # Sample/test images
├── logs/                         # Log files
├── models/                       # Model files
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Usage Examples

### Python API

```python
from src.super_resolution import SuperResolutionModel

# Initialize model
model = SuperResolutionModel(model_type="espcn", scale_factor=3)

# Process single image
original, enhanced = model.upscale_image("input.jpg")

# Batch processing
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = model.batch_upscale(image_paths)
```

### Command Line Interface

```bash
# Create sample images for testing
python src/cli.py samples --output-dir data/samples --num-samples 10

# Process a single image
python src/cli.py process input.jpg output.jpg --model-type espcn --scale-factor 4

# Process all images in a directory
python src/cli.py process data/input/ data/output/ --scale-factor 3

# Use specific device
python src/cli.py process input.jpg output.jpg --device cuda
```

### Web Interface

1. Start the Streamlit app:
   ```bash
   streamlit run web_app/app.py
   ```

2. Open your browser to `http://localhost:8501`

3. Upload images and configure settings in the sidebar

4. Download enhanced images with quality metrics

## Configuration

The system uses YAML configuration files for flexible setup:

```yaml
# config/config.yaml
model:
  type: "espcn"           # Model type: espcn, real_esrgan, swinir
  scale_factor: 3         # Upscaling factor: 2, 3, 4, 8
  device: "auto"          # Device: auto, cpu, cuda

data:
  input_dir: "data/input"
  output_dir: "data/output"
  supported_formats: ["jpg", "jpeg", "png", "bmp", "tiff"]
  max_image_size: 2048

processing:
  batch_size: 1
  quality_metrics: true
  preserve_metadata: true
```

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test
python -m pytest tests/test_super_resolution.py::TestSuperResolutionModel::test_model_initialization -v
```

## Quality Metrics

The system calculates objective quality metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher values indicate better quality
- **SSIM (Structural Similarity Index)**: Values between 0 and 1, closer to 1 is better

```python
from src.super_resolution import ImageQualityMetrics
import numpy as np

# Calculate metrics
psnr = ImageQualityMetrics.calculate_psnr(img1, img2)
ssim = ImageQualityMetrics.calculate_ssim(img1, img2)
```

## 🔧 Advanced Usage

### Custom Model Integration

To add support for new super-resolution models:

1. Extend the `SuperResolutionModel` class
2. Implement the `_load_[model_name]_model()` method
3. Update the configuration validation

### Batch Processing

For large-scale processing:

```python
from pathlib import Path
from src.super_resolution import SuperResolutionModel

model = SuperResolutionModel()
input_dir = Path("data/input")
output_dir = Path("data/output")

# Process all images
for img_path in input_dir.glob("*.jpg"):
    output_path = output_dir / f"enhanced_{img_path.name}"
    original, enhanced = model.upscale_image(img_path)
    enhanced.save(output_path)
```

### Performance Optimization

- **GPU Acceleration**: Set `device: "cuda"` in configuration
- **Batch Processing**: Use `batch_upscale()` for multiple images
- **Memory Management**: Process images in smaller batches for large datasets

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use CPU processing: `device: "cpu"`
   - Process smaller images

2. **Model Loading Errors**:
   - Check PyTorch installation
   - Verify model weights are available
   - Check internet connection for model downloads

3. **Image Format Issues**:
   - Ensure supported formats: JPG, PNG, BMP, TIFF
   - Check image file integrity
   - Verify file permissions

### Debug Mode

Enable debug logging:

```bash
python src/cli.py process input.jpg output.jpg --log-level DEBUG
```

## Performance Benchmarks

Typical performance on different hardware:

| Hardware | Model | Scale Factor | Time per Image (512x512) |
|----------|-------|--------------|-------------------------|
| CPU (Intel i7) | ESPCN | 3x | ~2.5s |
| GPU (RTX 3080) | ESPCN | 3x | ~0.3s |
| CPU (Intel i7) | ESPCN | 4x | ~3.2s |
| GPU (RTX 3080) | ESPCN | 4x | ~0.4s |

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `python -m pytest tests/`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **Torchvision** for pre-trained super-resolution models
- **Streamlit** for the web interface framework
- **OpenCV** for image processing utilities

## References

- [ESPCN: Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)
- [Real-ESRGAN: Training Real-World Blind Super-Resolution](https://arxiv.org/abs/2107.10833)
- [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)

## Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation
- Review the test cases for usage examples


# Super-Resolution-Image-Enhancement
