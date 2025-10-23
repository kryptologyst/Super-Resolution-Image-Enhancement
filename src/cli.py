"""
Command-line interface for Super-Resolution image enhancement.

Provides a CLI tool for batch processing images and various utility functions.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import time

import torch
from PIL import Image

from super_resolution import SuperResolutionModel, ImageQualityMetrics, create_sample_dataset
from config_manager import ConfigManager, Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str, log_file: Optional[str] = None):
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def process_single_image(
    input_path: Path,
    output_path: Path,
    model: SuperResolutionModel,
    config: Config
) -> bool:
    """
    Process a single image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image
        model: Super-resolution model
        config: Configuration object
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {input_path}")
        
        # Process image
        start_time = time.time()
        original, upscaled = model.upscale_image(input_path)
        processing_time = time.time() - start_time
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        upscaled.save(output_path, quality=95)
        
        # Calculate metrics if enabled
        if config.processing.quality_metrics:
            original_array = np.array(original.convert('L'))
            upscaled_array = np.array(upscaled.convert('L'))
            psnr = ImageQualityMetrics.calculate_psnr(original_array, upscaled_array)
            ssim = ImageQualityMetrics.calculate_ssim(original_array, upscaled_array)
            
            logger.info(f"Quality metrics - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        
        logger.info(f"Completed in {processing_time:.2f}s - Saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False


def process_batch_images(
    input_dir: Path,
    output_dir: Path,
    model: SuperResolutionModel,
    config: Config
) -> None:
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        model: Super-resolution model
        config: Configuration object
    """
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Find all supported image files
    image_files = []
    for ext in config.data.supported_formats:
        image_files.extend(input_dir.glob(f"*.{ext}"))
        image_files.extend(input_dir.glob(f"*.{ext.upper()}"))
    
    if not image_files:
        logger.warning(f"No supported image files found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    successful = 0
    failed = 0
    
    for i, input_file in enumerate(image_files, 1):
        logger.info(f"Processing {i}/{len(image_files)}: {input_file.name}")
        
        # Create output filename
        output_file = output_dir / f"enhanced_{input_file.name}"
        
        if process_single_image(input_file, output_file, model, config):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Batch processing complete: {successful} successful, {failed} failed")


def create_samples_command(args):
    """Handle sample creation command."""
    output_dir = Path(args.output_dir)
    num_samples = args.num_samples
    
    logger.info(f"Creating {num_samples} sample images in {output_dir}")
    
    try:
        sample_files = create_sample_dataset(output_dir, num_samples)
        logger.info(f"Successfully created {len(sample_files)} sample images")
        
        # List created files
        for file_path in sample_files:
            logger.info(f"Created: {file_path}")
            
    except Exception as e:
        logger.error(f"Error creating samples: {e}")
        sys.exit(1)


def process_command(args):
    """Handle image processing command."""
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    if not config_manager.validate_config(config):
        logger.error("Invalid configuration")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.model_type:
        config.model.type = args.model_type
    if args.scale_factor:
        config.model.scale_factor = args.scale_factor
    if args.device:
        config.model.device = args.device
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Initialize model
    logger.info("Initializing super-resolution model...")
    try:
        model = SuperResolutionModel(
            model_type=config.model.type,
            scale_factor=config.model.scale_factor,
            device=config.model.device if config.model.device != "auto" else None
        )
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)
    
    # Process images
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Single file processing
        logger.info("Processing single image")
        if not process_single_image(input_path, output_path, model, config):
            sys.exit(1)
    elif input_path.is_dir():
        # Batch processing
        logger.info("Processing directory")
        process_batch_images(input_path, output_path, model, config)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Super-Resolution Image Enhancement CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python cli.py process input.jpg output.jpg
  
  # Process all images in a directory
  python cli.py process input_dir/ output_dir/
  
  # Create sample images
  python cli.py samples --output-dir data/samples --num-samples 10
  
  # Use specific model and scale factor
  python cli.py process input.jpg output.jpg --model-type espcn --scale-factor 4
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process images')
    process_parser.add_argument('input', help='Input image file or directory')
    process_parser.add_argument('output', help='Output image file or directory')
    process_parser.add_argument('--model-type', choices=['espcn', 'real_esrgan', 'swinir'], 
                              help='Model type to use')
    process_parser.add_argument('--scale-factor', type=int, choices=[2, 3, 4, 8],
                              help='Upscaling factor')
    process_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'],
                              help='Processing device')
    process_parser.add_argument('--log-level', default='INFO',
                              choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                              help='Logging level')
    process_parser.add_argument('--log-file', help='Log file path')
    
    # Samples command
    samples_parser = subparsers.add_parser('samples', help='Create sample images')
    samples_parser.add_argument('--output-dir', default='data/samples',
                               help='Output directory for samples')
    samples_parser.add_argument('--num-samples', type=int, default=10,
                               help='Number of sample images to create')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'process':
        process_command(args)
    elif args.command == 'samples':
        create_samples_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
