"""
Unit tests for the super-resolution module.

Tests core functionality including model loading, image processing,
and quality metrics calculation.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from super_resolution import SuperResolutionModel, ImageQualityMetrics, create_sample_dataset
from config_manager import ConfigManager, Config


class TestSuperResolutionModel(unittest.TestCase):
    """Test cases for SuperResolutionModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.model = SuperResolutionModel(model_type="espcn", scale_factor=3, device="cpu")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        # Test default initialization
        model = SuperResolutionModel()
        self.assertIsNotNone(model.model)
        self.assertEqual(model.model_type, "espcn")
        self.assertEqual(model.scale_factor, 3)
        
        # Test custom parameters
        model = SuperResolutionModel(model_type="espcn", scale_factor=2, device="cpu")
        self.assertEqual(model.scale_factor, 2)
        self.assertEqual(model.device, "cpu")
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        with self.assertRaises(ValueError):
            SuperResolutionModel(model_type="invalid_model")
    
    def test_invalid_scale_factor(self):
        """Test initialization with invalid scale factor."""
        with self.assertRaises(ValueError):
            SuperResolutionModel(model_type="espcn", scale_factor=5)
    
    def test_preprocess_image(self):
        """Test image preprocessing functionality."""
        # Create a test image
        test_image = Image.new('RGB', (120, 120), color='red')
        test_path = self.temp_dir / "test.jpg"
        test_image.save(test_path)
        
        # Test preprocessing
        input_tensor, cb, cr, original = self.model.preprocess_image(test_path)
        
        self.assertIsInstance(input_tensor, torch.Tensor)
        self.assertEqual(input_tensor.shape[0], 1)  # Batch dimension
        self.assertIsInstance(cb, Image.Image)
        self.assertIsInstance(cr, Image.Image)
        self.assertIsInstance(original, Image.Image)
    
    def test_preprocess_nonexistent_image(self):
        """Test preprocessing with non-existent image."""
        nonexistent_path = self.temp_dir / "nonexistent.jpg"
        
        with self.assertRaises(FileNotFoundError):
            self.model.preprocess_image(nonexistent_path)
    
    def test_upscale_image(self):
        """Test image upscaling functionality."""
        # Create a test image
        test_image = Image.new('RGB', (120, 120), color='blue')
        test_path = self.temp_dir / "test.jpg"
        test_image.save(test_path)
        
        # Test upscaling
        original, upscaled = self.model.upscale_image(test_path)
        
        self.assertIsInstance(original, Image.Image)
        self.assertIsInstance(upscaled, Image.Image)
        
        # Check that upscaled image is larger
        self.assertGreater(upscaled.size[0], original.size[0])
        self.assertGreater(upscaled.size[1], original.size[1])
        
        # Check scale factor
        scale_x = upscaled.size[0] / original.size[0]
        scale_y = upscaled.size[1] / original.size[1]
        self.assertAlmostEqual(scale_x, self.model.scale_factor, delta=0.1)
        self.assertAlmostEqual(scale_y, self.model.scale_factor, delta=0.1)
    
    def test_batch_upscale(self):
        """Test batch upscaling functionality."""
        # Create multiple test images
        test_paths = []
        for i in range(3):
            test_image = Image.new('RGB', (60, 60), color=(i*50, 100, 150))
            test_path = self.temp_dir / f"test_{i}.jpg"
            test_image.save(test_path)
            test_paths.append(test_path)
        
        # Test batch processing
        results = self.model.batch_upscale(test_paths)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)


class TestImageQualityMetrics(unittest.TestCase):
    """Test cases for ImageQualityMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test images
        self.img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        self.img2 = self.img1.copy()  # Identical image
        self.img3 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  # Different image
    
    def test_calculate_psnr_identical_images(self):
        """Test PSNR calculation with identical images."""
        psnr = ImageQualityMetrics.calculate_psnr(self.img1, self.img2)
        self.assertEqual(psnr, float('inf'))
    
    def test_calculate_psnr_different_images(self):
        """Test PSNR calculation with different images."""
        psnr = ImageQualityMetrics.calculate_psnr(self.img1, self.img3)
        self.assertIsInstance(psnr, float)
        self.assertGreater(psnr, 0)
    
    def test_calculate_ssim_identical_images(self):
        """Test SSIM calculation with identical images."""
        ssim = ImageQualityMetrics.calculate_ssim(self.img1, self.img2)
        self.assertAlmostEqual(ssim, 1.0, places=5)
    
    def test_calculate_ssim_different_images(self):
        """Test SSIM calculation with different images."""
        ssim = ImageQualityMetrics.calculate_ssim(self.img1, self.img3)
        self.assertIsInstance(ssim, float)
        self.assertGreaterEqual(ssim, -1)
        self.assertLessEqual(ssim, 1)


class TestSampleDataset(unittest.TestCase):
    """Test cases for sample dataset creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        num_samples = 5
        sample_files = create_sample_dataset(self.temp_dir, num_samples)
        
        self.assertEqual(len(sample_files), num_samples)
        
        # Check that files exist and are valid images
        for file_path in sample_files:
            self.assertTrue(file_path.exists())
            self.assertTrue(file_path.suffix.lower() in ['.jpg', '.jpeg'])
            
            # Verify it's a valid image
            img = Image.open(file_path)
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.mode, 'RGB')
    
    def test_create_sample_dataset_zero_samples(self):
        """Test sample dataset creation with zero samples."""
        sample_files = create_sample_dataset(self.temp_dir, 0)
        self.assertEqual(len(sample_files), 0)
    
    def test_create_sample_dataset_nonexistent_directory(self):
        """Test sample dataset creation in non-existent directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        sample_files = create_sample_dataset(nonexistent_dir, 3)
        
        self.assertEqual(len(sample_files), 3)
        self.assertTrue(nonexistent_dir.exists())


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.yaml"
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        manager = ConfigManager()
        config = manager._create_default_config()
        
        self.assertIsInstance(config, Config)
        self.assertEqual(config.model.type, "espcn")
        self.assertEqual(config.model.scale_factor, 3)
        self.assertEqual(config.data.input_dir, "data/input")
    
    def test_config_validation(self):
        """Test configuration validation."""
        manager = ConfigManager()
        
        # Test valid config
        valid_config = manager._create_default_config()
        self.assertTrue(manager.validate_config(valid_config))
        
        # Test invalid model type
        invalid_config = manager._create_default_config()
        invalid_config.model.type = "invalid"
        self.assertFalse(manager.validate_config(invalid_config))
        
        # Test invalid scale factor
        invalid_config = manager._create_default_config()
        invalid_config.model.scale_factor = 5
        self.assertFalse(manager.validate_config(invalid_config))
    
    def test_config_save_and_load(self):
        """Test configuration saving and loading."""
        manager = ConfigManager(self.config_path)
        
        # Create and save config
        config = manager._create_default_config()
        manager.save_config(config)
        
        # Load config
        loaded_config = manager.load_config()
        
        self.assertEqual(config.model.type, loaded_config.model.type)
        self.assertEqual(config.model.scale_factor, loaded_config.model.scale_factor)
        self.assertEqual(config.data.input_dir, loaded_config.data.input_dir)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.model = SuperResolutionModel(model_type="espcn", scale_factor=2, device="cpu")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create sample dataset
        sample_files = create_sample_dataset(self.temp_dir, 2)
        
        # Process images
        results = self.model.batch_upscale(sample_files)
        
        # Verify results
        self.assertEqual(len(results), 2)
        for original, upscaled in results:
            self.assertIsNotNone(original)
            self.assertIsNotNone(upscaled)
            
            # Check dimensions
            self.assertGreater(upscaled.size[0], original.size[0])
            self.assertGreater(upscaled.size[1], original.size[1])
            
            # Check scale factor
            scale_x = upscaled.size[0] / original.size[0]
            scale_y = upscaled.size[1] / original.size[1]
            self.assertAlmostEqual(scale_x, 2, delta=0.1)
            self.assertAlmostEqual(scale_y, 2, delta=0.1)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
