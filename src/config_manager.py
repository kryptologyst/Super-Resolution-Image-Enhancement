"""
Configuration management for the super-resolution project.

Handles loading and validation of configuration files in YAML format.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    type: str = "espcn"
    scale_factor: int = 3
    device: str = "auto"


@dataclass
class DataConfig:
    """Data handling configuration parameters."""
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    sample_dir: str = "data/samples"
    supported_formats: list = field(default_factory=lambda: ["jpg", "jpeg", "png", "bmp", "tiff"])
    max_image_size: int = 2048


@dataclass
class ProcessingConfig:
    """Processing configuration parameters."""
    batch_size: int = 1
    save_intermediate: bool = False
    quality_metrics: bool = True
    preserve_metadata: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    level: str = "INFO"
    file: str = "logs/super_resolution.log"
    max_size: str = "10MB"
    backup_count: int = 5


@dataclass
class WebAppConfig:
    """Web application configuration parameters."""
    host: str = "0.0.0.0"
    port: int = 8501
    debug: bool = False
    max_file_size: str = "50MB"


@dataclass
class AdvancedConfig:
    """Advanced configuration parameters."""
    use_gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = False
    cache_models: bool = True


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    web_app: WebAppConfig = field(default_factory=WebAppConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        self.config_path = config_path or Path("config/config.yaml")
        self.config: Optional[Config] = None
    
    def load_config(self) -> Config:
        """
        Load configuration from YAML file.
        
        Returns:
            Loaded configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                return self._create_default_config()
            
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            self.config = self._parse_config(config_data)
            logger.info(f"Configuration loaded from {self.config_path}")
            return self.config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _parse_config(self, config_data: Dict[str, Any]) -> Config:
        """Parse configuration data into Config object."""
        try:
            # Parse model config
            model_data = config_data.get("model", {})
            model_config = ModelConfig(
                type=model_data.get("type", "espcn"),
                scale_factor=model_data.get("scale_factor", 3),
                device=model_data.get("device", "auto")
            )
            
            # Parse data config
            data_data = config_data.get("data", {})
            data_config = DataConfig(
                input_dir=data_data.get("input_dir", "data/input"),
                output_dir=data_data.get("output_dir", "data/output"),
                sample_dir=data_data.get("sample_dir", "data/samples"),
                supported_formats=data_data.get("supported_formats", ["jpg", "jpeg", "png", "bmp", "tiff"]),
                max_image_size=data_data.get("max_image_size", 2048)
            )
            
            # Parse processing config
            processing_data = config_data.get("processing", {})
            processing_config = ProcessingConfig(
                batch_size=processing_data.get("batch_size", 1),
                save_intermediate=processing_data.get("save_intermediate", False),
                quality_metrics=processing_data.get("quality_metrics", True),
                preserve_metadata=processing_data.get("preserve_metadata", True)
            )
            
            # Parse logging config
            logging_data = config_data.get("logging", {})
            logging_config = LoggingConfig(
                level=logging_data.get("level", "INFO"),
                file=logging_data.get("file", "logs/super_resolution.log"),
                max_size=logging_data.get("max_size", "10MB"),
                backup_count=logging_data.get("backup_count", 5)
            )
            
            # Parse web app config
            web_app_data = config_data.get("web_app", {})
            web_app_config = WebAppConfig(
                host=web_app_data.get("host", "0.0.0.0"),
                port=web_app_data.get("port", 8501),
                debug=web_app_data.get("debug", False),
                max_file_size=web_app_data.get("max_file_size", "50MB")
            )
            
            # Parse advanced config
            advanced_data = config_data.get("advanced", {})
            advanced_config = AdvancedConfig(
                use_gpu_memory_fraction=advanced_data.get("use_gpu_memory_fraction", 0.8),
                enable_mixed_precision=advanced_data.get("enable_mixed_precision", False),
                cache_models=advanced_data.get("cache_models", True)
            )
            
            return Config(
                model=model_config,
                data=data_config,
                processing=processing_config,
                logging=logging_config,
                web_app=web_app_config,
                advanced=advanced_config
            )
            
        except Exception as e:
            logger.error(f"Error parsing config data: {e}")
            raise
    
    def _create_default_config(self) -> Config:
        """Create default configuration."""
        logger.info("Creating default configuration")
        return Config()
    
    def validate_config(self, config: Config) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate model config
            if config.model.type not in ["espcn", "real_esrgan", "swinir"]:
                logger.error(f"Invalid model type: {config.model.type}")
                return False
            
            if config.model.scale_factor not in [2, 3, 4, 8]:
                logger.error(f"Invalid scale factor: {config.model.scale_factor}")
                return False
            
            if config.model.device not in ["auto", "cpu", "cuda"]:
                logger.error(f"Invalid device: {config.model.device}")
                return False
            
            # Validate data config
            if config.data.max_image_size <= 0:
                logger.error(f"Invalid max image size: {config.data.max_image_size}")
                return False
            
            # Validate processing config
            if config.processing.batch_size <= 0:
                logger.error(f"Invalid batch size: {config.processing.batch_size}")
                return False
            
            # Validate web app config
            if not (1 <= config.web_app.port <= 65535):
                logger.error(f"Invalid port: {config.web_app.port}")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return False
    
    def save_config(self, config: Config, output_path: Optional[Path] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration object to save
            output_path: Output path for config file
        """
        output_path = output_path or self.config_path
        
        try:
            config_dict = {
                "model": {
                    "type": config.model.type,
                    "scale_factor": config.model.scale_factor,
                    "device": config.model.device
                },
                "data": {
                    "input_dir": config.data.input_dir,
                    "output_dir": config.data.output_dir,
                    "sample_dir": config.data.sample_dir,
                    "supported_formats": config.data.supported_formats,
                    "max_image_size": config.data.max_image_size
                },
                "processing": {
                    "batch_size": config.processing.batch_size,
                    "save_intermediate": config.processing.save_intermediate,
                    "quality_metrics": config.processing.quality_metrics,
                    "preserve_metadata": config.processing.preserve_metadata
                },
                "logging": {
                    "level": config.logging.level,
                    "file": config.logging.file,
                    "max_size": config.logging.max_size,
                    "backup_count": config.logging.backup_count
                },
                "web_app": {
                    "host": config.web_app.host,
                    "port": config.web_app.port,
                    "debug": config.web_app.debug,
                    "max_file_size": config.web_app.max_file_size
                },
                "advanced": {
                    "use_gpu_memory_fraction": config.advanced.use_gpu_memory_fraction,
                    "enable_mixed_precision": config.advanced.enable_mixed_precision,
                    "cache_models": config.advanced.cache_models
                }
            }
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as file:
                yaml.dump(config_dict, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise


# Convenience function for easy config loading
def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration object
    """
    manager = ConfigManager(config_path)
    return manager.load_config()
