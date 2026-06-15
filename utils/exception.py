

import logging
import logging.config
import os
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, Any, Optional

# Safely check if JSON logger is available for cloud-native logging (ELK/Datadog)
try:
    from pythonjsonlogger import jsonlogger
    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False

# Constants
FORMATTER_JSON = 'json'
FORMATTER_DETAILED = 'detailed'
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 20
DEFAULT_LOGGER_NAME = 'CustomerProfitability'


class ContextFilter(logging.Filter):
    """
    Injects contextual DataOps data into logs.
    Used to track which batch/year (2025/2026) or job ID is being processed.
    """
    def __init__(self):
        super().__init__()
        # Captures run ID or job ID from environment
        self.job_id = os.getenv("ETL_JOB_ID", "manual-trigger")
        self.env = os.getenv("APP_ENV", "development")

    def filter(self, record):
        record.job_id = self.job_id
        record.env = self.env
        return True


class LoggerConfigurator:
    """
    Configures and manages ETL logging with environment-aware outputs.
    """
    _instance: Optional['LoggerConfigurator'] = None
    _configured: bool = False

    def __new__(cls, *args, **kwargs) -> 'LoggerConfigurator':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
            self,
            log_dir: Optional[str] = None,
            log_file_name: Optional[str] = None,
            max_bytes: int = DEFAULT_MAX_BYTES,
            backup_count: int = DEFAULT_BACKUP_COUNT,
            log_level: str = None,
    ):
        if self._configured:
            return

        self.log_dir = log_dir or os.getcwd()
        self.log_file_name = log_file_name
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_level = self._validate_log_level(log_level or os.getenv("LOG_LEVEL", "INFO"))

        # Enable file logging by default for Credit Risk Audit Trails
        self.enable_file_logging = os.getenv("LOG_TO_FILE", "true").lower() == "true"
        self.enable_console_logging = True

        self._configure_logging()
        self._configured = True

    def _validate_log_level(self, level: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        level_upper = level.upper()
        return level_upper if level_upper in valid_levels else 'INFO'

    def _get_formatters(self) -> Dict[str, Dict[str, Any]]:
        formatters = {
            FORMATTER_DETAILED: {
                'format': '[%(asctime)s][%(levelname)-7s] [%(env)s] %(name)s - %(module)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        }
        if HAS_JSON_LOGGER:
            formatters[FORMATTER_JSON] = {
                '()': jsonlogger.JsonFormatter,
                'format': '%(asctime)s %(name)s %(levelname)s %(module)s %(lineno)d %(message)s %(job_id)s %(env)s'
            }
        return formatters

    def _get_log_file_path(self) -> str:
        if self.log_file_name:
            log_file_name = self.log_file_name
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            log_file_name = f"credit_risk_etl_{timestamp}.log"

        log_path = Path(self.log_dir) / 'logs' / log_file_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return str(log_path)

    def _get_handlers(self) -> Dict[str, Dict[str, Any]]:
        handlers = {}
        if self.enable_file_logging:
            handlers['file'] = {
                'level': self.log_level,
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': self._get_log_file_path(),
                'maxBytes': self.max_bytes,
                'backupCount': self.backup_count,
                'encoding': 'utf8',
                'formatter': FORMATTER_DETAILED, # Use detailed for files to help debugging
                'filters': ['etl_context'],
                'delay': True
            }

        if self.enable_console_logging:
            is_prod = os.getenv("APP_ENV", "development").lower() == "production"
            formatter = FORMATTER_JSON if (is_prod and HAS_JSON_LOGGER) else FORMATTER_DETAILED
            handlers['console'] = {
                'level': self.log_level,
                'class': 'logging.StreamHandler',
                'formatter': formatter,
                'filters': ['etl_context'],
                'stream': sys.stdout
            }
        return handlers

    def _configure_logging(self) -> None:
        try:
            handlers = self._get_handlers()
            handler_names = list(handlers.keys())

            # Silence high-volume library logs
            noisy_loggers = ['urllib3', 'matplotlib', 'boto3', 'botocore', 's3transfer', 'pandas']
            loggers_config = {
                DEFAULT_LOGGER_NAME: {
                    'handlers': handler_names,
                    'level': self.log_level,
                    'propagate': False
                }
            }
            for logger_name in noisy_loggers:
                loggers_config[logger_name] = {'level': 'WARNING', 'propagate': True}

            logging_config = {
                'version': 1,
                'disable_existing_loggers': False,
                'filters': {'etl_context': {'()': ContextFilter}},
                'formatters': self._get_formatters(),
                'handlers': handlers,
                'root': {'handlers': handler_names, 'level': self.log_level},
                'loggers': loggers_config,
            }
            logging.config.dictConfig(logging_config)
        except Exception as e:
            logging.basicConfig(level=logging.INFO)
            logging.warning(f"Failed to configure custom logging: {e}")

    def get_logger(self, name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
        return logging.getLogger(name)

# Singleton Instance
logger_configurator = LoggerConfigurator()
logger = logger_configurator.get_logger()

def get_logger(module_name: str) -> logging.Logger:
    """Get a child logger for a specific ETL module (e.g. CustomerProfitabilityETL.customer_profile.yaml)."""
    return logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{module_name}")

def log_step(log_level: int = logging.INFO, logger_instance: Optional[logging.Logger] = None):
    """
    Decorator to log ETL stage entry, exit, and execution time.
    Critical for identifying bottlenecks in large data processing.
    """
    def decorator(func):
        nonlocal logger_instance
        active_logger = logger_instance or logger

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            active_logger.log(log_level, f">>> Starting ETL Step: {func_name}")
            try:
                start_time = datetime.now()
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                active_logger.log(log_level, f"<<< Finished ETL Step: {func_name} (Duration: {duration:.2f}s)")
                return result
            except Exception as e:
                active_logger.exception(f"CRITICAL FAILURE in ETL Step {func_name}: {str(e)}")
                raise
        return wrapper
    return decorator