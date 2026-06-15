
# etl/utils/commons.py
import os
import json
import yaml
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, List, Optional, Dict
import pandas as pd
import box

from utils.logger import logger
from utils.exception import Exception, ErrorCategory, ErrorSeverity

def read_yaml(path_to_yaml: Path) -> box.ConfigBox:
    """Reads YAML and returns ConfigBox for dot-notation access."""
    try:
        with open(path_to_yaml) as f:
            content = yaml.safe_load(f)
            if content is None:
                raise ValueError(f"YAML file is empty: {path_to_yaml}")
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return box.ConfigBox(content)
    except Exception as e:
        raise Exception(
            error=e,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH
        )

def create_directories(path_to_directories: List[Path], verbose: bool = True) -> None:
    """Creates multiple directories for ETL artifacts."""
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
    except Exception as e:
        raise Exception(
            error=e,
            category=ErrorCategory.IO_ERROR,
            severity=ErrorSeverity.MEDIUM
        )

def save_parquet(data: pd.DataFrame, path: Path) -> None:
    """Optimized persistence for massive ETL datasets."""
    try:
        os.makedirs(path.parent, exist_ok=True)
        data.to_parquet(path, index=False, compression='snappy')
        logger.info(f"Dataframe saved to Parquet: {path} [Rows: {len(data)}]")
    except Exception as e:
        raise Exception(
            error=e,
            category=ErrorCategory.IO_ERROR,
            severity=ErrorSeverity.HIGH,
            context={"path": str(path)}
        )

def load_parquet(path: Path) -> pd.DataFrame:
    """Loads Parquet files into Pandas."""
    try:
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded Parquet file: {path} [Rows: {len(df)}]")
        return df
    except Exception as e:
        raise Exception(
            error=e,
            category=ErrorCategory.IO_ERROR,
            severity=ErrorSeverity.HIGH
        )

def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Saves ETL metrics/stats to JSON."""
    try:
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON metrics saved: {path}")
    except Exception as e:
        raise Exception(
            error=e,
            category=ErrorCategory.IO_ERROR,
            severity=ErrorSeverity.MEDIUM
        )

def generate_data_hash(df: pd.DataFrame) -> str:
    """
    Generates a hash of the dataframe content.
    Crucial for Credit Risk auditability to ensure 2025 data hasn't
    changed between ETL and Prediction.
    """
    try:
        # We hash the string representation of a sample or the whole DF
        # For massive data, hashing the first/last 1000 rows is faster
        hash_val = hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()
        return hash_val
    except Exception as e:
        logger.warning(f"Could not generate data hash: {e}")
        return "hash_failed"

def get_size(path: Path) -> str:
    """Returns file size in human-readable format."""
    try:
        size_bytes = os.path.getsize(path)
        if size_bytes == 0: return "0B"
        size_name = ("B", "KB", "MB", "GB")
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"
    except Exception:
        return "Unknown Size"

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def validate_schema(df: pd.DataFrame, required_columns: List[str]):
    """
    Hard-stop validation for Credit Risk ETL.
    Ensures incoming data contains all features needed by the model.
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        from utils.exception import raise_schema_error
        raise_schema_error(
            message=f"Missing columns in source data: {missing_cols}",
            context={"received_columns": list(df.columns)}
        )
    logger.info("Schema validation passed.")