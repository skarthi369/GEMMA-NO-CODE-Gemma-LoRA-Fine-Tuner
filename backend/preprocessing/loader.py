"""
Dataset Loading and Preprocessing

Handles loading, validation, and preprocessing of various dataset formats
for fine-tuning Gemma models.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from backend.config import settings
from backend.models import DatasetFormat

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Load and preprocess datasets from various formats
    
    Supports:
    - CSV files
    - JSON/JSONL files
    - Plain text files
    """
    
    @staticmethod
    def load_csv(
        file_path: str,
        text_column: str = "text",
        label_column: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load dataset from CSV file
        
        Args:
            file_path: Path to CSV file
            text_column: Column name containing text
            label_column: Optional column name containing labels
            
        Returns:
            List of dictionaries with 'text' and optional 'label' keys
        """
        logger.info(f"Loading CSV dataset: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"  Loaded {len(df)} rows")
            
            # Validate columns exist
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV. Available: {list(df.columns)}")
            
            if label_column and label_column not in df.columns:
                logger.warning(f"Label column '{label_column}' not found, proceeding without labels")
                label_column = None
            
            # Drop rows with missing text
            original_len = len(df)
            df = df.dropna(subset=[text_column])
            dropped = original_len - len(df)
            if dropped > 0:
                logger.warning(f"  Dropped {dropped} rows with missing text")
            
            # Convert to list of dicts
            data = []
            for _, row in df.iterrows():
                item = {"text": str(row[text_column]).strip()}
                if label_column:
                    item["label"] = str(row[label_column]).strip()
                data.append(item)
            
            logger.info(f"  Processed {len(data)} samples")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {str(e)}")
            raise ValueError(f"CSV loading error: {str(e)}")
    
    @staticmethod
    def load_json(file_path: str, text_key: str = "text", label_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load dataset from JSON file
        
        Supports two formats:
        1. Array of objects: [{"text": "...", "label": "..."}, ...]
        2. Single object with array values: {"text": [...], "label": [...]}
        
        Args:
            file_path: Path to JSON file
            text_key: Key for text field
            label_key: Optional key for label field
            
        Returns:
            List of dictionaries with 'text' and optional 'label' keys
        """
        logger.info(f"Loading JSON dataset: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            data = []
            
            # Format 1: Array of objects
            if isinstance(json_data, list):
                for item in json_data:
                    if not isinstance(item, dict):
                        continue
                    
                    if text_key not in item:
                        logger.warning(f"Missing '{text_key}' key in item, skipping")
                        continue
                    
                    processed_item = {"text": str(item[text_key]).strip()}
                    if label_key and label_key in item:
                        processed_item["label"] = str(item[label_key]).strip()
                    
                    data.append(processed_item)
            
            # Format 2: Object with array values
            elif isinstance(json_data, dict):
                if text_key not in json_data:
                    raise ValueError(f"Text key '{text_key}' not found in JSON")
                
                texts = json_data[text_key]
                labels = json_data.get(label_key, []) if label_key else []
                
                if not isinstance(texts, list):
                    raise ValueError(f"'{text_key}' should be a list")
                
                for i, text in enumerate(texts):
                    item = {"text": str(text).strip()}
                    if labels and i < len(labels):
                        item["label"] = str(labels[i]).strip()
                    data.append(item)
            
            else:
                raise ValueError("JSON should be either an array or an object")
            
            logger.info(f"  Processed {len(data)} samples")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSON: {str(e)}")
            raise ValueError(f"JSON loading error: {str(e)}")
    
    @staticmethod
    def load_jsonl(file_path: str, text_key: str = "text", label_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load dataset from JSONL (JSON Lines) file
        
        Each line should be a valid JSON object:
        {"text": "...", "label": "..."}
        {"text": "...", "label": "..."}
        
        Args:
            file_path: Path to JSONL file
            text_key: Key for text field
            label_key: Optional key for label field
            
        Returns:
            List of dictionaries with 'text' and optional 'label' keys
        """
        logger.info(f"Loading JSONL dataset: {file_path}")
        
        try:
            data = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        
                        if text_key not in item:
                            logger.warning(f"Line {line_num}: Missing '{text_key}' key, skipping")
                            continue
                        
                        processed_item = {"text": str(item[text_key]).strip()}
                        if label_key and label_key in item:
                            processed_item["label"] = str(item[label_key]).strip()
                        
                        data.append(processed_item)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON, skipping - {e}")
                        continue
            
            logger.info(f"  Processed {len(data)} samples")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSONL: {str(e)}")
            raise ValueError(f"JSONL loading error: {str(e)}")
    
    @staticmethod
    def load_txt(file_path: str) -> List[Dict[str, Any]]:
        """
        Load dataset from plain text file
        
        Each line becomes a separate training sample.
        Empty lines are skipped.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of dictionaries with 'text' key
        """
        logger.info(f"Loading text dataset: {file_path}")
        
        try:
            data = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        data.append({"text": line})
            
            logger.info(f"  Processed {len(data)} samples")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load text file: {str(e)}")
            raise ValueError(f"Text file loading error: {str(e)}")
    
    @classmethod
    def load_dataset(
        cls,
        file_path: str,
        format: Optional[DatasetFormat] = None,
        text_column: str = "text",
        label_column: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load dataset from file, auto-detecting format if not specified
        
        Args:
            file_path: Path to dataset file
            format: Dataset format (auto-detected if None)
            text_column: Column/key name for text
            label_column: Optional column/key name for labels
            
        Returns:
            List of dictionaries with dataset samples
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Auto-detect format from extension if not provided
        if format is None:
            extension = path.suffix.lower()
            format_map = {
                '.csv': DatasetFormat.CSV,
                '.json': DatasetFormat.JSON,
                '.jsonl': DatasetFormat.JSONL,
                '.txt': DatasetFormat.TXT,
            }
            format = format_map.get(extension)
            if format is None:
                raise ValueError(f"Unsupported file extension: {extension}")
        
        # Load based on format
        if format == DatasetFormat.CSV:
            return cls.load_csv(str(path), text_column, label_column)
        elif format == DatasetFormat.JSON:
            return cls.load_json(str(path), text_column, label_column)
        elif format == DatasetFormat.JSONL:
            return cls.load_jsonl(str(path), text_column, label_column)
        elif format == DatasetFormat.TXT:
            return cls.load_txt(str(path))
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def split_dataset(
        data: List[Dict[str, Any]],
        test_size: float = 0.1,
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split dataset into train and validation sets
        
        Args:
            data: Dataset samples
            test_size: Fraction for validation set (0.0 to 1.0)
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data)
        """
        import random
        
        if shuffle:
            random.seed(seed)
            data = data.copy()
            random.shuffle(data)
        
        if test_size <= 0:
            return data, []
        
        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} validation")
        
        return train_data, val_data
    
    @staticmethod
    def save_dataset(data: List[Dict[str, Any]], output_path: str, format: DatasetFormat = DatasetFormat.JSONL):
        """
        Save dataset to file
        
        Args:
            data: Dataset samples
            output_path: Output file path
            format: Output format
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == DatasetFormat.CSV:
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
        
        elif format == DatasetFormat.JSON:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == DatasetFormat.JSONL:
            with open(path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        logger.info(f"Saved {len(data)} samples to {output_path}")
