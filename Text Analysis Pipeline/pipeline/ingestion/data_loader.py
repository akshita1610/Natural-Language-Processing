"""
Data Loader Module
Supports loading text from multiple sources: text files, CSV, and structured data
"""

import pandas as pd
import os
import glob
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and initial processing of text data from various sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataLoader with configuration
        
        Args:
            config: Configuration dictionary containing data source settings
        """
        self.config = config
        self.raw_data = []
        
    def load_all_sources(self) -> pd.DataFrame:
        """
        Load data from all configured sources
        
        Returns:
            DataFrame containing all loaded text data
        """
        all_data = []
        
        for source in self.config['data']['input_sources']:
            try:
                if source['type'] == 'text_files':
                    data = self._load_text_files(source['path'], source.get('extensions', ['.txt']))
                elif source['type'] == 'csv':
                    data = self._load_csv(source['path'], source.get('text_column', 'text'))
                else:
                    logger.warning(f"Unsupported source type: {source['type']}")
                    continue
                    
                if data is not None and not data.empty:
                    data['source_type'] = source['type']
                    all_data.append(data)
                    logger.info(f"Loaded {len(data)} records from {source['type']}")
                    
            except Exception as e:
                logger.error(f"Error loading from {source['type']}: {str(e)}")
                
        if not all_data:
            raise ValueError("No data could be loaded from any source")
            
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Apply preprocessing filters
        combined_data = self._apply_filters(combined_data)
        
        logger.info(f"Total loaded records: {len(combined_data)}")
        return combined_data
    
    def _load_text_files(self, directory: str, extensions: List[str]) -> Optional[pd.DataFrame]:
        """
        Load text from individual files
        
        Args:
            directory: Directory containing text files
            extensions: List of file extensions to include
            
        Returns:
            DataFrame with text data
        """
        texts = []
        file_paths = []
        
        for ext in extensions:
            pattern = os.path.join(directory, f"*{ext}")
            files = glob.glob(pattern)
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if content:  # Only add non-empty content
                        texts.append(content)
                        file_paths.append(file_path)
                        
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {str(e)}")
                    
        if not texts:
            logger.warning(f"No text files found in {directory}")
            return None
            
        return pd.DataFrame({
            'text': texts,
            'file_path': file_paths,
            'document_id': range(len(texts))
        })
    
    def _load_csv(self, file_path: str, text_column: str) -> Optional[pd.DataFrame]:
        """
        Load text data from CSV files
        
        Args:
            file_path: Path to CSV file or directory
            text_column: Column name containing text data
            
        Returns:
            DataFrame with text data
        """
        if os.path.isfile(file_path):
            csv_files = [file_path]
        else:
            csv_files = glob.glob(os.path.join(file_path, "*.csv"))
            
        all_data = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                if text_column not in df.columns:
                    logger.warning(f"Column '{text_column}' not found in {csv_file}")
                    continue
                    
                # Filter out rows with empty text
                df = df[df[text_column].notna() & (df[text_column].str.strip() != "")]
                
                if not df.empty:
                    df['source_file'] = csv_file
                    all_data.append(df)
                    logger.info(f"Loaded {len(df)} records from {csv_file}")
                    
            except Exception as e:
                logger.error(f"Error reading CSV {csv_file}: {str(e)}")
                
        if not all_data:
            return None
            
        combined = pd.concat(all_data, ignore_index=True)
        
        # Rename text column to 'text' if needed
        if text_column != 'text':
            combined = combined.rename(columns={text_column: 'text'})
            
        # Add document IDs
        combined['document_id'] = range(len(combined))
        
        return combined[['text', 'document_id'] + [col for col in combined.columns if col not in ['text', 'document_id']]]
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filtering rules from configuration
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        original_count = len(df)
        
        # Remove duplicates if configured
        if self.config['data']['preprocessing'].get('remove_duplicates', True):
            df = df.drop_duplicates(subset=['text'], keep='first')
            
        # Filter by text length
        min_length = self.config['data']['preprocessing'].get('min_text_length', 10)
        max_length = self.config['data']['preprocessing'].get('max_text_length', 10000)
        
        df = df[
            (df['text'].str.len() >= min_length) & 
            (df['text'].str.len() <= max_length)
        ]
        
        filtered_count = len(df)
        logger.info(f"Filtered {original_count - filtered_count} records. Remaining: {filtered_count}")
        
        return df.reset_index(drop=True)
    
    def save_raw_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save raw data to file
        
        Args:
            df: DataFrame to save
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            df.to_json(output_path, orient='records', indent=2)
        else:
            # Default to CSV
            df.to_csv(output_path.replace('.json', '.csv'), index=False)
            
        logger.info(f"Raw data saved to {output_path}")
