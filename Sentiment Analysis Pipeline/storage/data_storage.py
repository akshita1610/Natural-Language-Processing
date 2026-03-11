"""
Data Storage Module

Handles storage and retrieval of sentiment analysis results.
Supports CSV, SQLite, and PostgreSQL storage options.
"""

import pandas as pd
import sqlite3
import csv
import os
import logging
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import json

# Optional PostgreSQL support
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataStorage:
    """
    Data storage class for sentiment analysis results.
    
    Supports:
    - CSV file storage
    - SQLite database storage
    - PostgreSQL database storage (if available)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data storage with configuration.
        
        Args:
            config: Storage configuration dictionary
        """
        self.config = config
        self.storage_type = config.get('type', 'csv')
        
        # Initialize storage-specific settings
        if self.storage_type == 'csv':
            self.output_file = config.get('csv', {}).get('output_file', 'sentiment_results.csv')
        elif self.storage_type == 'sqlite':
            self.db_file = config.get('sqlite', {}).get('database_file', 'sentiment_analysis.db')
            self._initialize_sqlite()
        elif self.storage_type == 'postgresql':
            if not POSTGRESQL_AVAILABLE:
                raise ImportError("PostgreSQL support requires psycopg2. Install with: pip install psycopg2-binary")
            self.pg_config = config.get('postgresql', {})
            self._initialize_postgresql()
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
        
        logger.info(f"Initialized {self.storage_type} storage")
    
    def _initialize_sqlite(self):
        """Initialize SQLite database and create tables."""
        try:
            # Create database file directory if it doesn't exist
            db_dir = os.path.dirname(self.db_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # Connect to database
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Create main sentiment results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    processed_text TEXT,
                    sentiment TEXT NOT NULL,
                    confidence REAL,
                    compound REAL,
                    pos REAL,
                    neg REAL,
                    neu REAL,
                    polarity REAL,
                    subjectivity REAL,
                    source TEXT,
                    created_at TIMESTAMP,
                    analyzed_at TIMESTAMP,
                    metadata TEXT,
                    raw_data TEXT
                )
            ''')
            
            # Create trends table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time_period TEXT NOT NULL,
                    positive_count INTEGER,
                    negative_count INTEGER,
                    neutral_count INTEGER,
                    avg_confidence REAL,
                    avg_compound REAL,
                    total_items INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create keywords table for keyword-level sentiment
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS keyword_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    confidence REAL,
                    text_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (text_id) REFERENCES sentiment_results (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("SQLite database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
            raise
    
    def _initialize_postgresql(self):
        """Initialize PostgreSQL database connection and create tables."""
        try:
            # Connect to PostgreSQL
            conn = psycopg2.connect(
                host=self.pg_config.get('host', 'localhost'),
                port=self.pg_config.get('port', 5432),
                database=self.pg_config.get('database', 'sentiment_analysis'),
                user=self.pg_config.get('username', 'postgres'),
                password=self.pg_config.get('password', 'password')
            )
            cursor = conn.cursor()
            
            # Create main sentiment results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_results (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    processed_text TEXT,
                    sentiment TEXT NOT NULL,
                    confidence REAL,
                    compound REAL,
                    pos REAL,
                    neg REAL,
                    neu REAL,
                    polarity REAL,
                    subjectivity REAL,
                    source TEXT,
                    created_at TIMESTAMP,
                    analyzed_at TIMESTAMP,
                    metadata JSONB,
                    raw_data JSONB
                )
            ''')
            
            # Create trends table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_trends (
                    id SERIAL PRIMARY KEY,
                    time_period TEXT NOT NULL,
                    positive_count INTEGER,
                    negative_count INTEGER,
                    neutral_count INTEGER,
                    avg_confidence REAL,
                    avg_compound REAL,
                    total_items INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create keywords table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS keyword_sentiment (
                    id SERIAL PRIMARY KEY,
                    keyword TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    confidence REAL,
                    text_id INTEGER REFERENCES sentiment_results(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_results_sentiment ON sentiment_results(sentiment)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_results_created_at ON sentiment_results(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_keyword_sentiment_keyword ON keyword_sentiment(keyword)')
            
            conn.commit()
            conn.close()
            
            logger.info("PostgreSQL database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing PostgreSQL database: {e}")
            raise
    
    def store_results(self, data: Union[List[Dict], pd.DataFrame]) -> bool:
        """
        Store sentiment analysis results.
        
        Args:
            data: List of dictionaries or pandas DataFrame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Data must be a list of dictionaries or pandas DataFrame")
            
            if self.storage_type == 'csv':
                return self._store_csv(df)
            elif self.storage_type == 'sqlite':
                return self._store_sqlite(df)
            elif self.storage_type == 'postgresql':
                return self._store_postgresql(df)
            
        except Exception as e:
            logger.error(f"Error storing results: {e}")
            return False
    
    def _store_csv(self, df: pd.DataFrame) -> bool:
        """Store results to CSV file."""
        try:
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(self.output_file)
            
            # Append to existing file or create new one
            mode = 'a' if file_exists else 'w'
            header = not file_exists
            
            df.to_csv(self.output_file, mode=mode, header=header, index=False)
            
            logger.info(f"Stored {len(df)} records to CSV: {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing to CSV: {e}")
            return False
    
    def _store_sqlite(self, df: pd.DataFrame) -> bool:
        """Store results to SQLite database."""
        try:
            conn = sqlite3.connect(self.db_file)
            
            # Prepare data for insertion
            df_copy = df.copy()
            
            # Convert datetime columns to string if they exist
            datetime_columns = df_copy.select_dtypes(include=['datetime64[ns]']).columns
            for col in datetime_columns:
                df_copy[col] = df_copy[col].astype(str)
            
            # Handle complex objects by converting to JSON strings
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    df_copy[col] = df_copy[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x))
            
            # Insert data into main table
            df_copy.to_sql('sentiment_results', conn, if_exists='append', index=False)
            
            conn.close()
            
            logger.info(f"Stored {len(df)} records to SQLite database: {self.db_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing to SQLite: {e}")
            return False
    
    def _store_postgresql(self, df: pd.DataFrame) -> bool:
        """Store results to PostgreSQL database."""
        try:
            conn = psycopg2.connect(
                host=self.pg_config.get('host', 'localhost'),
                port=self.pg_config.get('port', 5432),
                database=self.pg_config.get('database', 'sentiment_analysis'),
                user=self.pg_config.get('username', 'postgres'),
                password=self.pg_config.get('password', 'password')
            )
            cursor = conn.cursor()
            
            # Prepare columns and data
            columns = df.columns.tolist()
            placeholders = ', '.join(['%s'] * len(columns))
            insert_query = f"INSERT INTO sentiment_results ({', '.join(columns)}) VALUES ({placeholders})"
            
            # Convert DataFrame to list of tuples
            data_tuples = [tuple(None if pd.isna(val) else val for val in row) for row in df.values]
            
            # Insert data
            cursor.executemany(insert_query, data_tuples)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored {len(df)} records to PostgreSQL database")
            return True
            
        except Exception as e:
            logger.error(f"Error storing to PostgreSQL: {e}")
            return False
    
    def retrieve_results(self, limit: Optional[int] = None, sentiment_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve stored sentiment analysis results.
        
        Args:
            limit: Maximum number of records to retrieve
            sentiment_filter: Filter by sentiment (positive, negative, neutral)
            
        Returns:
            Pandas DataFrame with results
        """
        try:
            if self.storage_type == 'csv':
                return self._retrieve_csv(limit, sentiment_filter)
            elif self.storage_type == 'sqlite':
                return self._retrieve_sqlite(limit, sentiment_filter)
            elif self.storage_type == 'postgresql':
                return self._retrieve_postgresql(limit, sentiment_filter)
            
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            return pd.DataFrame()
    
    def _retrieve_csv(self, limit: Optional[int], sentiment_filter: Optional[str]) -> pd.DataFrame:
        """Retrieve results from CSV file."""
        try:
            if not os.path.exists(self.output_file):
                return pd.DataFrame()
            
            df = pd.read_csv(self.output_file)
            
            # Apply filters
            if sentiment_filter and 'sentiment' in df.columns:
                df = df[df['sentiment'] == sentiment_filter]
            
            if limit:
                df = df.tail(limit)  # Get most recent records
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving from CSV: {e}")
            return pd.DataFrame()
    
    def _retrieve_sqlite(self, limit: Optional[int], sentiment_filter: Optional[str]) -> pd.DataFrame:
        """Retrieve results from SQLite database."""
        try:
            conn = sqlite3.connect(self.db_file)
            
            query = "SELECT * FROM sentiment_results"
            conditions = []
            params = []
            
            if sentiment_filter:
                conditions.append("sentiment = ?")
                params.append(sentiment_filter)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving from SQLite: {e}")
            return pd.DataFrame()
    
    def _retrieve_postgresql(self, limit: Optional[int], sentiment_filter: Optional[str]) -> pd.DataFrame:
        """Retrieve results from PostgreSQL database."""
        try:
            conn = psycopg2.connect(
                host=self.pg_config.get('host', 'localhost'),
                port=self.pg_config.get('port', 5432),
                database=self.pg_config.get('database', 'sentiment_analysis'),
                user=self.pg_config.get('username', 'postgres'),
                password=self.pg_config.get('password', 'password')
            )
            
            query = "SELECT * FROM sentiment_results"
            conditions = []
            params = []
            
            if sentiment_filter:
                conditions.append("sentiment = %s")
                params.append(sentiment_filter)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving from PostgreSQL: {e}")
            return pd.DataFrame()
    
    def store_trends(self, trends_data: List[Dict]) -> bool:
        """
        Store trend analysis results.
        
        Args:
            trends_data: List of trend dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df = pd.DataFrame(trends_data)
            
            if self.storage_type == 'sqlite':
                conn = sqlite3.connect(self.db_file)
                df.to_sql('sentiment_trends', conn, if_exists='append', index=False)
                conn.close()
            elif self.storage_type == 'postgresql':
                conn = psycopg2.connect(
                    host=self.pg_config.get('host', 'localhost'),
                    port=self.pg_config.get('port', 5432),
                    database=self.pg_config.get('database', 'sentiment_analysis'),
                    user=self.pg_config.get('username', 'postgres'),
                    password=self.pg_config.get('password', 'password')
                )
                cursor = conn.cursor()
                
                columns = df.columns.tolist()
                placeholders = ', '.join(['%s'] * len(columns))
                insert_query = f"INSERT INTO sentiment_trends ({', '.join(columns)}) VALUES ({placeholders})"
                
                data_tuples = [tuple(None if pd.isna(val) else val for val in row) for row in df.values]
                cursor.executemany(insert_query, data_tuples)
                
                conn.commit()
                conn.close()
            else:
                logger.warning(f"Trend storage not implemented for {self.storage_type}")
                return False
            
            logger.info(f"Stored {len(trends_data)} trend records")
            return True
            
        except Exception as e:
            logger.error(f"Error storing trends: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about stored data.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            if self.storage_type == 'csv':
                if not os.path.exists(self.output_file):
                    return {'total_records': 0, 'file_size': 0}
                
                df = pd.read_csv(self.output_file)
                file_size = os.path.getsize(self.output_file)
                
                stats = {
                    'total_records': len(df),
                    'file_size': file_size,
                    'file_path': self.output_file
                }
                
                if 'sentiment' in df.columns:
                    stats['sentiment_distribution'] = df['sentiment'].value_counts().to_dict()
                
                return stats
                
            elif self.storage_type == 'sqlite':
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                
                # Get total records
                cursor.execute("SELECT COUNT(*) FROM sentiment_results")
                total_records = cursor.fetchone()[0]
                
                # Get sentiment distribution
                cursor.execute("SELECT sentiment, COUNT(*) FROM sentiment_results GROUP BY sentiment")
                sentiment_dist = dict(cursor.fetchall())
                
                # Get database file size
                file_size = os.path.getsize(self.db_file)
                
                conn.close()
                
                return {
                    'total_records': total_records,
                    'file_size': file_size,
                    'database_path': self.db_file,
                    'sentiment_distribution': sentiment_dist
                }
                
            elif self.storage_type == 'postgresql':
                conn = psycopg2.connect(
                    host=self.pg_config.get('host', 'localhost'),
                    port=self.pg_config.get('port', 5432),
                    database=self.pg_config.get('database', 'sentiment_analysis'),
                    user=self.pg_config.get('username', 'postgres'),
                    password=self.pg_config.get('password', 'password')
                )
                cursor = conn.cursor()
                
                # Get total records
                cursor.execute("SELECT COUNT(*) FROM sentiment_results")
                total_records = cursor.fetchone()[0]
                
                # Get sentiment distribution
                cursor.execute("SELECT sentiment, COUNT(*) FROM sentiment_results GROUP BY sentiment")
                sentiment_dist = dict(cursor.fetchall())
                
                conn.close()
                
                return {
                    'total_records': total_records,
                    'database': self.pg_config.get('database'),
                    'sentiment_distribution': sentiment_dist
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    def clear_data(self, confirm: bool = False) -> bool:
        """
        Clear all stored data.
        
        Args:
            confirm: Confirmation flag to prevent accidental deletion
            
        Returns:
            True if successful, False otherwise
        """
        if not confirm:
            logger.warning("Data clearing not confirmed")
            return False
        
        try:
            if self.storage_type == 'csv':
                if os.path.exists(self.output_file):
                    os.remove(self.output_file)
                    logger.info(f"Deleted CSV file: {self.output_file}")
            
            elif self.storage_type == 'sqlite':
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM sentiment_results")
                cursor.execute("DELETE FROM sentiment_trends")
                cursor.execute("DELETE FROM keyword_sentiment")
                
                conn.commit()
                conn.close()
                logger.info("Cleared SQLite database")
            
            elif self.storage_type == 'postgresql':
                conn = psycopg2.connect(
                    host=self.pg_config.get('host', 'localhost'),
                    port=self.pg_config.get('port', 5432),
                    database=self.pg_config.get('database', 'sentiment_analysis'),
                    user=self.pg_config.get('username', 'postgres'),
                    password=self.pg_config.get('password', 'password')
                )
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM sentiment_results")
                cursor.execute("DELETE FROM sentiment_trends")
                cursor.execute("DELETE FROM keyword_sentiment")
                
                conn.commit()
                conn.close()
                logger.info("Cleared PostgreSQL database")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            return False
