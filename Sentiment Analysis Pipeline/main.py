"""
Sentiment Analysis Pipeline - Main Entry Point

This is the main script that orchestrates the entire sentiment analysis pipeline.
It handles data ingestion, preprocessing, sentiment analysis, storage, trend tracking,
and visualization based on the configuration.

Usage:
    python main.py [--config config.yaml] [--mode batch|stream] [--source twitter|reddit|audio]

Author: Sentiment Analysis Pipeline
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline modules
from ingestion import TwitterIngestion, RedditIngestion, YouTubeIngestion
try:
    from ingestion import AudioIngestion
    AUDIO_AVAILABLE = True
except ImportError:
    AudioIngestion = None
    AUDIO_AVAILABLE = False
from preprocessing import TextPreprocessor
from sentiment import SentimentAnalyzer
from storage import DataStorage
from trends import TrendAnalyzer
from visualization import SentimentVisualizer


class SentimentPipeline:
    """
    Main sentiment analysis pipeline class.
    
    Orchestrates the entire workflow from data ingestion to visualization.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize pipeline components
        self.ingestion = None
        self.preprocessor = None
        self.sentiment_analyzer = None
        self.storage = None
        self.trend_analyzer = None
        self.visualizer = None
        
        self._initialize_components()
        
        logger.info("Sentiment Analysis Pipeline initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'pipeline.log')
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        global logger
        logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Initialize data ingestion
            data_source_config = self.config.get('data_source', {})
            source_type = data_source_config.get('type', 'twitter')
            
            if source_type == 'twitter':
                self.ingestion = TwitterIngestion(data_source_config.get('twitter', {}))
            elif source_type == 'reddit':
                self.ingestion = RedditIngestion(data_source_config.get('reddit', {}))
            elif source_type == 'audio':
                self.ingestion = AudioIngestion(data_source_config.get('audio', {}))
            elif source_type == 'youtube':
                self.ingestion = YouTubeIngestion(data_source_config.get('youtube', {}))
            else:
                raise ValueError(f"Unsupported data source: {source_type}")
            
            # Initialize preprocessor
            self.preprocessor = TextPreprocessor(self.config.get('preprocessing', {}))
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentAnalyzer(self.config.get('sentiment', {}))
            
            # Initialize storage
            self.storage = DataStorage(self.config.get('storage', {}))
            
            # Initialize trend analyzer
            self.trend_analyzer = TrendAnalyzer(self.config.get('trends', {}))
            
            # Initialize visualizer
            self.visualizer = SentimentVisualizer(self.config.get('visualization', {}))
            
            logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def run_batch_mode(self) -> bool:
        """
        Run the pipeline in batch mode.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting batch mode execution")
            
            # Step 1: Data Ingestion
            logger.info("Step 1: Data Ingestion")
            raw_data = self._ingest_data()
            
            if not raw_data:
                logger.error("No data collected during ingestion")
                return False
            
            # Step 2: Preprocessing
            logger.info("Step 2: Text Preprocessing")
            processed_data = self._preprocess_data(raw_data)
            
            # Step 3: Sentiment Analysis
            logger.info("Step 3: Sentiment Analysis")
            sentiment_data = self._analyze_sentiment(processed_data)
            
            # Step 4: Storage
            logger.info("Step 4: Data Storage")
            self._store_data(sentiment_data)
            
            # Step 5: Trend Analysis
            logger.info("Step 5: Trend Analysis")
            trends_data = self._analyze_trends(sentiment_data)
            
            # Step 6: Visualization
            logger.info("Step 6: Visualization")
            self._create_visualizations(sentiment_data, trends_data)
            
            # Step 7: Generate Report
            logger.info("Step 7: Report Generation")
            self._generate_report(sentiment_data, trends_data)
            
            logger.info("Batch mode execution completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in batch mode execution: {e}")
            return False
    
    def run_stream_mode(self) -> bool:
        """
        Run the pipeline in streaming mode.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting streaming mode execution")
            
            # For streaming, we'll process data in real-time
            processed_count = 0
            
            # Get data stream based on source type
            data_stream = self._get_data_stream()
            
            for data_item in data_stream:
                try:
                    # Process individual item
                    processed_item = self._process_single_item(data_item)
                    
                    if processed_item:
                        # Store immediately
                        self._store_single_item(processed_item)
                        processed_count += 1
                        
                        # Log progress
                        if processed_count % 10 == 0:
                            logger.info(f"Processed {processed_count} items in streaming mode")
                
                except Exception as e:
                    logger.warning(f"Error processing item: {e}")
                    continue
            
            logger.info(f"Streaming mode completed. Processed {processed_count} items")
            return True
            
        except KeyboardInterrupt:
            logger.info("Streaming mode interrupted by user")
            return True
        except Exception as e:
            logger.error(f"Error in streaming mode execution: {e}")
            return False
    
    def _ingest_data(self) -> List[Dict]:
        """Ingest data based on the configured source."""
        try:
            source_type = self.config.get('data_source', {}).get('type', 'twitter')
            
            if source_type == 'twitter':
                return self.ingestion.collect_batch_tweets()
            elif source_type == 'reddit':
                return self.ingestion.collect_combined()
            elif source_type == 'audio':
                return self.ingestion.transcribe_batch()
            elif source_type == 'youtube':
                return self.ingestion.collect_from_search(
                    self.config.get('data_source', {}).get('youtube', {}).get('search_query', 'sentiment analysis'),
                    self.config.get('data_source', {}).get('youtube', {}).get('max_videos', 10),
                    self.config.get('data_source', {}).get('youtube', {}).get('max_comments_per_video', 50)
                )
            else:
                raise ValueError(f"Unknown data source: {source_type}")
                
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise
    
    def _get_data_stream(self):
        """Get data stream for streaming mode."""
        try:
            source_type = self.config.get('data_source', {}).get('type', 'twitter')
            
            if source_type == 'twitter':
                return self.ingestion.stream_tweets()
            elif source_type == 'reddit':
                return self.ingestion.monitor_new_posts()
            else:
                raise ValueError(f"Streaming not supported for source: {source_type}")
                
        except Exception as e:
            logger.error(f"Error getting data stream: {e}")
            raise
    
    def _preprocess_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Preprocess the raw data."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            
            # Determine text column based on data source
            text_column = self._get_text_column(df)
            
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in data")
            
            # Apply preprocessing
            processed_df = self.preprocessor.preprocess_dataframe(df, text_column)
            
            logger.info(f"Preprocessed {len(processed_df)} records")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise
    
    def _analyze_sentiment(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment in the processed data."""
        try:
            text_column = self._get_text_column(processed_data)
            processed_text_column = f"{text_column}_processed"
            
            if processed_text_column not in processed_data.columns:
                raise ValueError(f"Processed text column '{processed_text_column}' not found")
            
            # Apply sentiment analysis
            sentiment_df = self.sentiment_analyzer.analyze_dataframe(
                processed_data, processed_text_column
            )
            
            logger.info(f"Analyzed sentiment for {len(sentiment_df)} records")
            return sentiment_df
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            raise
    
    def _store_data(self, sentiment_data: pd.DataFrame):
        """Store sentiment analysis results."""
        try:
            success = self.storage.store_results(sentiment_data)
            
            if success:
                logger.info(f"Stored {len(sentiment_data)} sentiment analysis results")
            else:
                logger.error("Failed to store sentiment analysis results")
                
        except Exception as e:
            logger.error(f"Error during data storage: {e}")
            raise
    
    def _analyze_trends(self, sentiment_data: pd.DataFrame) -> Dict:
        """Analyze trends in the sentiment data."""
        try:
            trends = {}
            
            # Rolling sentiment
            trends['rolling'] = self.trend_analyzer.compute_rolling_sentiment(sentiment_data)
            
            # Time-based sentiment
            trends['time_based'] = self.trend_analyzer.analyze_sentiment_over_time(sentiment_data)
            
            # Volume trends
            trends['volume'] = self.trend_analyzer.analyze_volume_trends(sentiment_data)
            
            # Keyword sentiment (if keywords are configured)
            if self.trend_analyzer.keywords:
                trends['keywords'] = self.trend_analyzer.analyze_keyword_sentiment(sentiment_data)
            
            # Sentiment shifts
            trends['shifts'] = self.trend_analyzer.detect_sentiment_shifts(sentiment_data)
            
            # Summary
            trends['summary'] = self.trend_analyzer.get_trend_summary(sentiment_data)
            
            logger.info("Trend analysis completed")
            return trends
            
        except Exception as e:
            logger.error(f"Error during trend analysis: {e}")
            return {}
    
    def _create_visualizations(self, sentiment_data: pd.DataFrame, trends_data: Dict):
        """Create visualizations."""
        try:
            # Sentiment over time
            self.visualizer.plot_sentiment_over_time(sentiment_data)
            
            # Sentiment distribution
            self.visualizer.plot_sentiment_distribution(sentiment_data)
            
            # Volume analysis
            self.visualizer.plot_volume_analysis(sentiment_data)
            
            # Keyword sentiment (if available)
            if 'keywords' in trends_data and not trends_data['keywords'].empty:
                self.visualizer.plot_keyword_sentiment(trends_data['keywords'])
            
            # Sentiment heatmap
            self.visualizer.plot_sentiment_heatmap(sentiment_data)
            
            # Dashboard
            keyword_df = trends_data.get('keywords')
            time_df = trends_data.get('time_based')
            self.visualizer.create_dashboard(sentiment_data, time_df, keyword_df)
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _generate_report(self, sentiment_data: pd.DataFrame, trends_data: Dict):
        """Generate comprehensive report."""
        try:
            keyword_df = trends_data.get('keywords')
            time_df = trends_data.get('time_based')
            
            report_path = self.visualizer.generate_report(
                sentiment_data, time_df, keyword_df
            )
            
            if report_path:
                logger.info(f"Report generated: {report_path}")
            else:
                logger.warning("Report generation failed")
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def _process_single_item(self, data_item: Dict) -> Optional[Dict]:
        """Process a single data item in streaming mode."""
        try:
            # Convert to DataFrame for processing
            df = pd.DataFrame([data_item])
            
            # Get text column
            text_column = self._get_text_column(df)
            
            if text_column not in df.columns:
                return None
            
            # Preprocess
            processed_df = self.preprocessor.preprocess_dataframe(df, text_column)
            
            # Analyze sentiment
            sentiment_df = self.sentiment_analyzer.analyze_dataframe(
                processed_df, f"{text_column}_processed"
            )
            
            # Convert back to dictionary
            if not sentiment_df.empty:
                return sentiment_df.iloc[0].to_dict()
            
            return None
            
        except Exception as e:
            logger.warning(f"Error processing single item: {e}")
            return None
    
    def _store_single_item(self, processed_item: Dict):
        """Store a single processed item."""
        try:
            self.storage.store_results([processed_item])
        except Exception as e:
            logger.warning(f"Error storing single item: {e}")
    
    def _get_text_column(self, df: pd.DataFrame) -> str:
        """Determine the appropriate text column based on data source."""
        source_type = self.config.get('data_source', {}).get('type', 'twitter')
        
        if source_type == 'twitter':
            return 'text'
        elif source_type == 'reddit':
            return 'combined_text' if 'combined_text' in df.columns else 'text'
        elif source_type == 'audio':
            return 'transcription'
        elif source_type == 'youtube':
            return 'text' if 'text' in df.columns else 'description'
        else:
            # Try common column names
            for col in ['text', 'content', 'message', 'transcription']:
                if col in df.columns:
                    return col
            
            raise ValueError("Could not determine text column")
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status and statistics."""
        try:
            status = {
                'pipeline_initialized': True,
                'data_source': self.config.get('data_source', {}).get('type'),
                'sentiment_model': self.config.get('sentiment', {}).get('model'),
                'storage_type': self.config.get('storage', {}).get('type'),
                'storage_stats': self.storage.get_statistics() if self.storage else {}
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {'error': str(e)}


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Sentiment Analysis Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', choices=['batch', 'stream'], default='batch', help='Execution mode')
    parser.add_argument('--source', choices=['twitter', 'reddit', 'audio', 'youtube'], help='Data source override')
    parser.add_argument('--status', action='store_true', help='Show pipeline status')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = SentimentPipeline(args.config)
        
        # Override data source if specified
        if args.source:
            pipeline.config['data_source']['type'] = args.source
            logger.info(f"Data source overridden to: {args.source}")
        
        # Show status if requested
        if args.status:
            status = pipeline.get_pipeline_status()
            print("Pipeline Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
            return
        
        # Run pipeline
        if args.mode == 'batch':
            success = pipeline.run_batch_mode()
        else:  # stream
            success = pipeline.run_stream_mode()
        
        if success:
            print("✅ Pipeline execution completed successfully!")
            print(f"📊 Check the '{pipeline.visualizer.output_dir}' directory for visualizations")
            print(f"📄 Check the '{pipeline.config.get('logging', {}).get('file', 'pipeline.log')}' file for detailed logs")
        else:
            print("❌ Pipeline execution failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Pipeline execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
