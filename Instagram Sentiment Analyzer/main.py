"""
Main entry point for Instagram Sentiment Analyzer
Command-line interface for running the complete pipeline
"""

import argparse
import sys
import json
import logging
from pathlib import Path
import pandas as pd
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from scraper import InstagramScraper
from preprocess import TextPreprocessor
from sentiment_model import SentimentClassifier
from visualize import SentimentVisualizer
from utils import ConfigManager, setup_logging, DataValidator

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Instagram Sentiment Analyzer - Complete pipeline for Instagram sentiment analysis"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.json",
        help="Path to configuration file"
    )
    
    # Data collection
    parser.add_argument(
        "--scrape-profile", "-p",
        type=str,
        help="Instagram profile name to scrape"
    )
    
    parser.add_argument(
        "--scrape-hashtag", "-h",
        type=str,
        help="Hashtag to scrape (without #)"
    )
    
    parser.add_argument(
        "--max-posts", "-m",
        type=int,
        default=100,
        help="Maximum number of posts to scrape"
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input data file (CSV or JSON)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", "-M",
        type=str,
        choices=["logistic_regression", "svm", "random_forest", "naive_bayes"],
        default="logistic_regression",
        help="Machine learning model to use"
    )
    
    parser.add_argument(
        "--embedding", "-e",
        type=str,
        choices=["tfidf", "count"],
        default="tfidf",
        help="Text embedding method"
    )
    
    # Actions
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train model, skip scraping"
    )
    
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only create visualizations from existing data"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    
    return parser

def load_or_scrape_data(args, config) -> pd.DataFrame:
    """Load existing data or scrape new data"""
    
    # If input file specified, load it
    if args.input:
        logging.info(f"Loading data from {args.input}")
        if args.input.endswith('.csv'):
            return pd.read_csv(args.input)
        else:
            with open(args.input, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
    
    # Otherwise, scrape data
    scraper = InstagramScraper(config)
    
    if args.scrape_profile:
        logging.info(f"Scraping profile: {args.scrape_profile}")
        posts_data = scraper.get_profile_posts(args.scrape_profile, args.max_posts)
        scraper.save_data(posts_data, f"profile_{args.scrape_profile}")
        return pd.DataFrame(posts_data)
    
    elif args.scrape_hashtag:
        logging.info(f"Scraping hashtag: {args.scrape_hashtag}")
        posts_data = scraper.get_hashtag_posts(args.scrape_hashtag, args.max_posts)
        scraper.save_data(posts_data, f"hashtag_{args.scrape_hashtag}")
        return pd.DataFrame(posts_data)
    
    else:
        raise ValueError("No data source specified. Use --input, --scrape-profile, or --scrape-hashtag")

def preprocess_data(df: pd.DataFrame, config) -> pd.DataFrame:
    """Preprocess the data"""
    logging.info("Preprocessing text data...")
    
    preprocessor = TextPreprocessor(config)
    processed_df = preprocessor.preprocess_dataframe(df, 'caption')
    
    return processed_df

def train_sentiment_model(df: pd.DataFrame, config, model_type: str, embedding_type: str) -> SentimentClassifier:
    """Train sentiment classification model"""
    logging.info(f"Training {model_type} model with {embedding_type} embeddings...")
    
    # Update config
    config["sentiment_model"]["model_type"] = model_type
    config["sentiment_model"]["embedding_type"] = embedding_type
    
    classifier = SentimentClassifier(config)
    
    # For demo, create sample labels
    # In production, you would need actual labeled data
    import numpy as np
    np.random.seed(42)
    texts = df['processed_text'].tolist()
    sample_labels = np.random.choice(
        ['positive', 'negative', 'neutral'], 
        size=len(texts), 
        p=[0.4, 0.3, 0.3]
    ).tolist()
    
    # Train model
    metrics = classifier.train_model(texts, sample_labels)
    logging.info(f"Model trained. Accuracy: {metrics['accuracy']:.4f}")
    
    # Save model
    classifier.save_model()
    
    return classifier

def create_visualizations(df: pd.DataFrame, config, output_dir: str):
    """Create and save visualizations"""
    logging.info("Creating visualizations...")
    
    visualizer = SentimentVisualizer(config)
    saved_files = visualizer.export_visualizations(df, output_dir)
    
    logging.info(f"Visualizations saved to {output_dir}")
    return saved_files

def run_dashboard():
    """Launch Streamlit dashboard"""
    import subprocess
    import os
    
    dashboard_path = Path(__file__).parent / "app" / "streamlit_app.py"
    
    if not dashboard_path.exists():
        logging.error(f"Dashboard file not found: {dashboard_path}")
        return
    
    logging.info("Launching Streamlit dashboard...")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Run streamlit
    subprocess.run([
        "streamlit", "run", 
        str(dashboard_path.relative_to(project_dir)),
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

def main():
    """Main function"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Load configuration
    config = ConfigManager.load_config(args.config)
    
    try:
        # Launch dashboard if requested
        if args.dashboard:
            run_dashboard()
            return
        
        # Visualize only mode
        if args.visualize_only:
            if not args.input:
                logging.error("Input file required for visualize-only mode")
                return
            
            df = pd.read_csv(args.input) if args.input.endswith('.csv') else pd.DataFrame(json.load(open(args.input)))
            
            # Add sentiment predictions if model exists
            classifier = SentimentClassifier(config)
            if classifier.load_model():
                texts = df['processed_text'].tolist()
                predictions = classifier.predict(texts)
                df['sentiment'] = [pred['predicted_label'] for pred in predictions]
            
            create_visualizations(df, config, args.output)
            return
        
        # Load or scrape data
        if not args.train_only:
            df = load_or_scrape_data(args, config)
            
            # Validate data
            validation_results = DataValidator.validate_instagram_data(df)
            if not validation_results['is_valid']:
                logging.error("Data validation failed:")
                for error in validation_results['errors']:
                    logging.error(f"  - {error}")
                return
            
            # Preprocess data
            processed_df = preprocess_data(df, config)
        else:
            # Train only mode - load processed data
            if not args.input:
                logging.error("Input file required for train-only mode")
                return
            
            processed_df = pd.read_csv(args.input) if args.input.endswith('.csv') else pd.DataFrame(json.load(open(args.input)))
        
        # Train model
        classifier = train_sentiment_model(processed_df, config, args.model, args.embedding)
        
        # Add predictions to data
        texts = processed_df['processed_text'].tolist()
        predictions = classifier.predict(texts)
        
        results_df = processed_df.copy()
        results_df['sentiment'] = [pred['predicted_label'] for pred in predictions]
        results_df['confidence'] = [pred.get('confidence', 0) for pred in predictions]
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_csv = output_path / "sentiment_analysis_results.csv"
        results_df.to_csv(results_csv, index=False)
        logging.info(f"Results saved to {results_csv}")
        
        results_json = output_path / "sentiment_analysis_results.json"
        results_df.to_json(results_json, orient='records', indent=2)
        logging.info(f"Results saved to {results_json}")
        
        # Create visualizations
        create_visualizations(results_df, config, str(output_path / "visualizations"))
        
        # Print summary
        logging.info("\n=== ANALYSIS COMPLETE ===")
        logging.info(f"Total posts analyzed: {len(results_df)}")
        logging.info(f"Sentiment distribution:")
        for sentiment, count in results_df['sentiment'].value_counts().items():
            logging.info(f"  {sentiment}: {count} ({count/len(results_df)*100:.1f}%)")
        logging.info(f"Results saved to: {output_path}")
        
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
