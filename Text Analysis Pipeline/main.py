"""
Main NLP Pipeline Orchestrator
Coordinates all components of the text analysis pipeline
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

# Add pipeline modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.ingestion import DataLoader
from pipeline.preprocessing import TextPreprocessor
from pipeline.features import FeatureExtractor
from pipeline.models import SentimentAnalyzer, TextClassifier
from pipeline.evaluation import ModelEvaluator
from pipeline.visualization import DataVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    Main orchestrator for the NLP Text Analysis Pipeline
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the NLP pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_loader = None
        self.preprocessor = None
        self.feature_extractor = None
        self.sentiment_analyzer = None
        self.text_classifier = None
        self.evaluator = None
        self.visualizer = None
        
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.results = {}
        
        # Initialize components
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        
        self.data_loader = DataLoader(self.config)
        self.preprocessor = TextPreprocessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.text_classifier = TextClassifier(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.visualizer = DataVisualizer(self.config)
        
        logger.info("All components initialized successfully")
    
    def run_full_pipeline(self, output_dir: str = "output") -> Dict[str, Any]:
        """
        Run the complete NLP pipeline
        
        Args:
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing all pipeline results
        """
        logger.info("Starting full NLP pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Data Ingestion
            logger.info("Step 1: Data Ingestion")
            self.raw_data = self.data_loader.load_all_sources()
            
            # Save raw data
            raw_output_path = os.path.join(output_dir, "raw_data.csv")
            self.data_loader.save_raw_data(self.raw_data, raw_output_path)
            
            # Step 2: Preprocessing
            logger.info("Step 2: Text Preprocessing")
            self.processed_data = self.preprocessor.preprocess_dataframe(self.raw_data)
            
            # Save processed data
            processed_output_path = os.path.join(output_dir, "processed_data.csv")
            self.processed_data.to_csv(processed_output_path, index=False)
            
            # Step 3: Feature Extraction
            logger.info("Step 3: Feature Extraction")
            texts = self.processed_data['processed_text'].tolist()
            feature_types = ['tfidf', 'word2vec']
            
            self.features = self.feature_extractor.create_feature_matrix(texts, feature_types)
            
            # Save features
            feature_output_dir = os.path.join(output_dir, "features")
            self.feature_extractor.save_features(self.features, feature_output_dir)
            
            # Step 4: Sentiment Analysis
            logger.info("Step 4: Sentiment Analysis")
            original_texts = self.processed_data['text'].tolist()
            sentiment_results = self.sentiment_analyzer.analyze_sentiment_ensemble(original_texts)
            
            # Save sentiment results
            sentiment_output_path = os.path.join(output_dir, "sentiment_results.csv")
            sentiment_results.to_csv(sentiment_output_path, index=False)
            
            self.results['sentiment'] = sentiment_results
            
            # Step 5: Text Classification (if labels available)
            if 'label' in self.processed_data.columns:
                logger.info("Step 5: Text Classification")
                labels = self.processed_data['label'].tolist()
                
                # Prepare data
                X_train, X_test, y_train, y_test = self.text_classifier.prepare_data(
                    texts, labels
                )
                
                # Extract features for classification
                train_features = self.feature_extractor.extract_tfidf_features(
                    [texts[i] for i in X_train], fit=True
                )
                test_features = self.feature_extractor.extract_tfidf_features(
                    [texts[i] for i in X_test], fit=False
                )
                
                # Train models
                trained_models = self.text_classifier.train_all_models(train_features, y_train)
                
                # Evaluate models
                classification_results = self.text_classifier.evaluate_all_models(
                    test_features, y_test
                )
                
                # Save models
                model_output_dir = os.path.join(output_dir, "models")
                self.text_classifier.save_models(model_output_dir)
                
                self.results['classification'] = classification_results
            
            # Step 6: Evaluation
            logger.info("Step 6: Evaluation")
            
            # Evaluate sentiment analysis
            sentiment_eval = self.evaluator.evaluate_sentiment_analysis(sentiment_results)
            
            # Compare classification models if available
            if 'classification' in self.results:
                model_comparison = self.evaluator.compare_models(self.results['classification'])
                self.results['model_comparison'] = model_comparison
            
            self.results['sentiment_evaluation'] = sentiment_eval
            
            # Step 7: Visualization
            logger.info("Step 7: Visualization")
            self._create_visualizations()
            
            # Step 8: Generate Report
            logger.info("Step 8: Generating Report")
            report_path = os.path.join(output_dir, "reports", "pipeline_report.md")
            evaluation_report = self.evaluator.generate_evaluation_report(
                self.results, report_path
            )
            
            # Save complete results
            results_path = os.path.join(output_dir, "pipeline_results.json")
            self.evaluator.save_evaluation_results(self.results, results_path)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"Pipeline completed successfully in {duration}")
            
            # Add pipeline metadata
            self.results['pipeline_metadata'] = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': str(duration),
                'total_documents': len(self.raw_data),
                'processed_documents': len(self.processed_data)
            }
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _create_visualizations(self):
        """Create all visualizations"""
        logger.info("Creating visualizations...")
        
        texts = self.processed_data['processed_text'].tolist()
        original_texts = self.processed_data['text'].tolist()
        
        # Word frequency plot
        self.visualizer.plot_word_frequency(texts)
        
        # Word cloud
        self.visualizer.create_wordcloud(texts)
        
        # Sentiment distribution
        if 'sentiment' in self.results:
            sentiment_df = self.results['sentiment']
            self.visualizer.plot_sentiment_distribution(sentiment_df)
            self.visualizer.plot_sentiment_scores(sentiment_df)
        
        # Text length distribution
        self.visualizer.plot_text_length_distribution(original_texts)
        
        # Model performance comparison
        if 'classification' in self.results:
            self.visualizer.plot_model_performance(self.results['classification'])
        
        # Interactive plots
        if 'sentiment' in self.results:
            self.visualizer.create_interactive_sentiment_plot(self.results['sentiment'])
        
        logger.info("Visualizations completed")
    
    def run_sentiment_analysis_only(self, texts: List[str]) -> pd.DataFrame:
        """
        Run only sentiment analysis on provided texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with sentiment results
        """
        logger.info("Running sentiment analysis only...")
        
        # Preprocess texts
        preprocessed_texts = []
        for text in texts:
            processed = self.preprocessor._preprocess_single_text(text)
            preprocessed_texts.append(processed)
        
        # Analyze sentiment
        results = self.sentiment_analyzer.analyze_sentiment_ensemble(texts)
        
        return results
    
    def run_text_classification(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Run text classification on provided data
        
        Args:
            texts: List of texts
            labels: List of corresponding labels
            
        Returns:
            Classification results
        """
        logger.info("Running text classification...")
        
        # Preprocess texts
        preprocessed_texts = []
        for text in texts:
            processed = self.preprocessor._preprocess_single_text(text)
            preprocessed_texts.append(processed)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.text_classifier.prepare_data(
            preprocessed_texts, labels
        )
        
        # Extract features
        train_features = self.feature_extractor.extract_tfidf_features(
            [preprocessed_texts[i] for i in X_train], fit=True
        )
        test_features = self.feature_extractor.extract_tfidf_features(
            [preprocessed_texts[i] for i in X_test], fit=False
        )
        
        # Train and evaluate models
        trained_models = self.text_classifier.train_all_models(train_features, y_train)
        results = self.text_classifier.evaluate_all_models(test_features, y_test)
        
        return results


def main():
    """Main function to run the pipeline"""
    try:
        # Initialize pipeline
        pipeline = NLPPipeline()
        
        # Run full pipeline
        results = pipeline.run_full_pipeline()
        
        print("\n" + "="*50)
        print("NLP PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        # Print summary
        if 'pipeline_metadata' in results:
            metadata = results['pipeline_metadata']
            print(f"Total documents processed: {metadata['total_documents']}")
            print(f"Processing time: {metadata['duration']}")
        
        if 'sentiment_evaluation' in results:
            print("\nSentiment Analysis Summary:")
            sentiment_eval = results['sentiment_evaluation']
            if 'summary_statistics' in sentiment_eval:
                for method, stats in sentiment_eval['summary_statistics'].items():
                    if 'percentages' in stats:
                        print(f"  {method}:")
                        for sentiment, percentage in stats['percentages'].items():
                            print(f"    {sentiment}: {percentage}%")
        
        if 'model_comparison' in results:
            print("\nBest Classification Model:")
            comparison = results['model_comparison']
            if 'best_models' in comparison and 'best_overall' in comparison['best_models']:
                best = comparison['best_models']['best_overall']
                print(f"  Model: {best['model']}")
                print(f"  Score: {best['score']:.4f}")
        
        print(f"\nResults saved to 'output/' directory")
        print(f"Check 'output/reports/pipeline_report.md' for detailed analysis")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
