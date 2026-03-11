"""
Simple Demo Script for NLP Text Analysis Pipeline
Demonstrates core functionality without problematic dependencies
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add pipeline modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.ingestion import DataLoader
from pipeline.preprocessing import TextPreprocessor
from pipeline.models import SentimentAnalyzer
import yaml

def main():
    """Run simplified demonstration of NLP pipeline"""
    print("Simple NLP Pipeline Demo")
    print("=" * 40)
    
    try:
        # Load configuration
        print("Loading configuration...")
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        data_loader = DataLoader(config)
        preprocessor = TextPreprocessor(config)
        sentiment_analyzer = SentimentAnalyzer(config)
        
        # Load sample data
        print("Loading sample data...")
        sample_csv = "example_data/sample_dataset.csv"
        sample_txt = "example_data/sample_texts.txt"
        
        if not os.path.exists(sample_csv):
            print(f"❌ Sample CSV not found: {sample_csv}")
            return
        
        # Load CSV data
        raw_data = data_loader.load_all_sources()
        print(f"Loaded {len(raw_data)} documents")
        
        # Preprocess data
        print(f"Preprocessing text...")
        processed_data = preprocessor.preprocess_dataframe(raw_data)
        print(f"Processed {len(processed_data)} documents")
        
        # Show sample preprocessing
        print("Preprocessing Examples:")
        for i in range(min(3, len(processed_data))):
            original = processed_data.iloc[i]['text'][:100] + "..." if len(processed_data.iloc[i]['text']) > 100 else processed_data.iloc[i]['text']
            processed = processed_data.iloc[i]['processed_text'][:100] + "..." if len(processed_data.iloc[i]['processed_text']) > 100 else processed_data.iloc[i]['processed_text']
            print(f"  Original: '{original}'")
            print(f"  Processed: '{processed}'")
            print()
        
        # Sentiment Analysis
        print("Running Sentiment Analysis...")
        original_texts = processed_data['text'].tolist()
        sentiment_results = sentiment_analyzer.analyze_sentiment_ensemble(original_texts)
        
        # Show sentiment results
        print("Sentiment Analysis Results:")
        sentiment_counts = sentiment_results['vader_sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(sentiment_results)) * 100
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Show sample predictions
        print("Sample Sentiment Predictions:")
        sample_results = sentiment_results[['text', 'vader_sentiment', 'textblob_sentiment', 'ensemble_sentiment']].head(5)
        for idx, row in sample_results.iterrows():
            text_preview = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
            print(f"  '{text_preview}'")
            print(f"    VADER: {row['vader_sentiment']}, TextBlob: {row['textblob_sentiment']}, Ensemble: {row['ensemble_sentiment']}")
            print()
        
        # Save results
        print("Saving results...")
        os.makedirs("results", exist_ok=True)
        
        # Save processed data
        processed_data.to_csv("results/processed_data.csv", index=False)
        print("Saved processed data to results/processed_data.csv")
        
        # Save sentiment results
        sentiment_results.to_csv("results/sentiment_results.csv", index=False)
        print("Saved sentiment results to results/sentiment_results.csv")
        
        # Generate simple report
        report_content = f"""# NLP Pipeline Demo Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Processing Summary
- Total documents processed: {len(processed_data)}
- Original text length (avg): {processed_data['original_length'].mean():.1f} characters
- Processed text length (avg): {processed_data['processed_length'].mean():.1f} characters
- Average token count: {processed_data['token_count'].mean():.1f} words

## Sentiment Analysis Results
"""
        
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(sentiment_results)) * 100
            report_content += f"- {sentiment.capitalize()}: {count} documents ({percentage:.1f}%)\n"
        
        report_content += f"""
## Sample Predictions
"""
        
        for idx, row in sample_results.head(3).iterrows():
            text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
            report_content += f"- '{text_preview}'\n"
            report_content += f"  - VADER: {row['vader_sentiment']}\n"
            report_content += f"  - TextBlob: {row['textblob_sentiment']}\n"
            report_content += f"  - Ensemble: {row['ensemble_sentiment']}\n\n"
        
        with open("results/demo_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        print("Saved report to results/demo_report.md")
        
        print(f"Simple demo completed successfully!")
        print(f"Check the 'results/' directory for all output files")
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
