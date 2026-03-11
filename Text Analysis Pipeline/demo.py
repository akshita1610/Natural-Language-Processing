"""
Demo Script for NLP Text Analysis Pipeline
Demonstrates pipeline usage with sample data
"""

import os
import sys
import pandas as pd
from main import NLPPipeline

def main():
    """Run demonstration of the NLP pipeline"""
    print("🚀 NLP Text Analysis Pipeline Demo")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        print("\n📋 Initializing pipeline...")
        pipeline = NLPPipeline()
        
        # Demo 1: Run full pipeline with sample data
        print("\n🔄 Running full pipeline with sample data...")
        
        # Check if sample data exists
        sample_csv = "example_data/sample_dataset.csv"
        sample_txt = "example_data/sample_texts.txt"
        
        if not os.path.exists(sample_csv):
            print(f"❌ Sample CSV not found: {sample_csv}")
            return
        
        if not os.path.exists(sample_txt):
            print(f"❌ Sample text file not found: {sample_txt}")
            return
        
        print(f"✅ Found sample data:")
        print(f"   - CSV: {sample_csv}")
        print(f"   - Text: {sample_txt}")
        
        # Run pipeline
        results = pipeline.run_full_pipeline()
        
        # Display results
        print("\n📊 Pipeline Results Summary:")
        print("-" * 30)
        
        # Metadata
        if 'pipeline_metadata' in results:
            metadata = results['pipeline_metadata']
            print(f"📈 Processing Statistics:")
            print(f"   Total documents: {metadata['total_documents']}")
            print(f"   Processed documents: {metadata['processed_documents']}")
            print(f"   Processing time: {metadata['duration']}")
        
        # Sentiment results
        if 'sentiment' in results:
            sentiment_df = results['sentiment']
            print(f"\n😊 Sentiment Analysis Results:")
            sentiment_counts = sentiment_df['vader_sentiment'].value_counts()
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(sentiment_df)) * 100
                print(f"   {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
            
            # Show sample predictions
            print(f"\n📝 Sample Sentiment Predictions:")
            sample_predictions = sentiment_df[['text', 'vader_sentiment', 'textblob_sentiment', 'ensemble_sentiment']].head(3)
            for idx, row in sample_predictions.iterrows():
                text_preview = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
                print(f"   Text: '{text_preview}'")
                print(f"   VADER: {row['vader_sentiment']}, TextBlob: {row['textblob_sentiment']}, Ensemble: {row['ensemble_sentiment']}")
                print()
        
        # Classification results
        if 'classification' in results:
            print(f"\n🎯 Text Classification Results:")
            for model_name, model_results in results['classification'].items():
                print(f"   {model_name.replace('_', ' ').title()}:")
                print(f"     Accuracy: {model_results['accuracy']:.3f}")
                print(f"     F1-Score: {model_results['f1']:.3f}")
        
        # Model comparison
        if 'model_comparison' in results:
            comparison = results['model_comparison']
            if 'best_models' in results['model_comparison'] and 'best_overall' in comparison['best_models']:
                best = comparison['best_models']['best_overall']
                print(f"\n🏆 Best Classification Model:")
                print(f"   Model: {best['model']}")
                print(f"   Overall Score: {best['score']:.3f}")
        
        # Output locations
        print(f"\n📁 Output Files Generated:")
        print(f"   📄 Processed data: output/processed_data.csv")
        print(f"   📄 Sentiment results: output/sentiment_results.csv")
        print(f"   📊 Visualizations: output/visualizations/")
        print(f"   📋 Report: output/reports/pipeline_report.md")
        print(f"   💾 Models: output/models/")
        print(f"   🔢 Features: output/features/")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📖 Check the detailed report at: output/reports/pipeline_report.md")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return

def demo_individual_components():
    """Demonstrate individual pipeline components"""
    print("\n🔧 Individual Components Demo")
    print("=" * 30)
    
    try:
        pipeline = NLPPipeline()
        
        # Sample texts for demonstration
        sample_texts = [
            "I absolutely love this product! It's amazing and works perfectly.",
            "This is terrible. I hate it and want my money back.",
            "It's okay, nothing special but not bad either.",
            "Outstanding quality and excellent customer service!",
            "Poor performance, very disappointed with this purchase."
        ]
        
        print(f"\n📝 Sample texts for analysis:")
        for i, text in enumerate(sample_texts, 1):
            print(f"   {i}. {text}")
        
        # Sentiment analysis only
        print(f"\n😊 Sentiment Analysis Demo:")
        sentiment_results = pipeline.run_sentiment_analysis_only(sample_texts)
        
        for idx, row in sentiment_results.iterrows():
            text_preview = row['text'][:40] + "..." if len(row['text']) > 40 else row['text']
            print(f"   '{text_preview}'")
            print(f"   VADER: {row['vader_sentiment']} ({row['vader_compound']:.2f})")
            print(f"   TextBlob: {row['textblob_sentiment']} ({row['textblob_polarity']:.2f})")
            print(f"   Ensemble: {row['ensemble_sentiment']}")
            print()
        
        # Text classification demo
        print(f"🎯 Text Classification Demo:")
        
        # Sample classification data
        classification_texts = [
            "This product is fantastic and I love it!",
            "Terrible quality, worst purchase ever",
            "Great service and amazing quality",
            "Poor customer service, very disappointed",
            "Excellent value for money",
            "Complete waste of time and money"
        ]
        
        classification_labels = ["positive", "negative", "positive", "negative", "positive", "negative"]
        
        classification_results = pipeline.run_text_classification(classification_texts, classification_labels)
        
        print(f"   Classification Performance:")
        for model_name, results in classification_results.items():
            print(f"   {model_name.replace('_', ' ').title()}: {results['accuracy']:.3f} accuracy")
        
    except Exception as e:
        print(f"❌ Components demo failed: {str(e)}")

if __name__ == "__main__":
    # Run full pipeline demo
    main()
    
    # Optional: Run individual components demo
    response = input("\n🤔 Would you like to see individual components demo? (y/n): ")
    if response.lower() in ['y', 'yes']:
        demo_individual_components()
    
    print("\n🎉 Demo session completed!")
