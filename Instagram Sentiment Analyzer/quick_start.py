"""
Quick Start Script for Instagram Sentiment Analyzer
Demonstrates basic functionality without complex dependencies
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from pathlib import Path

class SimpleSentimentAnalyzer:
    """Simple sentiment analyzer using basic ML"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(random_state=42)
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove mentions and hashtags symbols
        text = re.sub(r'[@#]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, texts, labels):
        """Train the sentiment model"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create features
        X = self.vectorizer.fit_transform(processed_texts)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained with accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict(self, texts):
        """Predict sentiment for new texts"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                'text': texts[i],
                'predicted_sentiment': pred,
                'confidence': max(probs),
                'probabilities': {
                    'negative': float(probs[0]) if len(probs) > 0 else 0,
                    'neutral': float(probs[1]) if len(probs) > 1 else 0,
                    'positive': float(probs[2]) if len(probs) > 2 else 0
                }
            }
            results.append(result)
        
        return results
    
    def create_visualizations(self, results):
        """Create simple visualizations"""
        # Sentiment distribution
        sentiments = [r['predicted_sentiment'] for r in results]
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        # Create pie chart
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                colors=colors, autopct='%1.1f%%')
        plt.title('Sentiment Distribution')
        
        # Confidence distribution
        confidences = [r['confidence'] for r in results]
        plt.subplot(1, 2, 2)
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('data/quick_start_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved to data/quick_start_analysis.png")

def create_sample_data():
    """Create sample Instagram data for demonstration"""
    sample_posts = [
        {
            'caption': 'Amazing sunset today! Feeling so blessed and grateful for this beautiful view! 🌅 #sunset #blessed',
            'likes': 150,
            'comments_count': 25,
            'timestamp': '2024-01-15T18:30:00'
        },
        {
            'caption': 'Terrible service at this restaurant. Waited 2 hours for cold food. Very disappointed! 😠',
            'likes': 45,
            'comments_count': 12,
            'timestamp': '2024-01-14T20:15:00'
        },
        {
            'caption': 'Just an ordinary day at the office. Nothing exciting to report. #work #office',
            'likes': 78,
            'comments_count': 8,
            'timestamp': '2024-01-13T09:00:00'
        },
        {
            'caption': 'Absolutely love my new phone! The camera is incredible and battery life is amazing! 📱❤️ #newphone #tech',
            'likes': 230,
            'comments_count': 35,
            'timestamp': '2024-01-12T14:20:00'
        },
        {
            'caption': 'This movie was so boring and predictable. Waste of time and money. #badmovie',
            'likes': 32,
            'comments_count': 6,
            'timestamp': '2024-01-11T21:45:00'
        },
        {
            'caption': 'Great workout session today! Feeling energized and ready to conquer the week! 💪 #fitness #motivation',
            'likes': 180,
            'comments_count': 28,
            'timestamp': '2024-01-10T07:30:00'
        },
        {
            'caption': 'Coffee and books - perfect combination for a lazy Sunday morning. ☕📚 #coffeetime #reading',
            'likes': 95,
            'comments_count': 15,
            'timestamp': '2024-01-09T10:15:00'
        },
        {
            'caption': 'Stuck in traffic again. This commute is getting worse every day. #traffic #frustrated',
            'likes': 28,
            'comments_count': 4,
            'timestamp': '2024-01-08T17:45:00'
        }
    ]
    
    return pd.DataFrame(sample_posts)

def main():
    """Main demonstration function"""
    print("🚀 Instagram Sentiment Analyzer - Quick Start Demo")
    print("=" * 60)
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Create sample data
    print("📊 Creating sample Instagram data...")
    df = create_sample_data()
    print(f"✅ Created {len(df)} sample posts")
    
    # Prepare training data (in real scenario, you'd have labeled data)
    print("\n🏷️  Preparing training data...")
    
    # Create realistic labels based on content
    texts = df['caption'].tolist()
    labels = []
    
    for text in texts:
        text_lower = text.lower()
        if any(word in text_lower for word in ['amazing', 'love', 'beautiful', 'blessed', 'great', 'perfect']):
            labels.append('positive')
        elif any(word in text_lower for word in ['terrible', 'disappointed', 'boring', 'waste', 'stuck', 'frustrated']):
            labels.append('negative')
        else:
            labels.append('neutral')
    
    print(f"✅ Created {len(texts)} labeled examples")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Train model
    print("\n🤖 Training sentiment model...")
    analyzer = SimpleSentimentAnalyzer()
    accuracy = analyzer.train(texts, labels)
    
    # Make predictions
    print("\n🔍 Making predictions...")
    results = analyzer.predict(texts)
    
    # Display results
    print("\n📋 Prediction Results:")
    print("-" * 80)
    for i, result in enumerate(results):
        print(f"Post {i+1}: {result['predicted_sentiment'].upper()} (Confidence: {result['confidence']:.3f})")
        print(f"Text: {result['text'][:80]}...")
        print("-" * 80)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    analyzer.create_visualizations(results)
    
    # Save results
    print("\n💾 Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/quick_start_results.csv', index=False)
    results_df.to_json('data/quick_start_results.json', orient='records', indent=2)
    
    print("✅ Results saved to:")
    print("  - data/quick_start_results.csv")
    print("  - data/quick_start_results.json")
    print("  - data/quick_start_analysis.png")
    
    # Summary
    print(f"\n🎉 Quick Start Demo Complete!")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Posts Analyzed: {len(results)}")
    
    sentiment_dist = pd.Series([r['predicted_sentiment'] for r in results]).value_counts()
    print(f"Sentiment Distribution: {sentiment_dist.to_dict()}")
    
    print(f"\n📚 Next Steps:")
    print(f"1. Try the full dashboard: streamlit run app/streamlit_app.py")
    print(f"2. Use your own data: python main.py --input your_data.csv")
    print(f"3. Check README.md for advanced features")

if __name__ == "__main__":
    main()
