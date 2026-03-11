"""
Working ML Demo using only basic libraries
Demonstrates sentiment analysis without scikit-learn dependency issues
"""

import re
import math
import json
from collections import Counter, defaultdict
from pathlib import Path

class SimpleMLSentimentAnalyzer:
    """Simple ML-like sentiment analyzer using basic Python"""
    
    def __init__(self):
        # Sentiment word lists
        self.positive_words = {
            'good', 'great', 'amazing', 'love', 'excellent', 'wonderful', 'fantastic',
            'beautiful', 'awesome', 'perfect', 'best', 'happy', 'blessed', 'grateful',
            'incredible', 'energized', 'motivation', 'perfect', 'love', 'great',
            'fantastic', 'excellent', 'beautiful', 'awesome', 'wonderful'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'hate', 'worst', 'disappointed', 'frustrated',
            'boring', 'predictable', 'waste', 'stuck', 'traffic', 'cold', 'disappointed',
            'terrible', 'bad', 'awful', 'boring', 'predictable', 'waste', 'stuck',
            'horrible', 'disgusting', 'annoying', 'useless', 'pathetic'
        }
        
        self.negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere'}
        
        # Feature weights (simple learned weights)
        self.feature_weights = {}
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
        
        # Remove emojis and special chars
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text):
        """Extract features from text"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        features = {
            'positive_count': 0,
            'negative_count': 0,
            'negation_count': 0,
            'word_count': len(words),
            'char_count': len(processed_text),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_count': sum(1 for c in text if c.isupper()),
            'hashtag_count': len(re.findall(r'#\w+', text)),
            'mention_count': len(re.findall(r'@\w+', text))
        }
        
        # Count sentiment words with negation handling
        negation = False
        for i, word in enumerate(words):
            if word in self.negation_words:
                negation = not negation
                features['negation_count'] += 1
                continue
            
            if word in self.positive_words:
                if negation:
                    features['negative_count'] += 1
                else:
                    features['positive_count'] += 1
            elif word in self.negative_words:
                if negation:
                    features['positive_count'] += 1
                else:
                    features['negative_count'] += 1
        
        # Calculate ratios
        if features['word_count'] > 0:
            features['positive_ratio'] = features['positive_count'] / features['word_count']
            features['negative_ratio'] = features['negative_count'] / features['word_count']
        else:
            features['positive_ratio'] = 0
            features['negative_ratio'] = 0
        
        return features, processed_text
    
    def train_simple_classifier(self, texts, labels):
        """Train a simple classifier using basic statistics"""
        print("Training simple sentiment classifier...")
        
        # Extract features for all texts
        all_features = []
        for text in texts:
            features, _ = self.extract_features(text)
            all_features.append(features)
        
        # Calculate feature statistics by sentiment
        sentiment_stats = defaultdict(lambda: defaultdict(list))
        
        for features, label in zip(all_features, labels):
            for feature_name, value in features.items():
                sentiment_stats[label][feature_name].append(value)
        
        # Calculate average feature values for each sentiment
        self.feature_weights = {}
        for sentiment in sentiment_stats:
            self.feature_weights[sentiment] = {}
            for feature_name in sentiment_stats[sentiment]:
                values = sentiment_stats[sentiment][feature_name]
                self.feature_weights[sentiment][feature_name] = sum(values) / len(values)
        
        self.is_trained = True
        print("Training completed!")
        
        # Print feature statistics
        print("\nFeature Statistics by Sentiment:")
        print("-" * 50)
        for sentiment in self.feature_weights:
            print(f"\n{sentiment.upper()}:")
            for feature, avg_value in self.feature_weights[sentiment].items():
                if 'ratio' in feature or 'count' in feature:
                    print(f"  {feature}: {avg_value:.3f}")
    
    def predict_sentiment(self, text):
        """Predict sentiment using trained weights"""
        if not self.is_trained:
            # Fallback to rule-based if not trained
            return self.rule_based_predict(text)
        
        features, processed_text = self.extract_features(text)
        
        # Calculate similarity to each sentiment class
        sentiment_scores = {}
        for sentiment in self.feature_weights:
            score = 0
            for feature_name, value in features.items():
                if feature_name in self.feature_weights[sentiment]:
                    # Simple distance-based scoring
                    expected = self.feature_weights[sentiment][feature_name]
                    score += 1 / (1 + abs(value - expected))
            
            sentiment_scores[sentiment] = score
        
        # Normalize scores
        total_score = sum(sentiment_scores.values())
        if total_score > 0:
            for sentiment in sentiment_scores:
                sentiment_scores[sentiment] /= total_score
        
        # Get best prediction
        best_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        confidence = sentiment_scores[best_sentiment]
        
        return {
            'sentiment': best_sentiment,
            'confidence': confidence,
            'scores': sentiment_scores,
            'features': features,
            'processed_text': processed_text
        }
    
    def rule_based_predict(self, text):
        """Fallback rule-based prediction"""
        features, processed_text = self.extract_features(text)
        
        positive_score = features['positive_count'] * 2 + features['positive_ratio'] * 10
        negative_score = features['negative_count'] * 2 + features['negative_ratio'] * 10
        
        # Adjust for negations
        if features['negation_count'] > 0:
            positive_score, negative_score = negative_score, positive_score
        
        # Adjust for emphasis
        positive_score += features['exclamation_count'] * 0.1
        positive_score += features['uppercase_count'] * 0.05
        
        if positive_score > negative_score:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + (positive_score - negative_score) * 0.1)
        elif negative_score > positive_score:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + (negative_score - positive_score) * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'features': features,
            'processed_text': processed_text
        }
    
    def analyze_posts(self, posts):
        """Analyze multiple posts"""
        results = []
        for post in posts:
            analysis = self.predict_sentiment(post['caption'])
            result = {
                'original_text': post['caption'],
                'sentiment': analysis['sentiment'],
                'confidence': analysis['confidence'],
                'likes': post.get('likes', 0),
                'comments_count': post.get('comments_count', 0),
                'features': analysis['features'],
                'processed_text': analysis['processed_text']
            }
            results.append(result)
        
        return results

def create_sample_data():
    """Create sample Instagram data"""
    return [
        {
            'caption': 'Amazing sunset today! Feeling so blessed and grateful for this beautiful view! #sunset #blessed',
            'likes': 150,
            'comments_count': 25,
            'timestamp': '2024-01-15T18:30:00'
        },
        {
            'caption': 'Terrible service at this restaurant. Waited 2 hours for cold food. Very disappointed!',
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
            'caption': 'Absolutely love my new phone! The camera is incredible and battery life is amazing! #newphone #tech',
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
            'caption': 'Great workout session today! Feeling energized and ready to conquer the week! #fitness #motivation',
            'likes': 180,
            'comments_count': 28,
            'timestamp': '2024-01-10T07:30:00'
        },
        {
            'caption': 'Coffee and books - perfect combination for a lazy Sunday morning. #coffeetime #reading',
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

def create_visualization(results):
    """Create text-based visualization"""
    print("\n" + "="*80)
    print("ADVANCED SENTIMENT ANALYSIS RESULTS")
    print("="*80)
    
    # Sentiment distribution
    sentiment_counts = Counter([r['sentiment'] for r in results])
    total_posts = len(results)
    
    print(f"\nSentiment Distribution:")
    print("-" * 40)
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_posts) * 100
        bar_length = int(percentage / 2)
        bar = "#" * bar_length + "." * (50 - bar_length)
        print(f"{sentiment.upper():10} | {bar} {count} ({percentage:.1f}%)")
    
    # Feature analysis
    print(f"\nFeature Analysis:")
    print("-" * 40)
    
    all_features = defaultdict(list)
    for result in results:
        for feature, value in result['features'].items():
            all_features[feature].append(value)
    
    key_features = ['positive_count', 'negative_count', 'word_count', 'exclamation_count']
    for feature in key_features:
        values = all_features[feature]
        avg_val = sum(values) / len(values)
        print(f"{feature:15} | Avg: {avg_val:.2f} | Min: {min(values)} | Max: {max(values)}")
    
    # Detailed results
    print(f"\nDetailed Analysis:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        symbols = {"positive": ":)", "negative": ":(", "neutral": ":|"}.get(result['sentiment'], "?")
        
        print(f"\n{i}. {symbols} {result['sentiment'].upper()} (Confidence: {result['confidence']:.3f})")
        print(f"   Engagement: {result['likes']} likes, {result['comments_count']} comments")
        print(f"   Features: +{result['features']['positive_count']} -{result['features']['negative_count']} words: {result['features']['word_count']}")
        print(f"   Text: {result['original_text'][:70]}...")
    
    # Engagement by sentiment
    print(f"\nEngagement by Sentiment:")
    print("-" * 40)
    
    sentiment_engagement = defaultdict(list)
    for result in results:
        engagement = result['likes'] + result['comments_count']
        sentiment_engagement[result['sentiment']].append(engagement)
    
    for sentiment, engagements in sentiment_engagement.items():
        avg_engagement = sum(engagements) / len(engagements)
        print(f"{sentiment.upper():10} | Avg: {avg_engagement:.1f} | Range: {min(engagements)}-{max(engagements)}")

def save_results(results, filename="working_ml_results.json"):
    """Save results to file"""
    Path("data").mkdir(exist_ok=True)
    
    # Create summary
    summary = {
        'total_posts': len(results),
        'sentiment_distribution': dict(Counter([r['sentiment'] for r in results])),
        'average_confidence': sum(r['confidence'] for r in results) / len(results),
        'method': 'simple_ml_classifier',
        'features_used': list(results[0]['features'].keys()) if results else []
    }
    
    output_data = {
        'summary': summary,
        'results': results
    }
    
    filepath = Path("data") / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {filepath}")
    return filepath

def main():
    """Main demonstration function"""
    print("Instagram Sentiment Analyzer - Working ML Demo")
    print("=" * 60)
    print("Advanced sentiment analysis without ML library dependencies")
    print("=" * 60)
    
    # Create sample data
    print("\nCreating sample Instagram data...")
    posts = create_sample_data()
    print(f"Created {len(posts)} sample posts")
    
    # Initialize analyzer
    print("\nInitializing advanced sentiment analyzer...")
    analyzer = SimpleMLSentimentAnalyzer()
    
    # Create training labels from content
    print("Creating training labels...")
    texts = [post['caption'] for post in posts]
    labels = []
    
    for text in texts:
        text_lower = text.lower()
        if any(word in text_lower for word in ['amazing', 'love', 'beautiful', 'blessed', 'great', 'perfect', 'wonderful', 'fantastic']):
            labels.append('positive')
        elif any(word in text_lower for word in ['terrible', 'disappointed', 'boring', 'waste', 'stuck', 'frustrated', 'bad', 'awful']):
            labels.append('negative')
        else:
            labels.append('neutral')
    
    print(f"Training labels: {dict(Counter(labels))}")
    
    # Train the classifier
    analyzer.train_simple_classifier(texts, labels)
    
    # Analyze posts
    print(f"\nAnalyzing {len(posts)} posts with trained classifier...")
    results = analyzer.analyze_posts(posts)
    
    # Create visualization
    create_visualization(results)
    
    # Save results
    save_results(results)
    
    # Summary
    print(f"\nDemo Complete!")
    print(f"Posts Analyzed: {len(results)}")
    print(f"Sentiment Distribution: {dict(Counter([r['sentiment'] for r in results]))}")
    print(f"Average Confidence: {sum(r['confidence'] for r in results) / len(results):.3f}")
    
    print(f"\nFeatures Used: {list(results[0]['features'].keys())}")
    
    print(f"\nNext Steps:")
    print(f"1. Try with your own data")
    print(f"2. Experiment with different feature weights")
    print(f"3. Compare with rule-based approach: python minimal_demo.py")

if __name__ == "__main__":
    main()
