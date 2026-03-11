"""
Minimal Working Demo for Instagram Sentiment Analyzer
Uses only basic Python libraries to demonstrate the concept
"""

import re
import json
from collections import Counter, defaultdict
from pathlib import Path
import math

class MinimalSentimentAnalyzer:
    """Simple rule-based sentiment analyzer"""
    
    def __init__(self):
        # Simple sentiment word lists
        self.positive_words = {
            'good', 'great', 'amazing', 'love', 'excellent', 'wonderful', 'fantastic',
            'beautiful', 'awesome', 'perfect', 'best', 'happy', 'blessed', 'grateful',
            'incredible', 'energized', 'motivation', 'perfect', 'love', 'great'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'hate', 'worst', 'disappointed', 'frustrated',
            'boring', 'predictable', 'waste', 'stuck', 'traffic', 'cold', 'disappointed',
            'terrible', 'bad', 'awful', 'boring', 'predictable', 'waste', 'stuck'
        }
        
        self.negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nobody'}
    
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
        
        # Remove emojis (Unicode ranges)
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        positive_count = 0
        negative_count = 0
        
        # Check for negation
        negation = False
        for i, word in enumerate(words):
            if word in self.negation_words:
                negation = not negation  # Toggle negation
                continue
            
            if word in self.positive_words:
                if negation:
                    negative_count += 1
                else:
                    positive_count += 1
            elif word in self.negative_words:
                if negation:
                    positive_count += 1
                else:
                    negative_count += 1
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'processed_text': processed_text
        }
    
    def analyze_posts(self, posts):
        """Analyze multiple posts"""
        results = []
        for post in posts:
            analysis = self.analyze_sentiment(post['caption'])
            result = {
                'original_text': post['caption'],
                'sentiment': analysis['sentiment'],
                'confidence': analysis['confidence'],
                'likes': post.get('likes', 0),
                'comments_count': post.get('comments_count', 0),
                'positive_words_found': analysis['positive_words'],
                'negative_words_found': analysis['negative_words'],
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

def create_text_visualization(results):
    """Create simple text-based visualization"""
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*80)
    
    # Count sentiments
    sentiment_counts = Counter([r['sentiment'] for r in results])
    total_posts = len(results)
    
    print(f"\nOverall Sentiment Distribution:")
    print("-" * 40)
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_posts) * 100
        bar_length = int(percentage / 2)  # Scale bar to 50 chars max
        bar = "#" * bar_length + "." * (50 - bar_length)
        print(f"{sentiment.upper():10} | {bar} {count} ({percentage:.1f}%)")
    
    print(f"\nDetailed Analysis:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        sentiment_symbols = {"positive": ":)", "negative": ":(", "neutral": ":|"}.get(result['sentiment'], "?")
        
        print(f"\n{i}. {sentiment_symbols} {result['sentiment'].upper()} (Confidence: {result['confidence']:.2f})")
        print(f"   Likes: {result['likes']} | Comments: {result['comments_count']}")
        print(f"   Text: {result['original_text'][:80]}...")
        print(f"   Positive words: {result['positive_words_found']} | Negative words: {result['negative_words_found']}")
    
    # Engagement analysis
    print(f"\nEngagement Analysis:")
    print("-" * 40)
    
    sentiment_engagement = defaultdict(list)
    for result in results:
        engagement = result['likes'] + result['comments_count']
        sentiment_engagement[result['sentiment']].append(engagement)
    
    for sentiment, engagements in sentiment_engagement.items():
        avg_engagement = sum(engagements) / len(engagements)
        print(f"{sentiment.upper():10} | Avg Engagement: {avg_engagement:.1f}")
    
    # Top posts
    print(f"\nTop Posts by Engagement:")
    print("-" * 40)
    
    sorted_results = sorted(results, key=lambda x: x['likes'] + x['comments_count'], reverse=True)
    for i, result in enumerate(sorted_results[:3], 1):
        engagement = result['likes'] + result['comments_count']
        sentiment_symbol = {"positive": ":)", "negative": ":(", "neutral": ":|"}.get(result['sentiment'], "?")
        print(f"{i}. {sentiment_symbol} {result['sentiment'].upper()} - {engagement} engagement")
        print(f"   {result['original_text'][:60]}...")

def save_results(results, filename="minimal_sentiment_results.json"):
    """Save results to file"""
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    filepath = Path("data") / filename
    
    # Add summary statistics
    summary = {
        'total_posts': len(results),
        'sentiment_distribution': dict(Counter([r['sentiment'] for r in results])),
        'average_confidence': sum(r['confidence'] for r in results) / len(results),
        'total_likes': sum(r['likes'] for r in results),
        'total_comments': sum(r['comments_count'] for r in results)
    }
    
    output_data = {
        'summary': summary,
        'results': results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {filepath}")
    return filepath

def main():
    """Main demonstration function"""
    print("Instagram Sentiment Analyzer - Minimal Demo")
    print("=" * 60)
    print("This demo uses only basic Python libraries (no ML packages required)")
    print("=" * 60)
    
    # Create sample data
    print("\nCreating sample Instagram data...")
    posts = create_sample_data()
    print(f"Created {len(posts)} sample posts")
    
    # Initialize analyzer
    print("\nInitializing sentiment analyzer...")
    analyzer = MinimalSentimentAnalyzer()
    print("Analyzer ready (rule-based approach)")
    
    # Analyze posts
    print(f"\nAnalyzing {len(posts)} posts...")
    results = analyzer.analyze_posts(posts)
    print(f"Analysis completed")
    
    # Create visualization
    create_text_visualization(results)
    
    # Save results
    save_results(results)
    
    # Summary
    sentiment_counts = Counter([r['sentiment'] for r in results])
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    print(f"\nDemo Complete!")
    print(f"Posts Analyzed: {len(results)}")
    print(f"Sentiment Distribution: {dict(sentiment_counts)}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    print(f"\nNext Steps:")
    print(f"1. Install full dependencies: python setup.py")
    print(f"2. Try ML-based approach: python quick_start.py")
    print(f"3. Launch dashboard: streamlit run app/streamlit_app.py")
    print(f"4. Check README.md for advanced features")

if __name__ == "__main__":
    main()
