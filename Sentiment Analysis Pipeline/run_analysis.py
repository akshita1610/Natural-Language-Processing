"""
Run complete analysis of YouTube sentiment results
"""

from simple_explore import SimpleResultsExplorer

def run_complete_analysis():
    """Run all analysis functions and display results."""
    
    print("Starting complete YouTube sentiment analysis...")
    
    # Initialize explorer
    explorer = SimpleResultsExplorer()
    
    if explorer.df is None:
        print("No data found!")
        return
    
    # Run all analyses
    print("\n1. BASIC STATISTICS")
    explorer.basic_statistics()
    
    print("\n2. VIDEO ANALYSIS")
    explorer.analyze_by_video()
    
    print("\n3. SAMPLE COMMENTS")
    explorer.show_sample_comments(3)
    
    print("\n4. ENGAGEMENT ANALYSIS")
    explorer.analyze_engagement()
    
    print("\n5. CONTROVERSIAL VIDEOS")
    explorer.find_most_controversial()
    
    print("\n6. EXPORTING SUMMARY")
    explorer.export_summary()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    run_complete_analysis()
