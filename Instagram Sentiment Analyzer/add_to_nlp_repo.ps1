# PowerShell Script to Add Instagram Sentiment Analyzer to NLP Repository
# Run this script to automatically add your project as a subfolder

Write-Host "🚀 Adding Instagram Sentiment Analyzer to NLP Repository..." -ForegroundColor Green

# Set paths
$projectsPath = "C:\Users\akshi\OneDrive\Desktop\Projects"
$nlpRepoPath = "$projectsPath\Natural-Language-Processing"
$instagramProjectPath = "$projectsPath\Instagram Sentiment Analyzer"
$targetPath = "$nlpRepoPath\Instagram-Sentiment-Analyzer"

try {
    # Navigate to Projects directory
    Write-Host "📁 Navigating to Projects directory..." -ForegroundColor Yellow
    Set-Location $projectsPath

    # Check if NLP repo exists, clone if not
    if (!(Test-Path $nlpRepoPath)) {
        Write-Host "📥 Cloning NLP repository..." -ForegroundColor Yellow
        git clone https://github.com/akshita1610/Natural-Language-Processing.git
    } else {
        Write-Host "✅ NLP repository already exists" -ForegroundColor Green
    }

    # Navigate to NLP repo
    Set-Location $nlpRepoPath
    Write-Host "📂 Current directory: $(Get-Location)" -ForegroundColor Cyan

    # Create Instagram Sentiment Analyzer subfolder
    Write-Host "📁 Creating Instagram-Sentiment-Analyzer subfolder..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "Instagram-Sentiment-Analyzer" -Force

    # Copy all files from current project
    Write-Host "📋 Copying all files from Instagram Sentiment Analyzer project..." -ForegroundColor Yellow
    
    # Get all files and directories from source
    $sourceItems = Get-ChildItem -Path $instagramProjectPath -Force
    
    foreach ($item in $sourceItems) {
        $targetItem = Join-Path $targetPath $item.Name
        if ($item.PSIsContainer) {
            # Copy directory
            Copy-Item -Path $item.FullName -Destination $targetItem -Recurse -Force
            Write-Host "📁 Copied directory: $($item.Name)" -ForegroundColor Green
        } else {
            # Copy file
            Copy-Item -Path $item.FullName -Destination $targetItem -Force
            Write-Host "📄 Copied file: $($item.Name)" -ForegroundColor Green
        }
    }

    # Check git status
    Write-Host "🔍 Checking git status..." -ForegroundColor Yellow
    git status

    # Add files to git
    Write-Host "➕ Adding files to git..." -ForegroundColor Yellow
    git add Instagram-Sentiment-Analyzer/

    # Commit changes
    Write-Host "💾 Committing changes..." -ForegroundColor Yellow
    git commit -m "Add Instagram Sentiment Analyzer - Complete sentiment analysis system with interactive dashboard

Features:
- Interactive Streamlit dashboard
- Multiple sentiment analysis methods (rule-based, custom ML, advanced ML)
- Instagram data scraping capabilities
- Real-time sentiment processing
- Beautiful visualizations and analytics
- Export functionality (CSV, JSON)
- Comprehensive documentation
- Cross-platform compatibility

This adds a complete, production-ready sentiment analysis system to the NLP portfolio."

    # Push to GitHub
    Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Yellow
    git push origin main

    Write-Host "✅ SUCCESS! Instagram Sentiment Analyzer added to NLP repository!" -ForegroundColor Green
    Write-Host "🌐 View at: https://github.com/akshita1610/Natural-Language-Processing" -ForegroundColor Cyan
    Write-Host "📂 Navigate to: Natural-Language-Processing/Instagram-Sentiment-Analyzer/" -ForegroundColor Cyan

} catch {
    Write-Host "❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Manual steps available in ADD_TO_NLP_REPO_GUIDE.md" -ForegroundColor Yellow
}

Write-Host "🎉 Process completed!" -ForegroundColor Green
Write-Host "Press any key to continue..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
