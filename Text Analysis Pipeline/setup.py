"""
Setup script for NLP Text Analysis Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nlp-text-analysis-pipeline",
    version="1.0.0",
    author="CS Student Portfolio Project",
    author_email="student@example.com",
    description="A comprehensive NLP text analysis pipeline for educational purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/nlp-text-analysis-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "transformers": [
            "torch>=1.9.0",
            "transformers>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nlp-pipeline=main:main",
            "nlp-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md", "*.txt"],
    },
)
