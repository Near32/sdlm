from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sdlm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Straight-Through Gumbel-Softmax Differentiable Language Modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sdlm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        #"torch>=1.9.0",
        #"transformers>=4.0.0",
        "dspy",
        "sentence-transformers",
        "bert-score",
        "mauve-text",
        "matplotlib",
        "wandb",
    ],
    extras_require={
        "dev": [
            "ipdb",
            "pytest>=6.0",
            "black>=21.7b0",
            "isort>=5.9.0",
            "mypy>=0.910",
        ],
    },
)
