#!/usr/bin/env python3
"""
Setup script for Deep-TOON package.
"""

from setuptools import setup, find_packages
import os

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the version from __init__.py
def get_version():
    init_path = os.path.join("deep_toon", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="deep-toon",
    version=get_version(),
    author="Deep-TOON Contributors",
    author_email="",
    description="Deep Token-Oriented Object Notation - Efficient JSON compression for LLM applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csabakecskemeti/deep-toon-python",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Markup",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Zero dependencies - keep it lightweight!
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ]
    },
    project_urls={
        "Bug Reports": "https://github.com/csabakecskemeti/deep-toon-python/issues",
        "Source": "https://github.com/csabakecskemeti/deep-toon-python",
        "Documentation": "https://github.com/csabakecskemeti/deep-toon-python/blob/main/README.md",
    },
    keywords="json compression llm ai tokens optimization serialization",
    zip_safe=False,
)