#!/usr/bin/env python3
"""
Amazon Robotics Hackathon - Setup Script

This script installs the ar_hackathon package.
"""

from setuptools import setup, find_packages

setup(
    name="ar_hackathon",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "networkx",
        "plotly",
        "numpy",
        "bokeh",
        "kaleido",
        "scipy",
        "requests"
    ],
    scripts=[
        "scripts/run_game.py",
        "scripts/visualize.py",
    ],
    author="Amazon Robotics",
    author_email="example@amazon.com",
    description="Amazon Robotics Hackathon Package",
    keywords="amazon, robotics, hackathon",
    python_requires=">=3.7",
)
