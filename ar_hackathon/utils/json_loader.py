"""
Amazon Robotics Hackathon - JSON Loader

This module provides utilities for loading test cases from JSON files.
"""

import json
from typing import Dict, Any
from ar_hackathon.models.test_case import TestCase


def load_test_case(file_path: str) -> TestCase:
    """
    Load a test case from a JSON file.
    
    Args:
        file_path: Path to the JSON test case file
        
    Returns:
        TestCase object initialized with the test case data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return TestCase(data)
