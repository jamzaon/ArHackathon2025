"""
Amazon Robotics Hackathon - Connection Model

This module defines the Connection class for the Amazon Robotics Hackathon.
"""

from typing import Optional

class Connection:
    def __init__(self, from_fc: str, to_fc: str, weight: float, 
                 bandwidth: Optional[int] = None, available_bandwidth: Optional[int] = None):
        self.from_fc = from_fc
        self.to_fc = to_fc
        self.weight = weight
        self.bandwidth = bandwidth  # None means unlimited
        self.available_bandwidth = available_bandwidth if available_bandwidth is not None else bandwidth
    
    def __repr__(self):
        return f"Connection(from='{self.from_fc}', to='{self.to_fc}', weight={self.weight})"
    
    def deep_copy(self):
        """Create a deep copy of this Connection."""
        return Connection(
            from_fc=self.from_fc,
            to_fc=self.to_fc,
            weight=self.weight,
            bandwidth=self.bandwidth,
            available_bandwidth=self.available_bandwidth
        )
