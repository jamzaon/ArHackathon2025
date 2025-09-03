"""
Amazon Robotics Hackathon - FulfillmentCenter Model

This module defines the FulfillmentCenter class for the Amazon Robotics Hackathon.
"""

class FulfillmentCenter:
    def __init__(self, fc_id: str, name: str):
        self.id = fc_id
        self.name = name
    
    def __repr__(self):
        return f"FulfillmentCenter(id='{self.id}', name='{self.name}')"
    
    def deep_copy(self):
        """Create a deep copy of this FulfillmentCenter."""
        return FulfillmentCenter(
            fc_id=self.id,
            name=self.name
        )
