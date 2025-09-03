"""
Amazon Robotics Hackathon - GameState Model

This module defines the GameState class for the Amazon Robotics Hackathon.
"""

from typing import List, Optional
from ar_hackathon.models.fulfillment_center import FulfillmentCenter
from ar_hackathon.models.connection import Connection
from ar_hackathon.models.package import Package

class GameState:
    def __init__(self, current_time_step: int, 
                 fulfillment_centers: List[FulfillmentCenter], 
                 connections: List[Connection],
                 active_packages: List[Package]):
        self.current_time_step = current_time_step
        self.fulfillment_centers = fulfillment_centers
        self.connections = connections
        self.active_packages = active_packages
        self.delivered_packages = []
    
    def get_connection(self, from_fc: str, to_fc: str) -> Optional[Connection]:
        """
        Find a connection between two fulfillment centers.
        
        Args:
            from_fc: ID of the source FC
            to_fc: ID of the destination FC
            
        Returns:
            Connection object if found, None otherwise
        """
        for conn in self.connections:
            if conn.from_fc == from_fc and conn.to_fc == to_fc:
                return conn
        return None
    
    def deep_copy(self):
        """Create a deep copy of this GameState."""
        # Copy fulfillment centers
        fcs = [fc.deep_copy() for fc in self.fulfillment_centers]
        
        # Copy connections
        connections = [conn.deep_copy() for conn in self.connections]
        
        # Copy active packages
        active_packages = [pkg.deep_copy() for pkg in self.active_packages]
        
        # Create new GameState
        new_state = GameState(
            current_time_step=self.current_time_step,
            fulfillment_centers=fcs,
            connections=connections,
            active_packages=active_packages
        )
        
        # Copy delivered packages
        new_state.delivered_packages = [pkg.deep_copy() for pkg in self.delivered_packages]
        
        return new_state
