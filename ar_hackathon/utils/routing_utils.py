"""
Amazon Robotics Hackathon - Routing Utilities

This module provides utility functions for routing packages in the Amazon Robotics Hackathon.
"""

from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package

def is_valid_move(game_state: GameState, package: Package, next_fc: str) -> bool:
    """
    Check if moving a package to a specific FC is a valid move.
    
    Args:
        game_state: Current game state
        package: Package to move
        next_fc: ID of the FC to move the package to
        
    Returns:
        bool: True if the move is valid, False otherwise
    """
    # Check if the package is already in transit
    if package.in_transit:
        return False
    
    # Check if the package is already at the destination
    if package.current_fc == next_fc:
        return False
    
    # Find the connection between the current FC and the next FC
    connection = None
    for conn in game_state.connections:
        if conn.from_fc == package.current_fc and conn.to_fc == next_fc:
            connection = conn
            break
    
    # If no connection exists, the move is invalid
    if connection is None:
        return False
    
    # Check if the connection has available bandwidth
    if connection.bandwidth is not None:
        if connection.available_bandwidth <= 0:
            return False
    
    # All checks passed, the move is valid
    return True
