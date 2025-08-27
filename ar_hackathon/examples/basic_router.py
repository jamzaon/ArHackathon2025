"""
Amazon Robotics Hackathon - Basic Router Example

This module provides a sample implementation of the route_package function. This is not a good
algorithm and is intended as an example only.
"""

from typing import Optional, Dict, Set
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package

visited_fcs: Dict[str, Set[str]] = {}

def basic_router(state: GameState, package: Package) -> Optional[str]:
    if package.id not in visited_fcs:
        visited_fcs[package.id] = set()

    package_visited = visited_fcs[package.id]
    package_visited.add(package.current_fc)

    best_next_fc = None
    best_weight = float('inf')
    
    for connection in state.connections:
        if connection.from_fc == package.current_fc:
            if connection.to_fc in package_visited:
                continue

            if connection.to_fc == package.destination_fc:
                return connection.to_fc

            if connection.weight < best_weight:
                best_weight = connection.weight
                best_next_fc = connection.to_fc

    if best_next_fc is None:
        for connection in state.connections:
            if connection.from_fc == package.current_fc:
                if connection.weight < best_weight:
                    best_weight = connection.weight
                    best_next_fc = connection.to_fc

    return best_next_fc
