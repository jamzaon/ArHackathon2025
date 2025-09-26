"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the route_package function in this module.

*****IMPORTANT*****
Team name: Kaibo and Friends
Email address: fiona.cai899@gmail.com
*******************
"""

from typing import Optional, Dict, List, Tuple, Set
import heapq
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package
from ar_hackathon.utils.routing_utils import is_valid_move

def dijkstra_shortest_path(state: GameState, start_fc: str, end_fc: str) -> Optional[List[str]]:
    """
    Find the shortest path from start_fc to end_fc using Dijkstra's algorithm.
    
    Args:
        state: GameState object containing the network
        start_fc: Starting FC ID
        end_fc: Destination FC ID
        
    Returns:
        List of FC IDs representing the shortest path, or None if no path exists
    """
    # Build adjacency list
    graph = {}
    for fc in state.fulfillment_centers:
        graph[fc.id] = []
    
    for conn in state.connections:
        if conn.from_fc in graph:
            graph[conn.from_fc].append((conn.to_fc, conn.weight))
    
    # Check if start and end FCs exist in the graph
    if start_fc not in graph or end_fc not in graph:
        return None
    
    # Dijkstra's algorithm
    distances = {fc_id: float('inf') for fc_id in graph}
    distances[start_fc] = 0
    previous = {fc_id: None for fc_id in graph}
    
    # Priority queue: (distance, fc_id)
    pq = [(0, start_fc)]
    visited = set()
    
    while pq:
        current_dist, current_fc = heapq.heappop(pq)
        
        if current_fc in visited:
            continue
            
        visited.add(current_fc)
        
        if current_fc == end_fc:
            break
            
        for neighbor_fc, weight in graph[current_fc]:
            if neighbor_fc in visited or neighbor_fc not in graph:
                continue
                
            new_dist = current_dist + weight
            
            if new_dist < distances[neighbor_fc]:
                distances[neighbor_fc] = new_dist
                previous[neighbor_fc] = current_fc
                heapq.heappush(pq, (new_dist, neighbor_fc))
    
    # Reconstruct path
    if distances[end_fc] == float('inf'):
        return None
        
    path = []
    current = end_fc
    while current is not None:
        path.append(current)
        current = previous[current]
    
    path.reverse()
    return path

def get_available_connections(state: GameState, current_fc: str) -> List[Tuple[str, float]]:
    """
    Get all available connections from current_fc, considering bandwidth constraints.
    
    Args:
        state: GameState object
        current_fc: Current FC ID
        
    Returns:
        List of (destination_fc, weight) tuples for available connections
    """
    available = []
    
    for conn in state.connections:
        if conn.from_fc == current_fc:
            # Check bandwidth constraint
            if conn.bandwidth is None or conn.available_bandwidth > 0:
                available.append((conn.to_fc, conn.weight))
    
    return available

def calculate_congestion_penalty(state: GameState, fc_id: str) -> float:
    """
    Calculate a congestion penalty for an FC based on packages currently there.
    
    Args:
        state: GameState object
        fc_id: FC ID to check
        
    Returns:
        Congestion penalty (higher = more congested)
    """
    packages_at_fc = sum(1 for pkg in state.active_packages 
                        if pkg.current_fc == fc_id and not pkg.in_transit)
    
    # Base penalty increases with number of packages
    return packages_at_fc * 0.1

def route_package(state: GameState, package: Package) -> Optional[str]:
    """
    Determine the next FC to route a package to.
    
    This implementation uses Dijkstra's algorithm to find the shortest path,
    with congestion avoidance and bandwidth awareness.
    
    Args:
        state: GameState object containing the current state of the network
        package: Package object containing information about the package
        
    Returns:
        next_fc_id: ID of the next FC to route the package to, or None to stay at current FC
    """
    # If package is already at destination, don't move
    if package.current_fc == package.destination_fc:
        return None
    
    # Get all available connections from current FC
    available_connections = get_available_connections(state, package.current_fc)
    
    if not available_connections:
        return None
    
    # Find the best next step
    best_next_fc = None
    best_score = float('inf')
    
    for next_fc, weight in available_connections:
        # Check if this move is valid
        if not is_valid_move(state, package, next_fc):
            continue
            
        # If this is the destination, go there immediately
        if next_fc == package.destination_fc:
            return next_fc
        
        # Calculate path from this FC to destination
        path = dijkstra_shortest_path(state, next_fc, package.destination_fc)
        
        if path is not None:
            # Calculate total path length
            total_length = weight
            for i in range(len(path) - 1):
                conn = state.get_connection(path[i], path[i+1])
                if conn is not None:
                    total_length += conn.weight
            
            # Add congestion penalty
            congestion_penalty = calculate_congestion_penalty(state, next_fc)
            score = total_length + congestion_penalty
            
            if score < best_score:
                best_score = score
                best_next_fc = next_fc
    
    return best_next_fc
