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
from collections import defaultdict, deque
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package
from ar_hackathon.utils.routing_utils import is_valid_move

class MinCostMaxFlow:
    """
    Min Cost Max Flow implementation using Successive Shortest Path algorithm.
    """
    
    def __init__(self, n: int):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.capacity = defaultdict(int)
        self.cost = defaultdict(int)
        self.flow = defaultdict(int)
        
    def add_edge(self, u: int, v: int, cap: int, cost: int):
        """Add edge from u to v with capacity and cost."""
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.capacity[(u, v)] = cap
        self.cost[(u, v)] = cost
        self.cost[(v, u)] = -cost
        
    def bellman_ford(self, s: int, t: int) -> Tuple[List[int], List[int]]:
        """Bellman-Ford algorithm to find shortest path with negative weights."""
        dist = [float('inf')] * self.n
        parent = [-1] * self.n
        dist[s] = 0
        
        # Relax edges V-1 times
        for _ in range(self.n - 1):
            for u in range(self.n):
                if dist[u] == float('inf'):
                    continue
                for v in self.graph[u]:
                    if self.capacity[(u, v)] > self.flow[(u, v)]:
                        if dist[u] + self.cost[(u, v)] < dist[v]:
                            dist[v] = dist[u] + self.cost[(u, v)]
                            parent[v] = u
        
        return dist, parent
    
    def find_path(self, s: int, t: int) -> Tuple[List[int], int]:
        """Find augmenting path from s to t."""
        dist, parent = self.bellman_ford(s, t)
        
        if dist[t] == float('inf'):
            return [], 0
            
        # Reconstruct path
        path = []
        current = t
        while current != -1:
            path.append(current)
            current = parent[current]
        path.reverse()
        
        # Find minimum capacity along path
        min_cap = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            min_cap = min(min_cap, self.capacity[(u, v)] - self.flow[(u, v)])
            
        return path, min_cap
    
    def augment_path(self, path: List[int], flow_amount: int):
        """Augment flow along the given path."""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.flow[(u, v)] += flow_amount
            self.flow[(v, u)] -= flow_amount
    
    def min_cost_max_flow(self, s: int, t: int) -> Tuple[int, int]:
        """Find minimum cost maximum flow from s to t."""
        total_flow = 0
        total_cost = 0
        
        while True:
            path, flow_amount = self.find_path(s, t)
            if not path or flow_amount == 0:
                break
                
            # Calculate cost of this path
            path_cost = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                path_cost += self.cost[(u, v)]
            
            total_flow += flow_amount
            total_cost += path_cost * flow_amount
            self.augment_path(path, flow_amount)
            
        return total_flow, total_cost

def has_capacity_constraints(state: GameState) -> bool:
    """
    Check if the network has any capacity constraints.
    
    Args:
        state: GameState object
        
    Returns:
        True if any connection has bandwidth constraints
    """
    for conn in state.connections:
        if conn.bandwidth is not None:
            return True
    return False

def build_flow_network(state: GameState, packages: List[Package]) -> Tuple[MinCostMaxFlow, Dict[str, int], Dict[int, str]]:
    """
    Build a flow network for min cost max flow routing.
    
    Args:
        state: GameState object
        packages: List of packages to route
        
    Returns:
        Tuple of (flow_network, fc_to_node, node_to_fc)
    """
    # Create node mapping
    fc_to_node = {}
    node_to_fc = {}
    node_id = 0
    
    # Add source and sink nodes
    source = node_id
    fc_to_node['SOURCE'] = source
    node_to_fc[source] = 'SOURCE'
    node_id += 1
    
    sink = node_id
    fc_to_node['SINK'] = sink
    node_to_fc[sink] = 'SINK'
    node_id += 1
    
    # Add FC nodes (split into in and out nodes for capacity constraints)
    for fc in state.fulfillment_centers:
        # In node
        fc_in = node_id
        fc_to_node[f"{fc.id}_IN"] = fc_in
        node_to_fc[fc_in] = f"{fc.id}_IN"
        node_id += 1
        
        # Out node
        fc_out = node_id
        fc_to_node[f"{fc.id}_OUT"] = fc_out
        node_to_fc[fc_out] = f"{fc.id}_OUT"
        node_id += 1
    
    # Create flow network
    flow_network = MinCostMaxFlow(node_id)
    
    # Add edges from source to package starting FCs
    package_counts = defaultdict(int)
    for pkg in packages:
        if pkg.current_fc != pkg.destination_fc:
            package_counts[pkg.current_fc] += 1
    
    for fc_id, count in package_counts.items():
        if f"{fc_id}_IN" in fc_to_node:
            flow_network.add_edge(source, fc_to_node[f"{fc_id}_IN"], count, 0)
    
    # Add edges from package destination FCs to sink
    dest_counts = defaultdict(int)
    for pkg in packages:
        if pkg.current_fc != pkg.destination_fc:
            dest_counts[pkg.destination_fc] += 1
    
    for fc_id, count in dest_counts.items():
        if f"{fc_id}_OUT" in fc_to_node:
            flow_network.add_edge(fc_to_node[f"{fc_id}_OUT"], sink, count, 0)
    
    # Add internal edges within FCs (in to out)
    for fc in state.fulfillment_centers:
        fc_in = fc_to_node.get(f"{fc.id}_IN")
        fc_out = fc_to_node.get(f"{fc.id}_OUT")
        if fc_in is not None and fc_out is not None:
            # High capacity, low cost for internal flow
            flow_network.add_edge(fc_in, fc_out, 1000, 0)
    
    # Add connection edges between FCs
    for conn in state.connections:
        from_out = fc_to_node.get(f"{conn.from_fc}_OUT")
        to_in = fc_to_node.get(f"{conn.to_fc}_IN")
        
        if from_out is not None and to_in is not None:
            # Use available bandwidth as capacity
            capacity = conn.available_bandwidth if conn.bandwidth is not None else 1000
            cost = int(conn.weight * 100)  # Scale cost for integer arithmetic
            flow_network.add_edge(from_out, to_in, capacity, cost)
    
    return flow_network, fc_to_node, node_to_fc

def get_optimal_routing(state: GameState, packages: List[Package]) -> Dict[str, str]:
    """
    Get optimal routing for all packages using min cost max flow.
    
    Args:
        state: GameState object
        packages: List of packages to route
        
    Returns:
        Dictionary mapping package_id to next_fc
    """
    if not packages:
        return {}
    
    # Build flow network
    flow_network, fc_to_node, node_to_fc = build_flow_network(state, packages)
    
    # Find optimal flow
    source = fc_to_node['SOURCE']
    sink = fc_to_node['SINK']
    total_flow, total_cost = flow_network.min_cost_max_flow(source, sink)
    
    # Extract routing decisions from flow
    routing = {}
    
    # For each package, find the next FC based on flow
    for pkg in packages:
        if pkg.current_fc == pkg.destination_fc:
            routing[pkg.id] = None
            continue
            
        # Find which outgoing edge has flow
        best_next_fc = None
        best_cost = float('inf')
        
        for conn in state.connections:
            if conn.from_fc == pkg.current_fc:
                from_out = fc_to_node.get(f"{conn.from_fc}_OUT")
                to_in = fc_to_node.get(f"{conn.to_fc}_IN")
                
                if from_out is not None and to_in is not None:
                    if flow_network.flow[(from_out, to_in)] > 0:
                        # This connection has flow, consider it
                        cost = conn.weight
                        if cost < best_cost:
                            best_cost = cost
                            best_next_fc = conn.to_fc
        
        routing[pkg.id] = best_next_fc
    
    return routing

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
    
    This implementation uses min cost max flow when capacity constraints exist,
    otherwise falls back to Dijkstra's algorithm for shortest path routing.
    
    Args:
        state: GameState object containing the current state of the network
        package: Package object containing information about the package
        
    Returns:
        next_fc_id: ID of the next FC to route the package to, or None to stay at current FC
    """
    # If package is already at destination, don't move
    if package.current_fc == package.destination_fc:
        return None
    
    # Check if we have capacity constraints
    if has_capacity_constraints(state):
        # Use min cost max flow for optimal routing with capacity constraints
        active_packages = [pkg for pkg in state.active_packages 
                          if pkg.current_fc != pkg.destination_fc and not pkg.in_transit]
        
        if active_packages:
            optimal_routing = get_optimal_routing(state, active_packages)
            return optimal_routing.get(package.id)
    
    # Fall back to Dijkstra's algorithm for shortest path routing
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
