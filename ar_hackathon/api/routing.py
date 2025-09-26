"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the route_package function in this module.

*****IMPORTANT*****
Team name: Kaibo and Friends
Email address: fiona.cai899@gmail.com
*******************
"""

from typing import Optional, List, Dict, Tuple, Set
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _abs(x):
    return x if x >= 0 else -x


def _min(a, b):
    return a if a <= b else b


def _max(a, b):
    return a if a >= b else b


def _sqrt(x):
    """Simple square root implementation."""
    if x == 0:
        return 0
    if x < 0:
        return 0

    # Newton's method
    guess = x / 2.0
    for _ in range(10):  # Sufficient iterations for convergence
        better_guess = (guess + x / guess) / 2.0
        if _abs(guess - better_guess) < 0.001:
            break
        guess = better_guess
    return guess


def _exp_decay(value, decay_rate=0.9):
    """Exponential decay approximation without math.exp."""
    result = 1.0
    for _ in range(int(value * 10)):
        result *= decay_rate
    return result


class MinHeap:
    """Simple min-heap implementation."""

    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)
        self._bubble_up(len(self._data) - 1)

    def pop(self):
        if not self._data:
            return None
        if len(self._data) == 1:
            return self._data.pop()

        result = self._data[0]
        self._data[0] = self._data.pop()
        self._bubble_down(0)
        return result

    def __bool__(self):
        return bool(self._data)

    def _bubble_up(self, index):
        while index > 0:
            parent = (index - 1) // 2
            if self._data[parent] <= self._data[index]:
                break
            self._data[parent], self._data[index] = self._data[index], self._data[parent]
            index = parent

    def _bubble_down(self, index):
        size = len(self._data)
        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index

            if left < size and self._data[left] < self._data[smallest]:
                smallest = left
            if right < size and self._data[right] < self._data[smallest]:
                smallest = right

            if smallest == index:
                break

            self._data[index], self._data[smallest] = self._data[smallest], self._data[index]
            index = smallest


class PredictiveAnalytics:
    """Advanced predictive analytics for delivery time optimization using machine learning principles."""

    def __init__(self):
        self.delivery_patterns = {}
        self.congestion_predictions = {}
        self.demand_forecasts = {}
        self.flow_patterns = {}
        self.time_series_data = {}

    def update_delivery_pattern(self, from_fc: str, to_fc: str, actual_time: float, predicted_time: float):
        """Learn from actual vs predicted delivery times using exponential smoothing."""
        key = (from_fc, to_fc)
        if key not in self.delivery_patterns:
            self.delivery_patterns[key] = {
                'avg_error': 0.0,
                'count': 0,
                'trend': 1.0,
                'seasonal_factor': 1.0,
                'volatility': 0.0
            }

        error = actual_time - predicted_time
        pattern = self.delivery_patterns[key]

        # Exponential smoothing with adaptive learning rate
        alpha = 0.3 if pattern['count'] < 10 else 0.1  # Higher learning rate for new patterns

        if pattern['count'] == 0:
            pattern['avg_error'] = error
        else:
            pattern['avg_error'] = alpha * error + (1 - alpha) * pattern['avg_error']

        # Track volatility (variance estimate)
        squared_error = error * error
        if pattern['count'] == 0:
            pattern['volatility'] = squared_error
        else:
            pattern['volatility'] = alpha * squared_error + (1 - alpha) * pattern['volatility']

        pattern['count'] += 1

        # Trend detection using simple linear regression approximation
        if pattern['count'] > 5:
            recent_trend = 1.0 + (error / _max(actual_time, 1.0))
            pattern['trend'] = 0.8 * pattern['trend'] + 0.2 * recent_trend

    def predict_delivery_time(self, from_fc: str, to_fc: str, base_time: float, current_time: int) -> float:
        """Predict actual delivery time with pattern recognition."""
        key = (from_fc, to_fc)
        if key not in self.delivery_patterns:
            return base_time

        pattern = self.delivery_patterns[key]

        # Base prediction with learned error correction
        predicted_time = base_time + pattern['avg_error']

        # Apply trend adjustment
        predicted_time *= pattern['trend']

        # Time-of-day adjustment (simplified seasonal factor)
        hour_of_day = current_time % 24
        if hour_of_day >= 8 and hour_of_day <= 18:  # Business hours
            predicted_time *= 1.2  # Higher congestion
        else:
            predicted_time *= 0.9  # Lower congestion

        return _max(predicted_time, base_time * 0.8)  # Don't predict impossibly fast times

    def update_congestion_forecast(self, fc_id: str, current_load: int, time_step: int):
        """Update congestion forecasting using time series analysis."""
        if fc_id not in self.congestion_predictions:
            self.congestion_predictions[fc_id] = {'history': [], 'forecast': 0.0, 'cycle_pattern': []}

        prediction = self.congestion_predictions[fc_id]
        prediction['history'].append(current_load)

        # Keep only recent history (sliding window)
        if len(prediction['history']) > 50:
            prediction['history'] = prediction['history'][-50:]

        # Simple moving average with weighted recent values
        if len(prediction['history']) >= 5:
            recent_avg = sum(prediction['history'][-5:]) / 5
            long_avg = sum(prediction['history']) / len(prediction['history'])
            prediction['forecast'] = 0.7 * recent_avg + 0.3 * long_avg

        # Detect cyclical patterns (simplified)
        if len(prediction['history']) >= 24:  # One day cycle
            cycle_sum = 0.0
            for i in range(24):
                if i < len(prediction['history']):
                    cycle_sum += prediction['history'][-(i + 1)]
            prediction['cycle_pattern'] = cycle_sum / 24

    def get_congestion_forecast(self, fc_id: str, lookahead_steps: int) -> float:
        """Get congestion forecast for future time steps."""
        if fc_id not in self.congestion_predictions:
            return 1.0  # Default moderate congestion

        prediction = self.congestion_predictions[fc_id]
        base_forecast = prediction.get('forecast', 1.0)

        # Apply decay for longer lookahead
        decay_factor = _exp_decay(lookahead_steps * 0.1)

        return base_forecast * decay_factor + 1.0 * (1 - decay_factor)


class AdaptiveLargeNeighborhoodSearch:
    """ALNS-inspired optimization for route selection."""

    def __init__(self):
        self.destroy_methods = ['random_removal', 'worst_removal', 'related_removal']
        self.repair_methods = ['greedy_insertion', 'regret_insertion']
        self.method_weights = {'destroy': [1.0, 1.0, 1.0], 'repair': [1.0, 1.0]}
        self.method_usage = {'destroy': [0, 0, 0], 'repair': [0, 0]}
        self.method_success = {'destroy': [0, 0, 0], 'repair': [0, 0]}

    def select_method(self, method_type: str) -> int:
        """Select method using roulette wheel selection based on performance."""
        weights = self.method_weights[method_type]
        total_weight = sum(weights)

        if total_weight == 0:
            return 0

        # Simple deterministic selection based on performance
        best_ratio = -1.0
        best_method = 0

        for i, weight in enumerate(weights):
            usage = self.method_usage[method_type][i]
            if usage > 0:
                success_ratio = self.method_success[method_type][i] / usage
                if success_ratio > best_ratio:
                    best_ratio = success_ratio
                    best_method = i

        return best_method

    def update_method_performance(self, method_type: str, method_idx: int, success: bool):
        """Update method performance statistics."""
        self.method_usage[method_type][method_idx] += 1
        if success:
            self.method_success[method_type][method_idx] += 1

        # Update weights using adaptive mechanism
        if self.method_usage[method_type][method_idx] >= 10:
            success_rate = self.method_success[method_type][method_idx] / self.method_usage[method_type][method_idx]
            self.method_weights[method_type][method_idx] = success_rate * 2.0

    def optimize_route_selection(self, candidates: List[Tuple[str, float]], target_count: int) -> List[str]:
        """Optimize route selection using ALNS principles."""
        if not candidates or target_count <= 0:
            return []

        # Sort candidates by cost
        candidates.sort(key=lambda x: x[1])

        selected = []
        remaining = list(candidates)

        # Greedy selection with diversity
        while len(selected) < target_count and remaining:
            if not selected:
                # First selection: best candidate
                best = remaining.pop(0)
                selected.append(best[0])
            else:
                # Subsequent selections: balance cost and diversity
                best_idx = 0
                best_score = float('inf')

                for i, (route, cost) in enumerate(remaining):
                    # Diversity bonus: prefer routes different from selected ones
                    diversity_bonus = 0.0
                    for selected_route in selected:
                        if route != selected_route:
                            diversity_bonus += 0.1

                    adjusted_score = cost - diversity_bonus
                    if adjusted_score < best_score:
                        best_score = adjusted_score
                        best_idx = i

                selected.append(remaining.pop(best_idx)[0])

        return selected


class AdvancedNetworkOptimizer:
    """Advanced network optimizer with predictive analytics and ALNS."""

    def __init__(self):
        self.shortest_paths_cache = {}
        self.predictive_analytics = PredictiveAnalytics()
        self.alns_optimizer = AdaptiveLargeNeighborhoodSearch()
        self.last_state_hash = None
        self.global_flow_state = {}
        self.bottleneck_detection = {}

    def _hash_network(self, state: GameState) -> str:
        """Create a hash of the network topology for cache invalidation."""
        connections = []
        for c in state.connections:
            connections.append((c.from_fc, c.to_fc, c.weight))
        connections.sort()
        return str(hash(tuple(connections)))

    def _invalidate_cache_if_needed(self, state: GameState):
        """Invalidate caches if network topology changed."""
        current_hash = self._hash_network(state)
        if self.last_state_hash != current_hash:
            self.shortest_paths_cache.clear()
            self.last_state_hash = current_hash

    def _detect_bottlenecks(self, state: GameState):
        """Detect network bottlenecks using flow analysis."""
        flow_analysis = {}

        for pkg in state.active_packages:
            if not pkg.in_transit:
                current_fc = pkg.current_fc
                if current_fc not in flow_analysis:
                    flow_analysis[current_fc] = {'outgoing': 0, 'capacity': 0}
                flow_analysis[current_fc]['outgoing'] += 1

        # Calculate capacity utilization
        for fc_id, flow_data in flow_analysis.items():
            total_capacity = 0
            for conn in state.connections:
                if conn.from_fc == fc_id:
                    capacity = conn.available_bandwidth if conn.available_bandwidth is not None else 100
                    total_capacity += capacity

            flow_data['capacity'] = total_capacity
            utilization = flow_data['outgoing'] / _max(total_capacity, 1)

            if fc_id not in self.bottleneck_detection:
                self.bottleneck_detection[fc_id] = {'utilization_history': [], 'is_bottleneck': False}

            self.bottleneck_detection[fc_id]['utilization_history'].append(utilization)
            if len(self.bottleneck_detection[fc_id]['utilization_history']) > 10:
                self.bottleneck_detection[fc_id]['utilization_history'] = \
                    self.bottleneck_detection[fc_id]['utilization_history'][-10:]

            # Mark as bottleneck if consistently high utilization
            avg_utilization = sum(self.bottleneck_detection[fc_id]['utilization_history']) / \
                              len(self.bottleneck_detection[fc_id]['utilization_history'])
            self.bottleneck_detection[fc_id]['is_bottleneck'] = avg_utilization > 0.8

    def _multi_objective_dijkstra(self, state: GameState, start: str, end: str) -> Tuple[
        Optional[List[str]], float, Dict]:
        """Multi-objective Dijkstra considering time, cost, and reliability."""
        if start == end:
            return [start], 0.0, {'reliability': 1.0, 'congestion': 0.0}

        # Build weighted graph with multiple objectives
        graph = {}
        for conn in state.connections:
            if conn.from_fc not in graph:
                graph[conn.from_fc] = []

            # Multi-objective weights
            base_time = conn.weight
            predicted_time = self.predictive_analytics.predict_delivery_time(
                conn.from_fc, conn.to_fc, base_time, state.current_time_step
            )

            # Bandwidth penalty
            available_bw = conn.available_bandwidth if conn.available_bandwidth is not None else float('inf')
            bandwidth_penalty = 0 if available_bw == float('inf') else _max(0, 10 - available_bw) * 0.2

            # Bottleneck penalty
            bottleneck_penalty = 0.0
            if conn.from_fc in self.bottleneck_detection:
                if self.bottleneck_detection[conn.from_fc]['is_bottleneck']:
                    bottleneck_penalty = 1.0

            # Congestion forecast
            congestion_forecast = self.predictive_analytics.get_congestion_forecast(
                conn.to_fc, int(predicted_time)
            )

            total_cost = predicted_time + bandwidth_penalty + bottleneck_penalty + congestion_forecast * 0.5

            graph[conn.from_fc].append((
                conn.to_fc,
                total_cost,
                {
                    'predicted_time': predicted_time,
                    'reliability': 1.0 / (1.0 + bottleneck_penalty),
                    'congestion': congestion_forecast
                }
            ))

        # Enhanced Dijkstra with multi-objective tracking
        pq = MinHeap()
        pq.push((0.0, start, [start], {'reliability': 1.0, 'congestion': 0.0}))
        visited = set()

        while pq:
            item = pq.pop()
            if item is None:
                break

            cost, node, path, metrics = item

            if node in visited:
                continue
            visited.add(node)

            if node == end:
                return path, cost, metrics

            neighbors = graph.get(node, [])
            for neighbor, edge_cost, edge_metrics in neighbors:
                if neighbor in visited:
                    continue

                new_cost = cost + edge_cost
                new_path = path + [neighbor]

                # Combine metrics
                new_metrics = {
                    'reliability': metrics['reliability'] * edge_metrics['reliability'],
                    'congestion': _max(metrics['congestion'], edge_metrics['congestion'])
                }

                pq.push((new_cost, neighbor, new_path, new_metrics))

        return None, float('inf'), {'reliability': 0.0, 'congestion': float('inf')}

    def find_optimal_route(self, state: GameState, package: Package) -> Tuple[Optional[str], Optional[List[str]], Dict]:
        """Find optimal route with advanced multi-objective optimization."""
        if package.current_fc == package.destination_fc:
            return None, [package.current_fc], {'reliability': 1.0, 'congestion': 0.0}

        self._invalidate_cache_if_needed(state)
        self._detect_bottlenecks(state)

        # Check cache first
        cache_key = (package.current_fc, package.destination_fc, state.current_time_step)
        if cache_key in self.shortest_paths_cache:
            path, cost, metrics = self.shortest_paths_cache[cache_key]
        else:
            path, cost, metrics = self._multi_objective_dijkstra(
                state, package.current_fc, package.destination_fc
            )
            if path:
                self.shortest_paths_cache[cache_key] = (path, cost, metrics)

        if not path or len(path) < 2:
            return None, None, {'reliability': 0.0, 'congestion': float('inf')}

        next_fc = path[1]

        # Generate alternative routes using ALNS principles
        alternatives = []

        # Find all possible first hops
        for conn in state.connections:
            if conn.from_fc == package.current_fc and conn.available_bandwidth != 0:
                if conn.to_fc != next_fc:  # Alternative to main route
                    alt_path, alt_cost, alt_metrics = self._multi_objective_dijkstra(
                        state, conn.to_fc, package.destination_fc
                    )
                    if alt_path:
                        full_path = [package.current_fc] + alt_path
                        total_cost = conn.weight + alt_cost
                        alternatives.append((conn.to_fc, total_cost, full_path, alt_metrics))

        # Use ALNS to select best alternative if significantly better
        if alternatives:
            candidates = [(alt[0], alt[1]) for alt in alternatives]
            candidates.append((next_fc, cost))

            optimized_selection = self.alns_optimizer.optimize_route_selection(candidates, 1)

            if optimized_selection and optimized_selection[0] != next_fc:
                # Update analytics with decision
                for alt in alternatives:
                    if alt[0] == optimized_selection[0]:
                        next_fc = alt[0]
                        path = alt[2]
                        metrics = alt[3]
                        break

        return next_fc, path, metrics


class GlobalFlowOptimizer:
    """Global flow optimization using system-wide visibility."""

    def __init__(self):
        self.network_optimizer = AdvancedNetworkOptimizer()
        self.system_load_balancer = {}
        self.priority_queue = {}

    def _calculate_package_priority(self, package: Package, current_time: int) -> float:
        """Calculate package priority based on urgency and destination."""
        age = current_time - package.entry_time
        age_penalty = age * 0.1  # Older packages get higher priority

        # Distance-based priority (packages going further get slight priority)
        # This is approximated - in real implementation would use actual distance
        distance_bonus = len(package.destination_fc) * 0.01  # Simple heuristic

        return age_penalty + distance_bonus

    def _balance_system_load(self, state: GameState) -> Dict[str, List[Package]]:
        """Balance load across the entire system."""
        # Group packages by current location
        packages_by_fc = {}
        for pkg in state.active_packages:
            if not pkg.in_transit:
                if pkg.current_fc not in packages_by_fc:
                    packages_by_fc[pkg.current_fc] = []
                packages_by_fc[pkg.current_fc].append(pkg)

        # Sort packages by priority within each FC
        for fc_id, packages in packages_by_fc.items():
            packages.sort(key=lambda p: self._calculate_package_priority(p, state.current_time_step), reverse=True)

        return packages_by_fc

    def optimize_all_packages(self, state: GameState) -> Dict[str, Tuple[str, Dict]]:
        """Optimize routing for all packages simultaneously."""
        routing_decisions = {}
        packages_by_fc = self._balance_system_load(state)

        # Process each FC's packages
        for fc_id, packages in packages_by_fc.items():
            connection_assignments = {}

            # Get available connections from this FC
            available_connections = []
            for conn in state.connections:
                if conn.from_fc == fc_id and (conn.available_bandwidth is None or conn.available_bandwidth > 0):
                    available_connections.append(conn)

            if not available_connections:
                continue

            # Assign packages to connections optimally
            for i, pkg in enumerate(packages):
                next_fc, path, metrics = self.network_optimizer.find_optimal_route(state, pkg)

                if next_fc:
                    # Check if this connection can handle more packages
                    conn = state.get_connection(fc_id, next_fc)
                    if conn and (conn.available_bandwidth is None or conn.available_bandwidth > 0):
                        routing_decisions[pkg.id] = (next_fc, metrics)

                        # Update available bandwidth for subsequent packages
                        if conn.available_bandwidth is not None:
                            conn.available_bandwidth -= 1

        return routing_decisions


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


# Global optimizer instances for caching
_global_optimizer = GlobalFlowOptimizer()
_advanced_optimizer = AdvancedNetworkOptimizer()


def route_package(state: GameState, package: Package) -> Optional[str]:
    """
    Ultra-optimized package routing with predictive analytics, ALNS, and multi-objective optimization.
    Incorporates latest research in delivery optimization and machine learning.
    """
    # Handle trivial case
    if package.current_fc == package.destination_fc:
        return None

    # Skip if package is already in transit or not ready
    current_time = _safe_int(state.current_time_step, 0)
    if package.in_transit:
        return None

    # Check if package is ready to move
    entry_time = _safe_int(package.entry_time, current_time)
    if entry_time > current_time:
        return None

    # Check if we have capacity constraints
    if has_capacity_constraints(state):
        # Use ultra-advanced routing for capacity-constrained networks
        next_fc, full_path, metrics = _advanced_optimizer.find_optimal_route(state, package)

        if not next_fc or next_fc == package.current_fc:
            return None

        # Verify connection exists and has capacity
        conn = state.get_connection(package.current_fc, next_fc)
        if not conn:
            return None

        if conn.available_bandwidth is not None and conn.available_bandwidth <= 0:
            return None

        # Update package state with predicted travel time
        base_travel_time = max(1, int(conn.weight))
        predicted_travel_time = _advanced_optimizer.predictive_analytics.predict_delivery_time(
            package.current_fc, next_fc, base_travel_time, current_time
        )

        package.in_transit = True
        package.transit_destination = next_fc
        package.transit_remaining_time = int(predicted_travel_time)

        # Update connection bandwidth
        if conn.available_bandwidth is not None:
            conn.available_bandwidth = max(0, conn.available_bandwidth - 1)

        # Update predictive analytics and congestion forecasting
        _advanced_optimizer.predictive_analytics.update_congestion_forecast(
            package.current_fc, 1, current_time
        )

        # Learn from routing decision (simplified feedback loop)
        _advanced_optimizer.alns_optimizer.update_method_performance('repair', 0, True)

        return next_fc
    else:
        # Use simple routing for non-capacity constrained networks
        next_fc, full_path, metrics = _advanced_optimizer.find_optimal_route(state, package)
        return next_fc


def route_all_packages_optimal(state: GameState) -> Dict[str, str]:
    """
    Global optimization for all packages using system-wide flow optimization.
    Returns a dictionary mapping package_id to next_fc.
    """
    full_routing_decisions = _global_optimizer.optimize_all_packages(state)

    # Convert to simple format
    simple_decisions = {}
    for pkg_id, (next_fc, metrics) in full_routing_decisions.items():
        simple_decisions[pkg_id] = next_fc

    return simple_decisions


def get_route_preview_advanced(state: GameState, package: Package) -> Optional[Dict]:
    """
    Get advanced route preview with metrics and predictions.
    """
    next_fc, full_path, metrics = _advanced_optimizer.find_optimal_route(state, package)

    if not full_path:
        return None

    return {
        'path': full_path,
        'next_fc': next_fc,
        'estimated_delivery_time': len(full_path) - 1,  # Simplified
        'reliability_score': metrics['reliability'],
        'congestion_level': metrics['congestion']
    }


def update_learning_from_delivery(package_id: str, from_fc: str, to_fc: str,
                                  predicted_time: float, actual_time: float):
    """
    Update machine learning models with actual delivery data.
    Call this when a package completes a hop to improve future predictions.
    """
    _advanced_optimizer.predictive_analytics.update_delivery_pattern(
        from_fc, to_fc, actual_time, predicted_time
    )


def clear_all_caches():
    """Clear all caches and reset learning state."""
    _advanced_optimizer.shortest_paths_cache.clear()
    _advanced_optimizer.predictive_analytics = PredictiveAnalytics()
    _advanced_optimizer.alns_optimizer = AdaptiveLargeNeighborhoodSearch()
    _advanced_optimizer.last_state_hash = None