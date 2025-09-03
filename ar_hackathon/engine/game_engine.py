"""
Amazon Robotics Hackathon - Game Engine

This module implements the core game engine for the Amazon Robotics Hackathon.
"""

from typing import Dict, List, Optional, Any, Callable, TypeVar, Tuple
import concurrent.futures
import logging
import math
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package
from ar_hackathon.models.test_case import TestCase
from ar_hackathon.utils.json_loader import load_test_case
from ar_hackathon.utils.routing_utils import is_valid_move

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameEngine:
    """
    Game engine for the Amazon Robotics Hackathon.
    
    This class encapsulates the game simulation logic, allowing step-by-step
    execution or running the entire simulation at once.
    """
    
    def __init__(self, test_case_path: str, player_algorithm: Callable[[GameState, Package], Optional[str]]):
        """
        Initialize the game engine with a test case and player algorithm.
        
        Args:
            test_case_path: Path to the JSON test case file
            player_algorithm: Function implementing the routing algorithm
        """
        self.test_case_path = test_case_path
        self.player_algorithm = player_algorithm
        self.test_case = load_test_case(test_case_path)
        self.game_state = self._initialize_game_state()
        self.is_finished = False
        self.stats = {}
    
    def _initialize_game_state(self) -> GameState:
        """Initialize game state from the test case."""
        return GameState(
            current_time_step=0,
            fulfillment_centers=self.test_case.fulfillment_centers,
            connections=self.test_case.connections,
            active_packages=[]
        )
    
    def reset(self) -> None:
        """Reset the game to its initial state."""
        self.game_state = self._initialize_game_state()
        self.is_finished = False
        self.stats = {}
    
    def step(self) -> Tuple[GameState, bool]:
        """
        Advance the game by one time step.
        
        Returns:
            game_state: The updated game state
            is_finished: Whether the game is finished
        """
        if self.is_finished:
            return self.game_state, True
        
        # 1. Process new packages entering the system
        self._process_new_packages()
        
        # 2. For each active package that's not in transit, call the player's algorithm
        self._route_packages()
        
        # 3. Advance packages in transit
        self._advance_packages_in_transit()
        
        # 4. Check for delivered packages
        self._check_for_delivered_packages()
        
        # 5. Update connection weights based on time
        self._update_connection_weights()
        
        # 6. Advance time step
        self.game_state.current_time_step += 1
        
        # 7. Check if game is over
        self.is_finished = self._is_game_over()
        
        # 8. Calculate stats if game is over
        if self.is_finished:
            self.stats = self._calculate_score()
        
        return self.game_state, self.is_finished
    
    def run_until_finished(self) -> Dict[str, Any]:
        """
        Run the game until it's finished.
        
        Returns:
            stats: Dictionary containing the final score and statistics
        """
        while not self.is_finished:
            self.step()
        
        return self.stats
    
    def _is_game_over(self) -> bool:
        """Check if the game is over."""
        # Check if there are no more active packages
        if self.test_case.num_remaining_packages(self.game_state.current_time_step) == 0 and not self.game_state.active_packages:
            return True
        
        # Check if the maximum time steps have been reached
        return self.game_state.current_time_step >= self.test_case.max_time_steps
    
    def _process_new_packages(self) -> None:
        """Process packages that enter the system at the current time step."""
        # Get packages entering at the current time step
        new_packages = self.test_case.packages_by_time.get(self.game_state.current_time_step, [])
        
        # Add new packages to active packages
        self.game_state.active_packages.extend(new_packages)
    
    def _route_packages(self) -> None:
        """Route packages using the player's algorithm."""
        for package in self.game_state.active_packages:
            if not package.in_transit:
                next_fc = safe_execute(
                    self.player_algorithm,
                    self.game_state.deep_copy(),
                    package.deep_copy(),
                    timeout_seconds=1,
                    default_return_value=None  # Stay at current FC if timeout/exception
                )
                
                if next_fc and is_valid_move(self.game_state, package, next_fc):
                    self._move_package(package, next_fc)
    
    def _move_package(self, package: Package, next_fc: str) -> bool:
        """Move a package from its current FC to the next FC."""
        # Find the connection between the current FC and the next FC
        connection = self.game_state.get_connection(package.current_fc, next_fc)
        
        # If no connection exists, the move is invalid
        if connection is None:
            return False
        
        # Check if the connection has available bandwidth
        if connection.bandwidth is not None:
            if connection.available_bandwidth <= 0:
                return False
            # Decrease available bandwidth
            connection.available_bandwidth -= 1
        
        # Update package transit information
        package.in_transit = True
        package.transit_destination = next_fc
        package.transit_remaining_time = connection.weight
        
        return True
    
    def _advance_packages_in_transit(self) -> None:
        """Advance all packages that are in transit."""
        for package in self.game_state.active_packages:
            if package.in_transit:
                # Decrement remaining time
                package.transit_remaining_time -= 1
                
                # Check if the package has arrived
                if package.transit_remaining_time <= 0:
                    # Find the connection that was used
                    connection = self.game_state.get_connection(package.current_fc, package.transit_destination)
                    
                    # Update package location
                    package.current_fc = package.transit_destination
                    
                    # Reset transit data
                    package.in_transit = False
                    package.transit_destination = None
                    package.transit_remaining_time = 0
                    
                    # Increase available bandwidth if applicable
                    if connection and connection.bandwidth is not None:
                        connection.available_bandwidth += 1
    
    def _check_for_delivered_packages(self) -> None:
        """Check for packages that have reached their destination."""
        delivered = []

        for package in self.game_state.active_packages:
            if package.current_fc == package.destination_fc:
                package.delivery_time = self.game_state.current_time_step
                delivered.append(package)

        for package in delivered:
            self.game_state.active_packages.remove(package)
            self.game_state.delivered_packages.append(package)
    
    def _update_connection_weights(self) -> None:
        """Update connection weights based on time-based changes."""
        # Get weight changes at the current time step
        weight_changes = self.test_case.weight_changes_by_time.get(self.game_state.current_time_step, [])
        
        # Apply weight changes
        for weight_change in weight_changes:
            connection = self.game_state.get_connection(weight_change.from_fc, weight_change.to_fc)
            if connection:
                connection.weight = weight_change.new_weight
    
    def _calculate_score(self) -> Dict[str, Any]:
        """
        Calculate the final score based on delivered packages.
        
        Uses a per-package scoring scheme where each package gets points based on
        how quickly it was delivered. The total score is normalized to a 0-100 scale.
        """
        # Calculate basic stats
        delivered_packages = len(self.game_state.delivered_packages)
        total_packages = delivered_packages + len(self.game_state.active_packages)
        
        # Base score for delivering a package
        base_points = 100
        
        # Calculate score for each package
        total_score = 0
        total_delivery_time = 0
        
        for pkg in self.game_state.delivered_packages:
            # Calculate delivery duration
            delivery_duration = pkg.delivery_time - pkg.entry_time
            total_delivery_time += delivery_duration
            
            # Calculate time efficiency factor (decreases as delivery time increases)
            # Using an exponential decay function: e^(-delivery_duration/50)
            # This gives diminishing returns for faster deliveries
            time_factor = math.exp(-delivery_duration / 50)
            
            # Calculate package score
            package_score = base_points * time_factor
            total_score += package_score
        
        # Calculate average delivery time
        average_delivery_time = (total_delivery_time / delivered_packages) if delivered_packages > 0 else 0
        delivery_percentage = (delivered_packages / total_packages * 100) if total_packages > 0 else 0
        
        # Normalize the score (0-100)
        # Maximum possible score would be base_points * total_packages (if all delivered instantly)
        max_possible_score = base_points * total_packages
        normalized_score = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        return {
            "score": normalized_score,
            "raw_score": total_score,
            "delivered_packages": delivered_packages,
            "total_packages": total_packages,
            "delivery_percentage": delivery_percentage,
            "average_delivery_time": average_delivery_time,
            "total_time_steps": self.game_state.current_time_step
        }


T = TypeVar('T')
def safe_execute(func: Callable[..., T], *args: Any, timeout_seconds: int = 10, 
                default_return_value: Optional[Any] = None, **kwargs: Any) -> Optional[T]:
    """
    Execute a function safely with timeout and exception handling.
    
    Args:
        func: The function to execute
        args: Positional arguments to pass to the function
        timeout_seconds: Maximum execution time in seconds
        default_return_value: Value to return if the function times out or raises an exception
        kwargs: Keyword arguments to pass to the function
        
    Returns:
        The function's return value, or default_return_value if an exception occurs
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            return default_return_value
        except Exception as e:
            logger.warning(f"Function {func.__name__} raised an exception: {str(e)}")
            return default_return_value
