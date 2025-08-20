"""
Amazon Robotics Hackathon - TestCase Model

This module defines the TestCase class for the Amazon Robotics Hackathon.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from ar_hackathon.models.fulfillment_center import FulfillmentCenter
from ar_hackathon.models.connection import Connection
from ar_hackathon.models.package import Package


@dataclass
class WeightChange:
    """Represents a change in connection weight at a specific time."""
    from_fc: str
    to_fc: str
    new_weight: float
    duration: Optional[int] = None  # None means permanent


class TestCase:
    """
    Represents a test case from an input JSON file.
    
    This class models the structure of the input file according to the schema.
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize a TestCase from a dictionary loaded from a JSON file.
        
        Args:
            data: Dictionary containing the test case data
        """
        # Metadata
        self.metadata = data.get("metadata", {})
        self.max_time_steps = self.metadata.get("max_time_steps", 100)
        self.description = self.metadata.get("description", "")
        
        # Fulfillment centers - keep as a simple list matching the schema
        self.fulfillment_centers = []
        for fc_data in data.get("fulfillment_centers", []):
            fc = FulfillmentCenter(
                fc_id=fc_data["id"],
                name=fc_data.get("name", fc_data["id"])
            )
            self.fulfillment_centers.append(fc)
        
        # Connections - keep as a simple list matching the schema
        self.connections = []
        for conn_data in data.get("connections", []):
            conn = Connection(
                from_fc=conn_data["from_fc"],
                to_fc=conn_data["to_fc"],
                weight=conn_data["base_weight"],
                bandwidth=conn_data.get("bandwidth")
            )
            self.connections.append(conn)
        
        # Weight changes - store in a map indexed by time_step
        # Each entry is a list of WeightChange objects
        self.weight_changes_by_time = {}
        for conn_data in data.get("connections", []):
            for change in conn_data.get("weight_changes", []):
                weight_change = WeightChange(
                    from_fc=conn_data["from_fc"],
                    to_fc=conn_data["to_fc"],
                    new_weight=change["new_weight"],
                    duration=change.get("duration")
                )
                
                time_step = change["time_step"]
                if time_step not in self.weight_changes_by_time:
                    self.weight_changes_by_time[time_step] = []
                
                self.weight_changes_by_time[time_step].append(weight_change)
        
        # Packages - store in a map indexed by entry_time
        # Each entry is a list of Package objects
        self.packages_by_time = {}
        for pkg_data in data.get("packages", []):
            pkg = Package(
                package_id=pkg_data["id"],
                current_fc=pkg_data["source_fc"],
                destination_fc=pkg_data["destination_fc"],
                entry_time=pkg_data["entry_time"]
            )
            
            if pkg.entry_time not in self.packages_by_time:
                self.packages_by_time[pkg.entry_time] = []
            
            self.packages_by_time[pkg.entry_time].append(pkg)
    
    def num_remaining_packages(self, current_time: int) -> int:
        """
        Calculate the number of packages that have not yet entered the system at the given time.
        
        Args:
            current_time: The current time step in the simulation
            
        Returns:
            The total count of packages scheduled to enter after the current time step
        """
        remaining_count = 0
        
        # Iterate through all time steps and their associated packages
        for entry_time, packages in self.packages_by_time.items():
            # Only count packages with entry times in the future
            if entry_time > current_time:
                remaining_count += len(packages)
                
        return remaining_count
