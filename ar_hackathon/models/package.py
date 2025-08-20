"""
Amazon Robotics Hackathon - Package Model

This module defines the Package class for the Amazon Robotics Hackathon.
"""


class Package:
    def __init__(self, package_id: str, current_fc: str, destination_fc: str, entry_time: int):
        self.id = package_id
        self.current_fc = current_fc
        self.destination_fc = destination_fc  # Final destination
        self.entry_time = entry_time
        self.delivery_time = None  # Will be set when the package is delivered
        
        # Transit tracking attributes
        self.in_transit = False
        self.transit_destination = None  # Immediate next FC (may differ from final destination)
        self.transit_remaining_time = 0  # Time steps remaining until arrival at next FC
    
    def __repr__(self):
        status = "in transit" if self.in_transit else f"at {self.current_fc}"
        return f"Package(id='{self.id}', {status}, dest='{self.destination_fc}')"
    
    def deep_copy(self):
        """Create a deep copy of this Package."""
        new_pkg = Package(
            package_id=self.id,
            current_fc=self.current_fc,
            destination_fc=self.destination_fc,
            entry_time=self.entry_time
        )
        new_pkg.delivery_time = self.delivery_time
        new_pkg.in_transit = self.in_transit
        new_pkg.transit_destination = self.transit_destination
        new_pkg.transit_remaining_time = self.transit_remaining_time
        return new_pkg
