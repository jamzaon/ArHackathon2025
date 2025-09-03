"""
Amazon Robotics Hackathon - Base Visualizer

This module defines the abstract base class for visualizers.
"""

from abc import ABC, abstractmethod
from typing import Any, List

from ar_hackathon.models.game_state import GameState


class BaseVisualizer(ABC):
    """
    Abstract base class for game visualizers.
    
    This class defines the interface that all visualizer implementations must follow.
    """
    
    @abstractmethod
    def create_frame(self, game_state: GameState, frame_number: int) -> Any:
        """
        Create a visualization frame for the current game state.
        
        Args:
            game_state: Current game state
            frame_number: Frame number
            
        Returns:
            A frame object representing the visualization (type depends on implementation)
        """
        pass
    
    @abstractmethod
    def save_frame(self, frame: Any, output_dir: str, frame_number: int, 
                  save_html: bool, save_images: bool) -> None:
        """
        Save a frame to disk.
        
        Args:
            frame: The frame object returned by create_frame
            output_dir: Directory to save to
            frame_number: Frame number
            save_html: Whether to save as HTML
            save_images: Whether to save as image
        """
        pass
    
    @abstractmethod
    def create_animation(self, frames: List[Any], output_dir: str) -> None:
        """
        Create an animation from a list of frames.
        
        Args:
            frames: List of frame objects
            output_dir: Directory to save to
        """
        pass
