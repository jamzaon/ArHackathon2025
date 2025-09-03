"""
Amazon Robotics Hackathon - Simulation Runner

This module implements the simulation runner for the Amazon Robotics Hackathon game.
"""

import os
from typing import List, Optional, Any, Callable

from ar_hackathon.engine.game_engine import GameEngine
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package
from ar_hackathon.visualizers.base_visualizer import BaseVisualizer


class SimulationRunner:
    """
    Simulation runner for the Amazon Robotics Hackathon.
    
    This class handles running the simulation and using a visualizer to create frames.
    """
    
    def __init__(self, test_case_path: str, router: str = 'default'):
        """
        Initialize the simulation runner.
        
        Args:
            test_case_path: Path to the JSON test case file
            router: Routing algorithm to use ('default' or 'basic')
        """
        # Select the routing algorithm
        if router == 'basic':
            from ar_hackathon.examples.basic_router import basic_router
            self.router = basic_router
        else:
            from ar_hackathon.api.routing import route_package
            self.router = route_package
        
        # Create the game engine
        self.engine = GameEngine(test_case_path, self.router)
    
    def run_simulation(self, visualizer: BaseVisualizer, output_dir: Optional[str] = None, 
                      max_frames: Optional[int] = None,
                      save_html: bool = True, 
                      save_images: bool = True) -> List[Any]:
        """
        Run the simulation and use the provided visualizer to create frames.
        
        Args:
            visualizer: An instance of a visualizer that implements the BaseVisualizer interface
            output_dir: Directory to save visualization outputs
            max_frames: Maximum number of frames to generate (None for all)
            save_html: Whether to save frames as HTML files
            save_images: Whether to save frames as image files
            
        Returns:
            List of frame objects representing each frame
        """
        # Reset the game engine
        self.engine.reset()
        frames = []
        
        # Create output directory if needed
        if output_dir and (save_html or save_images):
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate initial frame
        frame = visualizer.create_frame(self.engine.game_state, 0)
        frames.append(frame)
        
        if output_dir:
            visualizer.save_frame(frame, output_dir, 0, save_html, save_images)
        
        # Run the simulation step by step
        frame_number = 1
        is_finished = False
        
        while not is_finished and (max_frames is None or frame_number <= max_frames):
            # Step the game engine
            game_state, is_finished = self.engine.step()
            
            # Create and save frame
            frame = visualizer.create_frame(game_state, frame_number)
            frames.append(frame)
            
            # if output_dir:
            #     visualizer.save_frame(frame, output_dir, frame_number, save_html, save_images)
            
            frame_number += 1
        
        # Create animation from frames
        if output_dir:
            visualizer.create_animation(frames, output_dir)
        
        return frames
