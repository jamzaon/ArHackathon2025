#!/usr/bin/env python3
"""
Amazon Robotics Hackathon - Visualization Script

This script visualizes the game state and package movements.
"""

import argparse
import sys
import os
from ar_hackathon.simulation_runner import SimulationRunner
from ar_hackathon.visualizers.visualizer_factory import VisualizerFactory


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Visualize the Amazon Robotics Hackathon game.')
    parser.add_argument('test_case', help='Path to the test case JSON file')
    parser.add_argument('--router', default='default', choices=['default', 'basic'],
                        help='Routing algorithm to use (default: default)')
    parser.add_argument('--output-dir', default='./visualization_output',
                        help='Directory to save visualization outputs')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to generate')
    parser.add_argument('--format', default='both', choices=['html', 'images', 'both'],
                        help='Output format (default: both)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode (step by step)')
    parser.add_argument('--visualizer', default='plotly', choices=['plotly', 'bokeh', 'network'],
                        help='Visualization backend to use (default: plotly)')
    args = parser.parse_args()
    
    # Check if the test case file exists
    if not os.path.isfile(args.test_case):
        print(f"Error: Test case file '{args.test_case}' not found.")
        sys.exit(1)
    
    # Create the simulation runner
    print(f"Visualizing game with test case: {args.test_case}")
    print(f"Using router: {args.router}")
    
    runner = SimulationRunner(args.test_case, args.router)
    
    if args.interactive:
        # Run the simulation step by step
        engine = runner.engine
        engine.reset()
        is_finished = False
        
        while not is_finished:
            game_state, is_finished = engine.step()
            print(f"Time step: {game_state.current_time_step}")
            print(f"Active packages: {len(game_state.active_packages)}")
            print(f"Delivered packages: {len(game_state.delivered_packages)}")
            
            if not is_finished:
                input("Press Enter to continue to the next step...")
        
        print("\nGame Results:")
        print(f"Delivered {engine.stats['delivered_packages']} out of {engine.stats['total_packages']} packages "
              f"({engine.stats['delivery_percentage']:.2f}%)")
        print(f"Average delivery time: {engine.stats['average_delivery_time']:.2f} time steps")
        print(f"Total time steps: {engine.stats['total_time_steps']}")
    else:
        # Create the visualizer
        visualizer = VisualizerFactory.create_visualizer(args.visualizer)
        
        # Calculate the layout first for visualizers that support it
        if args.visualizer == 'plotly':
            from ar_hackathon.visualizers.plotly_visualizer import PlotlyVisualizer
            if isinstance(visualizer, PlotlyVisualizer):
                visualizer.calculate_layout(
                    runner.engine.test_case.fulfillment_centers,
                    runner.engine.test_case.connections
                )
        elif args.visualizer == 'bokeh':
            from ar_hackathon.visualizers.bokeh_visualizer import BokehVisualizer
            if isinstance(visualizer, BokehVisualizer):
                visualizer.calculate_layout(
                    runner.engine.test_case.fulfillment_centers,
                    runner.engine.test_case.connections
                )
        elif args.visualizer == 'network':
            from ar_hackathon.visualizers.network_visualizer import NetworkVisualizer
            if isinstance(visualizer, NetworkVisualizer):
                visualizer.calculate_layout(
                    runner.engine.test_case.fulfillment_centers,
                    runner.engine.test_case.connections
                )
        
        # Run the full simulation and generate visualization
        runner.run_simulation(
            visualizer,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            save_html=(args.format in ['html', 'both']),
            save_images=(args.format in ['images', 'both'])
        )
        
        print(f"Visualization complete. Output saved to {args.output_dir}")
        print(f"Animation file: {os.path.join(args.output_dir, 'animation.html')}")


if __name__ == '__main__':
    main()
