#!/usr/bin/env python3
"""
Amazon Robotics Hackathon - Run Game Script

This script runs the game with a specified test case and routing algorithm.
"""

import argparse
import sys
import os
from ar_hackathon.engine.game_engine import GameEngine
from ar_hackathon.api.routing import route_package


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Run the Amazon Robotics Hackathon game.')
    parser.add_argument('test_case', help='Path to the test case JSON file')
    parser.add_argument('--router', default='default', choices=['default', 'basic'],
                        help='Routing algorithm to use (default: default)')
    parser.add_argument('--step-by-step', action='store_true',
                        help='Run the game step by step (for debugging)')
    args = parser.parse_args()
    
    # Check if the test case file exists
    if not os.path.isfile(args.test_case):
        print(f"Error: Test case file '{args.test_case}' not found.")
        sys.exit(1)
    
    # Select the routing algorithm
    router = route_package  # Default router (student implementation)
    if args.router == 'basic':
        from ar_hackathon.examples.basic_router import basic_router
        router = basic_router
    
    # Run the game
    print(f"Running game with test case: {args.test_case}")
    print(f"Using router: {args.router}")
    
    engine = GameEngine(args.test_case, router)
    if args.step_by_step:
        is_finished = False
        
        while not is_finished:
            game_state, is_finished = engine.step()
            print(f"Time step: {game_state.current_time_step}")
            print(f"Active packages: {len(game_state.active_packages)}")
            print(f"Delivered packages: {len(game_state.delivered_packages)}")
            
            if not is_finished:
                input("Press Enter to continue to the next step...")
        
        score = engine.stats
    else:
        score = engine.run_until_finished()
    
    # Print the results
    print("\nGame Results:")
    print(f"Score: {score['score']}")
    print(f"Delivered {score['delivered_packages']} out of {score['total_packages']} packages "
          f"({score['delivery_percentage']:.2f}%)")
    print(f"Average delivery time: {score['average_delivery_time']:.2f} time steps")
    print(f"Total time steps: {score['total_time_steps']}")


if __name__ == '__main__':
    main()
