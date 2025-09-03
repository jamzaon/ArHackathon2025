# Amazon Robotics Hackathon

This package contains the code for the Amazon Robotics Hackathon, a coding competition for undergraduate students.

## Overview

In this hackathon, students will implement an algorithm to route packages through a network of fulfillment centers (FCs). The goal is to deliver packages from their source FC to their destination FC as efficiently as possible.

## Project Structure

```
ar_hackathon/
├── ar_hackathon/                  # Main package directory
│   ├── models/                    # Data models
│   │   ├── fulfillment_center.py  # FulfillmentCenter class
│   │   ├── connection.py          # Connection class
│   │   ├── package.py             # Package class
│   │   ├── game_state.py          # GameState class
│   │   └── test_case.py           # TestCase class
│   ├── engine/                    # Game engine components
│   │   └── game_engine.py         # Main game engine implementation
│   ├── api/                       # API for students
│   │   └── routing.py             # Contains route_package function signature
│   ├── utils/                     # Utility functions
│   │   ├── json_loader.py         # Functions to load test cases
│   │   └── routing_utils.py       # Routing utility functions
│   ├── visualizers/               # Visualization components
│   │   ├── base_visualizer.py     # Base visualizer class
│   │   ├── network_visualizer.py  # Network visualization implementation
│   │   └── visualizer_factory.py  # Factory for creating visualizers
│   ├── examples/                  # Example implementations
│   │   └── basic_router.py        # Basic routing implementation
│   └── simulation_runner.py       # Simulation runner
├── test_cases/                    # Test case JSON files
│   ├── schema.json                # JSON schema for test cases
│   ├── level1/                    # Level 1 test cases
│   │   ├── test_case_1.json       # Level 1 test case 1
│   │   └── test_case_2.json       # Level 1 test case 2
│   ├── level2/                    # Level 2 test cases
│   │   └── test_case_3.json       # Level 2 test case
│   └── level3/                    # Level 3 test cases
│       └── test_case_4.json       # Level 3 test case
├── scripts/                       # Utility scripts
│   ├── run_game.py                # Script to run the game
│   └── visualize.py               # Visualization script
├── setup.py                       # Package setup and dependencies
├── submit.py                      # Submission script
├── team.json                      # Team information
└── README.md                      # This file
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
```

## Set up
```bash
# Install the package
pip install -e .
```

## Usage

### Running the Game

```bash
# Run the game with the default router
python scripts/run_game.py test_cases/level1/test_case_1.json

# Run the game with the basic router
python scripts/run_game.py test_cases/level1/test_case_1.json --router basic
```

### Visualizing the Game

#### Pre-Requisite

Kaleido requires Google Chrome to be installed. If Chrome is not installed, a static visualization can still be viewed by accessing frames in `visualization_output/` directly.

#### Running the Visualizer

Visualization output will be saved to the `visualization_output/` directory.

Example:

```bash
# Visualize the game with the default router
python scripts/visualize.py test_cases/level1/test_case_1.json

# Visualize the game with the basic router
python scripts/visualize.py test_cases/level1/test_case_1.json --router basic
```

## Implementing Your Own Router

To implement your own routing algorithm, modify the `route_package` function in `ar_hackathon/api/routing.py`:

```python
def route_package(state: GameState, package: Package) -> Optional[str]:
    """
    Determine the next FC to route a package to.
    
    Args:
        state: GameState object containing the current state of the network
        package: Package object containing information about the package
        
    Returns:
        next_fc_id: ID of the next FC to route the package to, or None to stay at current FC
    """
    # Your implementation here
    pass
```

## Difficulty Levels

1. **Level 1**: Fulfilment Centers are connected by **roads of the same length** (Simple weighted graph)
2. **Level 2**: Fulfilment Centers are connected by **roads of different lengths** (Weighted graph)
3. **Level 3**: Fulfilment Centers are connected by **roads of different lengths that have a limited capacity on how many packages can flow through the road at any given time** (Weighted graph with bandwidth)

## Scoring

Implementations are scored based on:
- Percentage of packages successfully delivered to the correct destination
- Average delivery time for all deliveries

## Submission
Update the `team.json` file to contains your or your team's name and then run `python submit.py` to submit your implementation