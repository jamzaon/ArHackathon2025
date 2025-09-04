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
cd ArHackathon2025
pip install -e .
```

## Usage

### Running the Game

```bash
# Run the game with your router
python scripts/run_game.py test_cases/level1/test_case_1.json

# Run the game with the example basic router
python scripts/run_game.py test_cases/level1/test_case_1.json --router basic
```

There are more test cases you can practice with in the `test_cases` directory. When you submit your solution, it will be evaluated
against additional hidden test cases.

### Visualizing the Game

There is a visualizer included to help you develop your solution.

#### Pre-Requisite

Kaleido requires Google Chrome to be installed. If Chrome is not installed, a static visualization can still be viewed by accessing frames in `visualization_output/` directly.

#### Running the Visualizer

Visualization output will be saved to the `visualization_output/` directory.

Example:

```bash
# Visualize the game your default router
python scripts/visualize.py test_cases/level1/test_case_1.json

# Visualize the game with the example basic router
python scripts/visualize.py test_cases/level1/test_case_1.json --router basic
```

This will output an animation file at `./visualization_output/animation.html`. You can view this in your browser by running the following
and pasting the output into your browser address bar:
```bash
echo "file:///$(pwd)/visualization_output/animation.html"
```

## Instructions

### Game Mechanics

Each game run is based on a test case file that defines the FCs, the connections between them and the packages that will arrive. The game loop
then runs as follows:
1. Spawn new packages.
1. For each active package that is not in transit, call the user's `route_package` function and move the packages.
1. Advance the packages that are in transit.
1. Check for delivered packages.

### Implementing Your Own Router

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

The `GameState` and `Package` classes are defined in the `models` directory.

### Rules
1. You can only modify `routing.py`. This is the only file that will be submitted.
1. You cannot use any external libraries as these may not be installed when your submission is evaluated.
1. If your function returns an invalid move for a package, the package will not get moved.

## Difficulty Levels

1. **Level 1**: Fulfilment Centers are connected by **roads of the same length**
2. **Level 2**: Fulfilment Centers are connected by **roads of different lengths**
3. **Level 3**: Fulfilment Centers are connected by **roads of different lengths that have a limited capacity on how many packages can flow through the road at any given time**

There are test cases of different difficulty levels. When your submission is evaluated, it will be run against all difficulty levels to compute the final score.

## Scoring

Scoring for a test case is calculated as `e^(-delivery_duration/50)` for each package delivered, and is then normalized so that it is out of 100. This gives points for number of packages delivered and the speed of delivery.

The final score will be the sum of the scores from each test case.

## Submission
Update the `team.json` file to contain your or your team's name and then run `python submit.py` to submit your implementation.