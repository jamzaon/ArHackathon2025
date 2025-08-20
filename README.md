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
│   │   └── game_state.py          # GameState class
│   ├── engine/                    # Game engine components
│   │   └── game_engine.py         # Main game engine implementation
│   ├── api/                       # API for students
│   │   └── routing.py             # Contains route_package function signature
│   ├── utils/                     # Utility functions
│   │   └── json_loader.py         # Functions to load test cases
│   └── examples/                  # Example implementations
│       └── basic_router.py        # Basic routing implementation
├── test_cases/                    # Test case JSON files
│   ├── schema.json                # JSON schema for test cases
│   ├── level1/                    # Level 1 test cases
│   └── level2/                    # Level 2 test cases
└── scripts/                       # Utility scripts
    ├── run_game.py                # Script to run the game
    └── visualize.py               # Visualization script
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ar_hackathon

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

1. **Level 1**: Simple unweighted graph
2. **Level 2**: Weighted graph (connections have different lengths)
3. **Level 3**: Dynamic weights (weights can change over time)
4. **Level 4**: Bandwidth limitations (connections have limited capacity)

## Scoring

Packages are scored based on:
- Number of packages delivered
- Average delivery time
- Percentage of packages delivered
