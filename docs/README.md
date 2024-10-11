# Hawk-Dove Model Simulation

## Overview

This simulation implements the Hawk-Dove game, a classic model in evolutionary game theory. The model explores the dynamics of conflict and cooperation in a population where individuals can adopt one of two strategies: Hawk (aggressive) or Dove (peaceful).

## How to Use

1. Ensure you have Python and the required libraries (Mesa, Matplotlib, NumPy, Pandas) installed.
2. Navigate to the `hawk-and-dove/src` directory.
3. Run the script using the command: `python main.py`
4. A web browser will open, displaying the simulation interface.
5. Adjust the parameters using the sliders on the left side of the interface.
6. Set a random seed value for reproducible results.
7. Click "Start" to run the simulation, and "Stop" to pause it.
8. Observe the grid, charts, and data table to analyze the results.

## Random Seed

The simulation includes a random seed option. This feature is crucial for:

1. Reproducibility: By using the same seed, you can recreate identical simulation runs.
2. Comparative Analysis: When changing parameters, using the same seed allows you to isolate the effects of the parameter changes from the randomness in the model.
3. Debugging: A fixed seed helps in identifying and fixing issues by providing consistent behavior.

To use the random seed:

1. Enter a number in the "Random Seed" input field before starting the simulation.
2. Use the same seed value to reproduce the exact same initial conditions and random events.
3. Change the seed to explore how different random initializations affect the outcomes.

## Purpose of the Simulation

The main purposes of this simulation are:

1. To demonstrate how different strategies (aggressive vs. peaceful) can coexist and evolve in a population.
2. To explore the concept of Evolutionary Stable Strategy (ESS) in the context of resource competition.
3. To visualize the dynamics of population growth, resource distribution, and strategy adoption over time.
4. To understand the impact of various parameters on the evolution of strategies.

## What Can Be Inferred from the Results

By running and analyzing this simulation, users can infer:

1. The relative success of Hawk and Dove strategies under different environmental conditions.
2. How the initial ratio of Hawks to Doves affects the long-term population dynamics.
3. The impact of resource availability and distribution on strategy success.
4. The role of factors such as injury cost, reproduction threshold, and maximum agent resources in shaping the population.
5. Emergent patterns of spatial distribution and resource exploitation.
6. The sensitivity of the model to initial conditions and random fluctuations (by varying the random seed).

## Learning Outcomes

Users of this simulation will learn:

1. Basic concepts of evolutionary game theory and the Hawk-Dove model.
2. How to interpret and analyze complex system dynamics through data visualization.
3. The importance of parameter tuning in agent-based models.
4. How small changes in initial conditions or rules can lead to significant differences in outcomes (sensitivity to initial conditions).
5. The concept of equilibrium in population dynamics and how it can be achieved or disrupted.
6. The trade-offs between aggressive (Hawk) and peaceful (Dove) strategies in resource competition.
7. How to think critically about the assumptions and limitations of simplified models of complex systems.
8. The importance of reproducibility in scientific simulations and how to achieve it using random seeds.

## Key Features of the Simulation

1. Multi-agent system with Hawks and Doves competing for resources.
2. Dynamic resource generation and consumption.
3. Spatial component allowing for local interactions and resource distribution patterns.
4. Reproduction mechanism influenced by resource availability and population density.
5. Metabolism and aging mechanics for agents.
6. Interactive web-based interface for easy parameter adjustment and real-time visualization.
7. Data collection and visualization of key metrics over time.
8. Random seed option for reproducible experiments and comparative analysis.

By exploring this simulation, users can gain insights into the complex dynamics of competition and cooperation in biological and social systems, and develop a deeper understanding of how simple rules can lead to complex, emergent behaviors in populations. The addition of the random seed feature allows for more rigorous scientific exploration and comparison of different parameter sets.
