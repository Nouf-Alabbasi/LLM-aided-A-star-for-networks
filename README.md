# LLM-aided A* for networks
[![arXiv](https://img.shields.io/badge/arXiv-2411.02617-b31b1b.svg)]()

This repository contains the source code for the paper ["<u>LLM-Aided A* Search in Non-Geometric Network Graphs</u>"]().

We propose a large language model (LLM)- aided A* algorithm in which an LLM generates intermediate waypoints that guide the A* expansion toward promising graph regions. On the solver side, we adopt the landmark- based (ALT) heuristic, which replaces geometric estimates with admissible lower bounds derived from precomputed shortest path distances to a small set of landmark nodes via the triangle inequality. On the LLM side, we inject the resulting heuristic estimates into the waypoint generation prompt, where they act as a surrogate coordinate system that restores to the model the distance-to-destination signal it loses on abstract graphs. Guided by this signal, the LLM proposes a set of intermediate waypoints, and the estimated distance to the current waypoint augments the A* evaluation function, biasing expansion toward promising regions of the graph while A* remains the underlying solver.

