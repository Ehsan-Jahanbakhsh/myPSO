# Particle Swarm Optimization Algorithm
This is a PSO Written in Numpy. Completly Pure-Function. 
## How to Use:
Entry Point is `run()` function. it takes: Function to optimized, adjacency matrix of connections, number of particles, size of dimensions, coefficients, upperbounds and lowerbounds for each dimension, and linear constraints.

NetworkX is preferred to create network topologies
## swarm size effect.ipynb
this files uses PSO to solve Knap-sack problem (1/0 and continuous variants). networkX is used to create Topologies.

## To Do
- Code can be cleaned up
- Add auto particle and dimension detection from adj_matrix
- remove extra calculation when neighbor best is turned off (c2 = 0) 