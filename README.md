# Scientific_Computing_Lab
The aim of the coursework is to integrate the work you do from each week into a single piece of software. The software should be a general numerical continuation code that can track, under variations of a parameter in the system, the limit cycle oscillations of arbitrary ordinary differential equations (of any number of dimensions) and steady-states of second-order diffusive PDEs.

The softwares architecture allows for easy maniplulation of each function in each script, allowing the user to have complete usability when solving ODEs and PDEs.

### Ordinary and Potential Differential Equation Definitions (ODE/PDE) 
The odes and pdes used in this software can be found in odes.py

### Ode Solver
All functions that are used to solve an ODE are found in ode_solver.py, this script has 2 different solving methods available: forward Euler and 4th order Runge-Kutta.

### Numerical Shooting 
The shooting method can be found in shooting.py, it contains two main processes, finding limit cycles and implementing numerical shooting

### Numerical Continuation
The numerical continuation code contains two main processes for ODEs with and without limit cycles: natural parameter continuation and pseudo arc length continuation.
Here the main interface for the software is described

### PDE Solver 
The PDE solver, found in the pde_solver.py file. It utilises three methods: 1. Forward Euler. 2. Backward Euler. 3. Crank-Nicholson Where each of these methods are seperate classes defined as forwardEuler, backwardEuler and crankNicholson.

### Shooting testing 
Found in shooting_test.py


### How to run
Please look at numerical_continuation.py to view the interface. Running this script with user inputted arguments
