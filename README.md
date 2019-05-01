# Overparameterization in Sum-Product Networks

This repository contains everything you need to reproduce the paper on "Overparameterization in Sum-Product Networks".

To run the experiments,

1. Install a running version of Julia
2. Add the SumProductNetworks package using `pkg> add SumProductNetworks`
3. Add the following additionally packages: `Statistics, Plots, PGFPlots`
4. Start Julia from the command line at the location of this repo and start jupyter, e.g.

	```
	using IJulia
	notebook(dir=pwd())
	```

5. Run the script using `julia experiment.jl` and wait for a while. Sorry for the slow code. :D

#### Dependencies

The following version for each package has been used.

```
Julia v1.1.0
SumProductNetworks v0.1.2
Plots v0.24.0
PGFPlots v3.0.3
```

New version may work too.
