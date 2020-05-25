# Optimisation of Overparametrized Sum-Product Networks

This repository contains everything you need to reproduce the paper on "Overparameterization in Sum-Product Networks".

To run the experiments,

1. Install a running version of Julia
2. Add the SumProductNetworks package using `pkg> add SumProductNetworks`
3. Add the following additionally packages: `Statistics, Plots, PGFPlots`
4. Run the script using `julia experiment.jl` and wait for a while. Sorry for the slow code. :D

## UPDATE
This repository now also contains an updated version of the experiments in the paper.
The updated implementation uses a bijection to optimise the parameters in an unconstrainted space, reducing artifacts in the optimisation and providing more stable results.
To run the updated script, make sure to use the provided `Project.toml` script to instantiate the julia environment. This is done by starting julia (version 1.3 or higher) inside the repository directory using `julia --project=.`  and calling `] instantiate` within the REPL.

To run the experiments, run the `tpm2019.jl` script.

Updated plots:
![nltcs](https://user-images.githubusercontent.com/7974003/82798312-a3d30080-9e78-11ea-96bb-dd16df977d1e.png)
![plants](https://user-images.githubusercontent.com/7974003/82798428-ccf39100-9e78-11ea-9fce-2eb234f007a2.png)
![audio](https://user-images.githubusercontent.com/7974003/82798301-9f0e4c80-9e78-11ea-9353-b08a6859a67f.png)


#### Dependencies (old script)

The following version for each package has been used.

```
Julia v1.1.0
SumProductNetworks v0.1.2
Plots v0.24.0
PGFPlots v3.0.3
```

New version may work too.

#### Citation
Martin Trapp, Robert Peharz, and Franz Pernkopf. "Optimisation of Overparametrized Sum-Product Networks", 3rd Workshop of Tractable Probabilistic Modeling at the International Conference on Machine Learning, 2019.
