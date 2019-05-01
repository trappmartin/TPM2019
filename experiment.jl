# Loading packages and modules.
using Statistics, SumProductNetworks, Plots, Random, DelimitedFiles

# Load NLTCS data.
data = readdlm("data/nltcs.ts.data", ',', Int);

##################################################################
# Generate SPN structure for experiment i.e., random partitions. #
##################################################################

"""
    build_product(D, dims, l, maxL)

Parameters:
* D::Int = Dimensionality of the data set.
* dims::Int[] = Scope of the new product node.
* l::Int = Current depth.
* maxL::Int = Maximum depth of network.

Return:
* node::FiniteProductNode = A new product node.

"""
function build_product(D::Int, dims::Vector{Int}, l::Int, maxL::Int)

    node = FiniteProductNode(D = D)
    setscope!(node, dims)

    if l >= maxL
        for d in dims
            S = build_sum(D, [d], l, maxL)
            add!(node, S)
        end
    elseif length(dims) > 2
        partition = rand(0:1, length(dims))
        while length(unique(partition)) != 2
            partition = rand(0:1, length(dims))
        end

        S0 = build_sum(D, dims[partition .== 0], l, maxL)
        add!(node, S0)

        S1 = build_sum(D, dims[partition .== 1], l, maxL)
        add!(node, S1)
    else
        S0 = build_sum(dims, l, maxL)
        add!(node, S0)
    end

    return node
end

"""
    build_sum(D, dims, l, maxL)

Parameters:
* D::Int = Dimensionality of the data set.
* dims::Int[] = Scope of the new product node.
* l::Int = Current depth.
* maxL::Int = Maximum depth of network.

Return:
* node::FiniteSumNode = A new sum node.

"""
function build_sum(D::Int, dims::Vector{Int}, l::Int, maxL::Int)

    node = FiniteSumNode(D = D)
    setscope!(node, dims)

    if l < maxL
        P0 = build_product(D, dims, l+1, maxL)
        add!(node, P0, log(1/2))

        P1 = build_product(D, dims, l+1, maxL)
        add!(node, P1, log(1/2))
    else
        @assert length(dims) == 1
        add!(node, IndicatorNode(0, first(dims)), log(1/2))
        add!(node, IndicatorNode(1, first(dims)), log(1/2))
    end

    return node
end

##################################################################################
# Compute the gradients of the log-likelihood w.r.t. the weights of the network. #
##################################################################################

"""
    ∇SPN(spn, llhvals, zllhval)

Parameters:
* spn::SumProductNetwork = The SPN to be diverentiated.
* llhvals::AxisArray = An array containing all LLH values i.e., for each node and observation.
* zllhval::AxisArray = An array containing the partition function value for each node.

Return:
* ∇w::Dict = Dictionary containing gradients for each sum nodes in `spn`.

"""
function ∇SPN(spn, llhvals, zllhval)

    id(node) = node.id

    nodes = values(spn)
    root = spn.root

    ∇S = Dict{Symbol,Vector{Float64}}()
    ∇Sz = Dict{Symbol,Float64}()
    ∇w = Dict{Symbol,Vector{Float64}}()

    ∇S[root.id] = log.(ones(N))
    ∇Sz[root.id] = log(1.0)

    for node in reverse(nodes)

        if node isa SumNode
            ∇wS = zeros(length(node))
            lw = logweights(node)
            for k in 1:length(node)
                # Propagate gradient.
                ∇S[node[k].id] = lw[k] .+ ∇S[node.id]
                ∇Sz[node[k].id] = lw[k] + ∇Sz[node.id]

                # Compute partial derivatives.
                ∇wSkx = ∇S[node.id] .+ llhvals[:,node[k].id] .- llhvals[:,root.id]
                ∇wSkz = ∇Sz[node.id] + zllhval[node[k].id] - zllhval[root.id]

                ∇wS[k] = mean(exp.(∇wSkx) .- exp(∇wSkz))
            end
            ∇w[node.id] = ∇wS
        elseif node isa ProductNode
            for k in 1:length(node)
                # Propagate gradient.
                cids = id.(node[setdiff(1:length(node), k)])
                ∇S[node[k].id] = ∇S[node.id] .+ vec(sum(llhvals[:,cids], dims = 2))
                ∇Sz[node[k].id] = ∇Sz[node.id] + sum(zllhval[cids])
            end
        else
            continue
        end
    end

    return ∇w
end

######################
# Actual Experiment. #
######################

# Number of groups.
K = 8

# Learning rate.
η = 0.01

# Number of iterations.
maxiter = 500

# Number of re-runs.
runs = 2

# Train llh values for shallow SPN.
lp_shallow = Matrix{Float64}(undef, runs, maxiter)

# Train llh values for deep SPN.
lp_deep = Vector{Matrix{Float64}}(undef, 2)
lp_deep[1] = Matrix{Float64}(undef, runs, maxiter)
lp_deep[2] = Matrix{Float64}(undef, runs, maxiter)

###########################
# Shallow SPN Experiment. #
###########################

(N, D) = size(data);

for run in 1:runs

    # Build shallow SPN.
    root = FiniteSumNode(D = D)
    setscope!(root, collect(1:D))

    for k in 1:K
        # Build product -> sum -> indicator structure.
        n = build_product(D, collect(1:D), 1, 1)
        add!(root, n, log(1e-4))
    end

    spn = SumProductNetwork(root)

    nodes = values(spn);
    snodes = filter(n -> n isa FiniteSumNode, nodes)

    for iteration in 1:maxiter

        # Compute predictions.
        llhvals = initllhvals(spn, data)
        for node in nodes
            logpdf!(node, data, llhvals)
        end

        # Compute normalization constants.
        zllhval = initllhvals(spn, ones(D) * NaN)
        for node in nodes
            logpdf!(node, ones(D) * NaN, zllhval)
        end

        # Compute normalization constant at root.
        z = zllhval[spn.root.id]

        # Evaluate SPN at current iteration
        lp_shallow[run, iteration] = mean(llhvals[:,spn.root.id] .- z)

        # Compute gradients.
        ∇w = ∇SPN(spn, llhvals, zllhval)

        # Update parameters.
        for snode in snodes
            wt = weights(snode) .+ η*∇w[snode.id]
            wt = map(wk -> max(wk, 1e-4), wt) # Clipping so that we don't get negative weights.
            snode.logweights[:] .= log.(wt)
        end
    end
end

#########################
# Deep SPNs Experiments #
#########################

# Number of terminal sum nodes (parents of C_k) required.
R = Int(K / 2)

for L in [1,2]

    for run in 1:runs

        # Build deep SPN.
        root = FiniteSumNode(D = D)
        setscope!(root, collect(1:D))

        # Build terminal sum nodes.
        leafsums = Vector()
        for r in 1:R
            S = FiniteSumNode(D = D)
            setscope!(S, collect(1:D))

            K_ = Int(R / L)
            for k in 1:K_
                n = build_product(D, collect(1:D), 1, 1)
                add!(S, n, log(1e-4))
            end

            push!(leafsums, S)
        end

        # Build hierarchy
        lastnodes = Vector()
        push!(lastnodes, root)
        for l in 1:(L-1)
            newlastnodes = Vector()
            for n in lastnodes

                S1 = FiniteSumNode(D = D)
                setscope!(S1, collect(1:D))
                add!(n, S1, log(1e-4))
                push!(newlastnodes, S1)

                S2 = FiniteSumNode(D = D)
                setscope!(S2, collect(1:D))
                add!(n, S2, log(1e-4))
                push!(newlastnodes, S2)
            end
            lastnodes = newlastnodes
        end

        # Add terminal sum nodes to lowest layer of SPN hierary.
        j = 1
        for i in 1:length(lastnodes)
            n = lastnodes[i]
            c1 = leafsums[j]
            c2 = leafsums[j+1]
            add!(n, c1, log(1e-4))
            add!(n, c2, log(1e-4))
            j += 2
        end

        spn = SumProductNetwork(root)

        nodes = values(spn)
        snodes = filter(n -> n isa FiniteSumNode, nodes)

        for iteration in 1:maxiter

            # Compute predictions.
            llhvals = initllhvals(spn, data)
            for node in nodes
                logpdf!(node, data, llhvals)
            end

            # Compute normalization constant.
            zllhval = initllhvals(spn, ones(D) * NaN)
            for node in nodes
                logpdf!(node, ones(D) * NaN, zllhval)
            end

            # Compute normalization constant at root.
            z = zllhval[spn.root.id]

            # Evaluate SPN at current iteration
            lp_deep[L][run, iteration] = mean(llhvals[:,spn.root.id] .- z)

            # Compute gradients.
            ∇w = ∇SPN(spn, llhvals, zllhval)

            # Update parameters.
            for snode in snodes
                wt = weights(snode) .+ η*∇w[snode.id]
                wt = map(wk -> max(wk, 1e-4), wt)
                snode.logweights[:] .= log.(wt)
            end
        end
    end
end

#########################
# Visualize Experiment. #
#########################

# Load PGFPlots for nicer looking plots.
pgfplots()

# Plot result
plot(title = "Overparameterization of SPNs on NLTCS",
    xlabel = "iteration", ylabel = "train LLH", legend = :bottomright)

# Estimate performance over all re-runs. Max takes the best performance.
# Change to mean if you are interested in the average performance instead.

plot!(vec(maximum(lp_shallow, dims=1)), label = "L=1 SPN")
plot!(vec(maximum(lp_deep[1], dims=1)), label = "L=2 SPN")
plot!(vec(maximum(lp_deep[2], dims=1)), label = "L=3 SPN")

# Save to disk.
savefig("nltcs_experiment.pdf")
