using Zygote # AD
using ProgressMeter # show progress bar
using DelimitedFiles # CSV reader
using Statistics, Distributions

# directories
datadir = "datasets"
resultdir = "results"

for dataset in ["nltcs", "plants", "baudio"]

  # read data
  x = readdlm(joinpath(datadir, "$(dataset).ts.data"), ',', Bool);
  xtest = readdlm(joinpath(datadir, "$(dataset).test.data"), ',', Bool);

  N, D = size(x)

  # number of components
  K = 8

  # initialise parameters
  function init(;s = 0.1)
    w = log.(rand(truncated(Normal(0, s), 0, Inf), K))
    θ0 = log.(rand(truncated(Normal(0, s), 0, Inf), D, K))
    θ1 = log.(rand(truncated(Normal(0, s), 0, Inf), D, K))
    return (θ0, θ1, w)
  end

  # functions for shallow spn
  function f(w1, w2, w3)
    r = sum(d -> log.(x[:,d] * w1[d,:]' + .!x[:,d] * w2[d,:]'), 1:D)
    return log.(exp.(r)*w3)
  end
  function ftest(w1, w2, w3)
    r = sum(d -> log.(xtest[:,d] * w1[d,:]' + .!xtest[:,d] * w2[d,:]'), 1:D)
    return log.(exp.(r)*w3)
  end
  z(w1, w2, w3) = first(prod(w1+w2,dims=1)*w3)

  likelihood(w1, w2, w3) = mean(f(w1, w2, w3) .- z(w1, w2, w3))
  llh(p1, p2, p3) = likelihood(exp.(p1), exp.(p2), exp.(p3))
  llhtest(p1, p2, p3) = mean(ftest(exp.(p1), exp.(p2), exp.(p3)) .- z(exp.(p1), exp.(p2), exp.(p3)))

  function buildw(ws...)
    if length(ws) == 1
      return first(ws)
    else
      w1, w2 = ws
      K = length(w1)
      Ks = length(w2)
      Cs = Int(K / Ks)
      return buildw(log.(reshape(reshape(exp.(w1), Cs, Ks) .* exp.(w2)', K)), ws[3:end]...)
    end
  end

  function llh(p1, p2, ps...)
    return llh(p1, p2, buildw(ps...) )
  end

  function llhtest(p1, p2, ps...)
    return llhtest(p1, p2, buildw(ps...) )
  end


  # -- Training -- #
  function train(p1, p2, p3; iterations = 10_000, η = 0.1)

    performance = zeros(iterations)
    performance_test = zeros(iterations)
    @showprogress 1 "Training..." for i in 1:iterations

      performance[i] = llh(p1, p2, p3)
      performance_test[i] = llhtest(p1, p2, p3)
      grad = gradient(Params([p1, p2, p3])) do
          llh(p1, p2, p3)
      end
      p1 += η * grad[p1]
      p2 += η * grad[p2]
      p3 += η * grad[p3]
    end
    return performance, performance_test
  end

  function train(p1, p2, p3, p4; iterations = 10_000, η = 0.1)

    performance = zeros(iterations)
    performance_test = zeros(iterations)
    @showprogress 1 "Training..." for i in 1:iterations
      performance[i] = llh(p1, p2, p3, p4)
      performance_test[i] = llhtest(p1, p2, p3, p4)
      grad = gradient(Params([p1, p2, p3, p4])) do
          llh(p1, p2, p3, p4)
      end

      p1 += η * grad[p1]
      p2 += η * grad[p2]
      p3 += η * grad[p3]
      p4 += η * grad[p4]
    end
    return performance, performance_test
  end

  function train(p1, p2, p3, p4, p5; iterations = 10_000, η = 0.1)

    performance = zeros(iterations)
    performance_test = zeros(iterations)
    @showprogress 1 "Training..." for i in 1:iterations
      performance[i] = llh(p1, p2, p3, p4, p5)
      performance_test[i] = llhtest(p1, p2, p3, p4, p5)
      grad = gradient(Params([p1, p2, p3, p4, p5])) do
          llh(p1, p2, p3, p4, p5)
      end

      p1 += η * grad[p1]
      p2 += η * grad[p2]
      p3 += η * grad[p3]
      p4 += η * grad[p4]
      p5 += η * grad[p5]
    end
    return performance, performance_test
  end

  # Experiment
  mkpath(resultdir)

  iterations = 1_000
  s = 0.1

  for run in 1:5
    θ0, θ1, w = init(;s = s)
    rtrain, rtest = train(θ0, θ1, w; iterations = iterations)

    # save results
    open(joinpath(resultdir, "$(dataset)_shallow_train.csv"), "a") do io
        writedlm(io, rtrain', ',')
    end
    open(joinpath(resultdir, "$(dataset)_shallow_test.csv"), "a") do io
        writedlm(io, rtest', ',')
    end

    θ0, θ1, w = init(;s = s)
    w1 = rand(Normal(0, s), 2)
    rtrain, rtest = train(θ0, θ1, w, w1; iterations = iterations)

    # save results
    open(joinpath(resultdir, "$(dataset)_deep1_train.csv"), "a") do io
        writedlm(io, rtrain', ',')
    end
    open(joinpath(resultdir, "$(dataset)_deep1_test.csv"), "a") do io
        writedlm(io, rtest', ',')
    end

    θ0, θ1, w = init(;s = s)
    w1 = rand(Normal(0, s), 4)
    w2 = rand(Normal(0, s), 2)
    rtrain, rtest = train(θ0, θ1, w, w1, w2; iterations = iterations)

    # save results
    open(joinpath(resultdir, "$(dataset)_deep2_train.csv"), "a") do io
        writedlm(io, rtrain', ',')
    end
    open(joinpath(resultdir, "$(dataset)_deep2_test.csv"), "a") do io
        writedlm(io, rtest', ',')
    end
  end
end
