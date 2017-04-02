## Julia implementation of a ConditionalRBM from Graham Taylor's PhD Thesis

##conditional.jl with temporal connections moved to the hidden layer. 

## Links:
##     Thesis - http://www.cs.nyu.edu/~gwtaylor/thesis/Taylor_Graham_W_200911_PhD_thesis.pdf
##     FCRBM - http://www.cs.toronto.edu/~fritz/absps/fcrbm_icml.pdf


# Input data layout for 2 batches and 3 steps of history
#
# NOTE: in the code below the history is referred to as `cond`
# since technically you can condition on any features not just
# previous time steps.
#
#      batch 1               batch 2
# |--------------------|--------------------|-----
# |  current visible   |  current visible   | ...
# |--------------------|--------------------|-----
# |  history step 1    |  history step 1    | ...
# |  history step 2    |  history step 2    | ...
# |  history step 3    |  history step 3    | ...
# |--------------------|--------------------|-----


import StatsBase: predict


@runonce type ConditionalRBM{T,V,H} <: AbstractRBM{T,V,H}
    W::Matrix{T}  # standard weights
    A::Matrix{T}  # autoregressive params (hid to hid)
    vbias::Vector{T}
    hbias::Vector{T}
    dyn_hbias::Array{T}
end

#Initialize conditionalRBM weights and biases.
function ConditionalRBM(T::Type, V::Type, H::Type,
                        n_vis::Int, n_hid::Int, n_cond::Int; sigma=0.01)
    ConditionalRBM{T,V,H}(
        map(T, rand(Normal(0, sigma), (n_hid, n_vis))),
        map(T, rand(Normal(0, sigma), (n_hid, n_cond))),
        zeros(T, n_vis),
        zeros(T, n_hid),
        zeros(T, n_hid),
    )
end

#automatically assign T as floats if type not provided
function ConditionalRBM(V::Type, H::Type,
                        n_vis::Int, n_hid::Int, n_cond::Int; sigma=0.01)
    ConditionalRBM(Float64, V, H, n_vis, n_hid, n_cond;
                   sigma=sigma)
end

#n_vis*steps = n_cond steps is the time steps and n_cond is the total number of conditional input
function ConditionalRBM(T::Type, V::Type, H::Type, n_vis::Int, n_hid::Int;
                        steps=5, sigma=0.01)
    ConditionalRBM(T, V, H, n_vis, n_hid, (n_hids * steps);
                   sigma=sigma)
end

function ConditionalRBM(V::Type, H::Type, n_vis::Int, n_hid::Int;
                        steps=5, sigma=0.01)
    ConditionalRBM(Float64, V, H, n_vis, n_hid, (n_hid * steps);
                   sigma=sigma)
end

#print the features of the ConditionalRBM
function Base.show{T,V,H}(io::IO, crbm::ConditionalRBM{T,V,H})
    n_vis = size(crbm.vbias, 1)
    n_hid = size(crbm.hbias, 1)
    n_cond = size(crbm.A, 2)
    print(io, "ConditionalRBM{$T,$V,$H}($n_vis, $n_hid, $n_cond)")
end

#split the visible matrix into current and conditional data
function split_vis(crbm::ConditionalRBM, vis::Mat)
    vis_size = length(crbm.vbias) #the number of visible units

    curr = vis[1:vis_size, :]
    cond = vis[(vis_size + 1):end, :] #the number of conditional units
    return curr, cond
end


function hidden_cond(crbm::ConditionalRBM, cond::Mat, ctx)
    n_vis = length(crbm.vbias)
    n_hid = length(crbm.hbias)
    steps = (size(cond,1) / n_vis)
    @assert steps/floor(steps) == 1.0
    rows = Int(steps*n_hid)
    columns = size(cond, 2)

    h_cond = Array(Float64, (rows, 0))
    for c=1:columns
        column = Array(Float64,(0, 1))
        for n=0:Int(steps)-1
            pattern = cond[(n*n_vis)+1:(n*n_vis)+n_vis, c]
            pattern = reshape(pattern, n_vis, 1)
#            sampler = @get_or_create(ctx, :sampler, persistent_contdiv) #assigns and returns the value of a key in the dict
            v_pos, h_pos, v_neg, h_neg = persistent_contdiv(crbm, pattern, ctx)
            column = vcat(column, h_pos)
        end
        h_cond = hcat(h_cond, column)
    end
    return h_cond
end


#bias = weights * conditional units + original bias
function dynamic_biases!(crbm::ConditionalRBM, cond::Mat)
    crbm.dyn_hbias = crbm.A * cond .+ crbm.hbias
end

#sample hiddens using dynamic temporal hidden bias
function hid_means(crbm::ConditionalRBM, vis::Mat)
    p = crbm.W * vis .+ crbm.dyn_hbias
    return logistic(p)
end

# sample visibles using dynamic temporal visible bias
function vis_means(crbm::ConditionalRBM, hid::Mat)
    p = crbm.W' * hid .+ crbm.vbias
    return logistic(p)
end


function gradient_classic{T}(crbm::ConditionalRBM{T}, X::Mat{T},
                          ctx::Dict)
    vis, cond = split_vis(crbm, X)
    sampler = @get_or_create(ctx, :sampler, persistent_contdiv) #assigns and returns the value of a key in the dict
    v_pos, h_pos, v_neg, h_neg = sampler(crbm, vis, ctx) #use persistent_contdiv to sample states
    n_obs = size(vis, 2)
    # updating weight matrix W
    dW = @get_array(ctx, :dW_buf, size(crbm.W), similar(crbm.W))
    # same as: dW = ((h_pos * v_pos') - (h_neg * v_neg')) / n_obs
    gemm!('N', 'T', T(1 / n_obs), h_neg, v_neg, T(0.0), dW)
    gemm!('N', 'T', T(1 / n_obs), h_pos, v_pos, T(-1.0), dW)

    return dW
end


function gradient_A{T}(crbm::ConditionalRBM{T}, X::Mat{T},
                          ctx::Dict)
    #to do: find way to get h_neg, h_pos, and cond
    vis, cond = split_vis(crbm, X)
    cond = hidden_cond(crbm, cond, ctx)
    println("hiddens at training: ", cond)
    sampler = @get_or_create(ctx, :sampler, persistent_contdiv) #assigns and returns the value of a key in the dict
    v_pos, h_pos, v_neg, h_neg = sampler(crbm, vis, ctx) #use persistent_contdiv to sample states
    n_obs = size(vis, 2)

    # updating hid to hid matrix A
    dA = @get_array(ctx, :dA_buf, size(crbm.A), similar(crbm.A))
    # same as: dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', T(1 / n_obs), h_neg, cond, T(0.0), dA)
    gemm!('N', 'T', T(1 / n_obs), h_pos, cond, T(-1.0), dA)

    # gradient for vbias and hbias
    ##    db = squeeze(sum(v_pos, 2) - su?m(v_neg, 2), 2) ./ n_obs what is su?m keep in case it's important
    dc = squeeze(sum(h_pos, 2) - sum(h_neg, 2), 2) ./ n_obs
    return dA, dc
end


function grad_apply_learning_rate!{T}(crbm::ConditionalRBM{T},
                                      X::Mat{T},
                                      d, ctx::Dict)
    lr = T(@get(ctx, :lr, 0.1))
    # same as: dW *= lr
    scal!(length(d), lr, d, 1)
end


function grad_apply_momentum!{T}(crbm::ConditionalRBM{T}, X::Mat{T},
                                 dW::Mat{T}, ctx::Dict)
    momentum = @get(ctx, :momentum, 0.9)
    dW_prev = @get_array(ctx, :dW_prev, size(dW), zeros(T, size(dW)))
    # same as: dW += momentum * dW_prev
    axpy!(momentum, dW_prev, dW)
end


function grad_apply_weight_decay!(rbm::ConditionalRBM,
                                     X::Mat,
                                     d::Mat,
                                     rbmt::Mat, ctx::Dict)
    # The decay penalty should drive all weights toward
    # zero by some small amount on each update.
    decay_kind = @get_or_return(ctx, :weight_decay_kind, nothing)
    decay_rate = @get(ctx, :weight_decay_rate,
                      throw(ArgumentError("If using :weight_decay_kind, weight_decay_rate should also be specified")))
    is_l2 = @get(ctx, :l2, false)
    if decay_kind == :l2
        # same as: dW -= decay_rate * W
        axpy!(-decay_rate, rbmt, d)
    elseif decay_kind == :l1
        # same as: dW -= decay_rate * sign(W)
        axpy!(-decay_rate, sign(rbmt), d)
    end
end

#the sparity penalty drives the units towards a target level of sparseness
function grad_apply_sparsity!{T}(rbm::ConditionalRBM{T}, X::Mat,
                                 d, ctx::Dict)
    # The sparsity constraint should only drive the weights
    # down when the mean activation of hidden units is higher
    # than the expected (hence why it isn't squared or the abs())
    cost = @get_or_return(ctx, :sparsity_cost, nothing)
    vis, cond = split_vis(rbm, X)
    target = @get(ctx, :sparsity_target,
                  throw(ArgumentError("If :sparsity_cost is used, :sparsity_target should also be defined")))
    curr_sparsity = mean(hid_means(rbm, vis))
    penalty = T(cost * (curr_sparsity - target))
    add!(d, -penalty)
end


function update_weights!(crbm::ConditionalRBM, dW::Mat, ctx::Dict)
    axpy!(1.0, dW, crbm.W)
    # save previous dW
    dW_prev = @get_array(ctx, :dW_prev, size(dW), similar(dW))
    copy!(dW_prev, dW)
end

function update_A!(crbm::ConditionalRBM, dA::Mat, dc::Any, ctx::Dict)
    axpy!(1.0, dA, crbm.A)
    crbm.hbias += dc
end


function update_classic!(crbm::ConditionalRBM, X::Mat,
                            dW::Mat, ctx::Dict)
    # apply gradient updaters. note, that updaters all have
    # the same signature and are thus composable
    grad_apply_learning_rate!(crbm, X, dW, ctx)
    grad_apply_momentum!(crbm, X, dW, ctx)
    grad_apply_weight_decay!(crbm, X, dW, crbm.W, ctx)
    grad_apply_sparsity!(crbm, X, dW, ctx)
    # add gradient to the weight matrix
    update_weights!(crbm, dW, ctx)
end

function update_classic_A!(crbm::ConditionalRBM, X::Mat,
                            dA::Mat, dc::Any, ctx::Dict)
    grad_apply_learning_rate!(crbm, X, dA, ctx)
    grad_apply_learning_rate!(crbm, X, dc, ctx)
    grad_apply_weight_decay!(crbm, X, dA, crbm.A, ctx)
    grad_apply_sparsity!(crbm, X, dA, ctx)
    grad_apply_sparsity!(crbm, X, dc, ctx)
    # add gradient to the weight matrix
    update_A!(crbm, dA, dc, ctx)
end

function free_energy(crbm::ConditionalRBM, vis::Mat)
#    vb = sum(vis .* crbm.W, 1)
    Wx_b_log = sum(log(1 + exp(crbm.W * vis .+ crbm.dyn_hbias)), 1)
    result = - Wx_b_log
    tofinite!(result)
    return result
end


function fit_batch!(crbm::ConditionalRBM, X::Mat, ctx = Dict())
    curr, cond = split_vis(crbm, X) #split current and conditional
    cond = hidden_cond(crbm, cond, ctx)
    println("hiddens at bias updates: ", cond)
    dynamic_biases!(crbm, cond) #calculate dynamic biases
    dW = gradient_classic(crbm, X, ctx) #calculate gradient
    update_classic!(crbm, X, dW, ctx)
    dA, dc = gradient_A(crbm, X, ctx)
    update_classic_A!(crbm, X, dA, dc, ctx)
    return crbm
end


function fit{T}(crbm::ConditionalRBM{T}, X::Mat, opts::Dict{Any,Any})
    @assert minimum(X) >= 0 && maximum(X) <= 1
    ctx = copy(opts)
    n_examples = size(X, 2)
    batch_size = @get(ctx, :batch_size, 1) #batch = the number of patterns being trained at once
    batch_idxs = split_evenly(n_examples, batch_size)
    if @get(ctx, :randomize, false)
        batch_idxs = sample(batch_idxs, length(batch_idxs); replace=false)
    end
    n_epochs = @get(ctx, :n_epochs, 100) #ten training epochs
    scorer = @get_or_create(ctx, :scorer, pseudo_likelihood)
    reporter = @get_or_create(ctx, :reporter, TextReporter())
    for epoch=1:n_epochs
        epoch_time = @elapsed begin
            for (batch_start, batch_end) in batch_idxs
                # BLAS.gemm! can't handle sparse matrices, so cheaper
                # to make it dense here
                batch = full(X[:, batch_start:batch_end])
                batch = ensure_type(T, batch)
                fit_batch!(crbm, batch, ctx)
            end
        end
        curr, cond = split_vis(crbm, X)
#        cond = hidden_cond(crbm, cond, ctx)
#        dynamic_biases!(crbm, cond) removed because it was interfering with vcat() in hidden_cond()

        # We convert to full, to avoid changing the the n_obs if
        # X is a sparse matrix
        score = scorer(crbm, full(curr))
        report(reporter, crbm, epoch, epoch_time, score)
    end

    return crbm
end

fit{T}(crbm::ConditionalRBM{T}, X::Mat; opts...) = fit(crbm, X, Dict(opts))

#calculate the current hidden unit states based on conditional data
function transform{T}(crbm::ConditionalRBM{T}, X::Mat)
    curr, cond = split_vis(crbm, ensure_type(T, X))
    cond = hidden_cond(crbm, cond, ctx)
    dynamic_biases!(crbm, cond)
    return hid_means(crbm, curr)
end

#generate patterns based on weights
function generate{T}(crbm::ConditionalRBM{T}, X::Mat, ctx::Dict; n_gibbs=1)
    curr, cond = split_vis(crbm, ensure_type(T, X))
    cond = hidden_cond(crbm, cond, ctx)
    println("hiddens at prediction: ", cond)
    dynamic_biases!(crbm, cond)
    return gibbs(crbm, curr; n_times=n_gibbs)[3]
end

generate{T}(crbm::ConditionalRBM{T}, vis::Vec; n_gibbs=1) =
    generate(crbm, reshape(ensure_type(T, vis), length(vis), 1); n_gibbs=n_gibbs)


function predict{T}(crbm::ConditionalRBM{T}, cond::Mat, opts::Dict{Any,Any})
    ctx = copy(opts)
    cond = ensure_type(T, cond)
    n_gibbs = @get(ctx, :n_gibbs, 20)
#    @assert size(cond, 1) == size(crbm.A, 2)

    curr = sub(cond, 1:length(crbm.vbias), :) #take part of conditional to act as next set of current
    vis = vcat(curr, cond) #add the original conditional to the set of current

    return generate(crbm, vis, ctx; n_gibbs=n_gibbs) #generate using visible
end

predict{T}(crbm::ConditionalRBM{T}, cond::Mat; opts...) = predict(crbm, cond, Dict(opts))

predict{T}(crbm::ConditionalRBM{T}, cond::Vec; n_gibbs=1) =
    predict(crbm, reshape(ensure_type(T, cond), length(cond), 1); n_gibbs=n_gibbs)


predict{T}(crbm::ConditionalRBM{T}, vis::Vec, cond::Vec{T}; n_gibbs=1) =
    predict(crbm, reshape(vis, length(vis), 1),
            reshape(ensure_type(T, cond), length(cond), 1); n_gibbs=n_gibbs)


function fit_dbn(sequence::Mat, n_vis::Int, n_hid::Int, ctx = Dict()) #for running dbns
    model = ConditionalRBM(Bernoulli, Bernoulli, n_vis, n_hid; steps=1)
#    tsteps = Int(size(sequence,1) / n_vis)
    data =  Array(Float64, ((2*n_vis), 0))
    tsteps = size(sequence, 2)
    for t=1:tsteps-1
        data1 = sequence[:, t]
        data2 = sequence[:, t+1]
        step = vcat(data2, data1)
        data = hcat(data, step)
    end
    fit_step(model, data; n_gibbs=5, sparsity_cost = 0.0, sparsity_target = 0.05)
    return model
end

function fit_sequence(sequence::Mat, n_vis::Int, n_hid::Int, n_epochs::Int, tsteps::Int)
    model = ConditionalRBM(Bernoulli, Bernoulli, n_vis, n_hid; steps=1)
#    tsteps = Int(size(sequence,1) / n_vis)
    data =  Array(Float64, ((2*n_vis), 0))
    for t=0:tsteps-2
        data1 = sequence[(t*n_vis)+1:(t*n_vis)+(n_vis), :]
        data2 = sequence[(t*n_vis)+(n_vis)+1:(t*n_vis)+(2*n_vis), :]
        step = vcat(data2, data1)
        data = hcat(data, step)
    end
    fit(model, data; n_gibbs=5, n_epochs = n_epochs, sparsity_cost = 0.0, sparsity_target = 0.05)
    return model
end

function predict_sequence(crbm::ConditionalRBM, data::Mat, tsteps::Int)
    sequence = deepcopy(data)
    for t=1:tsteps-1
        forecast = predict(crbm, data; n_gibbs=20)
        sequence = hcat(sequence, forecast)
        data = deepcopy(forecast)
    end
    return sequence
end
