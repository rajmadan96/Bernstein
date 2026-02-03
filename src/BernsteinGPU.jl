using CUDA
using BenchmarkTools
using Statistics
# CUDA.functional()   # should be true
# CUDA.versioninfo()

# CPU De Casteljau (scalar p)
function Bernstein(p, X::Vector{Float64})
    Y = copy(X)
    for i in length(Y)-1:-1:1
        for j in 1:i
            Y[j] = (1-p)*Y[j] + p*Y[j+1]
        end
    end
    return Y[1]
end


"""
    Bernstein_logsumexp(p, X)

Compute the Bernstein polynomial value at p using
a log-sum-exp stabilized O(n) algorithm.

- Numerically stable for large n
- O(n) time, O(1) extra memory
- Safe replacement for de Casteljau when n is large
"""
function Bernstein_logsumexp(p::Float64, X::Vector{Float64})
    n = length(X)

    # boundary safety
    if p <= 0.0
        return X[1]
    elseif p >= 1.0
        return X[end]
    end

    q = 1.0 - p
    lp = log(p)
    lq = log(q)

    # --- pass 1: find max log-weight ---
    # w₁ = q^(n-1)
    logw = (n - 1) * lq
    mlog = logw

    @inbounds for k = 1:n-1
        logw += log(n - k) - log(k) + lp - lq
        mlog = max(mlog, logw)
    end

    # --- pass 2: stabilized accumulation ---
    logw = (n - 1) * lq
    num = 0.0
    den = 0.0

    @inbounds begin
        w = exp(logw - mlog)
        num += w * X[1]
        den += w
    end

    @inbounds for k = 1:n-1
        logw += log(n - k) - log(k) + lp - lq
        w = exp(logw - mlog)
        num += w * X[k + 1]
        den += w
    end

    return num / den
end


# CPU: compute for all p in a grid (De Casteljau (scalar p))
# function bernstein_cpu_grid!(B::Vector{Float64}, R::Vector{Float64}, X::Vector{Float64})
#     @inbounds for ip in eachindex(R)
#         B[ip] = Bernstein(R[ip], X)
#     end
#     return B
# end

# CPU: compute for all p in a grid (logsumexp)
function bernstein_cpu_grid!(B::Vector{Float64}, R::Vector{Float64}, X::Vector{Float64})
    @inbounds for ip in eachindex(R)
        B[ip] = Bernstein_logsumexp(R[ip], X)
    end
    return B
end


function bernstein_gpu!(out::CuArray{Float64,1}, pgrid::CuArray{Float64,1}, X::CuArray{Float64,1})
    n = length(X)
    @cuda threads=256 blocks=cld(length(pgrid),256) kernel_bernstein!(out, pgrid, X, n)
    return out
end

function kernel_bernstein!(out, pgrid, X, n)
    # Compute the global index ip
    ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    # Guard for extra threads; one often launch a little more threads than needed (e.g. 1024 threads for 1000 p’s). Extra threads simply exit.
    if ip > length(pgrid); return; end 
    p = pgrid[ip]
    q = 1.0 - p

    w = q^(n-1)
    s = 0.0

    @inbounds for k = 1:n
        s += w * X[k]
        if k < n
            w *= (n - k) / k * (p / q)
        end
    end

    out[ip] = s
    return
end

function bernstein_gpu_logstable!(out::CuArray{Float64,1},
                                  pgrid::CuArray{Float64,1},
                                  X::CuArray{Float64,1})
    n = Int32(length(X))  # keep n as Int32 for GPU friendliness
    @cuda threads=256 blocks=cld(length(pgrid),256) kernel_bernstein_logstable!(out, pgrid, X, n)
    return out
end

function kernel_bernstein_logstable!(out, pgrid, X, n::Int32)
    ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if ip > length(pgrid); return; end

    p = pgrid[ip]
    q = 1.0 - p

    # guard (your grid is 0.1..0.9)
    if p <= 0.0
        out[ip] = X[1]; return
    elseif p >= 1.0
        out[ip] = X[Int(n)]; return
    end

    lp = CUDA.log(p)
    lq = CUDA.log(q)

    # logw = log w_k, starting at k=1: w1 = q^(n-1)
    logw = Float64(n - 1) * lq

    # pass 1: find maximum log-weight for stabilization
    mlog = logw
    @inbounds for k::Int32 = 1:n-1
        # logw_{k+1} = logw_k + log(n-k) - log(k) + log(p) - log(q)
        logw += CUDA.log(Float64(n - k)) - CUDA.log(Float64(k)) + lp - lq
        mlog = ifelse(logw > mlog, logw, mlog)
    end

    # pass 2: accumulate stabilized weights
    logw = Float64(n - 1) * lq
    num = 0.0
    den = 0.0

    # k = 1 term
    w = CUDA.exp(logw - mlog)
    @inbounds begin
        num += w * X[1]
        den += w
    end

    # k = 2..n terms
    @inbounds for k::Int32 = 1:n-1
        logw += CUDA.log(Float64(n - k)) - CUDA.log(Float64(k)) + lp - lq
        w = CUDA.exp(logw - mlog)
        idx = Int(k + 1)
        num += w * X[idx]
        den += w
    end

    out[ip] = num / den
    return
end



m = 1000
R_cpu = collect(range(0.1, 0.9, m))
X_cpu = sort!(randn(10_000))

B_cpu = zeros(Float64, m)

R_d = CuArray(R_cpu)
X_d = CuArray(X_cpu)
B_d = CUDA.zeros(Float64, m)

# Warm up (compile + JIT)
bernstein_cpu_grid!(B_cpu, R_cpu, X_cpu)

bernstein_gpu_logstable!(B_d, R_d, X_d)
CUDA.synchronize()   # IMPORTANT: ensure kernel finished


# CPU timing
t_cpu = @benchmark bernstein_cpu_grid!($B_cpu, $R_cpu, $X_cpu)
println("CPU median: ", median(t_cpu).time/1e6, " ms")

# GPU timing (kernel only, no host↔device copy)
t_gpu = @benchmark begin
    bernstein_gpu_logstable!($B_d, $R_d, $X_d)
    CUDA.synchronize()
end
println("GPU median (kernel only): ", median(t_gpu).time/1e6, " ms")


# --- Compute reference CPU values ---
bernstein_cpu_grid!(B_cpu, R_cpu, X_cpu)     # fills B_cpu

# --- Compute GPU values and bring back to CPU ---
#bernstein_gpu!(B_d, R_d, X_d)
#CUDA.synchronize()
#B_gpu = Array(B_d)
bernstein_gpu_logstable!(B_d, R_d, X_d)
CUDA.synchronize()
B_gpu = Array(B_d)


# --- Error diagnostics ---
abs_err = abs.(B_gpu .- B_cpu)
max_abs = maximum(abs_err)
mean_abs = mean(abs_err)

# relative error (safe: avoid divide-by-zero)
den = max.(abs.(B_cpu), eps(Float64))
rel_err = abs_err ./ den
max_rel = maximum(rel_err)
mean_rel = mean(rel_err)

println("Max absolute error:  ", max_abs)
println("Mean absolute error: ", mean_abs)
println("Max relative error:  ", max_rel)
println("Mean relative error: ", mean_rel)

# --- Optional: show where the worst error occurs ---
ip_worst = argmax(abs_err)#1000_000
println("\nWorst index ip = ", ip_worst)
println("p = ", R_cpu[ip_worst])
println("CPU value = ", B_cpu[ip_worst])
println("GPU value = ", B_gpu[ip_worst])
println("Abs error  = ", abs_err[ip_worst])
println("Rel error  = ", rel_err[ip_worst])

# bernstein_gpu_logstable!(B_d, R_d, X_d)
# CUDA.synchronize()
# B_gpu = Array(B_d)

# abs_err = abs.(B_gpu .- B_cpu)
# println("Max absolute error:  ", maximum(abs_err))
# println("Mean absolute error: ", mean(abs_err))


#For n=10,000 and m=1000 evaluation points, the GPU implementation took ~28 ms versus ~5.43 s on the CPU (≈195× speedup). Note that the GPU uses an O(n) weight-recursion formulation, while the CPU baseline used O(n²) De Casteljau.