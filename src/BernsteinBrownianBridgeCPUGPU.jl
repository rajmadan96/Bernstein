using Random, Distributions
using CUDA
using DelimitedFiles

include("BernsteinGPU.jl")

# ----------------------------
# CPU De Casteljau 
function Bernstein(p::Float64, X::Vector{Float64})
    Y = copy(X)
    for i in length(Y)-1:-1:1
        @inbounds for j in 1:i
            Y[j] = (1-p)*Y[j] + p*Y[j+1]
        end
    end
    return Y[1]
end

# CPU grid evaluation for all p in R
function bernstein_cpu_grid!(out::Vector{Float64}, R::Vector{Float64}, X::Vector{Float64})
    @inbounds for ip in eachindex(R)
        out[ip] = Bernstein(R[ip], X)
    end
    return out
end

# ----------------------------
# GPU log-stable kernel (assumes you already defined bernstein_gpu_logstable!)
# If not, include it here, or:
# include("bernstein_gpu_logstable.jl")

# ----------------------------
# Setup
m  = 1000
R  = collect(range(0.1, 0.9, m))
R_d = CuArray(R)

Dist = Normal()

Dcdf = Vector{Float64}(undef, m)
Dpdf = Vector{Float64}(undef, m)
for (i,p) in enumerate(R)
    Dcdf[i] = quantile(Dist, p)
    Dpdf[i] = pdf(Dist, Dcdf[i])
end

env = sqrt.(R .* (1 .- R))  # ± envelope for Brownian bridge plot

# output folder
outdir = joinpath(homedir(), "bb_gp1_data")
isdir(outdir) || mkpath(outdir)
println("Writing GP1 data to: ", outdir)

# buffers
B_gpu = Vector{Float64}(undef, m)
B_cpu = Vector{Float64}(undef, m)

B_d = CUDA.zeros(Float64, m)   # reuse GPU output buffer

ns = [10, 100, 1_000, 10_000, 100_000, 1000_000]

for n in ns
    t = @elapsed begin
        # sample + sort on CPU
        X = rand(Dist, n); sort!(X)

        # ---------------- GPU GP1
        X_d = CuArray(X)
        bernstein_gpu_logstable!(B_d, R_d, X_d)
        CUDA.synchronize()
        B_gpu .= Array(B_d)

        bridge_gpu = sqrt(n+1) .* (B_gpu .- Dcdf) .* Dpdf
        file_gpu = joinpath(outdir, "GP1_GPU_n$(n).csv")
        writedlm(file_gpu, hcat(R, env, -env, bridge_gpu), ',')

        # ---------------- CPU GP1
        # bernstein_cpu_grid!(B_cpu, R, X)
        # bridge_cpu = sqrt(n+1) .* (B_cpu .- Dcdf) .* Dpdf
        # file_cpu = joinpath(outdir, "GP1_CPU_n$(n).csv")
        # writedlm(file_cpu, hcat(R, env, -env, bridge_cpu), ',')
    end

    @show n, t
end
