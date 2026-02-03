"""
	AV@R 
	created: 2026, January
    author©: Rajmadan Lakshmanan
"""

using Dates
using Statistics, LinearAlgebra
using DataFrames
using YFinance
using Optim
using Gnuplot

# If you run on a server without display, keep false and save PNGs.
Gnuplot.options.gpviewer = false

# ----------------------------
# Download data & build returns
# ----------------------------
function get_adjclose(sym::String; startdt="2019-01-02", enddt="2020-12-31")
    # autoadjust=true -> "close" is adjusted
    od = get_prices(sym; startdt=startdt, enddt=enddt, interval="1d", autoadjust=true)
    df = DataFrame(od)
    select!(df, [:timestamp, :close])
    rename!(df, :close => Symbol(sym))
    return df
end

function aligned_logreturns(tickers::Vector{String}; startdt="2019-01-02", enddt="2020-12-31")
    @show dfs = [get_adjclose(s; startdt=startdt, enddt=enddt) for s in tickers]
    df = reduce((a,b)->innerjoin(a,b,on=:timestamp), dfs)
    P  = Matrix(select(df, Not(:timestamp)))              # T×d prices
    R  = log.(P[2:end,:] ./ P[1:end-1,:])                 # (T-1)×d log returns
    return df.timestamp[2:end], Array{Float64}(R)
end

# ----------------------------
# Bernstein quantile via De Casteljau
# ----------------------------
function Bernstein(p::Float64, X::Vector{Float64})
    Y = copy(X)
    for i in length(Y)-1:-1:1
        @inbounds for j in 1:i
            Y[j] = (1-p)*Y[j] + p*Y[j+1]
        end
    end
    return Y[1]
end

# ----------------------------
# Risk functionals on a sample of losses
# ----------------------------

# Classical empirical AVaR (a.k.a CVaR / ES)
# Uses the standard "min over eta" formula but evaluated directly.
function CVaR_empirical(alpha::Float64, L::Vector{Float64})
    # minimize over eta: eta + (1/((1-alpha)T)) * sum(max(L-eta,0))
    # For 1D, the minimizer eta is VaR_alpha (a quantile).
    # We'll compute it by sorting; then average tail beyond VaR.
    X = sort(L)
    T = length(X)
    k = clamp(ceil(Int, alpha*T), 1, T)   # index of empirical quantile
    η = X[k]
    tail = @view X[k:T]
    return mean(tail)   # average tail loss
end

# Bernstein-smoothed CVaR:
# BCVaR_alpha(L) = (1/(1-alpha)) * ∫_{alpha}^1 Q_B(p) dp
# where Q_B(p) is Bernstein smoothing of the empirical quantile function.
function BCVaR(alpha::Float64, L::Vector{Float64}; m_int::Int=200)
    X = sort(L)
    ps = range(alpha, 1.0; length=m_int)
    vals = Vector{Float64}(undef, length(ps))
    @inbounds for (i,p) in enumerate(ps)
        vals[i] = Bernstein(p, X)
    end
    # trapezoidal integration
    integ = 0.0
    @inbounds for i in 1:length(ps)-1
        integ += 0.5*(vals[i] + vals[i+1])*(ps[i+1] - ps[i])
    end
    return integ / (1 - alpha)
end

# ----------------------------
# Portfolio mapping + constraints
# ----------------------------

# Long-only and sum-to-1 via softmax parametrization:
# u ∈ R^d -> w = softmax(u), ensures w_i>=0 and sum(w)=1 automatically.
function softmax(u::Vector{Float64})
    # stabilize: subtract max
    umax = maximum(u)
    ex = exp.(u .- umax)
    return ex / sum(ex)
end

# Compute losses from returns matrix R (T×d) and weights w:
# loss L_t = -(r_t ⋅ w)
@inline function losses(R::Matrix{Float64}, w::Vector{Float64})
    return -(R * w)
end

# ----------------------------
# 5) Optimization for each alpha
# ----------------------------

# Objective choice: :cvar or :bcvar
function optimize_weights(R::Matrix{Float64}, alpha::Float64;
                          objective::Symbol=:bcvar,
                          m_int::Int=200,
                          maxiters::Int=800)

    d = size(R,2)
    u0 = zeros(d)  # starts at equal weights

    obj(u) = begin
        w = softmax(u)
        L = losses(R, w)
        if objective == :cvar
            return CVaR_empirical(alpha, L)
        elseif objective == :bcvar
            return BCVaR(alpha, L; m_int=m_int)
        else
            error("objective must be :cvar or :bcvar")
        end
    end

    # NelderMead is robust and avoids gradient complications (sorting makes gradients messy).
    res = optimize(obj, u0, NelderMead(), Optim.Options(iterations = maxiters))

    uopt = Optim.minimizer(res)
    wopt = softmax(uopt)
    fopt = Optim.minimum(res)
    return wopt, fopt
end

# ----------------------------
# 6) Experiment: weights vs alpha (and risk vs alpha)
# ----------------------------

tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX"]
startdt = "2019-01-02"
enddt   = "2020-12-31"

println("Downloading data...")
dates, R = aligned_logreturns(tickers; startdt=startdt, enddt=enddt)
T, d = size(R)
println("Built returns: T=$T scenarios, d=$d assets")

#alphas = collect(0.1:0.1:0.9)
alphas = collect(0.9:0.01:0.99)

W_cvar  = zeros(length(alphas), d)
W_bcvar = zeros(length(alphas), d)
J_cvar  = zeros(length(alphas))
J_bcvar = zeros(length(alphas))

# Integration resolution for BCVaR:
m_int = 200

for (i, a) in enumerate(alphas)
    println("alpha = ", a)

    w1, j1 = optimize_weights(R, a; objective=:cvar,  maxiters=600)
    w2, j2 = optimize_weights(R, a; objective=:bcvar, m_int=m_int, maxiters=600)

    W_cvar[i,:]  .= w1
    W_bcvar[i,:] .= w2
    J_cvar[i]    = j1
    J_bcvar[i]   = j2
end

# ----------------------------
# 7) Plot in Julia with Gnuplot
# ----------------------------

# # Plot A: weights vs alpha (CVaR)
# @gp :P1 "clear"
# @gp :- :P1 "set title \"Weights vs alpha (classical CVaR)\"" :-
# @gp :- :P1 "set xlabel \"alpha\"" "set ylabel \"weight\"" "set border 1" :-
# @gp :- :P1 "set key outside"

# for j in 1:d
#     @gp :- :P1 alphas W_cvar[:,j] "w l lw 2 title \"$(tickers[j])\""
# end

# W_bcvar is size (length(alphas), d) with weights that sum to 1 in each row
# tickers is length d






# Wpct = 100 .* W_bcvar
# Cum  = cumsum(Wpct, dims=2)
# d    = size(W_bcvar, 2)

# outbase = "bernstein_allocation"   # produces bernstein_allocation.tex and .pdf

# @gp :P2 "clear"
# @gp :- :P2 "set term cairolatex pdf input color size 5in,3.3in"
# @gp :- :P2 "set output '$(outbase).tex'"

# @gp :- :P2 "set title 'Weights vs alpha (Bernstein smoothed)'"
# @gp :- :P2 "set xlabel 'alpha'"
# @gp :- :P2 "set ylabel 'portfolio allocation (\\%)'"
# @gp :- :P2 "set border 1"
# @gp :- :P2 "set grid"
# @gp :- :P2 "set key outside"
# @gp :- :P2 "set yrange [0:100]"
# @gp :- :P2 "set style fill solid 0.20 border"
# @gp :- :P2 "set tics out"

# # first layer
# @gp :- :P2 alphas Cum[:,1] "w filledcurves y1=0 title '$(tickers[1])'"

# # remaining layers
# for j in 2:d
#     lower = Cum[:, j-1]
#     upper = Cum[:, j]
#     @gp :- :P2 alphas upper lower "w filledcurves title '$(tickers[j])'"
# end


# Gnuplot.save(:P2, "bernstein_allocation.tex";
#     term="cairolatex pdf input color dashed size 5in,3.3in"
# )


outbase = "bernstein_riskVsAlpha2"   # produces bernstein_riskVsAlpha.tex and .pdf

# Plot C: risk value vs alpha (compare)
@gp :P3 "clear"
@gp :- :P3 "set term cairolatex pdf input color size 5in,3.3in"
@gp :- :P3 "set output '$(outbase).tex'"

@gp :- :P3 "set title \"Risk vs alpha (CVaR vs Bernstein-CVaR)\"" :-
@gp :- :P3 "set xlabel \"alpha\"" "set ylabel \"risk (loss units)\"" "set border 1" :-
@gp :- :P3 "set border 1"
@gp :- :P3 "set grid"
#@gp :- :P3 "set key outside"
#@gp :- :P3 "set style fill solid 0.20 border"
@gp :- :P3 "set key off"          # <<< legend removed
@gp :- :P3 "set tics out"



# @gp :- :P3 alphas J_cvar  "w l lw 2 title \"CVaR\""
# @gp :- :P3 alphas J_bcvar "w l lw 2 dt 2 title \"Bernstein-CVaR\""


@gp :- :P3 alphas J_cvar  "w l ls 1 notitle"
@gp :- :P3 alphas J_bcvar "w l ls 2 notitle"

 Gnuplot.save(:P3, "bernstein_riskVsAlpha2.tex";
     term="cairolatex pdf input color dashed size 5in,3.3in"
 )
