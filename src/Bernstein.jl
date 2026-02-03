"""
	evaluate Bernstein’s polynomial
	created: 2025, February
    author©: Alois Pichler and Rajmadan Lakshmanan
"""

using Gnuplot; Gnuplot.options.gpviewer= true;	# external viewer
using Random, Distributions
using Revise


#BernsteinAccelerated evaluates the same Bernstein polynomial as De Casteljau,
#but collapses the control points in blocks of size Δ using binomial/Bernstein weights,
#effectively performing Δ De Casteljau steps at once.
# De Casteljau's algorithm with acceleration 
function BernsteinAccelerated(p, X::Vector{Float64}; Δ= 1020)
    Y= copy(X)  # copy to secure initial values
    # Δ= min(1000, max(1, Δ))
    i= length(Y)
    while i> 1
        if i> Δ; i= i- Δ else Δ= i-1; i= 1 end
        for j in 1:i
            t= 1.0
            for k in 1:Δ
                t*= p/ k* (Δ-k+1)
                Y[j]= (1-p)* Y[j]+ t* Y[j+k]
    end end end
    return Y[1]
end

# Bernstein(p, X)
# Evaluates the Bernstein polynomial at probability level p using the data vector X.
# If X is sorted (order statistics), the result is a smooth, distribution-free
# estimator of the p-quantile. The value is a convex (binomially weighted) average
# of all entries of X, computed via De Casteljau’s algorithm, without estimating
# densities or probabilities from data.
function Bernstein(p, X::Vector{Float64})
	Y= copy(X)  # copy to secure initial values
	for i in length(Y)-1:-1:1
		for j in 1:i
			Y[j]= (1-p)* Y[j]+ p* Y[j+1]
	end end
	return Y[1]
end

# BernsteinParallel(p, X)
# Parallel evaluation of the Bernstein polynomial at probability level p using
# De Casteljau’s algorithm. The function computes a convex (binomially weighted)
# average of the entries of X. If X is sorted (order statistics), the result is a
# smooth, distribution-free estimator of the p-quantile. The inner recursion is
# parallelized over indices j for improved performance on large vectors.

# Threads.@threads splits the loop iterations into contiguous chunks and
# executes them simultaneously on multiple CPU cores, so different values
# of j are computed in parallel, reducing wall-clock time for large loops.

# Example 4 CPU cores
#| Thread | j range    |
#| ------ | ---------- |
#| 1      | 1–2500     |
#| 2      | 2501–5000  |
#| 3      | 5001–7500  |
#| 4      | 7501–10000 |
function BernsteinParallel(p::Float64, X::Vector{Float64})
	Y= copy(X)  # copy to secure initial values
    Z= copy(Y)
    p1= 1.0 - p
	for i in length(Y)-1:-1:1
       Threads.@threads for j= 1:i 
            Z[j]= p1* Y[j]+ p* Y[j+1]
        end
        Y= Z
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


function Bernstein3(p::Float64, X::Array)
	n= length(X)
	w= zeros(Float64, n); w[1]= 1
	for i in 2:n
		w[2:i]= (1-p)*w[2:i]+p*w[1:i-1]
        w[1]*= 1-p
	end
    return w'*X
end

function Bernstein2(p::Float64, X::Array)
	n= length(X)
	w= zeros(Float64, n)
	w[n]= 1.0
	for i in n-1:(-1):1
		for j in i:n-1
			w[j]= p*w[j]+ (1-p)* w[j+1]
		end
		w[n]*= p
	end
	return w'* X
end

function Bernstein4(p::Float64, X::Array)
	n= length(X)
	w= zeros(Float64, n)
	w[1]= 1.0
	for _ in n-1:(-1):1
        t= 0.0
		for j in 1:n-1+1
            s=t; t= w[j]
			w[j]= (1-p)*w[j]+ p*s  # w[j]+= p*(s- w[j])
		end
	end
	return w'* X
end

# 	╭───────────────────────────────────────────────
# 	│	Brownian Bridge
m= 1000  # plotted points
R= range(0.1, 0.9, m)
B= Vector{Float64}(undef, m) # will hold the Bernstein-quantile estimate 
Bp= Vector{Float64}(undef, m) # will hold a proxy for 𝑄′(𝑝)
Bq= Vector{Float64}(undef, m) # will hold a true 𝑄′(𝑝)

# Dist= Exponential()
Dist= Normal()
# Dist= Uniform()
Dpdf= Vector{Float64}(undef, m)
Dcdf= Vector{Float64}(undef, m)
Dq=   Vector{Float64}(undef, m)
for (i, pi) in enumerate(R)
    Dcdf[i]= quantile(Dist, pi)
    Dpdf[i]= pdf(Dist, Dcdf[i])
end

# @gp :GP1
# @gp :GP2
@gp :GP1 "clear"
@gp :GP2 "clear"
@gp :- :GP1 "set title 'realizations'" "set xlabel 'p'" "set border 1" :-
@gp :- :GP2 "set title 'density'" "set xlabel 'p'" "set border 1" :-
@gp :- :GP1 R +sqrt.(R.*(1.0 .- R)) "w line lw 0.6 lc rgb 'black' notitle"
@gp :- :GP2 R +sqrt.(R.*(1.0 .- R)) "w line lw 0.6 lc rgb 'black' notitle"
@gp :- :GP1 R 0.0* R "w line lw 1 lc black notitle"
@gp :- :GP2 R 0.0* R "w line lw 1 lc black notitle"
@gp :- :GP1 R -sqrt.(R.*(1.0 .- R)) "w line lw 0.6 lc rgb 'black' notitle"
@gp :- :GP2 R -sqrt.(R.*(1.0 .- R)) "w line lw 0.6 lc rgb 'black' notitle"
println("De Casteljau's Bernstein begins")
for (j,n) in enumerate([10, 100, 1_000, 10_000,100_000,1000_000])  # realization total
    t= @elapsed begin
        X= rand(Dist, n); sort!(X)
        ΔX= n* (X[2:end]- X[1:end-1])
        # for (i,pi) in enumerate(R)
        Threads.@threads for ip in eachindex(R)
            B[ip]=  Bernstein_logsumexp(R[ip], X)
            Bp[ip]= Bernstein_logsumexp(R[ip], ΔX)
            Bq[ip]= 1.0/ pdf(Dist, quantile(Dist, R[ip]))
        end
        if j==5
            @gp :- :GP1 R sqrt(n+1)*(B- Dcdf).* Dpdf "w line title '$n samples' lw 2 lc rgb 'blue'"
        else
            @gp :- :GP1 R sqrt(n+1)*(B- Dcdf).* Dpdf "w line title '$n samples' lw ($j<5?2:3)" # GP1: As n increases, the curve should look more like a Brownian bridge path, living roughly inside ±√(p(1−p))
        end
        @gp :- :GP2 R (n+2)^0.0*(Bp - Bq) "w line title '$n samples' lw 2" # GP2 is basicaly how close is the Bernstein-smoothed spacing estimator to the true quantile derivative?
        # @gp :- :GP2 R 1.0 ./ Yp  "w line title '$n samples' lw 2"
    end
    @show n, t
end


println("logsumexp Bernstein begins")
for (j,n) in enumerate([10, 100, 1_000, 10_000,100_000])  # realization total
    t= @elapsed begin
        X= rand(Dist, n); sort!(X)
        ΔX= n* (X[2:end]- X[1:end-1])
        # for (i,pi) in enumerate(R)
        Threads.@threads for ip in eachindex(R)
            B[ip]=  Bernstein_logsumexp(R[ip], X)
            Bp[ip]= Bernstein_logsumexp(R[ip], ΔX)
            Bq[ip]= 1.0/ pdf(Dist, quantile(Dist, R[ip]))
        end        
    end
    @show n, t
end

