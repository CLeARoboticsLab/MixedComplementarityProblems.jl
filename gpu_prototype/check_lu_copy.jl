# Is `copy(::UmfpackLU)` a SAFE, independent deep copy (so we can compute the
# symbolic factorization once and copy it B times), and is it faster than B× lu?
using SparseArrays
using LinearAlgebra

n = 300
A0 = sprand(n, n, 0.04) + 10I          # strongly diagonally dominant -> nonsingular
F = lu(A0)

# A matrix with the SAME pattern but different values.
A1 = copy(A0)
A1.nzval .*= 1.7

# Copy F, then refactorize the copy with A1 (reusing the copy's symbolic).
Fc = copy(F)
lu!(Fc, A1)

b = rand(n)
println("copy independent?  original still solves A0 : ",
    norm(A0 * (F \ b) - b, Inf))                       # should stay ~0 if copy is independent
println("                   copy solves A1            : ",
    norm(A1 * (Fc \ b) - b, Inf))                      # should be ~0 if copy+lu! is correct

# Timing: B× lu  vs  one lu + B× copy.
B = 200
lu(A0); copy(F)                                         # warm up
t_recompute = @elapsed [lu(A0) for _ in 1:B]
t_copy = @elapsed (P = lu(A0); [copy(P) for _ in 1:B])
println("B× lu(A0)            : ", round(t_recompute * 1e3, digits = 2), " ms")
println("lu(A0) + B× copy     : ", round(t_copy * 1e3, digits = 2), " ms")
