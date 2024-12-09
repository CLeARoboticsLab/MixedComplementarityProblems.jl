"""
Utility for encoding functions which return sparse matrices.

Code from: https://github.com/JuliaGameTheoreticPlanning/ParametricMCPs.jl/blob/main/src/sparse_utils.jl
"""

struct SparseFunction{T1,T2}
    _f::T1
    result_buffer::T2
    rows::Vector{Int}
    cols::Vector{Int}
    size::Tuple{Int,Int}
    constant_entries::Vector{Int}
    function SparseFunction(_f::T1, rows, cols, size, constant_entries = Int[]) where {T1}
        length(constant_entries) <= length(rows) ||
            throw(ArgumentError("More constant entries than non-zero entries."))
        result_buffer = get_result_buffer(rows, cols, size)
        new{T1,typeof(result_buffer)}(
            _f,
            result_buffer,
            rows,
            cols,
            size,
            constant_entries,
        )
    end
end

(f::SparseFunction)(args...) = f._f(args...)
SparseArrays.nnz(f::SparseFunction) = length(f.rows)

function get_result_buffer(rows::Vector{Int}, cols::Vector{Int}, size::Tuple{Int,Int})
    data = zeros(length(rows))
    SparseArrays.sparse(rows, cols, data, size...)
end

function get_result_buffer(f::SparseFunction)
    get_result_buffer(f.rows, f.cols, f.size)
end
