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

(f::SparseFunction)(args...; kwargs...) = f._f(args...; kwargs...)
SparseArrays.nnz(f::SparseFunction) = length(f.rows)

function get_result_buffer(rows::Vector{Int}, cols::Vector{Int}, size::Tuple{Int,Int})
    data = zeros(length(rows))
    SparseArrays.sparse(rows, cols, data, size...)
end

function get_result_buffer(f::SparseFunction)
    get_result_buffer(f.rows, f.cols, f.size)
end

"Get the (sparse) linear indices of all entries that are constant in the symbolic matrix M w.r.t. symbolic vector z."
function get_constant_entries(
    M_symbolic::AbstractMatrix{<:Symbolics.Num},
    z_symbolic::AbstractVector{<:Symbolics.Num},
)
    _z_syms = Symbolics.tosymbol.(z_symbolic)
    findall(SparseArrays.nonzeros(M_symbolic)) do v
        _vars_syms = Symbolics.tosymbol.(Symbolics.get_variables(v))
        isempty(intersect(_vars_syms, _z_syms))
    end
end

function get_constant_entries(
    M_symbolic::AbstractMatrix{<:FD.Node},
    z_symbolic::AbstractVector{<:FD.Node},
)
    _z_syms = [zs.node_value for zs in FD.variables(z_symbolic)]
    # find all entries that are not a function of any of the symbols in z
    findall(SparseArrays.nonzeros(M_symbolic)) do v
        _vars_syms = [vs.node_value for vs in FD.variables(v)]
        isempty(intersect(_vars_syms, _z_syms))
    end
end
