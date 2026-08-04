using CUDA
n = 256*1024*1024 ÷ 8           # 256 MB of Float64
h = rand(Float64, n); d = CUDA.zeros(Float64, n)
# warmup
copyto!(d, h); copyto!(h, Array(d)); CUDA.synchronize()
println("running 60 large H2D/D2H transfers (sample nvidia-smi now)...")
t = @elapsed for i in 1:60
    copyto!(d, h)              # H2D
    copyto!(h, Array(d))       # D2H
    CUDA.synchronize()
end
gb = 60*2*sizeof(Float64)*n/1e9
println("transferred $(round(gb;digits=1)) GB in $(round(t;digits=2)) s => $(round(gb/t;digits=1)) GB/s aggregate")
