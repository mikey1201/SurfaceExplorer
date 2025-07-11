using StaticArrays, BenchmarkTools

@inline function adj_hardcoded(X)
    @SMatrix [ X[5]*X[9]-X[6]*X[8]  X[6]*X[7]-X[4]*X[9]  X[4]*X[8]-X[5]*X[7];
               X[3]*X[8]-X[2]*X[9]  X[1]*X[9]-X[3]*X[7]  X[2]*X[7]-X[1]*X[8];
               X[2]*X[6]-X[3]*X[5]  X[3]*X[4]-X[1]*X[6]  X[1]*X[5]-X[2]*X[4] ]
end

@inline adj_vcat(X) = vcat(cross(X[2,:], X[3,:])', cross(X[3,:], X[1,:])', cross(X[1,:], X[2,:])')

@inline adj_constructor(X) = SMatrix{3,3}(cross(X[2,:], X[3,:])..., cross(X[3,:], X[1,:])..., cross(X[1,:], X[2,:])...)'

X_bench = @SMatrix rand(3,3)
suite = BenchmarkGroup()
suite["Hard-Coded"] = @benchmarkable adj_hardcoded(X_bench)
suite["vcat Method"] = @benchmarkable adj_vcat(X_bench)
suite["Constructor Method"] = @benchmarkable adj_constructor(X_bench)

println("Running Adjugate Benchmarks...")
results = run(suite, verbose=true)
