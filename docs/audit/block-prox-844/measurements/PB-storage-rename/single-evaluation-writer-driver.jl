# PB-only diagnostic of the six writer cases above 5% in the initial rename run.
source=read(ARGS[1],String)
include_string(Main,first(split(source,"\nrepeats=")),ARGS[1])
function measure(label,op)
    startswith(label,"public_write/") || return
    op();op()
    # Use single evaluations here to detect a batching-specific result.
    trial=run(@benchmarkable($op());evals=1,samples=3000,seconds=.25)
    println(label,'\t',median(trial).time,'\t',trial.memory,'\t',trial.allocs)
    flush(stdout)
end
for rep in 1:4
    open("$(ENV["PB_BLOCK_BENCH_OUTPUT"])-$rep.tsv","w") do output
        redirect_stdout(output) do
            println("case\tmedian_ns\tbytes\tallocations")
            for T in (Float32,Float64), n in (8,32,256), strided in (false,true)
                block_cases(T,n,strided)
            end
        end
    end
end
