# (pointer(y) - pointer(x)) mod 4096 for y = similar(x) right after x, in a fresh process and after large arrays were freed
d(x, y) = Int(mod(UInt(pointer(y)) - UInt(pointer(x)), UInt(4096)))
function sweep(label)
  for T in (Float64, Float32)
    print(label, " ", T, ":")
    for n in (1000, 1024, 4096, 10_000, 16384, 65536, 100_000, 1 << 17, 1_000_000, 1 << 20, 10_000_000, 1 << 24)
      x = Vector{T}(undef, n); y = similar(x); z = similar(x)
      print("  n=", n, " ", d(x, y), "/", d(y, z))
    end
    println()
  end
end
sweep("fresh")
for k in 1:3; a = zeros(1 << 22); b = zeros(1 << 21); a = b = nothing; GC.gc(); end
sweep("after-free")
