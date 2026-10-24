import Metal
let d = MTLCreateSystemDefaultDevice()!
for (n, r) in [("apple9",1009),("apple10",1010),("apple11",1011),("metal4",5002)] { print(n, d.supportsFamily(MTLGPUFamily(rawValue: r)!)) }
