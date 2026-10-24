import Metal
let d = MTLCreateSystemDefaultDevice()!
print("name:", d.name, "| registryID", d.registryID)
print("unifiedMemory:", d.hasUnifiedMemory, "| recommendedMaxWorkingSet MB:", d.recommendedMaxWorkingSetSize >> 20, "| maxBufferLength MB:", d.maxBufferLength >> 20)
print("maxThreadgroupMemory KB:", d.maxThreadgroupMemoryLength / 1024, "| maxThreadsPerTG:", d.maxThreadsPerThreadgroup)
let fams: [(String, MTLGPUFamily)] = [("apple7", .apple7), ("apple8", .apple8), ("apple9", .apple9), ("metal3", .metal3)]
for (n, f) in fams { print(n, d.supportsFamily(f)) }
if #available(macOS 26.0, *) { print("metal4 family:", d.supportsFamily(.metal4)); print("apple10:", d.supportsFamily(MTLGPUFamily(rawValue: 1010)!)) }
print("simdgroup matrix / bfloat via lang version check below")
