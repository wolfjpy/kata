import Foundation
let d = try! Data(contentsOf: URL(fileURLWithPath: "/tmp/kata_head.bin"))
var p = 0
func rd<T>(_ t: T.Type) -> T { let v = d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: p, as: T.self) }; p += MemoryLayout<T>.size; return v }
func str() -> String { let n = Int(rd(UInt64.self)); let s = String(data: d[p..<p+n], encoding: .utf8) ?? "?"; p += n; return s }
func val(_ t: UInt32) -> String {
  switch t {
  case 0: return "\(rd(UInt8.self))"; case 1: return "\(rd(Int8.self))"
  case 2: return "\(rd(UInt16.self))"; case 3: return "\(rd(Int16.self))"
  case 4: return "\(rd(UInt32.self))"; case 5: return "\(rd(Int32.self))"
  case 6: return "\(rd(Float.self))"; case 7: return "\(rd(UInt8.self) != 0)"
  case 8: return str()
  case 9: let et = rd(UInt32.self); let n = Int(rd(UInt64.self)); var out: [String] = []
    for i in 0..<n { let s = val(et); if i < 6 { out.append(String(s.prefix(40))) } }
    return "[\(n)] \(out.joined(separator: ","))…"
  case 10: return "\(rd(UInt64.self))"; case 11: return "\(rd(Int64.self))"; case 12: return "\(rd(Double.self))"
  default: return "?"
  }
}
p = 4; let ver = rd(UInt32.self); let nt = Int(rd(UInt64.self)); let nkv = Int(rd(UInt64.self))
print("ver \(ver) tensors \(nt) kv \(nkv)")
for _ in 0..<nkv { let k = str(); let t = rd(UInt32.self); let v = val(t); print("KV \(k) = \(v.prefix(160))") }
var types: [UInt32: (Int, Int)] = [:]
for _ in 0..<nt {
  let n = str(); let nd = Int(rd(UInt32.self)); var dims: [UInt64] = []
  for _ in 0..<nd { dims.append(rd(UInt64.self)) }
  let ty = rd(UInt32.self); let off = rd(UInt64.self)
  let ne = dims.reduce(1, { $0 * Int($1) })
  let e = types[ty] ?? (0, 0); types[ty] = (e.0 + 1, e.1 + ne)
  print("T \(n) \(dims) type=\(ty) off=\(off)")
}
print("TYPES (type: count, elements)"); for (k, v) in types.sorted(by: { $0.key < $1.key }) { print(k, v) }
print("data starts at ~", (p + 31) / 32 * 32)
