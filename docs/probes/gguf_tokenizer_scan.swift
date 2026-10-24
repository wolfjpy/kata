import Foundation
let path = CommandLine.arguments[1]
let d = try! Data(contentsOf: URL(fileURLWithPath: path))
var p = 0
func rd<T>(_ t: T.Type) -> T { let v = d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: p, as: T.self) }; p += MemoryLayout<T>.size; return v }
func str() -> (Int, Int) { let n = Int(rd(UInt64.self)); let s = p; p += n; return (s, n) }
func fnv(_ a: Int, _ n: Int, _ h: inout UInt64) { for i in a..<(a+n) { h ^= UInt64(d[i]); h = h &* 1099511628211 }; h ^= 0xff; h = h &* 1099511628211 }
p = 4; _ = rd(UInt32.self); _ = rd(UInt64.self); let nkv = Int(rd(UInt64.self))
func skipVal(_ t: UInt32) { switch t { case 0,1,7: p += 1; case 2,3: p += 2; case 4,5,6: p += 4; case 10,11,12: p += 8; case 8: _ = str(); case 9: let et = rd(UInt32.self); let n = Int(rd(UInt64.self)); for _ in 0..<n { skipVal(et) }; default: break } }
for _ in 0..<nkv {
  let (ks, kn) = str(); let key = String(data: d[ks..<ks+kn], encoding: .utf8)!; let t = rd(UInt32.self)
  let want = key.hasPrefix("tokenizer.ggml") || key == "general.architecture" || key.hasSuffix("embedding_length") || key.hasSuffix("block_count") || key.hasSuffix("full_attention_interval") || key.hasSuffix("feed_forward_length")
  if want {
    if t == 9 { let et = rd(UInt32.self); let n = Int(rd(UInt64.self)); var h: UInt64 = 14695981039346656037
      if et == 8 { for _ in 0..<n { let (s, l) = str(); fnv(s, l, &h) } } else { for _ in 0..<n { skipVal(et) } }
      print("\(key): array[\(n)] fnv=\(String(h, radix: 16))") }
    else if t == 8 { let (s, l) = str(); print("\(key) = \(String(data: d[s..<s+l], encoding: .utf8)!.prefix(40))") }
    else if t == 4 { print("\(key) = \(rd(UInt32.self))") } else if t == 5 { print("\(key) = \(rd(Int32.self))") } else if t == 7 { print("\(key) = \(rd(UInt8.self))") } else { skipVal(t) }
  } else { skipVal(t) }
}
