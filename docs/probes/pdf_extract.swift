import PDFKit
import Foundation
let a = CommandLine.arguments
guard let doc = PDFDocument(url: URL(fileURLWithPath: a[1])) else { print("fail"); exit(1) }
var out = ""
for i in 0..<doc.pageCount { out += "\n=====PAGE \(i+1)=====\n" + (doc.page(at: i)?.string ?? "") }
try! out.write(toFile: a[2], atomically: true, encoding: .utf8)
print(a[1], doc.pageCount, "pages", out.count, "chars")
