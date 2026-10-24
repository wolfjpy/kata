// Qwen pre-tokenizer using ICU regular expressions (Foundation). Everything else in the tokenizer is plain C++.
#import <Foundation/Foundation.h>
#include "tokenizer.h"

static NSRegularExpression* qwenRegex() {
    static NSRegularExpression* re = nil;
    if (!re) {
        NSString* pat = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
        NSError* err = nil; re = [NSRegularExpression regularExpressionWithPattern:pat options:0 error:&err];
        if (!re) { fprintf(stderr, "pre-tokenizer regex failed: %s\n", [[err description] UTF8String]); exit(1); }
    }
    return re;
}
std::string pretokNfc(const std::string& utf8) {
    @autoreleasepool { NSString* s = [[NSString alloc] initWithBytes:utf8.data() length:utf8.size() encoding:NSUTF8StringEncoding]; if (!s) return utf8; return std::string([[s precomposedStringWithCanonicalMapping] UTF8String]); }
}
void pretokSplit(const std::string& utf8, std::vector<std::pair<uint32_t, uint32_t>>& spans) {
    spans.clear(); if (utf8.empty()) return;
    @autoreleasepool {
        NSString* s = [[NSString alloc] initWithBytes:utf8.data() length:utf8.size() encoding:NSUTF8StringEncoding];
        if (!s) { spans.push_back({0, (uint32_t)utf8.size()}); return; }
        // map UTF-16 unit offsets to UTF-8 byte offsets
        std::vector<uint32_t> u16ToU8; u16ToU8.reserve(utf8.size() + 1);
        for (size_t i = 0; i < utf8.size();) {
            unsigned char c = (unsigned char)utf8[i]; size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
            u16ToU8.push_back((uint32_t)i); if (len == 4) u16ToU8.push_back((uint32_t)i);   // surrogate pair: two UTF-16 units
            i += len;
        }
        u16ToU8.push_back((uint32_t)utf8.size());
        __block uint32_t last = 0;
        [qwenRegex() enumerateMatchesInString:s options:0 range:NSMakeRange(0, [s length]) usingBlock:^(NSTextCheckingResult* m, NSMatchingFlags, BOOL*) {
            uint32_t b = u16ToU8[m.range.location], e = u16ToU8[m.range.location + m.range.length];
            if (b > last) spans.push_back({last, b});      // Isolated: any gap is its own piece
            if (e > b) spans.push_back({b, e}); last = e;
        }];
        if (last < utf8.size()) spans.push_back({last, (uint32_t)utf8.size()});
    }
}
