// Source: PrismML-Eng/llama.cpp @ 73bfe6cc... ggml/src/ggml-metal/kernels/dequantize.h (PTQ1_0 parts; grep ptq1)
205-
206-template <typename type4x4>
207:void dequantize_ptq1_0(device const block_ptq1_0 * xb, short il, thread type4x4 & reg) {
208-    // A packed byte is a base-3 fraction of 256: with u = b/256, trit n is
209-    //   g_{n+1} - 3*g_n,   g_k = floor(3^k * u)
210-    // and 3^k*b <= 61965 is exact in fp32, so this matches the reference decode bit for
211-    // bit while staying in the float pipe; the table walk cost a load, a shift, a mask, a
212-    // subtract and a convert per element on an ISA that cannot co-issue integer work.
213-    //
214-    // The sixteen elements of a call are regular in il: il 0..4 are the 16-byte chunk at
215-    // trit il; il 5 and 6 are the 8-byte chunk at trits 2(il-5) and 2(il-5)+1, each byte
216-    // read once for both; il 7 is that chunk at trit 4 followed by the qh tail, which
217-    // packs four trits into each of its two bytes.
218-    const float pow3f[6] = {1.0f, 3.0f, 9.0f, 27.0f, 81.0f, 243.0f};
219-
220-    const float d = xb->d;
221-
222-    float4x4 reg_f;
223-
224-    if (il < 5) {
225-        device const uchar * qs = xb->qs;
226-        const float c0 = pow3f[il];
227-        const float c1 = 3.0f*c0;
228-        for (short k = 0; k < 16; k++) {
229-            const float u = (float) qs[k] * (1.0f/256.0f);
230-            reg_f[k/4][k%4] = d*(floor(c1*u) - 3.0f*floor(c0*u) - 1.0f);
231-        }
232-    } else if (il < 7) {
233-        device const uchar * qs = xb->qs + 16;
234-        const float c0 = pow3f[2*(il - 5)];
235-        const float c1 = 3.0f*c0;
236-        const float c2 = 3.0f*c1;
237-        for (short k = 0; k < 8; k++) {
--
261-
262-template <typename type4>
263:void dequantize_ptq1_0_t4(device const block_ptq1_0 * xb, short il, thread type4 & reg) {
264-    const float d = xb->d;
265-    const int base = il * 4;
266-
267-    float4 reg_f;
268-    for (int j = 0; j < 4; j++) {
269-        reg_f[j] = ptq1_0_elem(xb, base + j) * d;
270-    }
271-    reg = (type4) reg_f;
272-}
273-
274-template <typename type4x4>
275-void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
276-    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
277-    const float d1 = il ? (xb->d / 16.h) : xb->d;
278-    const float d2 = d1 / 256.f;
279-    const float md = -8.h * xb->d;
280-    const ushort mask0 = il ? 0x00F0 : 0x000F;
281-    const ushort mask1 = mask0 << 8;
282-
283-    float4x4 reg_f;
284-
285-    for (int i = 0; i < 8; i++) {
286-        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
287-        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
288-    }
289-
290-    reg = (type4x4) reg_f;
291-}
292-
293-template <typename type4>
