#!/bin/sh
# Regression test of the persistent prefix cache (needs the model; about 3 minutes).
#   sh tools/test_cache.sh [model.gguf]
# Checks: a restored run equals the uncached run (greedy and seeded sampling), the writing run equals it too, a whole-prompt hit equals the
# run that wrote it, the int8 path never uses an fp16 checkpoint, a truncated checkpoint is ignored and rewritten.
M=${1:-../../models/bonsai-2-27b-mtp-ptq1_0.gguf}; B=build/kata; D=$(mktemp -d); export KATA_CACHE_DIR=$D/cache
SYS=$(head -c 14000 docs/CHECKPOINT-001.md); Q="Describe the structure of the document in two sentences."
fail=0; check() { if [ "$2" = "$3" ]; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }
run() { $B run --model "$M" --system "$SYS" --prompt "$Q" --no-think --max-tokens 60 "$@" 2>"$D/err" ; }
restored() { grep -o 'restored [0-9]* of [0-9]*' "$D/err" | head -1; }
h() { md5 -q 2>/dev/null || md5sum | cut -d' ' -f1; }
A=$(run --temp 0 --no-cache | h);          check "uncached greedy run" "$(restored)" ""
W=$(run --temp 0 | h);                       check "writing run equals the uncached run" "$W" "$A"; [ -z "$(restored)" ] || { echo "FAIL  the writing run restored something"; fail=1; }
R=$(run --temp 0 | h);                       check "restored run equals the uncached run" "$R" "$A"; [ -n "$(restored)" ] || { echo "FAIL  nothing was restored"; fail=1; }
S0=$(run --seed 11 --no-cache | h); S1=$(run --seed 11 | h); check "seeded sampling: restored equals uncached" "$S1" "$S0"
K0=$(run --temp 0 --no-cache --kv int8 | h); K1=$(run --temp 0 --kv int8 | h); check "int8 KV: the writing run equals the uncached run" "$K1" "$K0"
K2=$(run --temp 0 --kv int8 | h); check "int8 KV: the restored run equals the uncached run" "$K2" "$K0"; [ -n "$(restored)" ] || { echo "FAIL  nothing was restored for int8 KV"; fail=1; }
FF0=$(run --temp 0 --no-cache --fast-ffn | h); FF1=$(run --temp 0 --fast-ffn | h); FF2=$(run --temp 0 --fast-ffn | h); check "--fast-ffn: writing and restored runs equal the uncached run" "$FF1/$FF2" "$FF0/$FF0"; [ -n "$(restored)" ] || { echo "FAIL  nothing was restored for --fast-ffn"; fail=1; }
F=$(run --temp 0 --fast | h); check "int8 path does not use the fp16 checkpoint" "$(restored)" ""
head -c 14000 docs/CHECKPOINT-001.md > "$D/raw.txt"; rm -rf "$KATA_CACHE_DIR"; P1=$($B run --model "$M" --raw --file "$D/raw.txt" --temp 0 --max-tokens 30 2>"$D/err" | h); P2=$($B run --model "$M" --raw --file "$D/raw.txt" --temp 0 --max-tokens 30 2>"$D/err" | h)
check "whole-prompt hit equals the run that wrote the checkpoint" "$P2" "$P1"; [ -n "$(restored)" ] || { echo "FAIL  whole-prompt hit did not restore"; fail=1; }
for f in "$KATA_CACHE_DIR"/*.ckpt; do truncate -s 100000 "$f"; done
$B run --model "$M" --raw --file "$D/raw.txt" --temp 0 --max-tokens 2 2>"$D/err" >/dev/null; check "truncated checkpoint is ignored" "$(restored)" ""
$B run --model "$M" --raw --file "$D/raw.txt" --temp 0 --max-tokens 2 2>"$D/err" >/dev/null; [ -n "$(restored)" ] && echo "PASS  truncated checkpoint was rewritten" || { echo "FAIL  truncated checkpoint was not rewritten"; fail=1; }
rm -rf "$D"; [ $fail = 0 ] && echo "ALL PASSED" || echo "FAILURES"; exit $fail
