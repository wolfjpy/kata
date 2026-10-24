// Speculative decoding with the model's own MTP head (depth 1, verify width 2).
#pragma once
#include "chat.h"
#include "engine.h"
#include <functional>

struct SpecStats { uint64_t cycles = 0, accepted = 0, tokens = 0; double draftSec = 0, verifySec = 0, sampleSec = 0, hostSec = 0; };

struct SpecResult { int unfed = -1; uint32_t pos = 0; };   // the last emitted token that has not been fed (or -1), number of tokens fed

// Continues generation after `pos` tokens have been fed (eng.feed with mtpTrack on). `emit` receives every generated token in order and returns false to stop.
// Exact: with a deterministic draft token d, sampling the target distribution and accepting when the sample equals d keeps the output distribution unchanged.
SpecResult specGenerate(Engine& eng, uint32_t pos, uint32_t ctx, int maxTokens, const SamplerParams& sp, std::mt19937_64& rng, const std::function<bool(int)>& isEog,
                        const std::function<void(int)>& emit, SpecStats* stats);
