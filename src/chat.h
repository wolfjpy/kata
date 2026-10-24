// Qwen3.8 chat template (the subset the engine needs) and the token sampler.
#pragma once
#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct Msg { std::string role, content, reasoning; };
struct ChatOpts { bool thinking = true; std::string effort = "xhigh"; bool addGenerationPrompt = true; };

// Renders exactly what the model's Jinja template renders for plain system/user/assistant messages (no tools, no vision).
std::string renderChat(const std::vector<Msg>& msgs, const ChatOpts& o);
// The reasoning instruction the template prepends to the system message ("" for medium effort or when thinking is off).
std::string reasoningInstructions(const ChatOpts& o);

struct SamplerParams { float temp = 1.0f; int topK = 20; float topP = 0.95f; float minP = 0.05f; uint64_t seed = 0; };
int sampleToken(const float* logits, int n, const SamplerParams& p, std::mt19937_64& rng);   // temp <= 0 -> greedy
