# Bonsai-M6 build. Plain make, no cmake.
export DEVELOPER_DIR ?= /Applications/Xcode.app/Contents/Developer
CXX      ?= clang++
CXXFLAGS ?= -std=c++20 -O2 -g -Wall -Wextra -Wno-unused-parameter -Iconfig -Ithird_party/metal-cpp -Isrc
LDFLAGS  ?= -framework Metal -framework Foundation -framework QuartzCore -framework CoreFoundation
METAL    := xcrun -sdk macosx metal -std=metal4.1 -O3 -fno-fast-math -Iconfig -Ishaders
METALLIB := xcrun -sdk macosx metallib
REF      ?= ../../refs/llama.cpp-prism

BUILD := build
SHADERS := $(wildcard shaders/*.metal)
AIRS := $(patsubst shaders/%.metal,$(BUILD)/%.air,$(SHADERS))
SRCS := $(filter-out src/tool_%.cpp,$(wildcard src/*.cpp))
MMSRCS := $(wildcard src/*.mm)
OBJS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS)) $(patsubst src/%.mm,$(BUILD)/%.o,$(MMSRCS))

.PHONY: all clean shaders ref_logits
all: $(BUILD)/bonsai $(BUILD)/bonsai.metallib

$(BUILD)/%.o: src/%.cpp $(wildcard src/*.h) config/bonsai_m6.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.mm $(wildcard src/*.h) config/bonsai_m6.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -x objective-c++ -fobjc-arc -c $< -o $@

$(BUILD)/bonsai: $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

$(BUILD)/%.air: shaders/%.metal $(wildcard shaders/*.h) config/bonsai_m6.h | $(BUILD)
	$(METAL) -c $< -o $@

$(BUILD)/bonsai.metallib: $(AIRS)
	$(METALLIB) $(AIRS) -o $@

shaders: $(BUILD)/bonsai.metallib

ref_logits: | $(BUILD)
	$(CXX) -std=c++17 -O2 -I$(REF)/include -I$(REF)/ggml/include tools/ref_logits.cpp -L$(REF)/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$(abspath $(REF))/build/bin -o $(BUILD)/ref_logits

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

# standalone tools
$(BUILD)/bench_gemv: tools/bench_gemv.cpp src/gguf.cpp src/gpu.cpp src/gguf.h src/gpu.h | $(BUILD)
	$(CXX) $(CXXFLAGS) tools/bench_gemv.cpp src/gguf.cpp src/gpu.cpp $(LDFLAGS) -o $@

$(BUILD)/bench_gemv2: tools/bench_gemv2.cpp src/gguf.cpp src/gpu.cpp src/gguf.h src/gpu.h | $(BUILD)
	$(CXX) $(CXXFLAGS) tools/bench_gemv2.cpp src/gguf.cpp src/gpu.cpp $(LDFLAGS) -o $@

$(BUILD)/alu_probe: tools/alu_probe.cpp src/gpu.cpp src/gpu.h | $(BUILD)
	$(CXX) $(CXXFLAGS) tools/alu_probe.cpp src/gpu.cpp $(LDFLAGS) -o $@

$(BUILD)/probe_flash: tools/probe_flash.cpp src/gpu.cpp src/gpu.h | $(BUILD)
	$(CXX) $(CXXFLAGS) tools/probe_flash.cpp src/gpu.cpp $(LDFLAGS) -o $@

$(BUILD)/layout_probe: tools/layout_probe.cpp src/gpu.cpp src/gpu.h | $(BUILD)
	$(CXX) $(CXXFLAGS) tools/layout_probe.cpp src/gpu.cpp $(LDFLAGS) -o $@
