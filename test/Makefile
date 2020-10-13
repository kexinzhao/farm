CXX=c++
CXXFLAGS=-O3 --std=c++11
BIN_DIR?=bin
MKDIR=mkdir -p

.PHONY: all gemm test clean
all: gemm test
gemm: $(BIN_DIR)/gemm_bench
test: $(BIN_DIR)/test_correctness

$(BIN_DIR)/gemm_bench: gemm_bench.cc
	$(MKDIR) $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/test_correctness: test_correctness.cc
	$(MKDIR) $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -rf $(BIN_DIR)
