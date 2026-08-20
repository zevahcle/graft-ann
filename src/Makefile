# forest-graph C++ port
#
# Linux/gcc:        make
# macOS quickest:   brew install libomp && make          (Apple clang + libomp)
# macOS with gcc:   brew install gcc && make CXX=g++-15  (any Homebrew g++-NN)
# serial fallback:  make OMP_C= OMP_L=                   (pragmas ignored)
CXX      ?= g++
BASE     := -O3 -funroll-loops -std=c++17 -Wall -Wextra
UNAME_S  := $(shell uname -s)
IS_CLANG := $(shell $(CXX) --version 2>/dev/null | grep -ci clang)

ifeq ($(UNAME_S),Darwin)
  # NEON is baseline on arm64, so SIMD kernels compile regardless; -mcpu=native
  # is probed because older Apple clang rejects it.
  ARCH := $(shell echo 'int main(){return 0;}' | $(CXX) -x c++ - -mcpu=native -o /dev/null 2>/dev/null && echo -mcpu=native)
  ifeq ($(IS_CLANG),0)
    OMP_C ?= -fopenmp
    OMP_L ?= -fopenmp
  else
    LIBOMP := $(shell brew --prefix libomp 2>/dev/null)
    ifeq ($(LIBOMP),)
      $(warning libomp not found -- 'brew install libomp' for the parallel build; compiling SERIAL)
      OMP_C ?=
      OMP_L ?=
    else
      OMP_C ?= -Xpreprocessor -fopenmp -I$(LIBOMP)/include
      OMP_L ?= -L$(LIBOMP)/lib -lomp
    endif
  endif
else
  ARCH  := -march=native
  OMP_C ?= -fopenmp
  OMP_L ?= -fopenmp
endif

CXXFLAGS := $(BASE) $(ARCH) $(OMP_C)
OBJ = main.o build.o search.o

fg: $(OBJ)
	$(CXX) $(OBJ) -o $@ $(OMP_L)

%.o: %.cpp fg.hpp npy.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

HNSWLIB ?= ../hnswlib
hnsw_bench: hnsw_bench.cpp fg.hpp npy.hpp
	$(CXX) $(CXXFLAGS) -isystem $(HNSWLIB) hnsw_bench.cpp -o $@ $(OMP_L)

clean:
	rm -f $(OBJ) fg hnsw_bench

.PHONY: clean
