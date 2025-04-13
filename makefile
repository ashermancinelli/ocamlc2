CMAKE            ?= cmake
NINJA            ?= ninja
# LLVM             ?= /opt/homebrew/opt/llvm@19
LLVM             ?= /usr/local
CBIN             ?= $(LLVM)/bin
CXX               = $(CBIN)/clang++
CC                = $(CBIN)/clang
BUILD_TYPE       ?= Debug
CXXFLAGS         += -fdiagnostics-color=always -stdlib=libc++
LDFLAGS          += -stdlib=libc++
PREFIX           ?= $(shell pwd)/install
ARCH              = $(shell uname -s)
ARGS             += -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_INSTALL_PREFIX=$(PREFIX)
ARGS             += -DCMAKE_CXX_COMPILER=$(CXX) -DCMAKE_C_COMPILER=$(CC)
ARGS             += -DLLVM_DIR=$(LLVM)/lib/cmake/llvm -DMLIR_DIR=$(LLVM)/lib/cmake/mlir

ifeq ($(ARCH),Darwin)
#CXXFLAGS         += -mmacosx-version-min=$(shell xcrun --sdk macosx --show-sdk-version)
LDFLAGS          += -Wl,-no_warn_duplicate_libraries
endif

ARGS             += -DCMAKE_EXE_LINKER_FLAGS="$(LDFLAGS)"
ARGS             += -DCMAKE_CXX_FLAGS="$(CXXFLAGS)"

RED    = "\e[41m"
YELLOW = "\e[33m"
CYAN   = "\e[36m"
GREEN  = "\e[32m"
CLR    = "\e[0m"

.DEFAULT_GOAL := build
.PHONY: all config build install clean init

clean:
	rm -rf build/* install/*||:

init:
	mkdir -p build install||:

config:
	rm build/CMakeCache.txt || :
	$(CMAKE) -B build -S . -G Ninja $(ARGS)

build:
	$(NINJA) -C build

install:
	$(NINJA) -C build install
