CMAKE            ?= cmake
LIT              ?= lit
# LLVM             ?= /opt/homebrew/opt/llvm@19
LLVM             ?= /usr/local
CBIN             ?= $(LLVM)/bin
CXX               = $(CBIN)/clang++
CC                = $(CBIN)/clang
BUILD_TYPE       ?= Debug
STDLIB           ?= -L$(LLVM)/lib -lc++ -lc++abi -lunwind -stdlib=libc++
CXXFLAGS         += -fdiagnostics-color=always -fno-omit-frame-pointer
CFLAGS           += -fdiagnostics-color=always
LDFLAGS          += $(STDLIB) -rpath $(LLVM)/lib
PREFIX           ?= $(shell pwd)/install
ARCH              = $(shell uname -s)
ARGS             += -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_INSTALL_PREFIX=$(PREFIX)
ARGS             += -DCMAKE_CXX_COMPILER=$(CXX) -DCMAKE_C_COMPILER=$(CC)
ARGS             += -DLLVM_DIR=$(LLVM)/lib/cmake/llvm -DMLIR_DIR=$(LLVM)/lib/cmake/mlir
J                ?= $(shell nproc)
NINJA            ?= ninja -j $(J)
BUILD_DIR        ?= $(shell pwd)/build
INSTALL_DIR      ?= $(shell pwd)/install

ifeq ($(ARCH),Darwin)
#CXXFLAGS         += -mmacosx-version-min=$(shell xcrun --sdk macosx --show-sdk-version)
LDFLAGS          += -Wl,-no_warn_duplicate_libraries
ARGS             += -DCMAKE_MACOSX_RPATH=1
endif

ARGS             += -DCMAKE_EXE_LINKER_FLAGS="$(LDFLAGS)"
ARGS             += -DCMAKE_CXX_FLAGS="$(CXXFLAGS)"
ARGS             += -DENABLE_COVERAGE=ON
ARGS             += -DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR)

RED    = "\e[41m"
YELLOW = "\e[33m"
CYAN   = "\e[36m"
GREEN  = "\e[32m"
CLR    = "\e[0m"

.DEFAULT_GOAL := build
.PHONY: all config build install clean init clean check test coverage config

clean:
	rm -rf $(BUILD_DIR)/* $(INSTALL_DIR)/*||:

init:
	mkdir -p $(BUILD_DIR) $(INSTALL_DIR)||:

config:
	rm $(BUILD_DIR)/CMakeCache.txt || :
	$(CMAKE) -B $(BUILD_DIR) -S . -G Ninja $(ARGS)

build:
	$(NINJA) -C $(BUILD_DIR)

all: config build test

i:
	$(NINJA) -C $(BUILD_DIR) stdlib-interfaces

test:
	ctest --test-dir $(BUILD_DIR) -VV

lit:
	$(LIT) -svv test

p3:
	$(NINJA) -C $(BUILD_DIR) p3

g3:
	$(NINJA) -C $(BUILD_DIR) g3

repl: p3
	$(BUILD_DIR)/bin/p3
