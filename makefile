CMAKE            ?= cmake
NINJA            ?= ninja
CBIN             ?= /usr/local/bin
CXX              ?= $(CBIN)/clang++
CC               ?= $(CBIN)/clang
BUILD_TYPE       ?= Debug
CXXFLAGS         += -fdiagnostics-color=always
PREFIX           ?= $(shell pwd)/install
ARCH              = $(shell uname -s)
ARGS             += -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_INSTALL_PREFIX=$(PREFIX)
ARGS             += -DCMAKE_CXX_COMPILER=$(CXX) -DCMAKE_C_COMPILER=$(CC)

ifeq ($(ARCH),Darwin)
	CXXFLAGS += -mmacosx-version-min=15.1
endif

ARGS += -DCMAKE_CXX_FLAGS="$(CXXFLAGS)"

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
	rm build/CMakeCache.txt||:
	$(CMAKE) -B build -S . -G Ninja $(ARGS)

build:
	$(NINJA) -C build

install:
	$(NINJA) -C build install
