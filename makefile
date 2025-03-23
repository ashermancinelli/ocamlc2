CMAKE  ?= cmake
NINJA  ?= ninja
CXX    ?= clang++
CC     ?= clang
PREFIX ?= $(shell pwd)/install
ARGS   += -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$(PREFIX)
ARGS   += -DCMAKE_CXX_COMPILER=$(CXX) -DCMAKE_C_COMPILER=$(CC)

RED    = "\e[41m"
YELLOW = "\e[33m"
CYAN   = "\e[36m"
GREEN  = "\e[32m"
CLR    = "\e[0m"

.DEFAULT_GOAL := all
.PHONY: all config build install clean init

clean:
	rm -rf build/* install/*

init:
	mkdir -p build install||:

config:
	$(CMAKE) -B build -S . -G Ninja $(ARGS)

build:
	$(NINJA) -C build

install: build
	$(NINJA) -C build install

all: config build
