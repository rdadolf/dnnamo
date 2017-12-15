# COMMON_DIR must be specified
ifeq ("$(UBENCH_COMMON)","")
$(error UBENCH_COMMON must be specified)
endif

# common_rules.mk must include this file. Enfore this.
COMMON_DEFS_IS_DEFINED=1

OS := $(shell uname)

ifeq ("$(OS)","Darwin")
CPPFLAGS += -arch x86_64
override LDFLAGS += -dynamiclib

else ifeq ("$(OS)","Linux")
override LDFLAGS += -shared

endif

CPP = c++
override CPPFLAGS += -I$(UBENCH_COMMON)

PYLIBDIR := $(shell python-config --exec-prefix)/lib
SWIG = swig
SWIG_CPPFLAGS = $(shell python-config --includes)
SWIG_LDFLAGS = -L$(PYLIBDIR) $(shell python-config --libs)
# SWIG *always* wants a .so, regardless of platform...
SWIG_SHLIB_EXT=.so

