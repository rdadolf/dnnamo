# common_defs.mk must have been included already. Enforce this.
ifeq ("$(COMMON_DEFS_IS_DEFINED)","")
$(error The file "common_defs.mk" must be included before "common_rules.mk".)
endif

# Requires the device name to be defined
# The top-level Makefile can do this
ifeq ("$(DEVICE_NAME)","")
$(error Must define DEVICE_NAME in the device's Makefile.)
endif

##### Rules #####
default: _ubench$(SWIG_SHLIB_EXT)

ubench.cpp: $(UBENCH_COMMON)/ubench.i
	$(SWIG) -python -c++ -o ubench.cpp $(UBENCH_COMMON)/ubench.i

ubench.o: ubench.cpp
	$(CPP) $(CPPFLAGS) $(SWIG_CPPFLAGS) -c -o ubench.o ubench.cpp
#@echo "CPPFLAGS:",$(CPPFLAGS)
#@echo "SHLIB_CPPFLAGS:",$(SHLIB_CPPFLAGS)
#@echo "SWIG_CPPFLAGS:",$(SWIG_CPPFLAGS)

ifeq ("$(OS)","Darwin")
_ubench$(SWIG_SHLIB_EXT) ubench.py: ubench.o $(DEVICE_NAME).o
	$(CPP) $(LDFLAGS) $(SWIG_LDFLAGS) -o _ubench$(SWIG_SHLIB_EXT) ubench.o $(DEVICE_NAME).o
	install_name_tool -change libpython2.7.dylib $(shell python-config --exec-prefix)/lib/libpython2.7.dylib _ubench$(SWIG_SHLIB_EXT)

else ifeq ("$(OS)","Linux")
_ubench$(SWIG_SHLIB_EXT): ubench.o $(DEVICE_NAME).o
	$(CPP) $(LDFLAGS) $(SWIG_LDFLAGS) -o _ubench$(SWIG_SHLIB_EXT) ubench.o $(DEVICE_NAME).o

endif

clean:
	rm -f ubench.cpp ubench.o _ubench$(SWIG_SHLIB_EXT) ubench.py ubench.pyc
	rm -f $(DEVICE_NAME).o
