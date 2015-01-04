noinst_LTLIBRARIES = libgi.la

bin_PROGRAMS = gi-gui gi

BUILT_SOURCES = cmdline.x lights.x hl-main.x

# libconfigfiles_la_SOURCES = grammar.yy  lexer.ll  parser.h  token.h lexer.h config_creator.h configfile.h configfile.cpp
libgi_la_SOURCES = 	\
					arealight-sampler.cu arealight-sampler.h \
					cgls-lights.cu cgls-lights.h \
					direct-lighting.cpp direct-lighting.h \
					dofrays.cu dofrays.h \
					hybrid-pt.cpp hybrid-pt.h \
					lights.cpp lights.h \
					material.cu material-conversion.cpp material.h \
					gi_algorithm.cpp gi_algorithm.h \
					gpu_cgls_lights.cpp gpu_cgls_lights.h \
					gpu-pt.cpp gpu-pt.h gpu-pt-kernels.cu gpu-pt-kernels.h \
					raygen.cu raygen.h \
					rayvis.cu rayvis.h \
					subd.cpp subd.h \
					tracers.cu tracers.h \
					util.cpp util-kernels.cu util.h

gi_gui_SOURCES = 	main.cpp \
					cmdline.cpp cmdline.h

gi_SOURCES = 	hl-main.cpp


# for bison/flex
# AM_YFLAGS = -d -p__yacc_prefix__
# AM_LFLAGS = -P__lex_prefix__ -o lex.yy.c

#library_includedir = $(includedir)/$(PACKAGE)-$(VERSION)
#library_include_HEADERS = barebones-cgls.h

# setting c++ flags:
# AM_CXXFLAGS = -Wall -ansi -pedantic -ggdb -pg

AM_CPPFLAGS = -D PACKAGE=\"$(PACKAGE)\" -D VERSION=\"$(VERSION)\" -D DATADIR=\"$(cglsdata)\" \
				@GUILE_DEF@ $(GUILE_CFLAGS) $(CGLS_CFLAGS) $(CGL_CFLAGS) \
				$(RTA_CFLAGS) $(OXR_CFLAGS)

libgi_la_CPPFLAGS =  $(AM_CPPFLAGS)
libgi_la_CXXFLAGS =  $(AM_CXXFLAGS) -fopenmp
libgi_la_LDFLAGS  = $(GUILE_LDFLAGS) $(CGLS_LIBS) $(CGL_LIBS) $(RTA_LIBS) -Wl,-rpath,$(RTA_PLUGIN_DIR) -fopenmp
libgi_la_LIBADD   = $(RTA_PLUGIN_DIR)/bbvh-cuda.a $(RTA_PLUGIN_DIR)/subdiv.a

gi_gui_CPPFLAGS = $(AM_CPPFLAGS)
gi_gui_CXXFLAGS = $(AM_CXXFLAGS) -fopenmp
gi_gui_LDFLAGS  = $(GUILE_LDFLAGS) $(CGLS_LIBS) $(CGL_LIBS) $(RTA_LIBS) -Wl,-rpath,$(RTA_PLUGIN_DIR) -fopenmp
gi_gui_LDADD    = libgi.la $(RTA_PLUGIN_DIR)/bbvh-cuda.a $(RTA_PLUGIN_DIR)/subdiv.a

gi_CPPFLAGS = $(AM_CPPFLAGS)
gi_CXXFLAGS = $(AM_CXXFLAGS) -fopenmp
gi_LDFLAGS  = $(GUILE_LDFLAGS) $(CGLS_LIBS) $(CGL_LIBS) $(RTA_LIBS) -Wl,-rpath,$(RTA_PLUGIN_DIR) -fopenmp
gi_LDADD    = libgi.la $(RTA_PLUGIN_DIR)/bbvh-cuda.a $(RTA_PLUGIN_DIR)/subdiv.a

SUFFIXES: .x
cmdline.x: cmdline.cpp
lights.x: lights.cpp
hl-main.x: hl-main.cpp

if CAN_USE_GUILE
snarf_opts = @GUILE_DEF@ $(AM_CPPFLAGS) $(CPPFLAGS) $(gi_CPPFLAGS)
.c.x:
	guile-snarf $< $(snarf_opts) > $@
.cpp.x:
	guile-snarf $< $(snarf_opts) > $@
endif

#scheme_def.c: cmdline.x

# setting c preprocessor flags for main project

#LINES=
LINES=-lineinfo
cuda_CPPFLAGS = $(DEFAULT_INCLUDES) $(gi_CPPFLAGS) $(CPPFLAGS) $(AM_CPPFLAGS)

# fuck you cuda.

material.o: material.cu material.h
	$(LIBTOOL) --tag=CXX --mode=compile $(top_srcdir)/nvcc-wrapper $(CXXFLAGS) -fopenmp $(cuda_CPPFLAGS) -gencode arch=compute_@compute_cap@,code=compute_@compute_cap@ -D COMP_CAP=@compute_cap@ -c -o $@ $< -v  -lineinfo  -use_fast_math
	
.cu.$(OBJEXT):
	$(LIBTOOL) --tag=CXX --mode=compile $(top_srcdir)/nvcc-wrapper $(CXXFLAGS)  -fopenmp $(cuda_CPPFLAGS) -gencode arch=compute_@compute_cap@,code=compute_@compute_cap@ -D COMP_CAP=@compute_cap@ -c -o $@ $< -v  -lineinfo  -use_fast_math

.cu.lo:
	$(LIBTOOL) --tag=CXX --mode=compile $(top_srcdir)/nvcc-wrapper $(CXXFLAGS)  -fopenmp $(cuda_CPPFLAGS) -gencode arch=compute_@compute_cap@,code=compute_@compute_cap@ -D COMP_CAP=@compute_cap@ -c -o $@ $< -v  -lineinfo  -use_fast_math


