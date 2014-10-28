
bin_PROGRAMS = gi

BUILT_SOURCES = cmdline.x
# libconfigfiles_la_SOURCES = grammar.yy  lexer.ll  parser.h  token.h lexer.h config_creator.h configfile.h configfile.cpp
gi_SOURCES = 	main.cpp \
				gi_algorithm.cpp gi_algorithm.h \
				gpu_cgls_lights.cu gpu_cgls_lights.h \
				cmdline.cpp cmdline.h

# for bison/flex
# AM_CPPFLAGS = -Isrc/
# AM_YFLAGS = -d -p__yacc_prefix__
# AM_LFLAGS = -P__lex_prefix__ -o lex.yy.c

#library_includedir = $(includedir)/$(PACKAGE)-$(VERSION)
#library_include_HEADERS = barebones-cgls.h

# setting c++ flags:
# AM_CXXFLAGS = -Wall -ansi -pedantic -ggdb -pg

SUFFIXES: .x
cmdline.x: cmdline.cpp

if CAN_USE_GUILE
snarf_opts = @GUILE_DEF@ $(AM_CPPFLAGS) $(CPPFLAGS) $(barebones_cgls_CPPFLAGS)
.c.x:
	guile-snarf $< $(snarf_opts) > $@
.cpp.x:
	guile-snarf $< $(snarf_opts) > $@
endif

#scheme_def.c: cmdline.x

# setting c preprocessor flags for main project
gi_CPPFLAGS = -D PACKAGE=\"$(PACKAGE)\" -D VERSION=\"$(VERSION)\" -D DATADIR=\"$(cglsdata)\" @GUILE_DEF@ $(GUILE_CFLAGS) $(CGLS_CFLAGS) $(CGL_CFLAGS) $(RTA_CFLAGS)
gi_LDFLAGS = $(GUILE_LDFLAGS) $(CGLS_LIBS) $(CGL_LIBS) $(RTA_LIBS)

#LINES=
LINES=-lineinfo
cuda_CPPFLAGS = $(DEFAULT_INCLUDES) $(gi_CPPFLAGS) $(CPPFLAGS)
.cu.o:
	$(LIBTOOL) --tag=CXX --mode=compile $(top_srcdir)/nvcc-wrapper $(CXXFLAGS) $(cuda_CPPFLAGS) -gencode arch=compute_@compute_cap@,code=compute_@compute_cap@ -D COMP_CAP=@compute_cap@ -c -o $@ $< -v  $(LINES) -use_fast_math


