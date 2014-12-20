#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
AC_PREREQ(2.61)
AC_INIT([barebones-cgls], [0.0.1])
AM_INIT_AUTOMAKE
AC_CONFIG_MACRO_DIR([m4])

# name one of your source files here
AC_CONFIG_SRCDIR([src/main.cpp])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_FILES(Makefile src/Makefile) #doc/Makefile doc/doxyfile doc/doxygen_mainpage.h

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

AX_ENABLE_GDB_OPTION
AX_STDCXX11_FLAG
CXXFLAGS="$CXXFLAGS $ac_cv_cxx_cxx11_spec"

AX_WITH_CUDA
AM_CONDITIONAL([cuda], [test "x$ac_cv_lib_cudart_main" == "xyes"])

AC_ARG_WITH(compute-capability,
			[AS_HELP_STRING([--with-compute-capability], 
				[What compute capability sould be used to compile the cuda files. We require at least 20, which is the default.])],
			compute_cap="$withval", compute_cap="20")
AC_SUBST([compute_cap])



# when using flex/bison:
# AC_PROG_LEX
# AC_PROG_YACC

# when building libraries
AC_PROG_LIBTOOL

AC_CHECK_LIB([mcm], [main])
AC_CHECK_LIB([glut], [main])

PKG_CHECK_MODULES(CGL, libcgl)
AC_SUBST(CGL_CFLAGS)
AC_SUBST(CGL_LDFLAGS)

PKG_CHECK_MODULES(CGLS, libcgls)
AC_SUBST(CGLS_CFLAGS)
AC_SUBST(CGLS_LDFLAGS)

PKG_CHECK_MODULES(GLEW, glew)
AC_SUBST(GLEW_CFLAGS)
AC_SUBST(GLEW_LDFLAGS)

AC_CHECK_LIB([hyb], [main])

cgldata="$(pkg-config --variable=datadir libcgl)"
AC_SUBST([cgldata])

cglsdata="$(pkg-config --variable=datadir libcgls)"
AC_SUBST([cglsdata])

AC_ARG_ENABLE(scheme,
			[AS_HELP_STRING([--disable-scheme], [Wether to use scheme. Default: yes.])],
			with_scheme="$enableval", with_scheme="yes")

if test "x$with_scheme" == "xyes" ; then
	ifdef([GUILE_FLAGS], [can_use_guile="yes"], [can_use_guile="no"])
	if test "x$can_use_guile" = "xyes" ; then
		GUILE_FLAGS
		GUILE_DEF="-DWITH_GUILE"
	else
		GUILE_DEF=""
	fi
else
	can_use_guile="no"
	GUILE_DEF=""
fi
AC_SUBST([GUILE_DEF])
AM_CONDITIONAL([CAN_USE_GUILE], [test "x$can_use_guile" = "xyes"])

# Checks for libraries.
PKG_CHECK_MODULES(RTA, rta)
AC_SUBST([RTA_CFLAGS])
AC_SUBST([RTA_LIBS])

RTA_PLUGIN_DIR="$(pkg-config rta --variable=plugindir)"
AC_SUBST([RTA_PLUGIN_DIR])

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([inttypes.h stdlib.h string.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

AC_OUTPUT
