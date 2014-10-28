AC_DEFUN([AX_ENABLE_GDB_OPTION], [
	AC_ARG_ENABLE(gdb,
				[AS_HELP_STRING([--enable-gdb], [Wether to use debugging flags. Default: no.])],
				with_debug_flags="$enableval", with_debug_flags="no")
	if test "x$with_debug_flags" == "xyes" ; then
		DEBUG_FLAGS="-ggdb3 -O0"
	else
		DEBUG_FLAGS=""
	fi
	CFLAGS="$CFLAGS $DEBUG_FLAGS"
	CXXFLAGS="$CXXFLAGS $DEBUG_FLAGS"
])



