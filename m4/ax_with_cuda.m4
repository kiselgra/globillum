AC_DEFUN([AX_WITH_CUDA], [
	AC_ARG_WITH(cuda,
				[AS_HELP_STRING([--with-cuda], [Where to find cuda. Default: /opt/nvidia/cuda])],
				cuda_location="$withval", cuda_location="/opt/nvidia/cuda")
	CUDA_CPPFLAGS="-I$cuda_location/include"
	CUDA_LDFLAGS="-L$cuda_location/lib64"
	CPPFLAGS="$CPPFLAGS $CUDA_CPPFLAGS"
	LDFLAGS="$LDFLAGS $CUDA_LDFLAGS -lcudart"
	AC_CHECK_LIB([cudart], [main])
	AC_SUBST(CUDA_CPPFLAGS)
	AC_SUBST(CUDA_LDFLAGS)
])

