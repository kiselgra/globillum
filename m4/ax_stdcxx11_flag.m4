AC_DEFUN([AX_STDCXX11_FLAG], [
  AC_CACHE_CHECK(if we should use gnu++0x or gnu++11 for c++11 support,
  ac_cv_cxx_cxx11_spec,
  [
  	AC_LANG_SAVE
  	AC_LANG_CPLUSPLUS
  	ac_save_CXXFLAGS="$CXXFLAGS"
  	CXXFLAGS="$CXXFLAGS -std=gnu++11"
  	AC_TRY_COMPILE(
		[
  		],, ac_cv_cxx_cxx11_spec_use_11=yes, ac_cv_cxx_cxx11_spec_use_11=no)
  	CXXFLAGS="$ac_save_CXXFLAGS"
  	CXXFLAGS="$CXXFLAGS -std=gnu++0x"
  	AC_TRY_COMPILE(
		[
  		],,
  		ac_cv_cxx_cxx11_spec_use_0x=yes, ac_cv_cxx_cxx11_spec_use_0x=no)
  	CXXFLAGS="$ac_save_CXXFLAGS"
  	AC_LANG_RESTORE

  	if test "$ac_cv_cxx_cxx11_spec_use_11" = yes ; then
  		ac_cv_cxx_cxx11_spec="-std=gnu++11";
  	else if test "$ac_cv_cxx_cxx11_spec_use_0x" = yes ; then
  		ac_cv_cxx_cxx11_spec="-std=gnu++0x";
  	else
  	  AC_MSG_ERROR("Your compiler accepts neither -std=gnu++0x nor -std=gnu++11!")
  	fi fi
  ])])


