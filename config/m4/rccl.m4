#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_RCCL],[
AS_IF([test "x$rccl_checked" != "xyes"],[
    rccl_happy="no"

    AC_ARG_WITH([rccl],
            [AS_HELP_STRING([--with-rccl=(DIR)], [Enable the use of RCCL (default is guess).])],
            [], [with_rccl=guess])

    AS_IF([test "x$with_rccl" != "xno"],
    [
        save_CPPFLAGS="$CPPFLAGS"
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"
        AC_MSG_RESULT([Trying RCCL dir: $with_rccl])

        AS_IF([test ! -z "$with_rccl" -a "x$with_rccl" != "xyes" -a "x$with_rccl" != "xguess"],
        [
            AS_IF([test ! -d $with_rccl],
                  [AC_MSG_ERROR([Provided "--with-rccl=${with_rccl}" location does not exist])], [AC_MSG_RESULT([Found RCCL directory])])
            check_rccl_dir="$with_rccl"
            check_rccl_libdir="$with_rccl/lib"
            CPPFLAGS="-I$with_rccl/include -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ $save_CPPFLAGS"
            LDFLAGS="-L$check_rccl_libdir $save_LDFLAGS"
        ])

        AS_IF([test ! -z "$with_rccl_libdir" -a "x$with_rccl_libdir" != "xyes"],
        [
            check_rccl_libdir="$with_rccl_libdir"
            LDFLAGS="-L$check_rccl_libdir $save_LDFLAGS"
        ])
        AC_MSG_RESULT([RCCL flags are $CPPFLAGS $LDFLAGS])
        AC_CHECK_HEADER([rccl.h],
        [
            AC_MSG_RESULT([Found rccl.h header])
            AC_CHECK_LIB([rccl], [ncclCommInitRank],
            [
                rccl_happy="yes"
                AC_MSG_RESULT([Found rccl library])
            ],
            [
                AC_MSG_RESULT([Did not find rccl library])
                rccl_happy="no"
            ])
        ],
        [
            AC_MSG_RESULT([Did not find rccl.h header])
            rccl_happy="no"
        ])

        AS_IF([test "x$rccl_happy" = "xyes"],
        [
            AS_IF([test "x$check_rccl_dir" != "x"],
            [
                AC_MSG_RESULT([RCCL dir: $check_rccl_dir])
                AC_SUBST(RCCL_CPPFLAGS, "-I$check_rccl_dir/include/")
            ])

            AS_IF([test "x$check_rccl_libdir" != "x"],
            [
                AC_SUBST(RCCL_LDFLAGS, "-L$check_rccl_libdir")
            ])

            AC_SUBST(RCCL_LIBADD, "-lrccl")
        ],
        [
            AS_IF([test "x$with_rccl" != "xguess"],
            [
                AC_MSG_ERROR([RCCL support is requested but RCCL packages cannot be found! $CPPFLAGS $LDFLAGS])
            ],
            [
                AC_MSG_WARN([RCCL not found])
            ])
        ])

        CFLAGS="$save_CFLAGS -I/opt/rocm/include -D__HIP_PLATFORM_AMD__"
        CPPFLAGS="$save_CPPFLAGS -I/opt/rocm/include -D__HIP_PLATFORM_AMD__"
        LDFLAGS="$save_LDFLAGS"

    ],
    [
        AC_MSG_WARN([RCCL was explicitly disabled])
    ])

    rccl_checked=yes
])
])
