#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#

HIPCC = hipcc
HIPCCFLAGS = ${AM_CPPFLAGS} ${UCS_CPPFLAGS} -I${UCC_TOP_BUILDDIR} -I${UCC_TOP_SRCDIR}/src -I${UCC_TOP_BUILDDIR}/src 
#--compiler-options -fno-rtti,-fno-exceptions

LINK = $(LIBTOOL) --mode=link $(CC) -o $@

.cu.o:
	$(HIPCC) -c $< -o $@  $(HIPCCFLAGS)

.cu.lo:
	/bin/bash $(top_srcdir)/rocm_lt.sh "$(LIBTOOL)" $@ $(HIPCC) -c  $< $(HIPCCFLAGS) $(HIPCC_ARCH)

comp_noinst = libucc_mc_rocm_kernels.la

libucc_mc_rocm_kernels_la_SOURCES  = mc_rocm_reduce.cu \
                                     mc_rocm_reduce_multi.cu \
                                     mc_rocm_wait_kernel.cu
libucc_mc_rocm_kernels_la_CPPFLAGS =

noinst_LTLIBRARIES = $(comp_noinst)
