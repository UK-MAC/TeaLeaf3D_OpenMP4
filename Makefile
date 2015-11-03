# Crown Copyright 2014 AWE.
#
# This file is part of TeaLeaf.
#
# TeaLeaf is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
#
# TeaLeaf is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# TeaLeaf. If not, see http://www.gnu.org/licenses/.
#
#  @brief Makefile for TeaLeaf
#  @author David Beckingsale, Wayne Gaudin, Matthew Martineau
#  @details Agnostic, platform independent makefile for the TeaLeaf benchmark code.
#
# At the time of release only Intel compilers support OpenMP 4.0 for Intel Xeon Phi
# Knights Corner targets, which is the intended target device. But flags are included
# for the GNU compiler suite, which states version 5.0 will support the Knights Landing
# devices
#
# To select a OpenMP compiler option, you can do this in the shell before typing make:-
#
#  		export COMPILER=INTEL
#  		export COMPILER=GNU
#
# or this works as well:-
#
#  		make COMPILER=INTEL
#  		make COMPILER=GNU
# 
# Don't forget to set the number of threads you want to use, like so
# 
# 		export OMP_NUM_THREADS=16
#
# USAGE:
# 		make                     # Makes with default or user defined parameters
#       make clean               # Cleans up the directory
#       make DEBUG=yes           # Selects debug related flags
#       make IEEE=yes            # Sets IEEE compliance flags
#		make CPROFILER=yes		 # Enables the C function level profiler
# 
# EXAMPLE:
#
# 		make COMPILER=INTEL MPI_F90=mpiifort MPI_C=mpiicc DEBUG=yes IEEE=yes
#
# will compile with the intel compiler with intel debug and ieee flags included

# Used specified parameters
COMPILER 	= INTEL
MPI_F90		= mpif90
MPI_C		= mpicc
CPROFILER 	= yes
DEBUG       = no
OPTIONS	    = #-qopt-report=5 -qopt-report-phase=vec

# Compiler-specific flags
FLAGS_INTEL     = -O3 -no-prec-div
FLAGS_GNU       = -O3 -march=native -funroll-loops
CFLAGS_INTEL    = -O3 -no-prec-div -restrict -fno-alias
CFLAGS_GNU       = -O3 -march=native -funroll-loops

OMP_INTEL	= -qopenmp
OMP_GNU		= -fopenmp
OMP         = $(OMP_$(COMPILER))

ifeq ($(DEBUG), yes)
  FLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  CFLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  FLAGS_INTEL   = -O0 -g -debug all -check all -traceback -check noarg_temp_created
  CFLAGS_INTEL  = -O0 -g -debug all -traceback
endif

ifeq ($(IEEE), yes)
  I3E_INTEL     = -fp-model strict -fp-model source -prec-div -prec-sqrt
  I3E_GNU       = -ffloat-store
  I3E=$(I3E_$(COMPILER))
endif

LDLIBS = -lstdc++
FLAGS  = $(FLAGS_$(COMPILER)) $(I3E) $(OMP) $(OPTIONS)
CFLAGS = $(CFLAGS_$(COMPILER)) $(I3E) $(OMP) $(OPTIONS) 

ifeq ($(CPROFILER), yes)
  CFLAGS += -std=gnu99 -DENABLE_PROFILING
else
  CFLAGS += -std=c99
endif

FOBJ=\
	 data.o						\
	 definitions.o				\
	 pack_kernel.o				\
	 tea.o						\
	 report.o					\
	 timer.o					\
	 parse.o					\
	 read_input.o				\
	 initialise_chunk_kernel.o	\
	 initialise_chunk.o			\
	 build_field.o				\
	 update_halo_kernel.o		\
	 update_halo.o				\
	 start.o					\
	 generate_chunk_kernel.o	\
	 generate_chunk.o			\
	 initialise.o				\
	 field_summary_kernel.o		\
	 field_summary.o			\
	 calc_dt.o					\
	 timestep.o					\
	 set_field_kernel.o         \
	 set_field.o                \
	 tea_leaf_jacobi.o          \
	 tea_leaf_cg.o             	\
	 tea_leaf_cheby.o           \
	 tea_leaf_ppcg.o            \
	 tea_solve.o                \
	 visit.o					\
	 tea_leaf.o					\
	 diffuse.o 

OBJ	 = $(patsubst %.c,%.o,$(wildcard *.c))
OBJ	+= $(FOBJ)

tea_leaf: Makefile $(OBJ)
	$(MPI_F90) $(FLAGS)	$(OBJ) $(LDLIBS) -o tea_leaf
	@echo $(MESSAGE)

include make.deps

%_module.mod: %.f90 %.o
	@true
%.o: %.f90 Makefile make.deps
	$(MPI_F90) $(FLAGS) -c $<
%.o: %.c Makefile make.deps
	$(MPI_C) $(CFLAGS) -c $<

.PHONY: clean

clean:
	rm -f *.o *.mod *genmod* *.lst *.cub *.ptx tea_leaf *.modmic

