!Crown Copyright 2014 AWE.
!
! This file is part of TeaLeaf.
!
! TeaLeaf is free software: you can redistribute it and/or modify it under 
! the terms of the GNU General Public License as published by the 
! Free Software Foundation, either version 3 of the License, or (at your option) 
! any later version.
!
! TeaLeaf is distributed in the hope that it will be useful, but 
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
! details.
!
! You should have received a copy of the GNU General Public License along with 
! TeaLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief Driver for the halo updates
!>  @author David Beckingsale, Wayne Gaudin
!>  @details Invokes the kernels for the internal and external halo cells for
!>  the fields specified.

MODULE update_halo_module

CONTAINS

    SUBROUTINE update_halo(fields, depth, c, x_min, x_max, y_min, y_max, z_min, &
            z_max, chunk_neighbours, density, u, energy0, energy1, &
            vector_p, vector_sd, offload, dev_id)

        USE tea_module
        USE update_halo_kernel_module

        IMPLICIT NONE

        INTEGER :: c,fields(NUM_FIELDS),depth
        INTEGER :: i,j,k,x_min,y_min,z_min,x_max,y_max,z_max,offload,dev_id
        INTEGER :: chunk_neighbours(6)
        REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: &
            density,u,energy0,energy1,vector_p,vector_sd

        CALL tea_exchange(fields,depth,c,offload)

        IF(chunks(c)%task.EQ.parallel%task) THEN
            IF(use_fortran_kernels)THEN
                CALL update_halo_kernel(x_min, x_max, y_min, y_max, z_min, z_max, &
                    chunk_neighbours, density, energy0, energy1, u, vector_p, &
                    vector_sd, fields, depth, offload, dev_id)
            ELSEIF(use_ext_kernels) THEN
                CALL ext_update_halo_kernel(c, density, energy0, energy1, u, &
                    vector_p, vector_sd, chunk_neighbours, fields, depth, offload)
            ENDIF
        ENDIF

    END SUBROUTINE update_halo

END MODULE update_halo_module
