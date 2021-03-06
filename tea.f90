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

!>  @brief Communication Utilities
!>  @author David Beckingsale, Wayne Gaudin
!>  @details Contains all utilities required to run TeaLeaf in a distributed
!>  environment, including initialisation, mesh decompostion, reductions and
!>  halo exchange using explicit buffers.
!>
!>  Note the halo exchange is currently coded as simply as possible and no
!>  optimisations have been implemented, such as post receives before sends or packing
!>  buffers with multiple data fields. This is intentional so the effect of these
!>  optimisations can be measured on large systems, as and when they are added.
!>
!>  Even without these modifications TeaLeaf weak scales well on moderately sized
!>  systems of the order of 10K cores.

MODULE tea_module

  USE data_module
  USE definitions_module
  !USE MPI
  USE OMP_LIB

  IMPLICIT NONE
  include "mpif.h"

CONTAINS

SUBROUTINE tea_barrier

  INTEGER :: err

  CALL MPI_BARRIER(MPI_COMM_WORLD,err)

END SUBROUTINE tea_barrier

SUBROUTINE tea_abort

  INTEGER :: ierr,err

  CALL MPI_ABORT(MPI_COMM_WORLD,ierr,err)

END SUBROUTINE tea_abort

SUBROUTINE tea_finalize

  INTEGER :: err

  CLOSE(g_out)
  CALL FLUSH(0)
  CALL FLUSH(6)
  CALL FLUSH(g_out)
  CALL MPI_FINALIZE(err)

END SUBROUTINE tea_finalize

SUBROUTINE tea_init_comms

  IMPLICIT NONE

  !INTEGER :: OMP_GET_NUM_DEVICES
  INTEGER :: err,rank,size, num_devices

  rank=0
  size=1

  CALL MPI_INIT(err)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD,rank,err)
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD,size,err)

  parallel%parallel=.TRUE.
  parallel%task=rank

  !$OMP PARALLEL
    num_devices = OMP_GET_NUM_DEVICES()
  !$OMP END PARALLEL

  IF(num_devices .GT. 0) THEN
    parallel%dev_id = MOD(rank, num_devices)
  ELSE
    WRITE(0,*) "No devices available for rank", parallel%task
  ENDIF

  IF(rank.EQ.0) THEN
    parallel%boss=.TRUE.
  ENDIF

  parallel%boss_task=0
  parallel%max_task=size

END SUBROUTINE tea_init_comms

SUBROUTINE tea_get_num_chunks(count)

  IMPLICIT NONE

  INTEGER :: count

! Should be changed so there can be more than one chunk per mpi task

  count=parallel%max_task

END SUBROUTINE tea_get_num_chunks

SUBROUTINE tea_decompose(x_cells,y_cells,z_cells,left,right,bottom,top,back,front)

  ! This decomposes the mesh into a number of chunks.
  ! The number of chunks may be a multiple of the number of mpi tasks
  ! Picks split with minimal surface area to volume ratio

  IMPLICIT NONE

  INTEGER :: x_cells,y_cells,z_cells,left(:),right(:),top(:),bottom(:),back(:),front(:)
  INTEGER :: c,delta_x,delta_y,delta_z

  REAL(KIND=8) :: surface,volume,best_metric,current_metric
  INTEGER  :: chunk_x,chunk_y,chunk_z,mod_x,mod_y,mod_z

  INTEGER  :: cx,cy,cz,chunk,add_x,add_y,add_z,add_x_prev,add_y_prev,add_z_prev
  INTEGER  :: div1,j,current_x,current_y,current_z

  ! 3D Decomposition of the mesh

  current_x = 1
  current_y = 1
  current_z = number_of_chunks

  ! Initialise metric
  surface = (((1.0*x_cells)/current_x)*((1.0*y_cells)/current_y)*2) &
          + (((1.0*x_cells)/current_x)*((1.0*z_cells)/current_z)*2) &
          + (((1.0*y_cells)/current_y)*((1.0*z_cells)/current_z)*2)
  volume  = ((1.0*x_cells)/current_x)*((1.0*y_cells)/current_y)*((1.0*z_cells)/current_z)
  best_metric = surface/volume
  chunk_x=current_x
  chunk_y=current_y
  chunk_z=current_z

  DO c=1,number_of_chunks

    ! If doesn't evenly divide loop
    IF(MOD(number_of_chunks,c).NE.0) CYCLE

    current_x=c

    div1 = number_of_chunks/c

    DO j=1,div1
      IF(MOD(div1,j).NE.0) CYCLE
      current_y = j

      IF(MOD(number_of_chunks,(c*j)).NE.0) CYCLE

      current_z = number_of_chunks/(c*j)

      surface = (((1.0*x_cells)/current_x)*((1.0*y_cells)/current_y)*2) &
              + (((1.0*x_cells)/current_x)*((1.0*z_cells)/current_z)*2) &
              + (((1.0*y_cells)/current_y)*((1.0*z_cells)/current_z)*2)
      volume  = ((1.0*x_cells)/current_x)*((1.0*y_cells)/current_y)*((1.0*z_cells)/current_z)

      current_metric = surface/volume

      IF(current_metric < best_metric) THEN
        chunk_x=current_x
        chunk_y=current_y
        chunk_z=current_z
        best_metric=current_metric
      ENDIF

    ENDDO

  ENDDO

  ! Set up chunk mesh ranges and chunk connectivity

  delta_x=x_cells/chunk_x
  delta_y=y_cells/chunk_y
  delta_z=z_cells/chunk_z
  mod_x=MOD(x_cells,chunk_x)
  mod_y=MOD(y_cells,chunk_y)
  mod_z=MOD(z_cells,chunk_z)
  add_x_prev=0
  add_y_prev=0
  add_z_prev=0
  chunk=1
  DO cz=1,chunk_z
    DO cy=1,chunk_y
      DO cx=1,chunk_x
        add_x=0
        add_y=0
        add_z=0
        IF(cx.LE.mod_x)add_x=1
        IF(cy.LE.mod_y)add_y=1
        IF(cz.LE.mod_z)add_z=1
        IF (chunk .EQ. parallel%task+1) THEN
          ! Mesh chunks
          left(1)=(cx-1)*delta_x+1+add_x_prev
          right(1)=left(1)+delta_x-1+add_x
          bottom(1)=(cy-1)*delta_y+1+add_y_prev
          top(1)=bottom(1)+delta_y-1+add_y
          back(1)=(cz-1)*delta_z+1+add_z_prev
          front(1)=back(1)+delta_z-1+add_z
          ! Chunk connectivity
          chunks(1)%chunk_neighbours(chunk_left)=  chunk-1
          chunks(1)%chunk_neighbours(chunk_right)= chunk+1
          chunks(1)%chunk_neighbours(chunk_bottom)=chunk-chunk_x
          chunks(1)%chunk_neighbours(chunk_top)=   chunk+chunk_x
          chunks(1)%chunk_neighbours(chunk_back)=  chunk-chunk_x*chunk_y
          chunks(1)%chunk_neighbours(chunk_front)= chunk+chunk_x*chunk_y
          IF(cx.EQ.1)chunks(1)%chunk_neighbours(chunk_left)=external_face
          IF(cx.EQ.chunk_x)chunks(1)%chunk_neighbours(chunk_right)=external_face
          IF(cy.EQ.1)chunks(1)%chunk_neighbours(chunk_bottom)=external_face
          IF(cy.EQ.chunk_y)chunks(1)%chunk_neighbours(chunk_top)=external_face
          IF(cz.EQ.1)chunks(1)%chunk_neighbours(chunk_back)=external_face
          IF(cz.EQ.chunk_z)chunks(1)%chunk_neighbours(chunk_front)=external_face
        ENDIF
        IF(cx.LE.mod_x)add_x_prev=add_x_prev+1
        chunk=chunk+1
      ENDDO
      add_x_prev=0
      IF(cy.LE.mod_y)add_y_prev=add_y_prev+1
    ENDDO
    add_x_prev=0
    add_y_prev=0
    IF(cz.LE.mod_z)add_z_prev=add_z_prev+1
  ENDDO

  IF(parallel%boss)THEN
    WRITE(g_out,*)
    WRITE(g_out,*)"Decomposing the mesh into ",chunk_x," by ",chunk_y," by ",chunk_z," chunks"
    WRITE(g_out,*)
  ENDIF

END SUBROUTINE tea_decompose

SUBROUTINE tea_allocate_buffers(chunk)

  IMPLICIT NONE

  INTEGER      :: chunk
  
  ! Unallocated buffers for external boundaries caused issues on some systems so they are now
  !  all allocated
  IF(parallel%task.EQ.chunks(chunk)%task)THEN
      ALLOCATE(chunks(chunk)%left_snd_buffer(19*2*(chunks(chunk)%field%y_max+5)*(chunks(chunk)%field%z_max+5)))
      ALLOCATE(chunks(chunk)%left_rcv_buffer(19*2*(chunks(chunk)%field%y_max+5)*(chunks(chunk)%field%z_max+5)))
      ALLOCATE(chunks(chunk)%right_snd_buffer(19*2*(chunks(chunk)%field%y_max+5)*(chunks(chunk)%field%z_max+5)))
      ALLOCATE(chunks(chunk)%right_rcv_buffer(19*2*(chunks(chunk)%field%y_max+5)*(chunks(chunk)%field%z_max+5)))
      ALLOCATE(chunks(chunk)%bottom_snd_buffer(19*2*(chunks(chunk)%field%x_max+5)*(chunks(chunk)%field%z_max+5)))
      ALLOCATE(chunks(chunk)%bottom_rcv_buffer(19*2*(chunks(chunk)%field%x_max+5)*(chunks(chunk)%field%z_max+5)))
      ALLOCATE(chunks(chunk)%top_snd_buffer(19*2*(chunks(chunk)%field%x_max+5)*(chunks(chunk)%field%z_max+5)))
      ALLOCATE(chunks(chunk)%top_rcv_buffer(19*2*(chunks(chunk)%field%x_max+5)*(chunks(chunk)%field%z_max+5)))
      ALLOCATE(chunks(chunk)%back_snd_buffer(19*2*(chunks(chunk)%field%x_max+5)*(chunks(chunk)%field%y_max+5)))
      ALLOCATE(chunks(chunk)%back_rcv_buffer(19*2*(chunks(chunk)%field%x_max+5)*(chunks(chunk)%field%y_max+5)))
      ALLOCATE(chunks(chunk)%front_snd_buffer(19*2*(chunks(chunk)%field%x_max+5)*(chunks(chunk)%field%y_max+5)))
      ALLOCATE(chunks(chunk)%front_rcv_buffer(19*2*(chunks(chunk)%field%x_max+5)*(chunks(chunk)%field%y_max+5)))
  ENDIF

END SUBROUTINE tea_allocate_buffers

SUBROUTINE tea_exchange(fields,depth,chunk,offload)

    IMPLICIT NONE

    INTEGER      :: fields(:),depth, chunk, offload
    INTEGER      :: left_right_offset(NUM_FIELDS),bottom_top_offset(NUM_FIELDS),back_front_offset(NUM_FIELDS)
    INTEGER      :: request(4)
    INTEGER      :: message_count,err
    INTEGER      :: status(MPI_STATUS_SIZE,4)
    INTEGER      :: end_pack_index_left_right, end_pack_index_bottom_top,end_pack_index_back_front,field

    ! Assuming 1 patch per task, this will be changed

    call flush(g_out)

    request=0
    message_count=0

    end_pack_index_left_right=0
    end_pack_index_bottom_top=0
    end_pack_index_back_front=0

    back_front_offset = -1
    left_right_offset = -1
    bottom_top_offset = -1

    DO field=1,NUM_FIELDS
      IF(fields(field).EQ.1) THEN
        left_right_offset(field)=end_pack_index_left_right
        bottom_top_offset(field)=end_pack_index_bottom_top
        back_front_offset(field)=end_pack_index_back_front
        end_pack_index_left_right=end_pack_index_left_right+depth* &
            (chunks(chunk)%field%y_max+5)*(chunks(chunk)%field%z_max+5)
        end_pack_index_bottom_top=end_pack_index_bottom_top+depth* &
            (chunks(chunk)%field%x_max+5)*(chunks(chunk)%field%z_max+5)
        end_pack_index_back_front=end_pack_index_back_front+depth* &
            (chunks(chunk)%field%x_max+5)*(chunks(chunk)%field%y_max+5)
      ENDIF
    ENDDO

    IF(chunks(chunk)%chunk_neighbours(chunk_left).NE.external_face) THEN
      ! do left exchanges
      IF(use_ext_kernels) THEN
         CALL ext_pack_message( &
            chunk,              &
            chunks(chunk)%field%density, &
            chunks(chunk)%field%energy0, &
            chunks(chunk)%field%energy1, &
            chunks(chunk)%field%u, &
            chunks(chunk)%field%vector_p, &
            chunks(chunk)%field%vector_sd, &
            fields,             &  
            left_right_offset,  & 
            depth,              &
            CHUNK_LEFT,         &
            chunks(chunk)%left_snd_buffer,&
            offload)
      ELSE
         CALL tea_pack_left(chunk, fields, depth, left_right_offset, offload,parallel%dev_id)
      ENDIF

      !send and recv messagse to the left
      CALL tea_send_recv_message_left(chunks(chunk)%left_snd_buffer,                      &
                                         chunks(chunk)%left_rcv_buffer,                      &
                                         chunk,end_pack_index_left_right,                    &
                                         1, 2,                                               &
                                         request(message_count+1), request(message_count+2))
      message_count = message_count + 2
    ENDIF

    IF(chunks(chunk)%chunk_neighbours(chunk_right).NE.external_face) THEN
      ! do right exchanges
      IF(use_ext_kernels) THEN
         CALL ext_pack_message( &
            chunk,              &
            chunks(chunk)%field%density, &
            chunks(chunk)%field%energy0, &
            chunks(chunk)%field%energy1, &
            chunks(chunk)%field%u, &
            chunks(chunk)%field%vector_p, &
            chunks(chunk)%field%vector_sd, &
            fields,             &  
            left_right_offset,  & 
            depth,              &
            CHUNK_RIGHT,        &
            chunks(chunk)%right_snd_buffer,&
            offload)
      ELSE
        CALL tea_pack_right(chunk, fields, depth, left_right_offset, offload,parallel%dev_id)
      ENDIF

      !send message to the right
      CALL tea_send_recv_message_right(chunks(chunk)%right_snd_buffer,                     &
                                          chunks(chunk)%right_rcv_buffer,                     &
                                          chunk,end_pack_index_left_right,                    &
                                          2, 1,                                               &
                                          request(message_count+1), request(message_count+2))
      message_count = message_count + 2
    ENDIF

    !make a call to wait / sync
    CALL MPI_WAITALL(message_count,request,status,err)

    !unpack in right direction
    IF(chunks(chunk)%chunk_neighbours(chunk_right).NE.external_face) THEN
      IF(use_ext_kernels) THEN
         CALL ext_unpack_message( &
            chunk,              &
            chunks(chunk)%field%density, &
            chunks(chunk)%field%energy0, &
            chunks(chunk)%field%energy1, &
            chunks(chunk)%field%u, &
            chunks(chunk)%field%vector_p, &
            chunks(chunk)%field%vector_sd, &
            fields,             &  
            left_right_offset,  & 
            depth,              &
            CHUNK_RIGHT,        &
            chunks(chunk)%right_rcv_buffer,&
            offload)
      ELSE
        CALL tea_unpack_right(fields, chunk, depth,                     &
                                 chunks(chunk)%right_rcv_buffer,           &
                                 left_right_offset, offload,parallel%dev_id)
      ENDIF
    ENDIF

    !unpack in left direction
    IF(chunks(chunk)%chunk_neighbours(chunk_left).NE.external_face) THEN
        IF(use_ext_kernels) THEN
            CALL ext_unpack_message( &
                chunk,              &
                chunks(chunk)%field%density, &
                chunks(chunk)%field%energy0, &
                chunks(chunk)%field%energy1, &
                chunks(chunk)%field%u, &
                chunks(chunk)%field%vector_p, &
                chunks(chunk)%field%vector_sd, &
                fields,             &  
                left_right_offset,  & 
                depth,              &
                CHUNK_LEFT,        &
                chunks(chunk)%left_rcv_buffer,&
            offload)
        ELSE
            CALL tea_unpack_left(fields, chunk, depth,                      &
                chunks(chunk)%left_rcv_buffer,             &
                left_right_offset, offload,parallel%dev_id)
        ENDIF
    ENDIF

    message_count = 0
    request = 0

    IF(chunks(chunk)%chunk_neighbours(chunk_bottom).NE.external_face) THEN
      ! do bottom exchanges
      IF(use_ext_kernels) THEN
         CALL ext_pack_message( &
            chunk,              &
            chunks(chunk)%field%density, &
            chunks(chunk)%field%energy0, &
            chunks(chunk)%field%energy1, &
            chunks(chunk)%field%u, &
            chunks(chunk)%field%vector_p, &
            chunks(chunk)%field%vector_sd, &
            fields,             &  
            bottom_top_offset,  & 
            depth,              &
            CHUNK_BOTTOM,       &
            chunks(chunk)%bottom_snd_buffer,&
            offload)
      ELSE
        CALL tea_pack_bottom(chunk, fields, depth, bottom_top_offset, offload,parallel%dev_id)
      ENDIF

      !send message downwards
      CALL tea_send_recv_message_bottom(chunks(chunk)%bottom_snd_buffer,                     &
                                           chunks(chunk)%bottom_rcv_buffer,                     &
                                           chunk,end_pack_index_bottom_top,                     &
                                           3, 4,                                                &
                                           request(message_count+1), request(message_count+2))
      message_count = message_count + 2
    ENDIF

    IF(chunks(chunk)%chunk_neighbours(chunk_top).NE.external_face) THEN
      ! do top exchanges
      IF(use_ext_kernels) THEN
         CALL ext_pack_message( &
            chunk,              &
            chunks(chunk)%field%density, &
            chunks(chunk)%field%energy0, &
            chunks(chunk)%field%energy1, &
            chunks(chunk)%field%u, &
            chunks(chunk)%field%vector_p, &
            chunks(chunk)%field%vector_sd, &
            fields,             &  
            bottom_top_offset,  & 
            depth,              &
            CHUNK_TOP,          &
            chunks(chunk)%top_snd_buffer,&
            offload)
      ELSE
        CALL tea_pack_top(chunk, fields, depth, bottom_top_offset, offload,parallel%dev_id)
      ENDIF

      !send message upwards
      CALL tea_send_recv_message_top(chunks(chunk)%top_snd_buffer,                           &
                                        chunks(chunk)%top_rcv_buffer,                           &
                                        chunk,end_pack_index_bottom_top,                        &
                                        4, 3,                                                   &
                                        request(message_count+1), request(message_count+2))
      message_count = message_count + 2
    ENDIF

    !need to make a call to wait / sync
    CALL MPI_WAITALL(message_count,request,status,err)

    !unpack in top direction
    IF( chunks(chunk)%chunk_neighbours(chunk_top).NE.external_face ) THEN
        IF(use_ext_kernels) THEN
            CALL ext_unpack_message( &
                chunk,              &
                chunks(chunk)%field%density, &
                chunks(chunk)%field%energy0, &
                chunks(chunk)%field%energy1, &
                chunks(chunk)%field%u, &
                chunks(chunk)%field%vector_p, &
                chunks(chunk)%field%vector_sd, &
                fields,             &  
                bottom_top_offset,  & 
                depth,              &
                CHUNK_TOP,        &
                chunks(chunk)%top_rcv_buffer,&
            offload)
        ELSE
            CALL tea_unpack_top(fields, chunk, depth,                       &
                chunks(chunk)%top_rcv_buffer,               &
                bottom_top_offset, offload,parallel%dev_id)
        ENDIF
    ENDIF

    !unpack in bottom direction
    IF(chunks(chunk)%chunk_neighbours(chunk_bottom).NE.external_face) THEN
        IF(use_ext_kernels) THEN
            CALL ext_unpack_message( &
                chunk,              &
                chunks(chunk)%field%density, &
                chunks(chunk)%field%energy0, &
                chunks(chunk)%field%energy1, &
                chunks(chunk)%field%u, &
                chunks(chunk)%field%vector_p, &
                chunks(chunk)%field%vector_sd, &
                fields,             &  
                bottom_top_offset,  & 
                depth,              &
                CHUNK_BOTTOM,        &
                chunks(chunk)%bottom_rcv_buffer,&
            offload)
        ELSE
            CALL tea_unpack_bottom(fields, chunk, depth,                   &
                chunks(chunk)%bottom_rcv_buffer,         &
                bottom_top_offset, offload,parallel%dev_id)
        ENDIF
    ENDIF

    message_count = 0
    request = 0

    IF(chunks(chunk)%chunk_neighbours(chunk_back).NE.external_face) THEN
      ! do back exchanges
      IF(use_ext_kernels) THEN
         CALL ext_pack_message( &
            chunk,              &
            chunks(chunk)%field%density, &
            chunks(chunk)%field%energy0, &
            chunks(chunk)%field%energy1, &
            chunks(chunk)%field%u, &
            chunks(chunk)%field%vector_p, &
            chunks(chunk)%field%vector_sd, &
            fields,             &  
            back_front_offset,  & 
            depth,              &
            CHUNK_BACK,         &
            chunks(chunk)%back_snd_buffer,&
            offload)
      ELSE
        CALL tea_pack_back(chunk, fields, depth, back_front_offset, offload,parallel%dev_id)
      ENDIF

      !send message downwards
      CALL tea_send_recv_message_back(chunks(chunk)%back_snd_buffer,                        &
                                           chunks(chunk)%back_rcv_buffer,                      &
                                           chunk,end_pack_index_back_front,                    &
                                           5, 6,                                               &
                                           request(message_count+1), request(message_count+2))
      message_count = message_count + 2
    ENDIF

    IF(chunks(chunk)%chunk_neighbours(chunk_front).NE.external_face) THEN
      ! do front exchanges
      IF(use_ext_kernels) THEN
         CALL ext_pack_message( &
            chunk,              &
            chunks(chunk)%field%density, &
            chunks(chunk)%field%energy0, &
            chunks(chunk)%field%energy1, &
            chunks(chunk)%field%u, &
            chunks(chunk)%field%vector_p, &
            chunks(chunk)%field%vector_sd, &
            fields,             &  
            back_front_offset,  & 
            depth,              &
            CHUNK_FRONT,        &
            chunks(chunk)%front_snd_buffer,&
            offload)
      ELSE
        CALL tea_pack_front(chunk, fields, depth, back_front_offset, offload,parallel%dev_id)
      ENDIF

      !send message upwards
      CALL tea_send_recv_message_front(chunks(chunk)%front_snd_buffer,                       &
                                        chunks(chunk)%front_rcv_buffer,                         &
                                        chunk,end_pack_index_back_front,                        &
                                        6, 5,                                                   &
                                        request(message_count+1), request(message_count+2))
      message_count = message_count + 2
    ENDIF

    !need to make a call to wait / sync
    CALL MPI_WAITALL(message_count,request,status,err)

    !unpack in front direction
    IF( chunks(chunk)%chunk_neighbours(chunk_front).NE.external_face ) THEN
        IF(use_ext_kernels) THEN
            CALL ext_unpack_message( &
                chunk,              &
                chunks(chunk)%field%density, &
                chunks(chunk)%field%energy0, &
                chunks(chunk)%field%energy1, &
                chunks(chunk)%field%u, &
                chunks(chunk)%field%vector_p, &
                chunks(chunk)%field%vector_sd, &
                fields,             &  
                back_front_offset,  & 
                depth,              &
                CHUNK_FRONT,        &
                chunks(chunk)%front_rcv_buffer,&
            offload)
        ELSE
            CALL tea_unpack_front(fields, chunk, depth,                       &
                chunks(chunk)%front_rcv_buffer,               &
                back_front_offset, offload,parallel%dev_id)
        ENDIF
    ENDIF

    !unpack in back direction
    IF(chunks(chunk)%chunk_neighbours(chunk_back).NE.external_face) THEN
        IF(use_ext_kernels) THEN
            CALL ext_unpack_message( &
                chunk,              &
                chunks(chunk)%field%density, &
                chunks(chunk)%field%energy0, &
                chunks(chunk)%field%energy1, &
                chunks(chunk)%field%u, &
                chunks(chunk)%field%vector_p, &
                chunks(chunk)%field%vector_sd, &
                fields,             &  
                back_front_offset,  & 
                depth,              &
                CHUNK_BACK,        &
                chunks(chunk)%back_rcv_buffer,&
            offload)
        ELSE

            CALL tea_unpack_back(fields, chunk, depth,                   &
                chunks(chunk)%back_rcv_buffer,         &
                back_front_offset, offload,parallel%dev_id)
        ENDIF
    ENDIF

END SUBROUTINE tea_exchange

SUBROUTINE tea_pack_left(chunk, fields, depth, left_right_offset, offload, dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER      :: fields(:),depth, chunk
  INTEGER      :: left_right_offset(:)
  INTEGER :: offload, dev_id

  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_pack_message_left(chunks(chunk)%field%x_min,                    &
                                    chunks(chunk)%field%x_max,                    &
                                    chunks(chunk)%field%y_min,                    &
                                    chunks(chunk)%field%y_max,                    &
                                    chunks(chunk)%field%z_min,                    &
                                    chunks(chunk)%field%z_max,                    &
                                    chunks(chunk)%field%density,                 &
                                    chunks(chunk)%left_snd_buffer,                &
                                    CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                    depth, CELL_DATA,                             &
                                    left_right_offset(FIELD_DENSITY), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_pack_message_left(chunks(chunk)%field%x_min,                    &
                                    chunks(chunk)%field%x_max,                    &
                                    chunks(chunk)%field%y_min,                    &
                                    chunks(chunk)%field%y_max,                    &
                                    chunks(chunk)%field%z_min,                    &
                                    chunks(chunk)%field%z_max,                    &
                                    chunks(chunk)%field%energy0,                  &
                                    chunks(chunk)%left_snd_buffer,                &
                                    CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                    depth, CELL_DATA,                             &
                                    left_right_offset(FIELD_ENERGY0), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_pack_message_left(chunks(chunk)%field%x_min,                    &
                                    chunks(chunk)%field%x_max,                    &
                                    chunks(chunk)%field%y_min,                    &
                                    chunks(chunk)%field%y_max,                    &
                                    chunks(chunk)%field%z_min,                    &
                                    chunks(chunk)%field%z_max,                    &
                                    chunks(chunk)%field%energy1,                  &
                                    chunks(chunk)%left_snd_buffer,                &
                                    CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                    depth, CELL_DATA,                             &
                                    left_right_offset(FIELD_ENERGY1), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_pack_message_left(chunks(chunk)%field%x_min,                    &
                                    chunks(chunk)%field%x_max,                    &
                                    chunks(chunk)%field%y_min,                    &
                                    chunks(chunk)%field%y_max,                    &
                                    chunks(chunk)%field%z_min,                    &
                                    chunks(chunk)%field%z_max,                    &
                                    chunks(chunk)%field%vector_p,                  &
                                    chunks(chunk)%left_snd_buffer,                &
                                    CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                    depth, CELL_DATA,                             &
                                    left_right_offset(FIELD_P), offload,dev_id)
    
  ENDIF

END SUBROUTINE tea_pack_left

SUBROUTINE tea_send_recv_message_left(left_snd_buffer, left_rcv_buffer,      &
                                         chunk, total_size,                     &
                                         tag_send, tag_recv,                    &
                                         req_send, req_recv)

  REAL(KIND=8)    :: left_snd_buffer(:), left_rcv_buffer(:)
  INTEGER         :: left_task
  INTEGER         :: chunk
  INTEGER         :: total_size, tag_send, tag_recv, err
  INTEGER         :: req_send, req_recv

  left_task =chunks(chunk)%chunk_neighbours(chunk_left) - 1

  CALL MPI_ISEND(left_snd_buffer,total_size,MPI_DOUBLE_PRECISION,left_task,tag_send &
                ,MPI_COMM_WORLD,req_send,err)

  CALL MPI_IRECV(left_rcv_buffer,total_size,MPI_DOUBLE_PRECISION,left_task,tag_recv &
                ,MPI_COMM_WORLD,req_recv,err)

END SUBROUTINE tea_send_recv_message_left

SUBROUTINE tea_unpack_left(fields, chunk, depth,                         &
                              left_rcv_buffer,                              &
                              left_right_offset, offload,dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER         :: fields(:), chunk, depth
  INTEGER         :: left_right_offset(:)
  REAL(KIND=8)    :: left_rcv_buffer(:)
  INTEGER :: offload,dev_id


  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_unpack_message_left(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%density,                 &
                                      chunks(chunk)%left_rcv_buffer,                &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                      depth, CELL_DATA,                             &
                                      left_right_offset(FIELD_DENSITY), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_unpack_message_left(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%energy0,                  &
                                      chunks(chunk)%left_rcv_buffer,                &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                      depth, CELL_DATA,                             &
                                      left_right_offset(FIELD_ENERGY0), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_unpack_message_left(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%energy1,                  &
                                      chunks(chunk)%left_rcv_buffer,                &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                      depth, CELL_DATA,                             &
                                      left_right_offset(FIELD_ENERGY1), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_unpack_message_left(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%vector_p,                  &
                                      chunks(chunk)%left_rcv_buffer,                &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                      depth, CELL_DATA,                             &
                                      left_right_offset(FIELD_P), offload,dev_id)
    
  ENDIF

END SUBROUTINE tea_unpack_left

SUBROUTINE tea_pack_right(chunk, fields, depth, left_right_offset,offload,dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER        :: chunk, fields(:), depth, tot_packr, left_right_offset(:)
  INTEGER :: offload,dev_id

  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_pack_message_right(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%density,                 &
                                     chunks(chunk)%right_snd_buffer,               &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     left_right_offset(FIELD_DENSITY), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_pack_message_right(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%energy0,                  &
                                     chunks(chunk)%right_snd_buffer,               &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     left_right_offset(FIELD_ENERGY0), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_pack_message_right(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%energy1,                  &
                                     chunks(chunk)%right_snd_buffer,               &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     left_right_offset(FIELD_ENERGY1), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_pack_message_right(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%vector_p,                  &
                                     chunks(chunk)%right_snd_buffer,               &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     left_right_offset(FIELD_P), offload,dev_id)
    
  ENDIF

END SUBROUTINE tea_pack_right

SUBROUTINE tea_send_recv_message_right(right_snd_buffer, right_rcv_buffer,   &
                                          chunk, total_size,                    &
                                          tag_send, tag_recv,                   &
                                          req_send, req_recv)

  IMPLICIT NONE

  REAL(KIND=8) :: right_snd_buffer(:), right_rcv_buffer(:)
  INTEGER      :: right_task
  INTEGER      :: chunk
  INTEGER      :: total_size, tag_send, tag_recv, err
  INTEGER      :: req_send, req_recv

  right_task=chunks(chunk)%chunk_neighbours(chunk_right) - 1

  CALL MPI_ISEND(right_snd_buffer,total_size,MPI_DOUBLE_PRECISION,right_task,tag_send, &
                 MPI_COMM_WORLD,req_send,err)

  CALL MPI_IRECV(right_rcv_buffer,total_size,MPI_DOUBLE_PRECISION,right_task,tag_recv, &
                 MPI_COMM_WORLD,req_recv,err)

END SUBROUTINE tea_send_recv_message_right

SUBROUTINE tea_unpack_right(fields, chunk, depth,                          &
                               right_rcv_buffer,                              &
                               left_right_offset, offload,dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER         :: fields(:), chunk, total_in_right_buff, depth, left_right_offset(:)
  REAL(KIND=8)    :: right_rcv_buffer(:)
  INTEGER :: offload,dev_id

  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_unpack_message_right(chunks(chunk)%field%x_min,                    &
                                       chunks(chunk)%field%x_max,                    &
                                       chunks(chunk)%field%y_min,                    &
                                       chunks(chunk)%field%y_max,                    &
                                       chunks(chunk)%field%z_min,                    &
                                       chunks(chunk)%field%z_max,                    &
                                       chunks(chunk)%field%density,                 &
                                       chunks(chunk)%right_rcv_buffer,               &
                                       CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                       depth, CELL_DATA,                             &
                                       left_right_offset(FIELD_DENSITY), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_unpack_message_right(chunks(chunk)%field%x_min,                    &
                                       chunks(chunk)%field%x_max,                    &
                                       chunks(chunk)%field%y_min,                    &
                                       chunks(chunk)%field%y_max,                    &
                                       chunks(chunk)%field%z_min,                    &
                                       chunks(chunk)%field%z_max,                    &
                                       chunks(chunk)%field%energy0,                  &
                                       chunks(chunk)%right_rcv_buffer,               &
                                       CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                       depth, CELL_DATA,                             &
                                       left_right_offset(FIELD_ENERGY0), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_unpack_message_right(chunks(chunk)%field%x_min,                    &
                                       chunks(chunk)%field%x_max,                    &
                                       chunks(chunk)%field%y_min,                    &
                                       chunks(chunk)%field%y_max,                    &
                                       chunks(chunk)%field%z_min,                    &
                                       chunks(chunk)%field%z_max,                    &
                                       chunks(chunk)%field%energy1,                  &
                                       chunks(chunk)%right_rcv_buffer,               &
                                       CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                       depth, CELL_DATA,                             &
                                       left_right_offset(FIELD_ENERGY1), offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_unpack_message_right(chunks(chunk)%field%x_min,                    &
                                       chunks(chunk)%field%x_max,                    &
                                       chunks(chunk)%field%y_min,                    &
                                       chunks(chunk)%field%y_max,                    &
                                       chunks(chunk)%field%z_min,                    &
                                       chunks(chunk)%field%z_max,                    &
                                       chunks(chunk)%field%vector_p,                  &
                                       chunks(chunk)%right_rcv_buffer,               &
                                       CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                       depth, CELL_DATA,                             &
                                       left_right_offset(FIELD_P), offload,dev_id)
    
  ENDIF
END SUBROUTINE tea_unpack_right

SUBROUTINE tea_pack_top(chunk, fields, depth, bottom_top_offset,offload,dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER        :: chunk, fields(:), depth, bottom_top_offset(:)
  INTEGER :: offload,dev_id

  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_pack_message_top(chunks(chunk)%field%x_min,                    &
                                   chunks(chunk)%field%x_max,                    &
                                   chunks(chunk)%field%y_min,                    &
                                   chunks(chunk)%field%y_max,                    &
                                   chunks(chunk)%field%z_min,                    &
                                   chunks(chunk)%field%z_max,                    &
                                   chunks(chunk)%field%density,                 &
                                   chunks(chunk)%top_snd_buffer,                 &
                                   CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                   depth, CELL_DATA,                             &
                                   bottom_top_offset(FIELD_DENSITY), &
                                   offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_pack_message_top(chunks(chunk)%field%x_min,                    &
                                   chunks(chunk)%field%x_max,                    &
                                   chunks(chunk)%field%y_min,                    &
                                   chunks(chunk)%field%y_max,                    &
                                   chunks(chunk)%field%z_min,                    &
                                   chunks(chunk)%field%z_max,                    &
                                   chunks(chunk)%field%energy0,                  &
                                   chunks(chunk)%top_snd_buffer,                 &
                                   CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                   depth, CELL_DATA,                             &
                                   bottom_top_offset(FIELD_ENERGY0), &
                                   offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_pack_message_top(chunks(chunk)%field%x_min,                    &
                                   chunks(chunk)%field%x_max,                    &
                                   chunks(chunk)%field%y_min,                    &
                                   chunks(chunk)%field%y_max,                    &
                                   chunks(chunk)%field%z_min,                    &
                                   chunks(chunk)%field%z_max,                    &
                                   chunks(chunk)%field%energy1,                  &
                                   chunks(chunk)%top_snd_buffer,                 &
                                   CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                   depth, CELL_DATA,                             &
                                   bottom_top_offset(FIELD_ENERGY1), &
                                   offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_pack_message_top(chunks(chunk)%field%x_min,                    &
                                   chunks(chunk)%field%x_max,                    &
                                   chunks(chunk)%field%y_min,                    &
                                   chunks(chunk)%field%y_max,                    &
                                   chunks(chunk)%field%z_min,                    &
                                   chunks(chunk)%field%z_max,                    &
                                   chunks(chunk)%field%vector_p,                  &
                                   chunks(chunk)%top_snd_buffer,                 &
                                   CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                   depth, CELL_DATA,                             &
                                   bottom_top_offset(FIELD_P), &
                                   offload,dev_id)
    
  ENDIF
END SUBROUTINE tea_pack_top

SUBROUTINE tea_send_recv_message_top(top_snd_buffer, top_rcv_buffer,     &
                                        chunk, total_size,                  &
                                        tag_send, tag_recv,                 &
                                        req_send, req_recv)

    IMPLICIT NONE

    REAL(KIND=8) :: top_snd_buffer(:), top_rcv_buffer(:)
    INTEGER      :: top_task
    INTEGER      :: chunk
    INTEGER      :: total_size, tag_send, tag_recv, err
    INTEGER      :: req_send, req_recv

    top_task=chunks(chunk)%chunk_neighbours(chunk_top) - 1

    CALL MPI_ISEND(top_snd_buffer,total_size,MPI_DOUBLE_PRECISION,top_task,tag_send, &
                   MPI_COMM_WORLD,req_send,err)

    CALL MPI_IRECV(top_rcv_buffer,total_size,MPI_DOUBLE_PRECISION,top_task,tag_recv, &
                   MPI_COMM_WORLD,req_recv,err)

END SUBROUTINE tea_send_recv_message_top

SUBROUTINE tea_unpack_top(fields, chunk, depth,                        &
                             top_rcv_buffer,                              &
                             bottom_top_offset, &
                             offload,dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER         :: fields(:), chunk, total_in_top_buff, depth, bottom_top_offset(:)
  REAL(KIND=8)    :: top_rcv_buffer(:)
  INTEGER :: offload,dev_id


  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_unpack_message_top(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%density,                 &
                                     chunks(chunk)%top_rcv_buffer,                 &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     bottom_top_offset(FIELD_DENSITY), &
                                     offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_unpack_message_top(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%energy0,                  &
                                     chunks(chunk)%top_rcv_buffer,                 &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     bottom_top_offset(FIELD_ENERGY0), &
                                     offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_unpack_message_top(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%energy1,                  &
                                     chunks(chunk)%top_rcv_buffer,                 &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     bottom_top_offset(FIELD_ENERGY1), &
                                     offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_unpack_message_top(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%vector_p,                  &
                                     chunks(chunk)%top_rcv_buffer,                 &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     bottom_top_offset(FIELD_P), &
                                     offload,dev_id)
    
  ENDIF
END SUBROUTINE tea_unpack_top

SUBROUTINE tea_pack_bottom(chunk, fields, depth, bottom_top_offset, &
        offload,dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER        :: chunk, fields(:), depth, tot_packb, bottom_top_offset(:)
  INTEGER :: offload,dev_id

  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_pack_message_bottom(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%density,                 &
                                      chunks(chunk)%bottom_snd_buffer,              &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                      depth, CELL_DATA,                             &
                                      bottom_top_offset(FIELD_DENSITY), &
                                      offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_pack_message_bottom(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%energy0,                  &
                                      chunks(chunk)%bottom_snd_buffer,              &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                      depth, CELL_DATA,                             &
                                      bottom_top_offset(FIELD_ENERGY0), &
                                      offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_pack_message_bottom(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%energy1,                  &
                                      chunks(chunk)%bottom_snd_buffer,              &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                      depth, CELL_DATA,                             &
                                      bottom_top_offset(FIELD_ENERGY1), &
                                      offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_pack_message_bottom(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%vector_p,                  &
                                      chunks(chunk)%bottom_snd_buffer,              &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                      depth, CELL_DATA,                             &
                                      bottom_top_offset(FIELD_P), &
                                      offload,dev_id)
    
  ENDIF
END SUBROUTINE tea_pack_bottom

SUBROUTINE tea_send_recv_message_bottom(bottom_snd_buffer, bottom_rcv_buffer,        &
                                           chunk, total_size,                           &
                                           tag_send, tag_recv,                          &
                                           req_send, req_recv)

  IMPLICIT NONE

  REAL(KIND=8) :: bottom_snd_buffer(:), bottom_rcv_buffer(:)
  INTEGER      :: bottom_task
  INTEGER      :: chunk
  INTEGER      :: total_size, tag_send, tag_recv, err
  INTEGER      :: req_send, req_recv

  bottom_task=chunks(chunk)%chunk_neighbours(chunk_bottom) - 1

  CALL MPI_ISEND(bottom_snd_buffer,total_size,MPI_DOUBLE_PRECISION,bottom_task,tag_send &
                ,MPI_COMM_WORLD,req_send,err)

  CALL MPI_IRECV(bottom_rcv_buffer,total_size,MPI_DOUBLE_PRECISION,bottom_task,tag_recv &
                ,MPI_COMM_WORLD,req_recv,err)

END SUBROUTINE tea_send_recv_message_bottom

SUBROUTINE tea_unpack_bottom(fields, chunk, depth,                        &
                             bottom_rcv_buffer,                              &
                             bottom_top_offset, &
                             offload,dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER         :: fields(:), chunk, depth, bottom_top_offset(:)
  REAL(KIND=8)    :: bottom_rcv_buffer(:)
  INTEGER :: offload,dev_id

  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_unpack_message_bottom(chunks(chunk)%field%x_min,                    &
                                        chunks(chunk)%field%x_max,                    &
                                        chunks(chunk)%field%y_min,                    &
                                        chunks(chunk)%field%y_max,                    &
                                        chunks(chunk)%field%z_min,                    &
                                        chunks(chunk)%field%z_max,                    &
                                        chunks(chunk)%field%density,                 &
                                        chunks(chunk)%bottom_rcv_buffer,              &
                                        CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                        depth, CELL_DATA,                             &
                                        bottom_top_offset(FIELD_DENSITY), &
                                        offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_unpack_message_bottom(chunks(chunk)%field%x_min,                    &
                                        chunks(chunk)%field%x_max,                    &
                                        chunks(chunk)%field%y_min,                    &
                                        chunks(chunk)%field%y_max,                    &
                                        chunks(chunk)%field%z_min,                    &
                                        chunks(chunk)%field%z_max,                    &
                                        chunks(chunk)%field%energy0,                  &
                                        chunks(chunk)%bottom_rcv_buffer,              &
                                        CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                        depth, CELL_DATA,                             &
                                        bottom_top_offset(FIELD_ENERGY0), &
                                        offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_unpack_message_bottom(chunks(chunk)%field%x_min,                    &
                                        chunks(chunk)%field%x_max,                    &
                                        chunks(chunk)%field%y_min,                    &
                                        chunks(chunk)%field%y_max,                    &
                                        chunks(chunk)%field%z_min,                    &
                                        chunks(chunk)%field%z_max,                    &
                                        chunks(chunk)%field%energy1,                  &
                                        chunks(chunk)%bottom_rcv_buffer,              &
                                        CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                        depth, CELL_DATA,                             &
                                        bottom_top_offset(FIELD_ENERGY1), &
                                        offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_unpack_message_bottom(chunks(chunk)%field%x_min,                    &
                                        chunks(chunk)%field%x_max,                    &
                                        chunks(chunk)%field%y_min,                    &
                                        chunks(chunk)%field%y_max,                    &
                                        chunks(chunk)%field%z_min,                    &
                                        chunks(chunk)%field%z_max,                    &
                                        chunks(chunk)%field%vector_p,                  &
                                        chunks(chunk)%bottom_rcv_buffer,              &
                                        CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                        depth, CELL_DATA,                             &
                                        bottom_top_offset(FIELD_P), &
                                        offload,dev_id)
    
  ENDIF
END SUBROUTINE tea_unpack_bottom

SUBROUTINE tea_pack_back(chunk, fields, depth, back_front_offset, &
        offload,dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER        :: chunk, fields(:), depth, back_front_offset(:)
  INTEGER :: offload,dev_id

  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_pack_message_back(chunks(chunk)%field%x_min,                    &
                                    chunks(chunk)%field%x_max,                    &
                                    chunks(chunk)%field%y_min,                    &
                                    chunks(chunk)%field%y_max,                    &
                                    chunks(chunk)%field%z_min,                    &
                                    chunks(chunk)%field%z_max,                    &
                                    chunks(chunk)%field%density,                 &
                                    chunks(chunk)%back_snd_buffer,                &
                                    CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                    depth, CELL_DATA,                             &
                                    back_front_offset(FIELD_DENSITY), &
                                    offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_pack_message_back(chunks(chunk)%field%x_min,                    &
                                    chunks(chunk)%field%x_max,                    &
                                    chunks(chunk)%field%y_min,                    &
                                    chunks(chunk)%field%y_max,                    &
                                    chunks(chunk)%field%z_min,                    &
                                    chunks(chunk)%field%z_max,                    &
                                    chunks(chunk)%field%energy0,                  &
                                    chunks(chunk)%back_snd_buffer,                &
                                    CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                    depth, CELL_DATA,                             &
                                    back_front_offset(FIELD_ENERGY0), &
                                    offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_pack_message_back(chunks(chunk)%field%x_min,                    &
                                    chunks(chunk)%field%x_max,                    &
                                    chunks(chunk)%field%y_min,                    &
                                    chunks(chunk)%field%y_max,                    &
                                    chunks(chunk)%field%z_min,                    &
                                    chunks(chunk)%field%z_max,                    &
                                    chunks(chunk)%field%energy1,                  &
                                    chunks(chunk)%back_snd_buffer,                &
                                    CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                    depth, CELL_DATA,                             &
                                    back_front_offset(FIELD_ENERGY1), &
                                    offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_pack_message_back(chunks(chunk)%field%x_min,                    &
                                    chunks(chunk)%field%x_max,                    &
                                    chunks(chunk)%field%y_min,                    &
                                    chunks(chunk)%field%y_max,                    &
                                    chunks(chunk)%field%z_min,                    &
                                    chunks(chunk)%field%z_max,                    &
                                    chunks(chunk)%field%vector_p,                  &
                                    chunks(chunk)%back_snd_buffer,                &
                                    CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                    depth, CELL_DATA,                             &
                                    back_front_offset(FIELD_P), &
                                    offload,dev_id)
    
  ENDIF
END SUBROUTINE tea_pack_back

SUBROUTINE tea_send_recv_message_back(back_snd_buffer, back_rcv_buffer,     &
                                         chunk, total_size,                  &
                                         tag_send, tag_recv,                 &
                                         req_send, req_recv)

  IMPLICIT NONE

  REAL(KIND=8) :: back_snd_buffer(:), back_rcv_buffer(:)
  INTEGER      :: back_task
  INTEGER      :: chunk
  INTEGER      :: total_size, tag_send, tag_recv, err
  INTEGER      :: req_send, req_recv

  back_task=chunks(chunk)%chunk_neighbours(chunk_back)-1

  CALL MPI_ISEND(back_snd_buffer,total_size,MPI_DOUBLE_PRECISION,back_task,tag_send, &
                 MPI_COMM_WORLD,req_send,err)

  CALL MPI_IRECV(back_rcv_buffer,total_size,MPI_DOUBLE_PRECISION,back_task,tag_recv, &
                 MPI_COMM_WORLD,req_recv,err)

END SUBROUTINE tea_send_recv_message_back

SUBROUTINE tea_unpack_back(fields, chunk, depth,                        &
                              back_rcv_buffer,                             &
                              back_front_offset, &
                              offload,dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER         :: fields(:), chunk, depth, back_front_offset(:)
  REAL(KIND=8)    :: back_rcv_buffer(:)
  INTEGER :: offload,dev_id

  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_unpack_message_back(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%density,                 &
                                      chunks(chunk)%back_rcv_buffer,                &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                      depth, CELL_DATA,                             &
                                      back_front_offset(FIELD_DENSITY), &
                                      offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_unpack_message_back(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%energy0,                  &
                                      chunks(chunk)%back_rcv_buffer,                &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                      depth, CELL_DATA,                             &
                                      back_front_offset(FIELD_ENERGY0), &
                                      offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_unpack_message_back(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%energy1,                  &
                                      chunks(chunk)%back_rcv_buffer,                &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                      depth, CELL_DATA,                             &
                                      back_front_offset(FIELD_ENERGY1), &
                                      offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_unpack_message_back(chunks(chunk)%field%x_min,                    &
                                      chunks(chunk)%field%x_max,                    &
                                      chunks(chunk)%field%y_min,                    &
                                      chunks(chunk)%field%y_max,                    &
                                      chunks(chunk)%field%z_min,                    &
                                      chunks(chunk)%field%z_max,                    &
                                      chunks(chunk)%field%vector_p,                  &
                                      chunks(chunk)%back_rcv_buffer,                &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                      depth, CELL_DATA,                             &
                                      back_front_offset(FIELD_P), &
                                      offload,dev_id)
    
  ENDIF
END SUBROUTINE tea_unpack_back

SUBROUTINE tea_pack_front(chunk, fields, depth, back_front_offset,offload,dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER        :: chunk, fields(:), depth, tot_packb, back_front_offset(:)
  INTEGER :: offload,dev_id

  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_pack_message_front(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%density,                 &
                                     chunks(chunk)%front_snd_buffer,               &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     back_front_offset(FIELD_DENSITY), &
                                     offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_pack_message_front(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%energy0,                  &
                                     chunks(chunk)%front_snd_buffer,               &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     back_front_offset(FIELD_ENERGY0), &
                                     offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_pack_message_front(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%energy1,                  &
                                     chunks(chunk)%front_snd_buffer,               &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     back_front_offset(FIELD_ENERGY1), &
                                     offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_pack_message_front(chunks(chunk)%field%x_min,                    &
                                     chunks(chunk)%field%x_max,                    &
                                     chunks(chunk)%field%y_min,                    &
                                     chunks(chunk)%field%y_max,                    &
                                     chunks(chunk)%field%z_min,                    &
                                     chunks(chunk)%field%z_max,                    &
                                     chunks(chunk)%field%vector_p,                  &
                                     chunks(chunk)%front_snd_buffer,               &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                     depth, CELL_DATA,                             &
                                     back_front_offset(FIELD_P), &
                                     offload,dev_id)
    
  ENDIF
END SUBROUTINE tea_pack_front

SUBROUTINE tea_send_recv_message_front(front_snd_buffer, front_rcv_buffer,        &
                                          chunk, total_size,                         &
                                          tag_send, tag_recv,                        &
                                          req_send, req_recv)

  IMPLICIT NONE

  REAL(KIND=8) :: front_snd_buffer(:), front_rcv_buffer(:)
  INTEGER      :: front_task
  INTEGER      :: chunk
  INTEGER      :: total_size, tag_send, tag_recv, err
  INTEGER      :: req_send, req_recv

  front_task=chunks(chunk)%chunk_neighbours(chunk_front)-1

  CALL MPI_ISEND(front_snd_buffer,total_size,MPI_DOUBLE_PRECISION,front_task,tag_send &
                ,MPI_COMM_WORLD,req_send,err)

  CALL MPI_IRECV(front_rcv_buffer,total_size,MPI_DOUBLE_PRECISION,front_task,tag_recv &
                ,MPI_COMM_WORLD,req_recv,err)

END SUBROUTINE tea_send_recv_message_front

SUBROUTINE tea_unpack_front(fields, chunk, depth,                        &
                               front_rcv_buffer,                            &
                               back_front_offset, &
                               offload,dev_id)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER         :: fields(:), chunk, depth, back_front_offset(:), offload,dev_id
  REAL(KIND=8)    :: front_rcv_buffer(:)

  IF(fields(FIELD_DENSITY).EQ.1) THEN
      CALL tea_unpack_message_front(chunks(chunk)%field%x_min,                    &
                                       chunks(chunk)%field%x_max,                    &
                                       chunks(chunk)%field%y_min,                    &
                                       chunks(chunk)%field%y_max,                    &
                                       chunks(chunk)%field%z_min,                    &
                                       chunks(chunk)%field%z_max,                    &
                                       chunks(chunk)%field%density,                 &
                                       chunks(chunk)%front_rcv_buffer,               &
                                       CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                       depth, CELL_DATA,                             &
                                       back_front_offset(FIELD_DENSITY), &
                                       offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY0).EQ.1) THEN
      CALL tea_unpack_message_front(chunks(chunk)%field%x_min,                    &
                                       chunks(chunk)%field%x_max,                    &
                                       chunks(chunk)%field%y_min,                    &
                                       chunks(chunk)%field%y_max,                    &
                                       chunks(chunk)%field%z_min,                    &
                                       chunks(chunk)%field%z_max,                    &
                                       chunks(chunk)%field%energy0,                  &
                                       chunks(chunk)%front_rcv_buffer,               &
                                       CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                       depth, CELL_DATA,                             &
                                       back_front_offset(FIELD_ENERGY0), &
                                       offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_ENERGY1).EQ.1) THEN
      CALL tea_unpack_message_front(chunks(chunk)%field%x_min,                    &
                                       chunks(chunk)%field%x_max,                    &
                                       chunks(chunk)%field%y_min,                    &
                                       chunks(chunk)%field%y_max,                    &
                                       chunks(chunk)%field%z_min,                    &
                                       chunks(chunk)%field%z_max,                    &
                                       chunks(chunk)%field%energy1,                  &
                                       chunks(chunk)%front_rcv_buffer,               &
                                       CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                       depth, CELL_DATA,                             &
                                       back_front_offset(FIELD_ENERGY1), &
                                       offload,dev_id)
    
  ENDIF
  IF(fields(FIELD_P).EQ.1) THEN
      CALL tea_unpack_message_front(chunks(chunk)%field%x_min,                    &
                                       chunks(chunk)%field%x_max,                    &
                                       chunks(chunk)%field%y_min,                    &
                                       chunks(chunk)%field%y_max,                    &
                                       chunks(chunk)%field%z_min,                    &
                                       chunks(chunk)%field%z_max,                    &
                                       chunks(chunk)%field%vector_p,                  &
                                       chunks(chunk)%front_rcv_buffer,               &
                                       CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA,&
                                       depth, CELL_DATA,                             &
                                       back_front_offset(FIELD_P), &
                                       offload,dev_id)
    
  ENDIF

END SUBROUTINE tea_unpack_front

SUBROUTINE tea_sum(value)

  ! Only sums to the master

  IMPLICIT NONE

  REAL(KIND=8) :: value

  REAL(KIND=8) :: total

  INTEGER :: err

  total=value

  CALL MPI_REDUCE(value,total,1,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,err)

  value=total

END SUBROUTINE tea_sum

SUBROUTINE tea_allsum(value)

  ! Global reduction for CG solver

  IMPLICIT NONE

  REAL(KIND=8) :: value

  REAL(KIND=8) :: total

  INTEGER :: err

  total=value

  CALL MPI_ALLREDUCE(value,total,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,err)

  value=total

END SUBROUTINE tea_allsum

SUBROUTINE tea_min(value)

  IMPLICIT NONE

  REAL(KIND=8) :: value

  REAL(KIND=8) :: minimum

  INTEGER :: err

  minimum=value

  CALL MPI_ALLREDUCE(value,minimum,1,MPI_DOUBLE_PRECISION,MPI_MIN,MPI_COMM_WORLD,err)

  value=minimum

END SUBROUTINE tea_min

SUBROUTINE tea_max(value)

  IMPLICIT NONE

  REAL(KIND=8) :: value

  REAL(KIND=8) :: maximum

  INTEGER :: err

  maximum=value

  CALL MPI_ALLREDUCE(value,maximum,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,err)

  value=maximum

END SUBROUTINE tea_max

SUBROUTINE tea_allgather(value,values)

  IMPLICIT NONE

  REAL(KIND=8) :: value

  REAL(KIND=8) :: values(parallel%max_task)

  INTEGER :: err

  values(1)=value ! Just to ensure it will work in serial

  CALL MPI_ALLGATHER(value,1,MPI_DOUBLE_PRECISION,values,1,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,err)

END SUBROUTINE tea_allgather

SUBROUTINE tea_check_error(error)

  IMPLICIT NONE

  INTEGER :: error

  INTEGER :: maximum

  INTEGER :: err

  maximum=error

  CALL MPI_ALLREDUCE(error,maximum,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,err)

  error=maximum

END SUBROUTINE tea_check_error


END MODULE tea_module
