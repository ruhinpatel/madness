# src/examples

set(EXAMPLE_SOURCES
    writecoeff writecoeff2 writecoeff3 writecoeff_json h2_write h2_write_json h2_write_json_2)

# Create executables for example applications
foreach(example ${EXAMPLE_SOURCES})
  add_mad_executable(${example} ${example}.cc "MADchem")
  add_dependencies(numerical-examples-madness ${example})
endforeach()

if(MADNESS_HAS_HDF5 AND TARGET MADNESS_HDF5)
  add_mad_executable(writecoeff_hdf5 writecoeff_hdf5.cc "MADchem")
  target_link_libraries(writecoeff_hdf5 PRIVATE MADNESS_HDF5)
  add_dependencies(numerical-examples-madness writecoeff_hdf5)

  if(BUILD_TESTING)
    add_test(
      NAME madness/test/examples/writecoeff_hdf5_serial/run
      COMMAND $<TARGET_FILE:writecoeff_hdf5>
              --h5=writecoeff_hdf5_serial.h5
              --dat=writecoeff_hdf5_serial.dat
              --max-roundtrip-error=1e-8)
    set_tests_properties(
      madness/test/examples/writecoeff_hdf5_serial/run
      PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        LABELS "examples;hdf5;short")

    if(ENABLE_MPI AND MPIEXEC_EXECUTABLE)
      add_test(
        NAME madness/test/examples/writecoeff_hdf5_mpi2/run
        COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:writecoeff_hdf5>
                --h5=writecoeff_hdf5_mpi2.h5
                --dat=writecoeff_hdf5_mpi2.dat
                --max-roundtrip-error=1e-8
                ${MPIEXEC_POSTFLAGS})
      set_tests_properties(
        madness/test/examples/writecoeff_hdf5_mpi2/run
        PROPERTIES
          WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
          LABELS "examples;hdf5;mpi;short")

      add_test(
        NAME madness/test/examples/writecoeff_hdf5_mpi2_threads2/run
        COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:writecoeff_hdf5>
                --h5=writecoeff_hdf5_mpi2_threads2.h5
                --dat=writecoeff_hdf5_mpi2_threads2.dat
                --max-roundtrip-error=1e-8
                ${MPIEXEC_POSTFLAGS})
      set_tests_properties(
        madness/test/examples/writecoeff_hdf5_mpi2_threads2/run
        PROPERTIES
          WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
          ENVIRONMENT "MAD_NUM_THREADS=2"
          LABELS "examples;hdf5;mpi;short")
    endif()
  endif()
endif()
