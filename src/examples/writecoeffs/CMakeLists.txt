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
endif()
