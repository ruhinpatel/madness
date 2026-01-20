# src/examples

set(EXAMPLE_SOURCES function_hdf5_v0_2d)

# Create executables for example applications
foreach(example ${EXAMPLE_SOURCES})
  add_mad_executable(${example} ${example}.cc "MADchem")
  add_dependencies(numerical-examples-madness ${example})
endforeach()
