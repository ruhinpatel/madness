if(ENABLE_HDF5)

  # Support HDF5 discovered from system paths, module environments, or
  # explicit hints such as HDF5_ROOT/HDF5_DIR.
  find_package(HDF5 COMPONENTS C)
  set_package_properties(HDF5 PROPERTIES
    TYPE OPTIONAL
    PURPOSE "Enables HDF5-backed I/O examples and support code")

  if(HDF5_FOUND)
    message(STATUS "Found optional package HDF5: version='${HDF5_VERSION}'")
    add_library(MADNESS_HDF5 INTERFACE)

    # Prefer imported targets when available.
    if(TARGET HDF5::HDF5)
      target_link_libraries(MADNESS_HDF5 INTERFACE HDF5::HDF5)
    elseif(TARGET hdf5::hdf5)
      target_link_libraries(MADNESS_HDF5 INTERFACE hdf5::hdf5)
    else()
      if(HDF5_INCLUDE_DIRS)
        target_include_directories(MADNESS_HDF5 INTERFACE ${HDF5_INCLUDE_DIRS})
      endif()
      if(HDF5_DEFINITIONS)
        target_compile_options(MADNESS_HDF5 INTERFACE ${HDF5_DEFINITIONS})
      endif()
      target_link_libraries(MADNESS_HDF5 INTERFACE ${HDF5_LIBRARIES})
    endif()

    set(MADNESS_HAS_HDF5 1)
  else()
    message(FATAL_ERROR "ENABLE_HDF5=ON but HDF5 was not found. Load an HDF5 module or set HDF5_ROOT/HDF5_DIR.")
  endif()

endif()
