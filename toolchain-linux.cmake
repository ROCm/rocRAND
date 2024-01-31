#set(CMAKE_MAKE_PROGRAM "nmake.exe")
#set(CMAKE_GENERATOR "Ninja")
# Ninja doesn't support platform
#set(CMAKE_GENERATOR_PLATFORM x64)

if (DEFINED ENV{ROCM_PATH})
  set(rocm_bin "$ENV{ROCM_PATH}/bin")
  set(llvm_bin "$ENV{ROCM_PATH}/llvm/bin")
else()
  set(rocm_bin "/opt/rocm/bin")
  set(llvm_bin "/opt/rocm/llvm/bin")
endif()


set(CMAKE_CXX_COMPILER "${llvm_bin}/clang++")
set(CMAKE_C_COMPILER "${llvm_bin}/clang")
