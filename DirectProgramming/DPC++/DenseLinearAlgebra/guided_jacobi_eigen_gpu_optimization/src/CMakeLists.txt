set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O0")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -DMKL_ILP64 -I$MKLROOT/include -L$MKLROOT/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl")

add_executable(1_guided_jacobi_eigen_solver_cpu 1_guided_jacobi_eigen_solver_cpu.cpp)
add_executable(2_guided_jacobi_eigen_solver_gpu_offload 2_guided_jacobi_eigen_solver_gpu_offload.cpp)
#add_executable(3_guided_jacobi_eigen_solver_multi_gpu 3_guided_jacobi_eigen_solver_multi_gpu.cpp)

target_link_libraries(1_guided_jacobi_eigen_solver_cpu OpenCL sycl)
target_link_libraries(2_guided_jacobi_eigen_solver_gpu_offload OpenCL sycl)
#target_link_libraries(3_guided_jacobi_eigen_solver_multi_gpu OpenCL sycl)

add_custom_target(run_1_cpu 1_guided_jacobi_eigen_solver_cpu)
add_custom_target(run_2_gpu 2_guided_jacobi_eigen_solver_gpu_offload)
#add_custom_target(run_3_multi_gpu 3_guided_jacobi_eigen_solver_multi_gpu)
