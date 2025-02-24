# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(Decoding STATIC Decoding.cc)
set_property(TARGET Decoding PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Decoding PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Decoding PUBLIC -lcublas -lcudart -lcurand Decoder decoding_kernels
                        BeamSearchLayer DynamicDecodeLayer tensor cuda_utils logger)

add_executable(decoding_gemm decoding_gemm.cc)
target_link_libraries(decoding_gemm PUBLIC -lcublas -lcublasLt -lcudart decoding_gemm_func memory_utils cuda_utils logger)
