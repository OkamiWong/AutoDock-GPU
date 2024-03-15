#pragma once

#include "preflight.hpp"

namespace preflight{
template <typename T>
inline cudaError_t wrappedCudaMalloc(T **ptr, size_t size) {
  auto error = cudaMalloc(ptr, size);
  registerArray(static_cast<void *>(*ptr), size);
  return error;
}
}
