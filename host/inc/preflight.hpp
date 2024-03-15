#pragma once

#include <cstdio>
#include <map>
#include <vector>

namespace preflight {
inline std::map<void *, size_t> arrayAddressToArraySizeMap;
inline std::vector<std::vector<void *>> kernelDataDependencies;

template <typename T>
inline void registerArray(T *array, size_t size) {
  arrayAddressToArraySizeMap[static_cast<void *>(array)] = size;
}

inline void registerKernel(std::vector<void *> arrays) {
  kernelDataDependencies.push_back(arrays);
}

inline void printResult() {
  size_t totalArraySize = 0;
  for (const auto &[_, size] : arrayAddressToArraySizeMap) {
    totalArraySize += size;
  }

  size_t bottleNeckKernelDataDependencySize = 0;
  int count = 0;
  for (const auto &arrays : kernelDataDependencies) {
    size_t size = 0;
    for (auto array : arrays) {
      size += arrayAddressToArraySizeMap[array];
    }
    printf("[preflight] kernel[%d]: %.4lf MiB\n", count++, static_cast<double>(size) / 1024.0 / 1024.0);
    bottleNeckKernelDataDependencySize = std::max(bottleNeckKernelDataDependencySize, size);
  }

  count = 0;
  for (const auto &[_, size] : arrayAddressToArraySizeMap) {
    printf("[preflight] array[%d]: %.4lf MiB\n", count++, static_cast<double>(size) / 1024.0 / 1024.0);
  }

  printf(
    "[preflight] Total array size (MiB): %.4lf\n",
    static_cast<double>(totalArraySize) / 1024.0 / 1024.0
  );
  printf(
    "[preflight] Bottleneck kernel data dependency size (MiB): %.4lf\n",
    static_cast<double>(bottleNeckKernelDataDependencySize) / 1024.0 / 1024.0
  );
  printf(
    "[preflight] Bottleneck / Total: %.4lf\n",
    static_cast<double>(bottleNeckKernelDataDependencySize) / static_cast<double>(totalArraySize)
  );
}
}  // namespace preflight
