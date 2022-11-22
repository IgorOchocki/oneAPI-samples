#include <bits/stdc++.h>

#include <CL/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <oneapi/dpl/random>
#include <vector>

#include "oneapi/mkl.hpp"

using namespace sycl;

// Selectors for specific targets
gpu_selector selector;

typedef float Real;

extern void PrintMatrix(char *desc, int m, int n, Real *a, int lda);

const Real kEpsilon = 0.0001;
const Real kDifferenceError = 1e-02;
const std::uint32_t kSeed = 666;
static const int kSize = 250;
static const int kMaxSweeps = kSize * kSize * kSize;

std::ofstream outfile;

/* Auxiliary routine: printing a matrix */
void PrintMatrix(char *desc, int m, int n, Real *a, int lda) {
  int i, j;
  printf("\n %s\n", desc);
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) printf(" %6.2f", a[i + j * lda]);
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  auto begin_runtime = std::chrono::high_resolution_clock::now();
  queue q(selector);

  // outfile.open("report.txt", std::ios_base::out);

  Real *A = malloc_shared<Real>(kSize * kSize, q);
  memset(A, 0, sizeof(*A) * kSize * kSize);
  Real *B = malloc_shared<Real>(kSize * kSize, q);
  memset(B, 0, sizeof(*B) * kSize * kSize);
  Real *matrix = malloc_shared<Real>(kSize * kSize, q);
  memset(matrix, 0, sizeof(*matrix) * kSize * kSize);

  std::cout << "Device : " << q.get_device().get_info<info::device::name>()
            << std::endl;

  auto begin_matrix = std::chrono::high_resolution_clock::now();

  q.submit([&](handler &h) {
    h.parallel_for(range<1>(kSize), [=](id<1> id) {
      int i = id;

      for (int j = i * kSize + i; j < i * kSize + kSize; ++j) {
        oneapi::dpl::minstd_rand engine(kSeed, j);
        oneapi::dpl::uniform_real_distribution<Real> distr(-0.5, 0.5);
        A[j] = distr(engine);
      }
    });
  });
  q.wait();

  q.submit([&](handler &h) {
    h.parallel_for(range<1>(kSize * kSize), [=](id<1> id) {
      int i = id / kSize;
      int j = id % kSize;
      B[id] = A[kSize * j + i];
    });
  });
  q.wait();

  q.submit([&](handler &h) {
    h.parallel_for(range<1>(kSize), [=](id<1> i) {
      for (int j = 0; j < kSize; j++) {
        Real tmp = 0;
        for (int l = 0; l < kSize; l++) {
          tmp += B[kSize * i + l] * A[kSize * l + j];
        }
        matrix[kSize * i + j] = tmp;
        if (i == j % kSize) matrix[kSize * i + i] *= 10;
      }
    });
  });
  q.wait();

  auto end_matrix = std::chrono::high_resolution_clock::now();
  auto elapsed_matrix = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_matrix - begin_matrix);

  std::cout << "\nMatrix generated, time elapsed: "
            << elapsed_matrix.count() * 1e-9 << " seconds.\n";
  Real ei[kSize * kSize];
  for (int i = 0; i < kSize * kSize; ++i) ei[i] = matrix[i];

  std::cout << std::endl;

  auto begin_computations = std::chrono::high_resolution_clock::now();

  int sweeps = 0;

  Real *eigen_vectors = malloc_shared<Real>(kSize * kSize, q);
  memset(eigen_vectors, 0, sizeof(*eigen_vectors) * kSize * kSize);
  Real *eigen_values = malloc_shared<Real>(kSize, q);
  memset(eigen_values, 0, sizeof(*eigen_values) * kSize);
  Real *calculation_matrix = malloc_shared<Real>(kSize * kSize, q);
  memset(calculation_matrix, 0, sizeof(*calculation_matrix) * kSize * kSize);
  int max_element_index;
  Real max_element;

  // The rotor (eigen_vectors) accumulation
  Real *eigen_vector_i_row = malloc_shared<Real>(kSize, q);
  Real *eigen_vector_j_row = malloc_shared<Real>(kSize, q);

  for (int n = 0; n < kSize * kSize; n++) calculation_matrix[n] = matrix[n];

  // Setting up the eigenvector matrix
  for (int i = 0; i < kSize * kSize; i++) {
    eigen_vectors[i] = 0;
  }
  for (int i = 0; i < kSize; i++) {
    eigen_vectors[i * kSize + i] = 1.0;
  }

  max_element = 0;
  int indexOfMaxElement = 0;
  int vectorLength = kSize * kSize;
  int n = 1;
  for (int i = 1; i < vectorLength; i++) {
    if (fabs(max_element) < fabs(calculation_matrix[i])) {
      max_element = calculation_matrix[i];
      max_element_index = i;
    }
    if (i == n * kSize - 1) {
      n++;
      i = i + n;
    }
  }

  do {
    Real cosine, sine;
    int i_max = max_element_index / kSize;
    int j_max = max_element_index - max_element_index / kSize * kSize;

    // Angle calculations. // Function to find the values of cos and sin
    if (calculation_matrix[j_max * kSize + i_max] != 0.0) {
      Real theta, tangent;
      theta = (calculation_matrix[j_max * kSize + j_max] -
               calculation_matrix[i_max * kSize + i_max]) /
              (2.0 * calculation_matrix[j_max * kSize + i_max]);
      if (theta > 0) {
        tangent = 1.0 / (theta + sqrt(1.0 + theta * theta));
      } else {
        tangent = -1.0 / (-theta + sqrt(1.0 + theta * theta));
      }
      cosine = 1 / sqrt(1 + tangent * tangent);
      sine = cosine * tangent;
    } else {
      cosine = 1.0;
      sine = 0.0;
    }

    q.submit([&](handler &h) {
      h.parallel_for(range<1>(kSize), [=](id<1> j) {
        eigen_vector_i_row[j] = eigen_vectors[i_max * kSize + j];
        eigen_vector_j_row[j] = eigen_vectors[j_max * kSize + j];

        eigen_vectors[i_max * kSize + j] =
            eigen_vector_i_row[j] * cosine - eigen_vector_j_row[j] * sine;
        eigen_vectors[j_max * kSize + j] =
            eigen_vector_i_row[j] * sine + eigen_vector_j_row[j] * cosine;
      });
    });
    // calculation_matrix calculations
    Real tmp_ii = calculation_matrix[i_max * kSize + i_max];
    Real tmp_jj = calculation_matrix[j_max * kSize + j_max];
    // Real tmp_ik, tmp_jk;

    // changing the matrix elements with indices i_max and j_max
    calculation_matrix[i_max * kSize + i_max] =
        cosine * cosine * tmp_ii -
        2.0 * sine * cosine * calculation_matrix[i_max * kSize + j_max] +
        sine * sine * tmp_jj;
    calculation_matrix[j_max * kSize + j_max] =
        sine * sine * tmp_ii +
        2.0 * sine * cosine * calculation_matrix[i_max * kSize + j_max] +
        cosine * cosine * tmp_jj;
    calculation_matrix[i_max * kSize + j_max] = 0.0;
    calculation_matrix[j_max * kSize + i_max] = 0.0;

    q.submit([&](handler &h) {
      h.parallel_for(range<1>(kSize), [=](id<1> l) {
        if (l != i_max && l != j_max) {
          Real tmp_ik = calculation_matrix[i_max * kSize + l];
          Real tmp_jk = calculation_matrix[j_max * kSize + l];

          calculation_matrix[i_max * kSize + l] =
              cosine * tmp_ik - sine * tmp_jk;
          calculation_matrix[l * kSize + i_max] =
              calculation_matrix[i_max * kSize + l];

          calculation_matrix[j_max * kSize + l] =
              sine * tmp_ik + cosine * tmp_jk;
          calculation_matrix[l * kSize + j_max] =
              calculation_matrix[j_max * kSize + l];
        }
      });
    });

    max_element = 0;
    indexOfMaxElement = 0;
    vectorLength = kSize * kSize;
    n = 1;

    q.wait();

    for (int i = 1; i < vectorLength; i++) {
      if (fabs(max_element) < fabs(calculation_matrix[i])) {
        max_element = calculation_matrix[i];
        max_element_index = i;
      }
      if (i == n * kSize - 1) {
        n++;
        i = i + n;
      }
    }

    sweeps++;
    // std::cout << "Sweep: " << sweeps << std::endl;
  } while (sweeps < kMaxSweeps &&
           fabs(calculation_matrix[max_element_index]) > kEpsilon);

  for (int n = 0; n < kSize; n++)
    eigen_values[n] = calculation_matrix[n * kSize + n];

  for (int i = 0; i < kSize; ++i) {
    for (int j = i + 1; j < kSize; ++j) {
      if (eigen_values[i] > eigen_values[j]) {
        Real tmp = eigen_values[i];
        eigen_values[i] = eigen_values[j];
        eigen_values[j] = tmp;
      }
    }
  }

  // PrintMatrix((char *)"Eigenvalues", 1, n, eigen_values, 1);

  std::cout << std::endl;
  auto end_computations = std::chrono::high_resolution_clock::now();
  auto elapsed_computations =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_computations -
                                                           begin_computations);

  std::cout << "Computations complete, time elapsed: "
            << elapsed_computations.count() * 1e-9 << " seconds.\n";
  std::cout << "Total number of sweeps: " << sweeps << std::endl;
  std::cout << "Checking results\n";

  Real w[kSize];
  std::int64_t scratchpad_size = kSize * kSize;
  Real scratchpad[scratchpad_size];
  printf(" DSYEVD Example Program Results\n");

  oneapi::mkl::lapack::syevd(q, oneapi::mkl::job::novec,
                             oneapi::mkl::uplo::upper, kSize, ei, kSize, w,
                             scratchpad, scratchpad_size);
  q.wait();
  /* Print eigenvalues */
  // PrintMatrix((char *)"Eigenvalues", 1, n, w, 1);

  auto begin_check = std::chrono::high_resolution_clock::now();

  bool all_is_correct = true;

  for (int i = 0; i < kSize; ++i)
    if (fabs(eigen_values[i] - w[i]) > kDifferenceError) {
      std::cout << fabs(eigen_values[i] - w[i]) << std::endl;
      all_is_correct = false;
    }

  if (all_is_correct)
    std::cout << "All values are correct\n";
  else
    std::cout << "There have been some errors\n";

  auto end_check = std::chrono::high_resolution_clock::now();
  auto elapsed_check = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_check - begin_check);

  std::cout << "\nCheck complete, time elapsed: "
            << elapsed_check.count() * 1e-9 << " seconds.\n";

  auto end_runtime = std::chrono::high_resolution_clock::now();
  auto elapsed_runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_runtime - begin_runtime);

  std::cout << "Total runtime is " << elapsed_runtime.count() * 1e-9
            << " seconds.\n";

  free(matrix, q);
  free(A, q);
  free(B, q);
  free(eigen_vectors, q);
  free(eigen_values, q);
  free(calculation_matrix, q);
  // outfile.close();

  return 0;
}
