#pragma once

#include "pch.hpp"

class Tensor;

void sgemm(uint32_t m, uint32_t n, uint32_t k, float alpha, float *a, float *b, float beta, float *c);

Tensor* sin(Tensor *in);

Tensor* cos(Tensor *in);
