
/*
 * MIT License
 *
 * Copyright (c) 2018 Red-Portal
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */ 

#include <cstdio>
#include <random>
#include <blaze/Blaze.h>

#include "STARS.hpp"

std::random_device seed;
std::mt19937 rng(seed());
std::normal_distribution<double> dist(0, 1);

double one_dim(double x)
{
    double noise = dist(rng);
    double sigma = 1;
    return (x - 5) * (x - 3) + sigma * noise;
}

auto b = blaze::DynamicVector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
double ten_dim(blaze::DynamicVector<double> x)
{
    double noise = dist(rng);
    double sigma = 1;
    return blaze::dot(x, x) + blaze::dot(x, b) + sigma * noise;
}

int main()
{
    {
        printf(" *** R^1 optimization problem ***\n");
        double initial = 10.0;
        size_t max_iter = 100;
        double L1 = 15;
        double sigma = 2;
        double a = 0.3;
        double k0 = 1;
        double alpha = 0.6;
        bool L1_clipping = true;
        size_t verbose = 2;

        auto ret = dfopt::stars_onedim(one_dim,
                                       initial, 
                                       max_iter,
                                       L1,
                                       sigma,
                                       a,
                                       k0,
                                       alpha,
                                       rng,
                                       verbose,
                                       L1_clipping);
        printf("error of solution: %f\n\n", std::abs(4.0 - ret));
        
    }
    {
        printf(" *** R^10 optimization problem ***\n");
        auto initial = blaze::DynamicVector<double>{1,1,1,1,1,1,1,1,1,1};

        size_t max_iter = 100;
        double L1 = 10;
        double sigma = 3;
        double a = 0.1;
        double k0 = 1;
        double alpha = 0.6;
        bool L1_clipping = false;
        size_t verbose = 2;

        auto ret = dfopt::stars(ten_dim,
                                initial, 
                                max_iter,
                                L1,
                                sigma,
                                a,
                                k0,
                                alpha,
                                rng,
                                verbose,
                                L1_clipping);
        auto answer = (-0.5) * b; 
        printf("error of solution: %f\n\n", dfopt::l2norm(ret - answer));
    }
}
