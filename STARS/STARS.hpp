
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

#ifndef _DERIVATIVE_FREE_STARS_HPP_
#define _DERIVATIVE_FREE_STARS_HPP_

#include <iostream>
#include <random>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <blaze/math/DynamicVector.h>

namespace dfopt
{
    enum verbose { };

    template<typename Float>
    inline bool
    quad_root(Float a, Float b, Float c, Float* fst, Float* snd)
    {
        Float inner = (b * b) - (4 * a * c);
        if(inner < 0)
            return false;

        *fst = (sqrt(inner) - b) / ( 2 * a );
        *snd = (sqrt(inner) + b) / ( 2 * a );
        return true;
    }

    template<typename Vec, typename Rng, typename Dist>
    inline void
    randn(Vec& vec, Rng& rng, Dist& dist, size_t n)
    {
        for(size_t i = 0; i < n; ++i) {
            vec[i] = dist(rng);
        }
    }

    template<typename Vec>
    inline double
    l2norm(Vec const& vec)
    {
        return sqrt(blaze::dot(vec, vec));
    }

    template<typename F, typename Rng, typename Float>
    Float
    stars_onedim(F&& f,
                 Float initial,
                 size_t max_iter,
                 Float L1,
                 Float sigma,
                 Float a,
                 Float k0,
                 Float alpha,
                 Rng& rng,
                 size_t verbose = 1,
                 bool L1_clipping = false)
    /*
     * Copyright Red-Portal 2018
     *
     * STARS Algorithm by R. Chen, S. Wild
     * https://arxiv.org/pdf/1507.03332.pdf
     * with annealing step size
     *
     * f        : Objective function
     * max iter : maximum iteration
     * L1       : Lipschitz constant
     * sigma    : Estimated stdandard deviation of noise in objective function observation
     * 
     * for, iteration k
     * the step size is calculated as: a / ((k0 + k) ^ alpha)
     * a     : step size
     * k0    : smoothing constant
     * alpha : step size decrease rate
     * 
     * rng         : C++ standard conforming pseudo-random number generator
     * L1 Clipping : Regects approximated derivative bigger than L1
     * verbose     : Output log verboseness level 
     */
    {
        assert(a > 0 && "coefficient \'a\' should conform \'a\' > 0");
        assert(L1 > 0 && "coefficient \'L1\' should conform \'L1\' > 0");
        assert(sigma > 0 && "coefficient \'sigma\' should conform \'sigma\' > 0");
        assert(0.5 < alpha && alpha <= 1 &&
               "alpha should conform 0.5 <= \'alpha\' < 1");

        auto dist = std::normal_distribution<Float>(0, 1);
        Float bwd = 0;
        Float cost = 0;
        Float fwd = 0;
        Float g = 0;
        Float h = 0;
        Float mu = pow((8 * sigma * sigma)/ (L1 * L1 * 343), 0.25);
        Float p = 0;
        Float point = initial;
        size_t iter = 1;

        if(verbose > 0)
        {
            printf(" ---- STARS: STep Approximation in Radomized Search ----\n\n");
            printf(" L1    : %f\n", L1);
            printf(" sigma : %f\n", sigma);
            printf(" a     : %f\n", a);
            printf(" k0    : %f\n", k0);
            printf(" alpha : %f\n", alpha);
            printf(" initial cost  %f\n\n", f(initial));
            printf("------------------------\n");
            printf(" iter       cost");
            if(verbose >= 2)
                printf("      forward      stepsize");
            printf("\n");
        }

        while(iter <= max_iter)
        {
            do
            { p = dist(rng); } while(p == 0);

            cost = f(point + p * mu);

            fwd = f(point + p * mu);
            bwd = f(point);
            g = (fwd - bwd) / (mu * p);

            if(L1_clipping && std::abs(g) > L1)
                continue;
            
            h = a / pow(iter + k0, alpha);
            point -= h * g;

            if(verbose > 0)
            {
                printf(" %5d  %10.5f", iter, bwd);
                if(verbose >= 2)
                    printf("  %10.5f  %10.5f", fwd, h);
                printf("\n");
            }
            ++iter;
        }

        if(verbose > 0)
        {
            printf("------------------------\n");
            printf(" optimization terminated \n");
            printf(" final cost %5f \n\n", bwd);
        }
        return point;
    }


    template<typename Float>
    using Vector = blaze::DynamicVector<Float>;

    template<typename F, typename Rng, typename Float>
    Vector<Float> 
    stars(F&& f,
          Vector<Float> const& initial,
          size_t max_iter,
          Float L1,
          Float sigma,
          Float a,
          Float k0,
          Float alpha,
          Rng& rng,
          size_t verbose = 1,
          bool L1_clipping = false)
    /*
     * Copyright Red-Portal 2018
     *
     * STARS Algorithm by R. Chen, S. Wild
     * https://arxiv.org/pdf/1507.03332.pdf
     * with annealing step size
     *
     * f        : Objective function
     * dim_n    : dimension of problm
     * max iter : maximum iteration
     * L1       : Lipschitz constant
     * sigma    : Estimated stdandard deviation of noise in objective function observation
     * 
     * for, iteration k
     * the step size is calculated as: a / ((k0 + k) ^ alpha)
     * a     : step size
     * k0    : smoothing constant
     * alpha : step size decrease rate
     * 
     * rng         : C++ standard conforming pseudo-random number generator
     * L1 Clipping : Regects approximated derivative bigger than L1
     * verbose     : Output log verboseness level 
     */
    {
        assert(a > 0 && "coefficient \'a\' should conform \'a\' > 0");
        assert(L1 > 0 && "coefficient \'L1\' should conform \'L1\' > 0");
        assert(sigma > 0 && "coefficient \'sigma\' should conform \'sigma\' > 0");
        assert(0.5 < alpha && alpha <= 1 &&
               "alpha should conform 0.5 <= \'alpha\' < 1");

        auto dist = std::normal_distribution<Float>(0, 1);
        auto point = initial;
        auto p = Vector<Float>(point.size());
        auto g = Vector<Float>(point.size());
        Float n = initial.size();
        Float bwd = 0;
        Float cost = 0;
        Float fwd = 0;
        Float h = 0;
        Float temp = (n + 6);
        Float mu = pow((8 * sigma * sigma)/ (L1 * L1 * temp * temp * temp), 0.25);
        size_t iter = 1;

        if(verbose > 0)
        {
            printf(" ---- STARS: STep Approximation in Radomized Search ----\n\n");
            printf(" L1    : %f\n", L1);
            printf(" sigma : %f\n", sigma);
            printf(" a     : %f\n", a);
            printf(" k0    : %f\n", k0);
            printf(" alpha : %f\n", alpha);
            printf(" initial cost  %f\n\n", f(initial));
            printf("------------------------\n");
            printf(" iter       cost");
            if(verbose >= 2)
                printf("      forward     stepsize");
            printf("\n");
        }

        while(iter <= max_iter)
        {
            randn(p, rng, dist, p.size());
            fwd = f(point + (p * mu));
            bwd = f(point);
            g = ((fwd - bwd) * p) / mu;

            if(L1_clipping && l2norm(g) > L1)
                continue;
            
            h = a / pow(iter + k0, alpha);
            point = point - h * g;

            if(verbose > 0)
            {
                printf(" %5d  %10.5f", iter, bwd);
                if(verbose >= 2)
                    printf("  %10.5f  %10.5f", fwd, h);
                printf("\n");
            }
            ++iter;
        }

        if(verbose > 0)
        {
            printf("------------------------\n");
            printf(" optimization terminated \n");
            printf(" final cost %5f \n\n", bwd);
        }
        return point;
    }
}

#endif
