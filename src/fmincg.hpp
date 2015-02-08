/*******************************************************************************
 * Copyright (c) 2015 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: fmincg.hpp
 *
 * Description:
 *      description
 *
 * Authors:
 *          Wojciech Migda (wm)
 *
 *******************************************************************************
 * History:
 * --------
 * Date         Who  Ticket     Description
 * ----------   ---  ---------  ------------------------------------------------
 * 2015-02-04   wm              Initial version
 *
 ******************************************************************************/

/*
 * Minimize a continuous differentialble multivariate function. Starting point <br/>
 * is given by "X" (D by 1), and the function named in the string "f", must<br/>
 * return a function value and a vector of partial derivatives. The Polack-<br/>
 * Ribiere flavour of conjugate gradients is used to compute search directions,<br/>
 * and a line search using quadratic and cubic polynomial approximations and the<br/>
 * Wolfe-Powell stopping criteria is used together with the slope ratio method<br/>
 * for guessing initial step sizes. Additionally a bunch of checks are made to<br/>
 * make sure that exploration is taking place and that extrapolation will not<br/>
 * be unboundedly large. The "length" gives the length of the run: if it is<br/>
 * positive, it gives the maximum number of line searches, if negative its<br/>
 * absolute gives the maximum allowed number of function evaluations. You can<br/>
 * (optionally) give "length" a second component, which will indicate the<br/>
 * reduction in function value to be expected in the first line-search (defaults<br/>
 * to 1.0). The function returns when either its length is up, or if no further<br/>
 * progress can be made (ie, we are at a minimum, or so close that due to<br/>
 * numerical problems, we cannot get any closer). If the function terminates<br/>
 * within a few iterations, it could be an indication that the function value<br/>
 * and derivatives are not consistent (ie, there may be a bug in the<br/>
 * implementation of your "f" function). The function returns the found<br/>
 * solution "X", a vector of function values "fX" indicating the progress made<br/>
 * and "i" the number of iterations (line searches or function evaluations,<br/>
 * depending on the sign of "length") used.<br/>
 * <br/>
 * Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)<br/>
 * <br/>
 * See also: checkgrad <br/>
 * <br/>
 * Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13<br/>
 * <br/>
 * <br/>
 * (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen <br/>
 * Permission is granted for anyone to copy, use, or modify these<br/>
 * programs and accompanying documents for purposes of research or<br/>
 * education, provided this copyright notice is retained, and note is<br/>
 * made of any changes that have been made.<br/>
 * <br/>
 * These programs and documents are distributed without any warranty,<br/>
 * express or implied. As the programs were written for research<br/>
 * purposes only, they have not been tested to the degree that would be<br/>
 * advisable in any important application. All use of these programs is<br/>
 * entirely at the user's own risk.<br/>
 * <br/>
 */

#ifndef FMINCG_HPP_
#define FMINCG_HPP_

#include <utility>
#include <valarray>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace num
{

// based on:
// https://github.com/thomasjungblut/tjungblut-math-cpp/blob/master/tjungblut-math%2B%2B/source/src/Fmincg.cpp
template<typename _ValueType>
std::valarray<_ValueType>
fmincg(
    std::function<std::pair<_ValueType, std::valarray<_ValueType>> (const std::valarray<_ValueType>)> cost_gradient_fn,
    std::valarray<_ValueType> theta,
    int maxiter,
    bool verbose=false
)
{
    typedef _ValueType value_type;
    typedef std::valarray<value_type> vector;

    // number of extrapolation runs, set to a higher value for smaller ravine landscapes
    constexpr value_type EXT = 3.0;
    // a bunch of constants for line searches
    constexpr value_type RHO = 0.01;
    // RHO and SIG are the constants in the Wolfe-Powell conditions
    constexpr value_type SIG = 0.5;
    // don't reevaluate within 0.1 of the limit of the current bracket
    constexpr value_type INT = 0.1;
    // max 20 function evaluations per line search
    constexpr int MAX = 20;
    // maximum allowed slope ratio
    constexpr value_type RATIO = 100.0;

    // we start by setting up all memory that we will need in terms of vectors,
    // while calculating we will just fill this memory (overloaded << uses memcpy)

    // input will be the pointer to our current active parameter set
    vector & input(theta);
    vector X0 = vector(input);

    // search directions
    vector s = vector(input.size());
    // gradients
    vector df0 = vector(input.size());
    vector df1 = vector(input.size());
    vector df2 = vector(input.size());

    // define some integers for bookkeeping and then start
    int M = 0;
    int i = 0; // zero the run length counter
    constexpr int red = 1; // starting point
    int ls_failed = 0; // no previous line search has failed

    const std::pair<value_type, vector> cost_gradient = cost_gradient_fn(input);
    value_type f1 = cost_gradient.first;
    df1 = std::move(cost_gradient.second);

    i = i + (maxiter < 0 ? 1 : 0);
    // search direction is steepest
    s = -df1;

    value_type d1 = -(s * s).sum(); // this is the slope
    value_type z1 = red / (1.0 - d1); // initial step is red/(|s|+1)

    while (i < std::abs(maxiter)) // while not finished
    {
        i = i + (maxiter > 0 ? 1 : 0); // count iterations?!
        // make a copy of current values
        X0 = input;
        value_type f0 = f1;
        df0 = df1;

        // begin line search
        // fill our new line searched parameters
        input = input + (s * z1);
        const std::pair<value_type, vector> evaluateCost2 = cost_gradient_fn(input);
        value_type f2 = evaluateCost2.first;
        df2 = std::move(evaluateCost2.second);
        i = i + (maxiter < 0 ? 1 : 0); // count epochs
        value_type d2 = (df2 * s).sum();

        // initialize point 3 equal to point 1
        value_type f3 = f1;
        value_type d3 = d1;
        value_type z3 = -z1;
        if (maxiter > 0)
        {
            M = MAX;
        }
        else
        {
            M = std::min(MAX, - maxiter - i);
        }
        // initialize quantities
        int success = 0;
        value_type limit = -1;

        while (true)
        {
            while (((f2 > f1 + z1 * RHO * d1) | (d2 > -SIG * d1)) && (M > 0))
            {
                // tighten the bracket
                limit = z1;
                value_type z2 = 0.0;
                if (f2 > f1)
                {
                    // quadratic fit
                    z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
                }
                else
                {
                    // cubic fit
                    const value_type A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
                    const value_type B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                    // numerical error possible - ok!
                    z2 = (std::sqrt(B * B - A * d2 * z3 * z3) - B) / A;
                }
                if (std::isnan(z2) || !std::isfinite(z2))
                {
                    // if we had a numerical problem then bisect
                    z2 = z3 / 2.0;
                }
                // don't accept too close to limits
                z2 = std::max(std::min(z2, INT * z3), (1 - INT) * z3);
                // update the step
                z1 = z1 + z2;
                input += (s * z2);
                std::pair<value_type, vector> evaluateCost3 = cost_gradient_fn(input);
                f2 = evaluateCost3.first;
                df2 = std::move(evaluateCost3.second);
                M = M - 1;
                i = i + (maxiter < 0 ? 1 : 0); // count epochs
                d2 = (df2 * s).sum();
                // z3 is now relative to the location of z2
                z3 = z3 - z2;
            }

            if (f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1)
            {
                break; // this is a failure
            }
            else if (d2 > SIG * d1)
            {
                success = 1;
                break; // success
            }
            else if (M == 0)
            {
                break; // failure
            }
            // make cubic extrapolation
            const value_type A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
            const value_type B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
            value_type z2 = -d2 * z3 * z3 / (B + std::sqrt(B * B - A * d2 * z3 * z3));
            // num prob or wrong sign?
            if (std::isnan(z2) || !std::isfinite(z2) || z2 < 0)
            {
                // if we have no upper limit
                if (limit < -0.5)
                {
                    // the extrapolate the maximum amount
                    z2 = z1 * (EXT - 1);
                }
                else
                {
                    // otherwise bisect
                    z2 = (limit - z1) / 2;
                }
            }
            else if ((limit > -0.5) && (z2 + z1 > limit))
            {
                // extraplation beyond max?
                z2 = (limit - z1) / 2; // bisect
            }
            else if ((limit < -0.5) && (z2 + z1 > z1 * EXT))
            {
                // extrapolationbeyond limit
                z2 = z1 * (EXT - 1.0); // set to extrapolation limit
            }
            else if (z2 < -z3 * INT)
            {
                z2 = -z3 * INT;
            }
            else if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - INT)))
            {
                // too close to the limit
                z2 = (limit - z1) * (1.0 - INT);
            }
            // set point 3 equal to point 2
            f3 = f2;
            d3 = d2;
            z3 = -z2;
            z1 = z1 + z2;
            // update current estimates
            input += (s * z2);
            const std::pair<value_type, vector> evaluateCost3 = cost_gradient_fn(input);
            f2 = evaluateCost3.first;
            df2 = std::move(evaluateCost3.second);
            M = M - 1;
            i = i + (maxiter < 0 ? 1 : 0); // count epochs?!
            d2 = (df2 * s).sum();
        } // end of line search

        if (success == 1) // if line search succeeded
        {
            f1 = f2;
            if (verbose)
            {
                std::cout << "Iteration " << i << " | Cost: " << f1 << std::endl;
            }
            // Polack-Ribiere direction: s =
            // (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;
            const value_type df2len = (df2 * df2).sum();
            const value_type df12len = (df1* df2).sum();
            const value_type df1len = (df1 * df1).sum();
            const value_type numerator = (df2len - df12len) / df1len;
            s = (s * numerator) - df2;
            std::swap(df1, df2); // swap derivatives
            d2 = (df1 * s).sum();
            // new slope must be negative
            if (d2 > 0)
            {
                // otherwise use steepest direction
                s = -df1;
                d2 = -(s * s).sum();
            }
            // realmin in octave = 2.2251e-308
            // slope ratio but max RATIO
            const value_type thres = d1 / (d2 - std::numeric_limits<value_type>::min());
            z1 = z1 * std::min(RATIO, thres);
            d1 = d2;
            ls_failed = 0; // this line search did not fail
        }
        else
        {
            // restore data from the beginning of the iteration
            input = X0;
            f1 = f0;
            df1 = df0; // restore point from before failed line search
            // line search failed twice in a row?
            if (ls_failed == 1 || i > std::abs(maxiter))
            {
                break; // or we ran out of time, so we give up
            }
            // swap derivatives
            std::swap(df1, df2);
            // try steepest
            s = -df1;
            d1 = -(s * s).sum();
            z1 = 1.0 / (1.0 - d1);
            ls_failed = 1; // this line search failed
        }
    }

    return theta;
}

}

#endif /* FMINCG_HPP_ */
