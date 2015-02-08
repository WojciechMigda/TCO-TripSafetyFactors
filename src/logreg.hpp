/*******************************************************************************
 * Copyright (c) 2015 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: logreg.hpp
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
 * 2015-02-06   wm              Initial version
 *
 ******************************************************************************/

#ifndef LOGREG_HPP_
#define LOGREG_HPP_

#include "array2d.hpp"
#include "sigmoid.hpp"
#include "fmincg.hpp"
#include <utility>
#include <valarray>
#include <cassert>
#include <functional>
#include <cmath>

namespace num
{

template<typename _ValueType>
void
logreg_cost_grad(
    /// out
    _ValueType & out_cost,
    std::valarray<_ValueType> & out_grad,
    std::valarray<_ValueType> & tcol,
    /// in
    const std::valarray<_ValueType> theta,
    const array2d<_ValueType> X,
    const std::valarray<_ValueType> y,
    const _ValueType C
)
{
    typedef _ValueType value_type;
    typedef std::valarray<value_type> vector;

    const shape_type X_shape = X.shape();

    assert(y.size() == X_shape.first);
    assert(out_grad.size() == X_shape.second);
    assert(theta.size() == X_shape.second);

    assert(tcol.size() >= X_shape.first);

    vector & H = tcol;

    //    H = sigmoid(theta' * X')';
    for (size_type r{0}; r < X_shape.first; ++r)
    {
        H[r] = (X[X.row(r)] * theta).sum();
    }
    H = sigmoid(H);

    //    theta_for_reg = [0; theta(2:size(theta))];

    //    grad = theta_for_reg' / C;
    out_grad = theta / C;
    out_grad[0] = 0.0;

    //    sigma_i = -y' * log(H) - (1 - y') * log(1 - H);
    const value_type Sigma = -(y * std::log(H)).sum() - (((value_type)1.0 - y) * std::log((value_type)1.0 - H)).sum();
    //    J = sigma_i / m + sum(theta_for_reg.^2) / (2 * C * m);
    out_cost = (theta[std::slice(1, X_shape.second - 1, 1)] * theta[std::slice(1, X_shape.second - 1, 1)]).sum() / (2.0 * C * X_shape.first);
    out_cost += Sigma / X_shape.first;

    //    grad += (H - y)' * X;
    H -= y;
    for (size_type r{0}; r < X_shape.second; ++r)
    {
        out_grad[r] += (X[X.column(r)] * H).sum();
    }
    //    grad /= m;
    out_grad /= X_shape.first;
}


template<typename _ValueType>
std::pair<_ValueType, std::valarray<_ValueType>>
logreg_cost_grad(
    const std::valarray<_ValueType> theta,
    const array2d<_ValueType> X,
    const std::valarray<_ValueType> y,
    const _ValueType C)
{
    typedef _ValueType value_type;
    typedef std::valarray<value_type> vector;

    const shape_type X_shape = X.shape();

    vector temp(X_shape.first);

    const double cost;
    const vector grad(X_shape.first);

    logreg_cost_grad(cost, grad, temp, theta, X, y, C);

    return std::make_pair(cost, grad);
}

template<typename _ValueType>
class LogisticRegression
{
public:
    typedef _ValueType value_type;
    typedef std::valarray<value_type> vector_type;
    typedef array2d<value_type> array_type;

    LogisticRegression(
        array_type && X,
        vector_type && y,
        vector_type && theta0,
        value_type C,
        size_type max_iter
    );

    vector_type
    fit(void) const;

    vector_type
    predict(const array_type & X, const vector_type & theta, bool round = true) const;

    vector_type
    predict(array_type && X, vector_type && theta, bool round = true) const;

private:
    const array_type m_X;
    const vector_type m_y;
    const vector_type m_theta0;
    const value_type m_C;
    const size_type m_max_iter;
};

template<typename _ValueType>
LogisticRegression<_ValueType>::LogisticRegression(
    array_type && X,
    vector_type && y,
    vector_type && theta0,
    value_type C,
    size_type max_iter
)
:
    m_X{std::move(X)},
    m_y{std::move(y)},
    m_theta0{m_theta0.size() == m_X.shape().second ? std::move(theta0) : vector_type(m_X.shape().second)},
    m_C{C},
    m_max_iter{max_iter}
{
}

template<typename _ValueType>
typename LogisticRegression<_ValueType>::vector_type
LogisticRegression<_ValueType>::fit(void) const
{
    vector_type tcol(m_y.size());

    std::function<std::pair<value_type, vector_type> (vector_type)>

    /* NOTE: Capturing member variables is always done via capturing this */
    cost_fn = [this, &tcol](const vector_type theta) -> std::pair<value_type, vector_type>
    {
        value_type cost;
        vector_type grad(theta.size());

        num::logreg_cost_grad(cost, grad, tcol, theta, this->m_X, this->m_y, this->m_C);

        return std::make_pair(cost, grad);
    };

    const vector_type theta = num::fmincg(cost_fn, m_theta0, m_max_iter, true);

    return theta;
}

template<typename _ValueType>
typename LogisticRegression<_ValueType>::vector_type
LogisticRegression<_ValueType>::predict(const array_type & X, const vector_type & theta, bool round) const
{
    assert(theta.size() == X.shape().second);
    vector_type H(X.shape().first);

    for (size_type r{0}; r < X.shape().first; ++r)
    {
        H[r] = (X[X.row(r)] * theta).sum();
    }

    if (round)
    {
        H = sigmoid(H).apply(std::round);
    }
    else
    {
        H = sigmoid(H);
    }

    return H;
}

template<typename _ValueType>
typename LogisticRegression<_ValueType>::vector_type
LogisticRegression<_ValueType>::predict(array_type && X, vector_type && theta, bool round) const
{
    return predict(theta, round);
}

}  // namespace num

#endif /* LOGREG_HPP_ */
