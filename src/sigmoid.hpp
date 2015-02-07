/*******************************************************************************
 * Copyright (c) 2015 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: sigmoid.hpp
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

#ifndef SIGMOID_HPP_
#define SIGMOID_HPP_

#include <cmath>

namespace num
{

template<typename _ValueType>
_ValueType
sigmoid(const _ValueType z)
{
    return 1.0 / (1.0 + std::exp(-z));
}

} // namespace num

#endif /* SIGMOID_HPP_ */
