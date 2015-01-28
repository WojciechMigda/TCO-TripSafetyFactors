/*******************************************************************************
 * Copyright (c) 2015 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: TripSafetyFactors.hpp
 *
 * Description:
 *      TripSafetyFactors class
 *
 * Authors:
 *          Wojciech Migda (wm)
 *
 *******************************************************************************
 * History:
 * --------
 * Date         Who  Ticket     Description
 * ----------   ---  ---------  ------------------------------------------------
 * 2015-01-27   wm              Initial version
 *
 ******************************************************************************/

#ifndef TRIPSAFETYFACTORS_HPP_
#define TRIPSAFETYFACTORS_HPP_

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

struct TripSafetyFactors
{
    std::vector<int> predict(
        std::vector<std::string> train_data,
        std::vector<std::string> test_data) const;
};

std::vector<int>
TripSafetyFactors::predict(
    std::vector<std::string> train_data,
    std::vector<std::string> test_data) const
{
    std::vector<int> result(test_data.size());

    std::cerr << train_data[0] << std::endl;
    std::cerr << train_data[1] << std::endl;
    std::cerr << train_data.back() << std::endl;
    std::cerr << std::endl;
    std::cerr << test_data[0] << std::endl;
    std::cerr << test_data[1] << std::endl;
    std::cerr << test_data.back() << std::endl;

    std::iota(result.begin(), result.end(), 1);

    return result;
}

#endif /* TRIPSAFETYFACTORS_HPP_ */
