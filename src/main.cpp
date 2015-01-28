/*******************************************************************************
 * Copyright (c) 2015 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: main.cpp
 *
 * Description:
 *      Trip Safety Factors
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

#include "TripSafetyFactors.hpp"

#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <cassert>

int main(int argc, char **argv)
{
    const int SEED = (argc == 2 ? std::atoi(argv[1]) : 1);
    const char * FNAME = (argc == 3 ? argv[2] : "../data/exampleData.csv");

    std::cout << "SEED: " << SEED << ", CSV: " << FNAME << std::endl;

    std::ifstream fcsv(FNAME);
    std::vector<std::string> vcsv;

    for (std::string line; std::getline(fcsv, line);)
    {
        vcsv.push_back(line);
    }
    fcsv.close();

    std::cout << "Read " << vcsv.size() << " lines" << std::endl;

    {
        std::mt19937 g(SEED);
        std::shuffle(vcsv.begin(), vcsv.end(), g);
    }

    const std::size_t PIVOT = 0.67 * vcsv.size();

    std::vector<std::string> test_data(vcsv.cbegin(), vcsv.cbegin() + PIVOT);
    std::vector<std::string> train_data(vcsv.cbegin() + PIVOT, vcsv.cend());

    for (auto & item : train_data)
    {
        constexpr std::size_t TEST_NCOL{29};
        std::size_t ncomma{0};

        item.resize(std::distance(item.cbegin(), std::find_if(item.cbegin(), item.cend(),
            [&ncomma](const char & ch)
            {
                if (ch == ',' && ncomma == (TEST_NCOL - 1))
                {
                    return true;
                }
                else
                {
                    ncomma += (ch == ',');
                    return false;
                }
            }
        )));
        assert(std::count(item.cbegin(), item.cend(), ',') == (TEST_NCOL - 1));
    }

    TripSafetyFactors worker;
    std::vector<int> prediction = worker.predict(test_data, train_data);

    return 0;
}
