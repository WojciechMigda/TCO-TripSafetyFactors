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
#include <valarray>
#include <iterator>

int main(int argc, char **argv)
{
    const int SEED = (argc == 2 ? std::atoi(argv[1]) : 1);
    const char * FNAME = (argc == 3 ? argv[2] : "../data/exampleData.csv");

    std::cerr << "SEED: " << SEED << ", CSV: " << FNAME << std::endl;

    std::ifstream fcsv(FNAME);
    std::vector<std::string> vcsv;

    for (std::string line; std::getline(fcsv, line);)
    {
        vcsv.push_back(line);
    }
    fcsv.close();

    std::cerr << "Read " << vcsv.size() << " lines" << std::endl;

    {
        std::mt19937 g(SEED);
        std::shuffle(vcsv.begin(), vcsv.end(), g);
    }

    const std::size_t PIVOT = 0.67 * vcsv.size();

    std::vector<std::string> train_data(vcsv.cbegin(), vcsv.cbegin() + PIVOT);
    std::vector<std::string> test_data(vcsv.cbegin() + PIVOT, vcsv.cend());

    const std::size_t N = std::count_if(test_data.cbegin(), test_data.cend(),
        [](const std::string & line) -> bool
        {
            const std::size_t pos = line.rfind(',');
            assert(pos != std::string::npos);
            const int last_val = std::atoi(line.c_str() + pos + 1);
            return last_val > 0;
        }
    );
    std::cerr << "N: " << N << std::endl;

    const std::size_t M = std::count_if(test_data.cbegin(), test_data.cend(),
        [](const std::string & line) -> bool
        {
            const std::size_t pos = line.rfind(',');
            assert(pos != std::string::npos);
            const int last_val = std::atoi(line.c_str() + pos + 1);
            return last_val > 1;
        }
    );
    std::cerr << "M: " << M << std::endl;

    std::sort(test_data.begin(), test_data.end(),
        [](const std::string & lhs, const std::string & rhs) -> bool
        {
            const std::size_t lpos = lhs.rfind(',');
            const std::size_t rpos = rhs.rfind(',');
            assert(lpos != std::string::npos);
            assert(rpos != std::string::npos);

            const int last_l_val = std::atoi(lhs.c_str() + lpos + 1);
            const int last_r_val = std::atoi(rhs.c_str() + rpos + 1);

            return last_l_val > last_r_val;
        }
    );

    for (auto & item : test_data)
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

    ////////////////////////////////////////////////////////////////////////////
    TripSafetyFactors worker;
    std::vector<int> prediction = worker.predict(train_data, test_data);
    ////////////////////////////////////////////////////////////////////////////

    std::valarray<float> scores(test_data.size());
    std::size_t index = 1;
    std::generate(std::begin(scores), std::end(scores),
        [&N, &index]()
        {
            return std::max((2.0f * N - index++) / (2 * N), 0.0f);
        }
    );
    std::valarray<float> bonuses(test_data.size());
    index = 1;
    std::generate(std::begin(bonuses), std::end(bonuses),
        [&M, &index]()
        {
            return std::max(0.3f * (2.0f * M - index++) / (2 * M), 0.0f);
        }
    );

    const float MAX_POINTS =
        std::accumulate(std::begin(scores), std::begin(scores) + N, 0.0f) +
        std::accumulate(std::begin(bonuses), std::begin(bonuses) + M, 0.0f);

    const float POINTS =
        std::accumulate(prediction.cbegin(), prediction.cbegin() + N, 0.0f,
            [&scores](const float & a, const int & i)
            {
                return scores[i - 1] + a;
            }
        ) +
        std::accumulate(prediction.cbegin(), prediction.cbegin() + M, 0.0f,
            [&bonuses](const float & a, const int & i)
            {
                return bonuses[i - 1] + a;
            }
        );

    std::cerr << "MAX_POINTS: " << MAX_POINTS << std::endl;
    std::cerr << "POINTS: " << POINTS << std::endl;

    std::cerr << "SCORE: " << (int)std::round(1000000 * POINTS / MAX_POINTS) << std::endl;

    return 0;
}
