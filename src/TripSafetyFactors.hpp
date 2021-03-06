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

#include "array2d.hpp"
#include "logreg.hpp"
#include "num.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <iterator>
#include <map>
#include <tuple>

typedef double real_type;

std::map<real_type, real_type>
map_event_density(
    const std::valarray<real_type> & feat,
    const std::valarray<real_type> & y
)
{
    std::map<real_type, std::pair<num::size_type, num::size_type>> event_count;

    assert(feat.size() == y.size());

    for (num::size_type r{0}; r < feat.size(); ++r)
    {
        event_count[feat[r]].first++;
        event_count[feat[r]].second += y[r];
    }

    std::map<real_type, real_type> result;

    for (auto count : event_count)
    {
        result[count.first] = (real_type)count.second.second / count.second.first;
    }

    return result;
}

std::vector<int> do_log_reg(
    num::array2d<real_type> && i_X_train,
    std::valarray<real_type> && i_y_train,
    num::array2d<real_type> && i_X_test
)
{
    typedef num::array2d<real_type> array_type;
    typedef std::valarray<real_type> vector_type;

    const num::size_type NUM_FEAT{i_X_train.shape().second + 1};

    // let's map input training features onto what we'll work with
    // first column will be 1s for the intercept
    array_type X_train = num::ones<real_type>({i_X_train.shape().first, NUM_FEAT});

    // the rest will be copied from i_X_train
    // X_train[:, 1:] = i_X_train[:, :]
    X_train[X_train.columns(1, -1)] = i_X_train[i_X_train.columns(0, -1)];

    // same with test features
    array_type X_test = num::ones<real_type>({i_X_test.shape().first, NUM_FEAT});

    // X_test[:, 1:] = i_X_test[:, :]
    X_test[X_test.columns(1, -1)] = i_X_test[i_X_test.columns(0, -1)];

    // we got train and test features mapped

    // lets keep columnd indices under control
    enum col : num::size_type
    {
        INTERCEPT,
        SOURCE, // *
        DIST,
        CYCLES,
        COMPLEXITY,
        CARGO, // *
        STOPS,
        START_DAY,
        START_MONTH,
        START_DAY_OF_MONTH,
        START_DAY_OF_WEEK,
        START_TIME,
        DAYS,
        PILOT, // *
        PILOT2, // *
        PILOT_EXP,
        PILOT_VISITS_PREV,
        PILOT_HOURS_PREV,
        PILOT_DUTY_HOURS_PREV,
        PILOT_DIST_PREV,
        ROUTE_RISK_1,
        ROUTE_RISK_2,
        WEATHER, // *
        VISIBILITY,
        TRAF0,
        TRAF1,
        TRAF2,
        TRAF3,
        TRAF4,
    };

    // There are some features (marked with asterix above) which do not have
    // a meaning in a monotonic domain. Let's do the following experiment:
    // for each of those features map to each occuring value a pair of values -
    // a number of occurences and a sum of events,
    // then we remap the original values to event density

    for (auto COLUMN :
        std::initializer_list<int>
        {
            col::SOURCE,
            col::PILOT,
            col::START_MONTH,
            col::CYCLES,
            col::PILOT_EXP,
        }
    )
    {
        auto event_density = map_event_density(X_train[X_train.column(COLUMN)], i_y_train);
        std::cerr << "event density size: " << event_density.size() << std::endl;

        vector_type mapped_train_col = X_train[X_train.column(COLUMN)];
        std::transform(std::begin(mapped_train_col), std::end(mapped_train_col), std::begin(mapped_train_col),
            [&event_density](const real_type & x) -> real_type
            {
                return event_density[x];
            }
        );
        X_train[X_train.column(COLUMN)] = mapped_train_col;

        vector_type mapped_test_col = X_test[X_test.column(COLUMN)];
        std::transform(std::begin(mapped_test_col), std::end(mapped_test_col), std::begin(mapped_test_col),
            [&event_density](const real_type & x) -> real_type
            {
                return event_density[x];
            }
        );
        X_test[X_test.column(COLUMN)] = mapped_test_col;
    }


    vector_type y_train = i_y_train.apply(
        [](real_type v)
        {
            return v > 1.0 ? 1.0 : v;
        }
    );

    vector_type theta(0.0, X_train.shape().second);

    // standardization
    for (num::size_type c{1}; c < X_train.shape().second; ++c)
    {
        const vector_type & col = X_train[X_train.column(c)];
        const vector_type & colt = X_test[X_test.column(c)];

        const real_type mu = num::mean<real_type>(col);
        const real_type dev = num::std<real_type>(col);

        X_train[X_train.column(c)] = col - mu;
        X_train[X_train.column(c)] = col / dev;

        X_test[X_test.column(c)] = colt - mu;
        X_test[X_test.column(c)] = colt / dev;
    }

    num::LogisticRegression<real_type> logRegClassifier(
        num::LogisticRegression<real_type>::array_type{X_train},
        num::LogisticRegression<real_type>::vector_type{y_train},
        num::LogisticRegression<real_type>::vector_type{theta},
        0.02,
        200
    );

    auto fit_theta = logRegClassifier.fit();

//    std::copy(std::begin(fit_theta), std::end(fit_theta), std::ostream_iterator<real_type>(std::cerr, "\n"));

    auto pred = logRegClassifier.predict(X_test, fit_theta, false);
//    std::copy(std::begin(pred), std::end(pred), std::ostream_iterator<real_type>(std::cerr, "\n"));
    std::cerr << "pred.max() " << pred.max() << std::endl;

    std::vector<int> result(i_X_test.shape().first);

    std::vector<std::tuple<num::size_type, num::size_type, real_type>> zipped;
    zipped.reserve(pred.size());

    for (num::size_type idx = 0; idx < pred.size(); ++idx)
    {
        zipped.emplace_back(idx + 1, 0, pred[idx]);
    }
    std::sort(zipped.begin(), zipped.end(),
        [](const std::tuple<num::size_type, num::size_type, real_type> & p, const std::tuple<num::size_type, num::size_type, real_type> & q)
        {
            return std::get<2>(p) > std::get<2>(q);
        }
    );
    for (num::size_type idx = 0; idx < pred.size(); ++idx)
    {
        std::get<1>(zipped[idx]) = idx + 1;
    }
    std::sort(zipped.begin(), zipped.end(),
        [](const std::tuple<num::size_type, num::size_type, real_type> & p, const std::tuple<num::size_type, num::size_type, real_type> & q)
        {
            return std::get<0>(p) < std::get<0>(q);
        }
    );
    std::transform(zipped.cbegin(), zipped.cend(), result.begin(),
        [](const std::tuple<num::size_type, num::size_type, real_type> & p)
        {
            return (int)std::get<1>(p);
        }
    );

    std::copy(result.cbegin(), result.cbegin() + 10, std::ostream_iterator<int>(std::cerr, " "));
    std::cerr << std::endl;
    std::copy(result.cend() - 10, result.cend(), std::ostream_iterator<int>(std::cerr, " "));
    std::cerr << std::endl;

    return result;
}

struct TripSafetyFactors
{
    std::vector<int> predict(
        std::vector<std::string> i_train_data,
        std::vector<std::string> i_test_data) const;
};

std::vector<int>
TripSafetyFactors::predict(
    std::vector<std::string> i_train_data,
    std::vector<std::string> i_test_data) const
{
    enum col : num::size_type
    {
        ID,
        SOURCE,
        DIST,
        CYCLES,
        COMPLEXITY,
        CARGO,
        STOPS,
        START_DAY,
        START_MONTH,
        START_DAY_OF_MONTH,
        START_DAY_OF_WEEK,
        START_TIME,
        DAYS,
        PILOT,
        PILOT2,
        PILOT_EXP,
        PILOT_VISITS_PREV,
        PILOT_HOURS_PREV,
        PILOT_DUTY_HOURS_PREV,
        PILOT_DIST_PREV,
        ROUTE_RISK_1,
        ROUTE_RISK_2,
        WEATHER,
        VISIBILITY,
        TRAF0,
        TRAF1,
        TRAF2,
        TRAF3,
        TRAF4,
        /////////////
        ACCEL_CNT,
        DECEL_CNT,
        SPEED_CNT,
        STABILITY_CNT,
        EVT_CNT
    };

    typedef num::array2d<real_type> array_type;
    typedef std::valarray<real_type> vector_type;

    const num::size_type TRAIN_ROWS{i_train_data.size()};
    constexpr num::size_type TRAIN_COLS{34};

    const num::size_type TEST_ROWS{i_test_data.size()};
    constexpr num::size_type TEST_COLS{29};

    const num::shape_type TRAIN_SHAPE{TRAIN_ROWS, TRAIN_COLS};
    const num::shape_type TEST_SHAPE{TEST_ROWS, TEST_COLS};

    ////////////////////////////////////////////////////////////////////////////

    auto time_xlt = [](const char * str) -> real_type
    {
        char * next;
        long int result = std::strtol(str, &next, 10) * 60;
        if (*next == ':')
        {
//            result += std::strtol(next + 1, nullptr, 10);
        }
        return result;
    };
    array_type train_data =
        num::loadtxt(
            std::move(i_train_data),
            std::move(
                num::loadtxtCfg<real_type>()
                .delimiter(',')
                .converters({{col::START_TIME, time_xlt}})
            )
        );
    std::cerr << train_data.shape() << std::endl;

    array_type test_data =
        num::loadtxt(
            std::move(i_test_data),
            std::move(
                num::loadtxtCfg<real_type>()
                .delimiter(',')
                .converters({{col::START_TIME, time_xlt}})
            )
        );
    std::cerr << test_data.shape() << std::endl;

    ////////////////////////////////////////////////////////////////////////////

    array_type X_train_data({train_data.shape().first, col::TRAF4 - col::SOURCE + 1}, 0.0);

    X_train_data[X_train_data.columns(0, -1)] = train_data[train_data.columns(col::SOURCE, col::TRAF4)];

    vector_type y_train_data = train_data[train_data.column(col::EVT_CNT)];

    array_type X_test_data({test_data.shape().first, test_data.shape().second - 1}, 0.0);

    X_test_data[X_test_data.columns(0, -1)] = test_data[test_data.columns(col::SOURCE, col::TRAF4)];

    ////////////////////////////////////////////////////////////////////////////

    std::vector<int> result(TEST_ROWS);

    result = do_log_reg(std::move(X_train_data), std::move(y_train_data), std::move(X_test_data));

    return result;
}

//#include <functional>
//#include "fmincg.hpp"
//#include <iterator>
//#include <fstream>
//void foo()
//{
//    typedef double real;
//    std::vector<std::string> Xtxt;
//    std::vector<std::string> ytxt;
//    {
//        std::ifstream fcsv("xxx.txt");
//
//        for (std::string line; std::getline(fcsv, line);)
//        {
//            Xtxt.push_back(line);
//        }
//        fcsv.close();
//    }
//    {
//        std::ifstream fcsv("xxy.txt");
//
//        for (std::string line; std::getline(fcsv, line);)
//        {
//            ytxt.push_back(line);
//        }
//        fcsv.close();
//    }
//    num::array2d<real> X =
//        num::loadtxt(
//            std::move(Xtxt),
//            std::move(
//                num::loadtxtCfg<real>()
//                .delimiter(',')
//            )
//        );
//    std::cerr << X.shape() << std::endl;
//    num::array2d<real> ym =
//        num::loadtxt(
//            std::move(ytxt),
//            std::move(
//                num::loadtxtCfg<real>()
//                .delimiter(',')
//            )
//        );
//    std::cerr << ym.shape() << std::endl;
//    std::valarray<real> y = ym[ym.column(0)];
//
//    std::valarray<real> theta(0.0, X.shape().second);
//
//    num::LogisticRegression<real> logRegClassifier(
//        num::LogisticRegression<real>::array_type{X},
//        num::LogisticRegression<real>::vector_type{y},
//        num::LogisticRegression<real>::vector_type{theta},
//        1.0,
//        400
//    );
//
//    auto fit_theta = logRegClassifier.fit();
//    std::copy(std::begin(fit_theta), std::end(fit_theta), std::ostream_iterator<real>(std::cerr, "\n"));
//
//    auto prediction = logRegClassifier.predict(X, fit_theta);
//    std::copy(std::begin(prediction), std::end(prediction), std::ostream_iterator<real>(std::cerr, "\n"));
//
//    return;
//}

#endif /* TRIPSAFETYFACTORS_HPP_ */
