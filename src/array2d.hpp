/*******************************************************************************
 * Copyright (c) 2015 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: array2d.hpp
 *
 * Description:
 *      Let's try to have a numpy-like environment
 *
 * Authors:
 *          Wojciech Migda (wm)
 *
 *******************************************************************************
 * History:
 * --------
 * Date         Who  Ticket     Description
 * ----------   ---  ---------  ------------------------------------------------
 * 2015-01-30   wm              Initial version
 *
 ******************************************************************************/

#ifndef ARRAY2D_HPP_
#define ARRAY2D_HPP_

#include "num.hpp"
#include <cstdlib>
#include <utility>
#include <valarray>
#include <map>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <string>
#include <sstream>
#include <type_traits>
#include <unordered_set>

namespace num
{

typedef std::pair<size_type, size_type> shape_type;

template<typename _Type>
class array2d
{
public:
    enum class Axis
    {
        Row,
        Column
    };

    typedef _Type value_type;
    typedef std::size_t size_type;
    typedef std::pair<size_type, size_type> shape_type;
    typedef std::valarray<value_type> varray_type;

    array2d(shape_type shape, value_type initializer);

    shape_type shape(void) const;

    std::slice row(size_type n) const;
    std::slice column(size_type n) const;
    std::slice stripe(size_type n, enum Axis axis) const;

    std::gslice columns(int p, int q) const;

    std::valarray<value_type> operator[](std::slice slicearr) const;
    std::slice_array<value_type> operator[](std::slice slicearr);
    std::valarray<value_type> operator[](const std::gslice & gslicearr) const;
    std::gslice_array<value_type> operator[](const std::gslice & gslicearr);

private:
    shape_type m_shape;
    varray_type m_varray;
};

template<typename _Type>
inline
array2d<_Type>::array2d(shape_type shape, array2d<_Type>::value_type initializer)
:
    m_shape(shape),
    m_varray(initializer, shape.first * shape.second)
{

}

template<typename _Type>
inline
shape_type
array2d<_Type>::shape(void) const
{
    return m_shape;
}

template<typename _Type>
inline
std::slice
array2d<_Type>::row(size_type n) const
{
    return std::slice(n * m_shape.second, m_shape.second, 1);
}

template<typename _Type>
inline
std::slice
array2d<_Type>::column(size_type n) const
{
    return std::slice(n, m_shape.first, m_shape.second);
}

template<typename _Type>
inline
std::gslice
array2d<_Type>::columns(int p, int q) const
{
    if (p < 0)
    {
        assert(-p < m_shape.second);
        p = m_shape.second + p;
    }
    if (q < 0)
    {
        assert(-q < m_shape.second);
        q = m_shape.second + q;
    }

    return std::gslice(
        p,
        {m_shape.first, {q - p + 1u}},
        {m_shape.second, 1u}
    );
}

template<typename _Type>
inline
std::slice
array2d<_Type>::stripe(size_type n, enum Axis axis) const
{
    return axis == Axis::Row ? row(n) : column(n);
}

template<typename _Type>
inline
std::valarray<_Type>
array2d<_Type>::operator[](std::slice slicearr) const
{
    return m_varray[slicearr];
}

template<typename _Type>
inline
std::slice_array<_Type>
array2d<_Type>::operator[](std::slice slicearr)
{
    return m_varray[slicearr];
}

template<typename _Type>
inline
std::valarray<_Type>
array2d<_Type>::operator[](const std::gslice & gslicearr) const
{
    return m_varray[gslicearr];
}

template<typename _Type>
inline
std::gslice_array<_Type>
array2d<_Type>::operator[](const std::gslice & gslicearr)
{
    return m_varray[gslicearr];
}

template<typename _Type>
inline
array2d<_Type>
zeros(shape_type shape)
{
    return array2d<_Type>(shape, 0.0);
}

template<typename _Type>
inline
array2d<_Type>
ones(shape_type shape)
{
    return array2d<_Type>(shape, 1.0);
}

template<typename _Type = double>
struct loadtxtCfg
{
    typedef std::map<size_type, _Type(*)(const char *)> converters_type;

    loadtxtCfg()
    :
        m_comments{'#'},
        m_delimiter{' '},
        m_converters{},
        m_skip_header{0},
        m_skip_footer{0},
        m_use_cols{}
    {}

    loadtxtCfg & comments(char _comments)
    {
        m_comments = _comments;
        return *this;
    }

    char delimiter(void) const
    {
        return m_delimiter;
    }

    loadtxtCfg & delimiter(char _delimiter)
    {
        m_delimiter = _delimiter;
        return *this;
    }

    const converters_type & converters(void) const
    {
        return m_converters;
    }

    loadtxtCfg & converters(converters_type && _converters)
    {
        m_converters = std::move(_converters);
        return *this;
    }

    size_type skip_header(void) const
    {
        return m_skip_header;
    }

    loadtxtCfg & skip_header(size_type _skip_header)
    {
        m_skip_header = _skip_header;
        return *this;
    }

    size_type skip_footer(void) const
    {
        return m_skip_footer;
    }

    loadtxtCfg & skip_footer(size_type _skip_footer)
    {
        m_skip_footer = _skip_footer;
        return *this;
    }

    loadtxtCfg & use_cols(std::unordered_set<size_type> && _use_cols)
    {
        m_use_cols = std::move(_use_cols);
        return *this;
    }

    char m_comments;
    char m_delimiter;
    converters_type m_converters;
    size_type m_skip_header;
    size_type m_skip_footer;
    std::unordered_set<size_type> m_use_cols;
};

template<typename _Type>
array2d<_Type>
loadtxt(
    std::vector<std::string> && txt,
    loadtxtCfg<_Type> && cfg
)
{
    typedef _Type value_type;
    const bool skip_empty = false;

    auto count_delimiters = [&skip_empty](std::string where, char delim) -> size_type
    {
        auto predicate = [&delim](char _this, char _that)
        {
            return _this == _that && _this == delim;
        };
        if (skip_empty)
        {
            std::unique(where.begin(), where.end(), predicate);
        }

        return std::count(where.cbegin(), where.cend(), delim);
    };

    assert(txt.size() >= (cfg.skip_header() + cfg.skip_footer()));
    const size_type NROWS = txt.size() - cfg.skip_header() - cfg.skip_footer();
    if (NROWS == 0)
    {
        return zeros<value_type>(shape_type(0, 0));
    }

    // TODO use_ncols
    const size_type NCOLS = 1 + count_delimiters(txt.front(), cfg.delimiter());

    array2d<_Type> result = zeros<value_type>(shape_type(NROWS, NCOLS));

    for (size_type ridx{0}; ridx < NROWS; ++ridx)
    {
        std::valarray<value_type> row(NROWS);
        std::stringstream ss(txt[ridx + cfg.skip_header()]);
        std::string item;

        for (size_type cidx{0}; cidx < NCOLS && std::getline(ss, item, cfg.delimiter()); ++cidx) // TODO
        {
            if (cfg.converters().find(cidx) != cfg.converters().cend())
            {
                row[cidx] = cfg.converters().at(cidx)(item.c_str());
            }
            else if (std::is_convertible<value_type, long double>::value)
            {
                row[cidx] = std::strtold(item.c_str(), nullptr);
            }
            else if (std::is_convertible<value_type, long long>::value)
            {
                row[cidx] = std::atoll(item.c_str());
            }
            else
            {
                std::stringstream item_ss(item);
                item_ss >> row[cidx];
            }
        }

        result[result.row(ridx)] = row;
    }

    return result;
}

} // namespace num

namespace std
{

std::ostream & operator<<(std::ostream & os, const num::shape_type & shape)
{
    os << '(' << shape.first << ',' << shape.second << ')';
    return os;
}

} // namespace std

#endif /* ARRAY2D_HPP_ */
