// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCRAND_RNG_STATE_DISPATCHER_HPP_
#define ROCRAND_RNG_STATE_DISPATCHER_HPP_

#include "common.hpp"
#include "rocrand/rocrand.h"

#include <hip/hip_runtime.h>

#include "cpp_utils.hpp"

#include <array>
#include <map>
#include <tuple>
#include <utility>

namespace rocrand_host::detail
{

template<template<class T> class ConfigProvider, class State>
class state_dispatcher
{
public:
    template<class EngineInitializer>
    rocrand_status init(const hipStream_t      stream,
                        const rocrand_ordering ordering,
                        EngineInitializer&&    engine_initializer)
    {
        m_config_to_state_map.clear();
        rocrand_status status = ROCRAND_STATUS_SUCCESS;
        cpp_utils::visit_tuple(
            [&](auto&& val)
            {
                using T = std::decay_t<decltype(val)>;
                if(status == ROCRAND_STATUS_SUCCESS)
                    status = initialize_for_type<T>(
                        stream,
                        ordering,
                        std::forward<EngineInitializer>(engine_initializer));
            },
            all_generated_types{});

        return status;
    }

    template<class T>
    const State& get_state() const
    {
        return m_config_to_state_map.at(
            m_type_to_config_map[cpp_utils::tuple_type_index_v<T, all_generated_types>]);
    }

    template<class T, class UpdateFunctor>
    void update_state(UpdateFunctor&& update_functor)
    {
        const auto& config
            = m_type_to_config_map[cpp_utils::tuple_type_index_v<T, all_generated_types>];
        update_functor(config, m_config_to_state_map[config]);
    }

private:
    struct config_comparator
    {
        constexpr bool operator()(const generator_config& lhs, const generator_config& rhs) const
        {
            return (lhs.blocks != rhs.blocks) ? (lhs.blocks < rhs.blocks)
                                              : (lhs.threads < rhs.threads);
        }
    };

    using all_generated_types = std::
        tuple<unsigned int, unsigned char, unsigned short, unsigned long long, float, half, double>;

    template<class T, class EngineInitializer>
    rocrand_status initialize_for_type(const hipStream_t      stream,
                                       const rocrand_ordering ordering,
                                       EngineInitializer&&    engine_initializer)
    {
        generator_config config{};
        const hipError_t error = ConfigProvider<T>::host_config(stream, ordering, config);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        m_type_to_config_map[cpp_utils::tuple_type_index_v<T, all_generated_types>] = config;
        if(m_config_to_state_map.find(config) == m_config_to_state_map.end())
        {
            return engine_initializer(T{}, config, m_config_to_state_map[config]);
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    std::array<generator_config, std::tuple_size_v<all_generated_types>> m_type_to_config_map{};
    std::map<generator_config, State, config_comparator>                 m_config_to_state_map{};
};

template<class Engine>
struct common_engine_state
{
    Engine*      m_engines{};
    unsigned int m_start_engine_id{};

    common_engine_state()                           = default;
    common_engine_state(const common_engine_state&) = delete;
    common_engine_state(common_engine_state&&)      = default;

    common_engine_state& operator=(const common_engine_state&) = delete;
    common_engine_state& operator=(common_engine_state&&)      = default;

    ~common_engine_state()
    {
        if(m_engines != nullptr)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(m_engines));
        }
    }
};

} // end namespace rocrand_host::detail

#endif // ROCRAND_RNG_STATE_DISPATCHER_HPP_
