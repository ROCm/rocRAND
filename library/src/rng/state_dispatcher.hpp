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

#include "config_types.hpp"
#include "cpp_utils.hpp"

#include <array>
#include <map>
#include <tuple>
#include <utility>

namespace rocrand_host::detail
{

/// @brief Manages states for generators based on different kernel launch configs.
/// \c ConfigProvider loads \ref generator_config for all generated value types. A
/// state object is initialized for every unique entry in this config list. The states
/// can be retrieved and updated via member functions, but ultimately their lifetime is
/// managed through \c this object.
/// @tparam ConfigProvider Specifies the \ref generator_config for the generated value types.
/// @tparam State The type of the state to manage.
template<class ConfigProvider, class State>
class state_dispatcher
{
public:
    /// @brief Initializes the states for the unique kernel launch configs.
    /// @tparam StateInitializer Functor type that initializes the state object.
    /// The signature must be alike:
    /// rocrand_status operator()(T /*for dispatch*/, generator_config, State&)
    /// @param stream HIP stream used to deduce the device architecture.
    /// @param ordering The ordering currently being in use.
    /// @param state_initializer Functor that initializes the state object.
    /// @return \ref ROCRAND_STATUS_SUCCESS if all states were successfully initialized.
    /// A different error code otherwise.
    template<class StateInitializer>
    rocrand_status init(const hipStream_t      stream,
                        const rocrand_ordering ordering,
                        StateInitializer&&     state_initializer)
    {
        m_config_to_state_map.clear();
        rocrand_status status = ROCRAND_STATUS_SUCCESS;
        cpp_utils::visit_tuple(
            [&](auto&& val)
            {
                // If there was an error previously, skip all subsequent types.
                if(status != ROCRAND_STATUS_SUCCESS)
                    return;
                using T = std::decay_t<decltype(val)>;
                status  = initialize_for_type<T>(stream,
                                                ordering,
                                                std::forward<StateInitializer>(state_initializer));
            },
            all_generated_types{});

        return status;
    }

    /// @brief Returns a reference to the \c State object that is associated with type \c T.
    /// @tparam T Generated value type.
    /// @return Reference to the state object.
    template<class T>
    const State& get_state() const
    {
        constexpr std::size_t tuple_idx = cpp_utils::tuple_type_index_v<T, all_generated_types>;
        static_assert(tuple_idx != std::numeric_limits<std::size_t>::max(),
                      "Requested type is not present in the list of generated types");
        return m_config_to_state_map.at(m_type_to_config_map[tuple_idx]);
    }

    /// @brief Update the state object that is associated with type \c T.
    /// @tparam T Generated value type.
    /// @tparam UpdateFunctor Type of the functor that is invoked to update the \c State.
    /// @param update_functor Functor object that can be invoked to update the \c State
    /// associated with \c T. The signature must be alike:
    /// void operator()(generator_config /*associated_config*/, State&)
    template<class T, class UpdateFunctor>
    void update_state(UpdateFunctor&& update_functor)
    {
        constexpr std::size_t tuple_idx = cpp_utils::tuple_type_index_v<T, all_generated_types>;
        static_assert(tuple_idx != std::numeric_limits<std::size_t>::max(),
                      "Requested type is not present in the list of generated types");
        const auto& config = m_type_to_config_map[tuple_idx];
        update_functor(config, m_config_to_state_map.at(config));
    }

private:
    struct config_comparator
    {
        constexpr bool operator()(const generator_config& lhs, const generator_config& rhs) const
        {
            // In order to store the configs in a \ref std::map, we must define an ordering.
            return (lhs.blocks != rhs.blocks) ? (lhs.blocks < rhs.blocks)
                                              : (lhs.threads < rhs.threads);
        }
    };

    using all_generated_types = std::
        tuple<unsigned int, unsigned char, unsigned short, unsigned long long, float, half, double>;

    /// \brief The function below is executed for all types in \ref all_generated_types when
    /// \ref init is called.
    template<class T, class StateInitializer>
    rocrand_status initialize_for_type(const hipStream_t      stream,
                                       const rocrand_ordering ordering,
                                       StateInitializer&&     state_initializer)
    {
        // Get the config for the current stream, ordering, value type
        generator_config config{};
        const hipError_t error = ConfigProvider{}.template host_config<T>(stream, ordering, config);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        // Store the config in the type->config map
        constexpr std::size_t tuple_idx = cpp_utils::tuple_type_index_v<T, all_generated_types>;
        static_assert(tuple_idx != std::numeric_limits<std::size_t>::max(),
                      "Requested type is not present in the list of generated types");
        m_type_to_config_map[tuple_idx] = config;

        // If the config is not yet present in the config->State map, then add it
        if(m_config_to_state_map.find(config) == m_config_to_state_map.end())
        {
            return state_initializer(T{}, config, m_config_to_state_map[config]);
        }

        return ROCRAND_STATUS_SUCCESS;
    }

    // The two fields below are a simple way to implement a mapping from type to State, where
    // multiple types can reference the same State object.

    /// @brief type->config map. Indices to this map are always known at compile time, based on
    /// \ref cpp_utils::tuple_type_index_v.
    std::array<generator_config, std::tuple_size_v<all_generated_types>> m_type_to_config_map{};

    /// @brief config->State map.
    std::map<generator_config, State, config_comparator> m_config_to_state_map{};
};

/// @brief A number of the generators require a state that consists of a pointer to the
/// device engine array, and the ID of the start engine. An instantiation class template can
/// be used as the \c State template argument for \ref state_dispatcher in those generators.
/// Additionally, it manages (frees) the device memory when destructed.
/// @tparam Engine The type of the device engine to manage.
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
