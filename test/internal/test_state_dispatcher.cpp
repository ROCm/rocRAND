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

#include "rng/state_dispatcher.hpp"
#include <gmock/gmock.h>

using rocrand_host::detail::generator_config;
using rocrand_host::detail::state_dispatcher;
using testing::_;

namespace rocrand_host::detail
{

inline bool operator==(const generator_config& lhs, const generator_config& rhs)
{
    return lhs.blocks == rhs.blocks && lhs.threads == rhs.threads;
}

} // namespace rocrand_host::detail

namespace
{

struct mock_state
{
    int                       value;
    static inline std::size_t num_instantiations{};
    static inline std::size_t num_destructions{};

    mock_state(const int v = 0) : value(v)
    {
        ++num_instantiations;
    }

    mock_state(const mock_state&) = delete;
    mock_state(mock_state&&)      = default;

    mock_state& operator=(const mock_state&) = delete;
    mock_state& operator=(mock_state&&)      = default;

    ~mock_state()
    {
        ++num_destructions;
    }
};

struct mock_config_provider
{
    // We don't care about device configs in these tests, but the two members below
    // are part of the ConfigProvider "concept". Commented out to avoid a warning.
    // template<class T>
    // static constexpr generator_config dynamic_device_config = {};
    // template<class T>
    // static constexpr generator_config static_device_config = {};

    template<class T>
    hipError_t host_config(const hipStream_t /*stream*/,
                           const rocrand_ordering ordering,
                           generator_config&      config) const
    {
        if(rocrand_host::detail::is_ordering_dynamic(ordering))
        {
            if constexpr(std::is_integral_v<T>)
            {
                if constexpr(sizeof(T) < 4)
                {
                    config = {128, 256};
                }
                else
                {
                    config = {512, 128};
                }
            }
            else
            {
                // Deliberately the same total thread count as in previous.
                config = {128, 512};
            }
        }
        else
        {
            config = {128, 256};
        }

        return hipSuccess;
    }
};

struct mock_state_initializer
{
    template<class T>
    rocrand_status operator()(T val, const generator_config& config, mock_state& state)
    {
        return op(static_cast<int>(val), config, state);
    }

    MOCK_METHOD(rocrand_status, op, (int, const generator_config&, mock_state&));
};

using tested_state_dispatcher = state_dispatcher<mock_config_provider, mock_state>;

} // end namespace

struct rocrand_state_dispatcher_test : public testing::Test
{
    void SetUp() override
    {
        mock_state::num_instantiations = 0;
        mock_state::num_destructions   = 0;
    }
};

TEST_F(rocrand_state_dispatcher_test, init)
{
    tested_state_dispatcher dispatcher;
    {
        testing::StrictMock<mock_state_initializer> initializer;
        // When the ordering is not dynamic, there is a single config.
        EXPECT_CALL(initializer, op(_, (generator_config{128, 256}), _)).Times(1);
        dispatcher.init(0, ROCRAND_ORDERING_PSEUDO_LEGACY, initializer);
    }
    {
        testing::StrictMock<mock_state_initializer> initializer;
        // When the ordering is dynamic, there are 3 distinct configs.
        EXPECT_CALL(initializer, op(_, (generator_config{128, 256}), _)).Times(1);
        EXPECT_CALL(initializer, op(_, (generator_config{512, 128}), _)).Times(1);
        EXPECT_CALL(initializer, op(_, (generator_config{128, 512}), _)).Times(1);
        dispatcher.init(0, ROCRAND_ORDERING_PSEUDO_DYNAMIC, initializer);
    }

    ASSERT_EQ(4, mock_state::num_instantiations);
    ASSERT_EQ(1, mock_state::num_destructions);
}

TEST_F(rocrand_state_dispatcher_test, retrieve_state)
{
    tested_state_dispatcher                   dispatcher;
    testing::NiceMock<mock_state_initializer> initializer;
    dispatcher.init(0, ROCRAND_ORDERING_PSEUDO_DYNAMIC, initializer);

    const auto& uint_state  = dispatcher.get_state<unsigned int>();
    const auto& ulong_state = dispatcher.get_state<unsigned long long>();
    const auto& float_state = dispatcher.get_state<float>();

    ASSERT_EQ(&uint_state, &ulong_state) << "Matching config must share state";
    ASSERT_NE(&uint_state, &float_state) << "Different config must not share state";

    ASSERT_EQ(0, uint_state.value);
    ASSERT_EQ(0, float_state.value);

    dispatcher.update_state<unsigned int>([](const auto& /*config*/, auto& state)
                                          { ++state.value; });
    ASSERT_EQ(1, uint_state.value);
    ASSERT_EQ(0, float_state.value);

    ASSERT_EQ(3, mock_state::num_instantiations);
}

TEST_F(rocrand_state_dispatcher_test, initialization_destructs_all_state)
{
    tested_state_dispatcher                   dispatcher;
    testing::NiceMock<mock_state_initializer> initializer;

    dispatcher.init(0, ROCRAND_ORDERING_PSEUDO_DYNAMIC, initializer);
    ASSERT_EQ(3, mock_state::num_instantiations);
    ASSERT_EQ(0, mock_state::num_destructions);

    dispatcher.init(0, ROCRAND_ORDERING_PSEUDO_DYNAMIC, initializer);
    ASSERT_EQ(6, mock_state::num_instantiations);
    ASSERT_EQ(3, mock_state::num_destructions);
}

TEST_F(rocrand_state_dispatcher_test, retrieve_uninitialized_state_throws)
{
    ASSERT_THROW(
        []
        {
            tested_state_dispatcher dispatcher;
            const auto&             state = dispatcher.get_state<unsigned int>();
            (void)state;
        }(),
        std::out_of_range);
}

TEST_F(rocrand_state_dispatcher_test, update_uninitialized_state_throws)
{
    ASSERT_THROW(
        []
        {
            tested_state_dispatcher dispatcher;
            dispatcher.update_state<unsigned int>([](const auto& /*config*/, auto& state)
                                                  { ++state.value; });
        }(),
        std::out_of_range);
}
