// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_common.hpp"
#include "unit_test_utils/mocks/mock_allocator.hpp"

using namespace InferenceEngine;
using namespace ::testing;

using LockedMemoryTest = testing::Test;

TEST_F(LockedMemoryTest, canUnlockMemoryAfterUsage) {
    std::unique_ptr<MockAllocator> allocator(new MockAllocator());
    char array [] = {1,2,3};

    EXPECT_CALL(*allocator.get(), lock((void*)1, _)).WillRepeatedly(Return((void*)array));
    EXPECT_CALL(*allocator.get(), unlock(_)).Times(1);
    {
        auto x = LockedMemory<char>(allocator.get(), (void *) 1, 1);
        //force locking of memory
        auto t = x[0];
        (void)t;
    }
}

TEST_F(LockedMemoryTest, canReadFromLockedMemory) {
    std::unique_ptr<MockAllocator> allocator(new MockAllocator());
    char array [] = {1,2,3,4,5};

    EXPECT_CALL(*allocator.get(), lock((void*)1, _)).WillRepeatedly(Return((void*)array));
    EXPECT_CALL(*allocator.get(), unlock(_)).Times(1);
    {
        auto x = LockedMemory<char>(allocator.get(), (void *) 1, 0);
        //we are getting first element
        ASSERT_EQ(1, x[0]);
    }
}

TEST_F(LockedMemoryTest, canWriteToLockedMemory) {
    std::unique_ptr<MockAllocator> allocator(new MockAllocator());
    char array [] = {1,2,3,4,5};

    EXPECT_CALL(*allocator.get(), lock((void*)1, _)).WillRepeatedly(Return((void*)array));
    EXPECT_CALL(*allocator.get(), unlock(_)).Times(1);
    {
        auto x = LockedMemory<char>(allocator.get(), (void *) 1, 0);

        //we are getting first element
        ASSERT_EQ(std::distance(array, &x[0]), 0);
        x[0] = 5;
    }
    EXPECT_EQ(array[0], 5);

}
