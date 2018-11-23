/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "auto_tuner.h"
#include "auto_tuner_offline.h"
namespace kernel_selector 
{
    // SKL GT4e
    void tuning_cache_193B(tuning_data& td)
    {
        tuning_cache_193B_B1_B16(td);
        tuning_cache_193B_B8(td);
        tuning_cache_193B_B32_B64(td);
    }
}