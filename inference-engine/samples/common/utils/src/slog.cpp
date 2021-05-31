// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

#include "samples/slog.hpp"

namespace slog {

LogStream info("INFO", std::cout);
LogStream warn("WARNING", std::cout);
LogStream err("ERROR", std::cerr);

// Specializing for LogStreamEndLine to support slog::endl
LogStream& LogStream::operator<<(const LogStreamEndLine& /*arg*/) {
    _new_line = true;

    (*_log_stream) << std::endl;
    return *this;
}

// Specializing for LogStreamBoolAlpha to support slog::boolalpha
LogStream& LogStream::operator<<(const LogStreamBoolAlpha& /*arg*/) {
    (*_log_stream) << std::boolalpha;
    return *this;
}

}  // namespace slog