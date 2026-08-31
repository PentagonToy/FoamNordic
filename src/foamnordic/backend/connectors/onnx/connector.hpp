#pragma once

#include <memory>

#include "foamnordic/backend/connectors/connector.hpp"

namespace foamnordic::backend {

[[nodiscard]] std::unique_ptr<ModelConnector> make_onnx_connector();

}  // namespace foamnordic::backend
