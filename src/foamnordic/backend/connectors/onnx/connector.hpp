// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\ /  \|  V  | |  \| |/__\| _ \ _\| |/ _/
 * | _| \/ | /\ | \_/ | | | ' | \/ | v / v | | \__
 * |_| \__/|_||_|_| |_| |_|\__|\__/|_|_\__/|_|\__/
 *
 * FoamNordic
 */
// clang-format on
#pragma once

#include <memory>

#include "foamnordic/backend/connectors/connector.hpp"

namespace foamnordic::backend {

[[nodiscard]] std::unique_ptr<ModelConnector> make_onnx_connector();

}  // namespace foamnordic::backend
