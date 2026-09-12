// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\\ /  \\|  V  | |  \\| |/__\\| _ \\ _\\| |/ _/
 * | _| \\/ | /\\ | \\_/ | | | ' | \\/ | v / v | | \\__
 * |_| \\__/|_||_|_| |_| |_|\\__|\\__/|_|_\\__/|_|\\__/
 *
 * FoamNordic
 */
// clang-format on

#pragma once

#include "bindnordic/artifact.hpp"
#include "foamnordic/fjord/tensor.hpp"
#include "foamnordic/backend/inference/scaler.hpp"

namespace foamnordic::inference {

using bindnordic::FieldContract;
using bindnordic::ModelArtifact;
using bindnordic::ModelFormat;
using bindnordic::ProgramContract;
using bindnordic::TreeLeaf;
using bindnordic::name;

}  // namespace foamnordic::inference
