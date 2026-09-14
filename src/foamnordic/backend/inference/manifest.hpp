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

#include "bindnordic/manifest.hpp"

namespace foamnordic::inference {

using bindnordic::BundlePayloadRegion;
using bindnordic::bundle_payload_region;
using bindnordic::decode_manifest;
using bindnordic::encode_manifest;
using bindnordic::extract_bundle_payload;
using bindnordic::is_bundle;
using bindnordic::read_bundle_payload;
using bindnordic::map_bundle_payload;
using bindnordic::MappedPayload;
using bindnordic::read_manifest;
using bindnordic::write_bundle;

}  // namespace foamnordic::inference
