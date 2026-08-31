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

#ifndef Foam_foamNordic_operations_expression_H
#define Foam_foamNordic_operations_expression_H

#include <string>
#include <vector>

namespace Foam::foamNordic::operations {

struct Expression {
    std::string name;
    std::vector<Expression> arguments;

    [[nodiscard]] bool primitive() const noexcept;
    [[nodiscard]] std::string canonical() const;

    [[nodiscard]] static Expression parse(const std::string& source);
};

}  // namespace Foam::foamNordic::operations

#endif
