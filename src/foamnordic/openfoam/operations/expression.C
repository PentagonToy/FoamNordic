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

#include "expression.H"

#include <cctype>
#include <stdexcept>
#include <utility>

namespace Foam::foamNordic::operations {
namespace {

class Parser {
    const std::string& source_;
    std::size_t position_{0};

    void whitespace() {
        while (position_ < source_.size()
               && std::isspace(
                   static_cast<unsigned char>(source_[position_]))) {
            ++position_;
        }
    }

    std::string token() {
        whitespace();
        const auto start = position_;
        while (position_ < source_.size()) {
            const auto character = source_[position_];
            if (character == '(' || character == ')' || character == ','
                || std::isspace(static_cast<unsigned char>(character))) {
                break;
            }
            ++position_;
        }
        if (start == position_) {
            throw std::invalid_argument(
                "Expected an OpenFOAM field or operation at position "
                + std::to_string(position_) + ".");
        }
        return source_.substr(start, position_ - start);
    }

    Expression node() {
        auto name = token();
        whitespace();
        if (position_ == source_.size() || source_[position_] != '(') {
            return {std::move(name), {}};
        }
        ++position_;
        whitespace();
        if (position_ == source_.size() || source_[position_] == ')') {
            throw std::invalid_argument(
                "OpenFOAM operation " + name + " requires an argument.");
        }
        std::vector<Expression> arguments;
        while (true) {
            arguments.push_back(node());
            whitespace();
            if (position_ == source_.size()) {
                throw std::invalid_argument(
                    "Missing ')' in OpenFOAM operation expression.");
            }
            if (source_[position_] == ')') {
                ++position_;
                break;
            }
            if (source_[position_] != ',') {
                throw std::invalid_argument(
                    "Expected ',' or ')' in OpenFOAM operation expression.");
            }
            ++position_;
        }
        return {std::move(name), std::move(arguments)};
    }

public:
    explicit Parser(const std::string& source) : source_(source) {}

    Expression parse() {
        whitespace();
        if (source_.empty() || position_ == source_.size()) {
            throw std::invalid_argument(
                "OpenFOAM operation expression must not be empty.");
        }
        auto result = node();
        whitespace();
        if (position_ != source_.size()) {
            throw std::invalid_argument(
                "Unexpected content at position "
                + std::to_string(position_) + " in OpenFOAM expression.");
        }
        return result;
    }
};

}  // namespace

bool Expression::primitive() const noexcept {
    return arguments.empty();
}

std::string Expression::canonical() const {
    if (primitive()) {
        return name;
    }
    std::string result = name + "(";
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        if (index != 0) {
            result += ',';
        }
        result += arguments[index].canonical();
    }
    return result + ')';
}

Expression Expression::parse(const std::string& source) {
    return Parser(source).parse();
}

}  // namespace Foam::foamNordic::operations
