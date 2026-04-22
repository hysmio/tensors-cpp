#pragma once

#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace flags {

enum class Type { Bool, Int, Float, String };

using Value = std::variant<bool, int, float, std::string>;

struct FlagOption {
    std::string name;
    char short_name;
    Type type;
    Value default_value;
    std::optional<float> min;
    std::optional<float> max;
    std::string description;

    FlagOption(std::string name, char short_name, Type type, Value default_value,
               std::optional<float> min = {}, std::optional<float> max = {},
               std::string description = "")
        : name(std::move(name)), short_name(short_name), type(type),
          default_value(std::move(default_value)), min(min), max(max),
          description(std::move(description)) {}

    FlagOption(std::string name, Type type, Value default_value, std::optional<float> min = {},
               std::optional<float> max = {}, std::string description = "")
        : FlagOption(std::move(name), '\0', type, std::move(default_value), min, max,
                     std::move(description)) {}
};

class Parser {
  public:
    explicit Parser(std::vector<FlagOption> options) : options_(std::move(options)) {
        for (const auto &opt : options_) {
            values_[opt.name] = opt.default_value;
        }
    }

    bool parse(int argc, char *argv[]) {
        program_name_ = argv[0];

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];

            if (arg == "--help" || arg == "-h") {
                print_help();
                return false;
            }

            std::string flag_name;
            FlagOption *opt = nullptr;

            if (arg.substr(0, 2) == "--") {
                flag_name = arg.substr(2);
                opt = find_option(flag_name);
            } else if (arg.size() == 2 && arg[0] == '-') {
                opt = find_option_short(arg[1]);
                if (opt) {
                    flag_name = opt->name;
                }
            } else {
                std::cerr << "Error: Expected flag starting with '--' or '-', got: " << arg << "\n";
                return false;
            }

            if (!opt) {
                std::cerr << "Error: Unknown flag: " << arg << "\n";
                return false;
            }

            if (opt->type == Type::Bool) {
                values_[opt->name] = true;
                continue;
            }

            if (i + 1 >= argc) {
                std::cerr << "Error: Flag '" << arg << "' requires a value\n";
                return false;
            }

            std::string value_str = argv[++i];

            if (!parse_value(flag_name, value_str, *opt)) {
                return false;
            }
        }

        return true;
    }

    template <typename T> T get(const std::string &name) const {
        auto it = values_.find(name);
        if (it == values_.end()) {
            throw std::runtime_error("Unknown flag: " + name);
        }
        return std::get<T>(it->second);
    }

    bool get_bool(const std::string &name) const { return get<bool>(name); }
    int get_int(const std::string &name) const { return get<int>(name); }
    float get_float(const std::string &name) const { return get<float>(name); }
    std::string get_string(const std::string &name) const { return get<std::string>(name); }

    void print_help() const {
        std::cout << "Usage: " << program_name_ << " [OPTIONS]\n\n";
        std::cout << "Options:\n";
        std::cout << "  --help, -h          Show this help message\n";

        for (const auto &opt : options_) {
            std::cout << "  --" << opt.name;
            if (opt.short_name != '\0') {
                std::cout << ", -" << opt.short_name;
            }

            switch (opt.type) {
            case Type::Bool:
                std::cout << " (bool)";
                break;
            case Type::Int:
                std::cout << " <int>";
                break;
            case Type::Float:
                std::cout << " <float>";
                break;
            case Type::String:
                std::cout << " <string>";
                break;
            }

            std::cout << "\n";

            if (!opt.description.empty()) {
                std::cout << "      " << opt.description << "\n";
            }

            std::cout << "      Default: ";
            print_value(opt.default_value);

            if (opt.min.has_value() || opt.max.has_value()) {
                std::cout << "      Range: ";
                if (opt.min.has_value()) {
                    std::cout << "[" << opt.min.value();
                } else {
                    std::cout << "(-inf";
                }
                std::cout << ", ";
                if (opt.max.has_value()) {
                    std::cout << opt.max.value() << "]";
                } else {
                    std::cout << "inf)";
                }
                std::cout << "\n";
            }
        }
    }

  private:
    std::vector<FlagOption> options_;
    std::unordered_map<std::string, Value> values_;
    std::string program_name_;

    FlagOption *find_option(const std::string &name) {
        for (auto &opt : options_) {
            if (opt.name == name) {
                return &opt;
            }
        }
        return nullptr;
    }

    FlagOption *find_option_short(char short_name) {
        for (auto &opt : options_) {
            if (opt.short_name == short_name) {
                return &opt;
            }
        }
        return nullptr;
    }

    bool parse_value(const std::string &name, const std::string &value_str, const FlagOption &opt) {
        try {
            switch (opt.type) {
            case Type::Bool: {
                if (value_str == "true" || value_str == "1") {
                    values_[name] = true;
                } else if (value_str == "false" || value_str == "0") {
                    values_[name] = false;
                } else {
                    std::cerr << "Error: Invalid bool value for --" << name << ": " << value_str
                              << "\n";
                    return false;
                }
                break;
            }
            case Type::Int: {
                int val = std::stoi(value_str);
                if (!validate_range(name, static_cast<float>(val), opt)) {
                    return false;
                }
                values_[name] = val;
                break;
            }
            case Type::Float: {
                float val = std::stof(value_str);
                if (!validate_range(name, val, opt)) {
                    return false;
                }
                values_[name] = val;
                break;
            }
            case Type::String: {
                values_[name] = value_str;
                break;
            }
            }
        } catch (const std::exception &e) {
            std::cerr << "Error: Failed to parse value for --" << name << ": " << value_str << "\n";
            return false;
        }

        return true;
    }

    bool validate_range(const std::string &name, float value, const FlagOption &opt) {
        if (opt.min.has_value() && value < opt.min.value()) {
            std::cerr << "Error: Value for --" << name << " (" << value
                      << ") is below minimum (" << opt.min.value() << ")\n";
            return false;
        }
        if (opt.max.has_value() && value > opt.max.value()) {
            std::cerr << "Error: Value for --" << name << " (" << value
                      << ") is above maximum (" << opt.max.value() << ")\n";
            return false;
        }
        return true;
    }

    void print_value(const Value &val) const {
        std::visit(
            [](auto &&arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, bool>) {
                    std::cout << (arg ? "true" : "false");
                } else if constexpr (std::is_same_v<T, std::string>) {
                    std::cout << "\"" << arg << "\"";
                } else {
                    std::cout << arg;
                }
            },
            val);
        std::cout << "\n";
    }
};

} // namespace flags
