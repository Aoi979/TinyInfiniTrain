#pragma once

#include <iostream>
#include <map>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "device.h"

namespace infini_train {
class KernelFunction {
public:
    template <typename FuncT> explicit KernelFunction(FuncT &&func) : func_ptr_(reinterpret_cast<void *>(func)) {}

    template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
        // =================================== 作业 ===================================
        // FINISHED：实现通用kernel调用接口
        // 功能描述：将存储的函数指针转换为指定类型并调用
        // =================================== 作业 ===================================

        using FuncT = RetT (*)(ArgsT...);
        auto f = reinterpret_cast<FuncT>(func_ptr_);
        return f(std::forward<ArgsT>(args)...);
    }

private:
    void *func_ptr_ = nullptr;
};

class Dispatcher {
public:
    using KeyT = std::pair<DeviceType, std::string>;

    static Dispatcher &Instance() {
        static Dispatcher instance;
        return instance;
    }

    const KernelFunction &GetKernel(KeyT key) const {
        CHECK(key_to_kernel_map_.contains(key))
            << "Kernel not found: " << key.second << " on device: " << static_cast<int>(key.first);
        return key_to_kernel_map_.at(key);
    }

    template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
        // =================================== 作业 ===================================
        // FINISHED：实现kernel注册机制
        // 功能描述：将kernel函数与设备类型、名称绑定
        // =================================== 作业 ===================================
        auto [it, inserted] = key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
        CHECK(inserted) << "Kernel already registered: " << key.second
                        << " on device: " << static_cast<int>(key.first);
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};
} // namespace infini_train

struct KernelRegister {
    KernelRegister(infini_train::DeviceType device, const std::string &name, auto func) {
        infini_train::Dispatcher::Instance().Register({device, name}, func);
    }
};
#define REGISTER_KERNEL(device, kernel_name, kernel_func) \
static KernelRegister _##kernel_name##_reg(device, #kernel_name, kernel_func);
    // =================================== 作业 ===================================
    // FINISHED：实现自动注册宏
    // 功能描述：在全局静态区注册kernel，避免显式初始化代码
    // =================================== 作业 ===================================
