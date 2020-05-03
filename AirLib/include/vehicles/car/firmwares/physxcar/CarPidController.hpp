#pragma once

#include <cstdlib>
#include <algorithm>
#include "common/ClockBase.hpp"

template<class T>
class CarPidController
{
public:
    CarPidController()
    {
    }

    void setGoal(const T& goal)
    {
        goal_ = goal;
    }
    const T& getGoal() const
    {
        return goal_;
    }

    void setMeasured(const T& measured)
    {
        measured_ = measured;
    }
    const T& getMeasured() const
    {
        return measured_;
    }

    T getOutput()
    {
        return output_;
    }

    void reset()
    {
        goal_ = T();
        measured_ = T();
        last_time_ = clock_get_millis();
        integrator_reset();
        last_error_ = goal_ - measured_;
        min_dt_ = config_.time_scale * config_.time_scale;
    }

    void update()
    {
        const T error = goal_ - measured_;

        float dt = (clock_get_millis() - last_time_) * config_.time_scale;

        float pterm = error * config_.kp;
        float dterm = 0;
        if (dt > min_dt_) {
            update_integrator(dt, error, last_time_);

            float error_der = (error - last_error_) / dt;
            dterm = error_der * config_.kd;
            last_error_ = error;
        }

        output_ = config_.output_bias + pterm + get_integrator_output() + dterm;

        //limit final output
        //output_ = std::clamp(output_, config_.min_output, config_.max_output);
        output_ = clip(output_, config_.min_output, config_.max_output);

        last_time_ = clock_get_millis();
    }

private:
    void update_integrator(float dt, T error, uint64_t last_time_)
    {
        iterm_int_ = iterm_int_ * config_.iterm_discount + dt * error * config_.ki;
        //iterm_int_ = std::clamp(iterm_int_, config_.min_output, config_.max_output);
        iterm_int_ = clip(iterm_int_, config_.min_output, config_.max_output);
    }

    static T clip(T val, T min_value, T max_value)
    {
        return std::max(min_value, std::min(val, max_value));
    }

    void integrator_reset()
    {
        iterm_int_ = T();
    }

    T get_integrator_output()
    {
        return iterm_int_;
    }


private:
    T goal_, measured_;
    T output_;
    uint64_t last_time_;

    float last_error_;
    float min_dt_;
    float iterm_int_;

    uint64_t clock_get_millis()
    {
        return clock()->nowNanos() / 1e6;
    }

    const msr::airlib::ClockBase* clock() const
    {
        return msr::airlib::ClockFactory::get();
    }

    struct PidConfig
    {
        PidConfig(float kp_val = 2.0f, float ki_val = 0.0f, float kd_val = 0.5f,
            T min_output_val = -1, T max_output_val = 1,
            float time_scale_val = 1.0f / 1000,
            bool enabled_val = true, T output_bias_val = T(), float iterm_discount_val = 1)
            : kp(kp_val), ki(ki_val), kd(kd_val),
            time_scale(time_scale_val),
            min_output(min_output_val), max_output(max_output_val),
            enabled(enabled_val), output_bias(output_bias_val), iterm_discount(iterm_discount_val)
        {}

        float kp, ki, kd;
        float time_scale;
        T min_output, max_output;
        bool enabled;
        T output_bias;
        float iterm_discount;
    } config_;
};