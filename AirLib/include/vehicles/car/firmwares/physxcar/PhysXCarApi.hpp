// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef msr_airlib_PhysXCarController_hpp
#define msr_airlib_PhysXCarController_hpp

#include "vehicles/car/api/CarApiBase.hpp"
#include "CarPidController.hpp"
namespace msr { namespace airlib {

class PhysXCarApi : public CarApiBase {
public:
    PhysXCarApi(const AirSimSettings::VehicleSetting* vehicle_setting, std::shared_ptr<SensorFactory> sensor_factory, 
                const Kinematics::State& state, const Environment& environment, const msr::airlib::GeoPoint& home_geopoint)
    : CarApiBase(vehicle_setting, sensor_factory, state, environment),
      home_geopoint_(home_geopoint), state_(state)
    {
        pid_.reset(new CarPidController<float>());
    }

    ~PhysXCarApi()
    {}

protected:
    virtual void resetImplementation() override
    {
        pid_->reset();
        CarApiBase::resetImplementation();
    }

public:
    virtual void update() override
    {
        CarApiBase::update();
    }

    virtual const SensorCollection& getSensors() const override
    {
        return CarApiBase::getSensors();
    }

    // VehicleApiBase Implementation
    virtual void enableApiControl(bool is_enabled) override
    {
        if (api_control_enabled_ != is_enabled) {
            last_controls_ = CarControls();
            api_control_enabled_ = is_enabled;
        }
    }

    virtual bool isApiControlEnabled() const override
    {
        return api_control_enabled_;
    }

    virtual GeoPoint getHomeGeoPoint() const override
    {
        return home_geopoint_;
    }

    virtual bool armDisarm(bool arm) override
    {
        //TODO: implement arming for car
        unused(arm);
        return true;
    }


public:
    virtual void setCarSpeed(float speed) override
    {
        pid_->setGoal(speed);
        target_speed_ = speed;
        speed_control_enabled_ = true;
    }

    virtual void setCarControls(const CarControls& controls) override
    {
        last_controls_ = controls;  
    }

    virtual void updateCarState(const CarState& car_state) override
    {
        last_car_state_ = car_state;
    }

    virtual const CarState& getCarState() const override
    {
        return last_car_state_;
    }

    virtual const CarControls& getCarControls() override
    {
        if (!speed_control_enabled_)
            return last_controls_;
        else
            updateCarControls();            

        return last_controls_;
    }

private:
    bool api_control_enabled_ = false;
    GeoPoint home_geopoint_;
    CarControls last_controls_;
    const Kinematics::State& state_;
    CarState last_car_state_;
    std::unique_ptr<CarPidController<float>> pid_;
    bool speed_control_enabled_;
    float target_speed_;

    void updateCarControls()
    {
        float curr_speed = last_car_state_.speed;
        pid_->setMeasured(curr_speed);

        pid_->update();

        double throttleComp = pid_->getOutput();

        if (throttleComp < 0.0) {
            last_controls_.throttle = 0.0;
            last_controls_.brake = -1 * throttleComp;
        }
        else {
            last_controls_.throttle = throttleComp;
            last_controls_.brake = 0;
        }
    }
};

}}

#endif
