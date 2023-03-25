#include "TomlSettings.hpp"
#include "toml.hpp"

TomlSettings::TomlSettings(std::string pathToSettingsFile = "Settings.toml") 
{
    this->pathToSettingsFile = pathToSettingsFile;
}

void TomlSettings::parseFromSettings(ParamsCarrier *paramsCarrier)
{
    const auto data = toml::parse(pathToSettingsFile);
    
    const auto& defaultValues = toml::find(data, "default_values");

    const auto k0 = toml::find<double>(defaultValues, "K0");
    const auto V = toml::find<double>(defaultValues, "V");
    const auto dt = toml::find<double>(defaultValues, "dt");
    const auto rInitial = toml::find<double>(defaultValues, "r_injection");
    const auto uniformEnergyInjectionMaximum = toml::find<double>(defaultValues, "uniform_energy_injection_maximum");
    
	paramsCarrier->putFloat("K0_default", k0 * 4.4683705e-27); 
    paramsCarrier->putFloat("V_default", V * 6.68458712e-9);
    paramsCarrier->putFloat("dt_default", dt);
    paramsCarrier->putFloat("r_injection", rInitial);
    paramsCarrier->putFloat("uniform_energy_injection_maximum", uniformEnergyInjectionMaximum);
    
    const auto& twoDimensionalModelsCommonSettings = toml::find(data, "2d_models_common_settings");

    const auto thetaInjection = toml::find<double>(twoDimensionalModelsCommonSettings, "theta_injection");
    const auto useUniformInjection = toml::find<bool>(twoDimensionalModelsCommonSettings, "use_uniform_injection");

    paramsCarrier->putFloat("theta_injection", thetaInjection);
    paramsCarrier->putInt("use_uniform_injection", useUniformInjection);

    const auto& solarPropSettings = toml::find(data, "SolarProp_like_model_settings");
    const auto solarPropRatio = toml::find<double>(solarPropSettings, "SolarProp_ratio");

    const auto& geliosphereSettings = toml::find(data, "Geliosphere_model_settings");
    const auto geliosphereRatio = toml::find<double>(geliosphereSettings, "Geliosphere_ratio");
    const auto K0Ratio = toml::find<double>(geliosphereSettings, "K0_ratio");
    const auto delta0Ratio = toml::find<double>(geliosphereSettings, "C_delta");
    const auto defaultTiltAngle = toml::find<double>(geliosphereSettings, "default_tilt_angle");

    const auto& advancedSettings = toml::find(data, "advanced_settings");
    const auto removeLogFilesAfterSimulation = toml::find<bool>(advancedSettings, "remove_log_files_after_simulation");

    paramsCarrier->putFloat("solarPropRatio", solarPropRatio);
    paramsCarrier->putFloat("geliosphereRatio", geliosphereRatio);
    paramsCarrier->putFloat("K0_ratio", K0Ratio);
    paramsCarrier->putFloat("C_delta", delta0Ratio);
    paramsCarrier->putFloat("default_tilt_angle", defaultTiltAngle);
    paramsCarrier->putInt("remove_log_files_after_simulation", removeLogFilesAfterSimulation);
}