#include "TomlSettings.hpp"
#include "toml.hpp"

TomlSettings::TomlSettings(std::string pathToSettingsFile = "Settings.toml") 
{
    this->pathToSettingsFile = pathToSettingsFile;
}

void TomlSettings::parseFromSettings(ParamsCarrier *paramsCarrier)
{
    const auto data = toml::parse(pathToSettingsFile);
    
    const auto& defaultValues = toml::find(data, "DefaultValues");

    const auto k0 = toml::find<double>(defaultValues, "K0");
    const auto V = toml::find<double>(defaultValues, "V");
    const auto dt = toml::find<double>(defaultValues, "dt");
    
	paramsCarrier->putFloat("K0_default", k0 * 4.4683705e-27); 
    paramsCarrier->putFloat("V_default", V * 6.68458712e-9);
    paramsCarrier->putFloat("dt_default", dt);

    const auto& solarPropSettings = toml::find(data, "SolarPropLikeModelSettings");
    const auto solarPropRatio = toml::find<double>(solarPropSettings, "solarPropRatio");

    const auto& geliosphereSettings = toml::find(data, "GeliosphereModelSettings");
    const auto geliosphereRatio = toml::find<double>(geliosphereSettings, "geliosphereRatio");
    const auto K0Ratio = toml::find<double>(geliosphereSettings, "K0_ratio");
    const auto delta0Ratio = toml::find<double>(geliosphereSettings, "C_delta");
    const auto tiltAngle = toml::find<double>(geliosphereSettings, "tilt_angle");

    const auto& advancedSettings = toml::find(data, "AdvancedSettings");
    const auto removeLogFilesAfterSimulation = toml::find<bool>(advancedSettings, "remove_log_files_after_simulation");

    paramsCarrier->putFloat("solarPropRatio", solarPropRatio);
    paramsCarrier->putFloat("geliosphereRatio", geliosphereRatio);
    paramsCarrier->putFloat("K0_ratio", K0Ratio);
    paramsCarrier->putFloat("C_delta", delta0Ratio);
    paramsCarrier->putFloat("tilt_angle", tiltAngle);
    paramsCarrier->putInt("remove_log_files_after_simulation", removeLogFilesAfterSimulation);
}