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

    const auto& advancedSettings = toml::find(data, "AdvancedSettings");
    
    const auto solarPropRatio = toml::find<double>(advancedSettings, "solarPropRatio");
    const auto ratio = toml::find<double>(advancedSettings, "ratio");
    const auto K0Ratio = toml::find<double>(advancedSettings, "K0_ratio");
    const auto delta0Ratio = toml::find<double>(advancedSettings, "delta0_ratio");
    const auto removeLogFilesAfterSimulation = toml::find<bool>(advancedSettings, "remove_log_files_after_simulation");

    paramsCarrier->putFloat("solarPropRatio", solarPropRatio);
    paramsCarrier->putFloat("ratio", ratio);
    paramsCarrier->putFloat("K0_ratio", K0Ratio);
    paramsCarrier->putFloat("delta0_ratio", delta0Ratio);
    paramsCarrier->putInt("remove_log_files_after_simulation", removeLogFilesAfterSimulation);
}