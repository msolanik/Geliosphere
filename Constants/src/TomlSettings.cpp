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
}