/**
 * @file ParamsCarrier.hpp
 * @author Michal Solanik
 * @brief Universal map-like structure.
 * @version 0.1
 * @date 2021-07-13
 * 
 * @details Based on discussions in following threads:
 * https://stackoverflow.com/questions/47404870/stdmap-with-different-data-types-for-values
 * https://stackoverflow.com/questions/24702235/c-stdmap-holding-any-type-of-value
 * 
 */

#ifndef COSMIC_SINGLETON_H
#define COSMIC_SINGLETON_H

#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <stdio.h>

/**
 * @brief ParamsCarrier is responsible for storing parameters for simulations.
 * 
 */
class ParamsCarrier
{
	/**
	 * @brief Data structure that can contain multiple data types. Supports
	 * std::string, int, float and double data types.
	 * 
	 */
	struct any
	{
		/**
		 * @brief Enum that represents data type. 
		 * 
		 */
		enum any_type : char
		{
			string_t = 0,
			int_t = 1,
			float_t = 2,
			double_t = 3
		};
		
		/**
		 * @brief Construct a new any object
		 * 
		 */
		any()
		{
		}

		/**
		 * @brief Construct a new any object
		 * 
		 * @param a 
		 */
		any(const any &a)
		{
			this->type = a.type;
			switch (this->type)
			{
			case any_type::string_t:
				new (&(this->str)) std::string(a.str);
				break;
			case any_type::int_t:
				this->i = a.i;
				break;
			case any_type::float_t:
				this->f = a.f;
				break;
			case any_type::double_t:
				this->d = a.d;
				break;
			}
		}

		/**
		 * @brief Destroy the any object
		 * 
		 */
		~any()
		{
			switch (this->type)
			{
			case any_type::string_t:
			{
				if (str.size())
				{
					str.std::string::~string();
				}
			}
			break;
			default:;
			}
		}

		/**
		 * @brief Holds type of stored data
		 * 
		 */
		any_type type;
		
		/**
		 * @brief Holds data defined by type
		 * 
		 */
		union
		{
			std::string str;
			int i;
			float f;
			double d;
		};
	};
	using any_t = any::any_type;

private:
	/**
	 * @brief Single tone instance
	 * 
	 */
	static ParamsCarrier *INSTANCE;

	/**
	 * @brief Construct a new Params Carrier object
	 * 
	 */
	ParamsCarrier();

	/**
	 * @brief Holds value for given key 
	 * 
	 */
	std::map<std::string, any> m;

public:
	/**
	 * @brief Implementation of Single tone pattern
	 * 
	 * @return ParamsCarrier* instance of this class
	 */
	static ParamsCarrier *instance()
	{
		if (!INSTANCE)
			INSTANCE = new ParamsCarrier();
		return INSTANCE;
	}

	/**
	 * @brief Put string with given value for given key into ParamsCarrier. 
	 * 
	 * @param key Key is used for storing value in map encapsulated in ParamsCarrier.
	 * @param value Value that should be put into map. 
	 */
	void putString(std::string key, std::string value)
	{
		any a;
		a.type = any_t::string_t;
		new (&(a.str)) std::string(value);
		m.insert({key, a});
	}

	/**
	 * @brief Put float with given value for given key into ParamsCarrier. 
	 * 
	 * @param key Key is used for storing value in map encapsulated in ParamsCarrier.
	 * @param value Value that should be put into map. 
	 */
	void putFloat(std::string key, float value)
	{
		any a;
		a.type = any_t::float_t;
		a.f = value;
		m.insert({key, a});
	}

	/**
	 * @brief Put integer with given value for given key into ParamsCarrier. 
	 * 
	 * @param key Key is used for storing value in map encapsulated in ParamsCarrier.
	 * @param value Value that should be put into map. 
	 */
	void putInt(std::string key, int value)
	{
		any a;
		a.type = any_t::int_t;
		a.i = value;
		m.insert({key, a});
	}

	/**
	 * @brief Put double with given value for given key into ParamsCarrier. 
	 * 
	 * @param key Key is used for storing value in map encapsulated in ParamsCarrier.
	 * @param value Value that should be put into map. 
	 */
	void putDouble(std::string key, double value)
	{
		any a;
		a.type = any_t::double_t;
		a.d = value;
		m.insert({key, a});
	}

	/**
	 * @brief Get the integer value for key. If record for
	 * given key is not present function will return default
	 * value.
	 * 
	 * @param key Key of record
	 * @param defaultValue Default value that should be returned
	 * in case of incompatible type or missing record.
	 * @return found value or default value if record is missing
	 * or type does not match.
	 */
	int getInt(std::string key, int defaultValue)
	{
		auto search = m.find(key);
		if (search != m.end())
		{
			if (search->second.type == any_t::int_t)
			{
				return search->second.i;
			}
			return defaultValue;
		}
		return defaultValue;
	}

	/**
	 * @brief Get the float value for key. If record for
	 * given key is not present function will return default
	 * value.
	 * 
	 * @param key Key of record
	 * @param defaultValue Default value that should be returned
	 * in case of incompatible type or missing record.
	 * @return found value or default value if record is missing
	 * or type does not match.
	 */
	float getFloat(std::string key, float defaultValue)
	{
		auto search = m.find(key);
		if (search != m.end())
		{
			if (search->second.type == any_t::float_t)
			{
				return search->second.f;
			}
			return defaultValue;
		}
		return defaultValue;
	}

	/**
	 * @brief Get the double value for key. If record for
	 * given key is not present function will return default
	 * value.
	 * 
	 * @param key Key of record
	 * @param defaultValue Default value that should be returned
	 * in case of incompatible type or missing record.
	 * @return found value or default value if record is missing
	 * or type does not match.
	 */
	double getDouble(std::string key, double defaultValue)
	{
		auto search = m.find(key);
		if (search != m.end())
		{
			if (search->second.type == any_t::double_t)
			{
				return search->second.d;
			}
			return defaultValue;
		}
		return defaultValue;
	}

	/**
	 * @brief Get the string value for key. If record for
	 * given key is not present function will return default
	 * value.
	 * 
	 * @param key Key of record
	 * @param defaultValue Default value that should be returned
	 * in case of incompatible type or missing record.
	 * @return found value or default value if record is missing
	 * or type does not match.
	 */
	std::string getString(std::string key, std::string defaultValue)
	{
		auto search = m.find(key);
		if (search != m.end())
		{
			if (search->second.type == any_t::string_t)
			{
				return search->second.str;
			}
			return defaultValue;
		}
		return defaultValue;
	}
};

#endif