/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Fwd/hkcstdio.h>
#include <Common/Base/Fwd/hkcstdarg.h>
#include <Common/Base/Config/hkConfigVersion.h>

#include <Common/Base/System/Io/OptionParser/hkOptionParser.h>

const static int s_consoleWidth = 80;
const static int s_maxErrorMessageLength = 256;
const static int s_maxDefaultTextLength = 512;

// A specific assert is being defined here because the option parser code may be executed before the base system and memory manager have been set up.
// HK_ASSERT relies on creating an object (using said memory manager) which means that we cannot use it here.
#ifdef HK_DEBUG
	#define OPTION_PARSER_ASSERT(x, s) {if (!(x)) {HK_BREAKPOINT(0); outputAssertMessage(s);}}
	#define OPTION_PARSER_ASSERT_RETURN(x, s) {if (!(x)) {HK_BREAKPOINT(0); outputAssertMessage(s); return false;}}
#else
	#define OPTION_PARSER_ASSERT(x, s) {}
	#define OPTION_PARSER_ASSERT_RETURN(x, s) {}
#endif

namespace
{
	void outputAssertMessage(const char* format, ...);

	enum OptionFlagInfoFlags
	{
		FLAGINFO_LONG	= 1,	// --flag
		FLAGINFO_EQUALS = 2,	// --flag=VALUE | -f=VALUE
		FLAGINFO_CONCAT = 4		// --flagVALUE | -fVALUE
	};

	struct OptionFlagInfo
	{
		const char* m_name;
		int	m_nameLength;
		const char* m_value;
		int	m_valueLength;
		int	m_flags;
	};

	void nullIfEmpty(const char*& str)
	{
		if (str != HK_NULL && hkString::strLen(str) == 0)
		{
			str = HK_NULL;
		}
	}

	bool isDigitChar(char c)
	{
		return (c >= '0' && c <= '9');
	}

	bool isLiteralChar(char c)
	{
		return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
	}

	bool isLiteral(const char* flag)
	{
		int i = 0;

		while (flag[i] != '\0')
		{
			if (!isLiteralChar(flag[i]))
			{
				return false;
			}
			i ++;
		}
		return true;
	}

	bool isInteger(const char* value, int valueLength, bool allowNegative = true)
	{
		int offset = 0;

		if (value[0] == '-')
		{
			if (allowNegative)
			{
				offset = 1;
			}
			else
			{
				return false;
			}
		}

		bool valid = true;
		bool hex = false;

		if (value[offset] == '0')
		{
			if ((value[1 + offset] == 'x') && valueLength >= 3)
			{
				hex = true;
			}
			else
			{
				// Allow a single 0
				if (valueLength != 1)
				{
					valid = false;
				}
			}
		}

		if (valid)
		{
			for (int i = offset; i < valueLength; i++)
			{
				// Can skip 'x' if hexadecimal
				if (hex && i == 1 + offset)
				{
					continue;
				}

				// Standard digit test
				if (!isDigitChar(value[i]))
				{
					valid = false;
				}

				// Check for hex
				if (hex && i > 1 + offset)
				{
					if ((value[i] >= 'A' && value[i] <= 'F') || (value[i] >= 'a' && value[i] <= 'f'))
					{
						valid = true;
					}
				}
			}
		}

		return valid;
	}

	void printTextBlock(const char* textBlock, int startPosition = 0, bool absolute = false)
	{
		printf("%s\n", textBlock);
// 		OPTION_PARSER_ASSERT(startPosition >= 0, "Start position must not be negative");
// 
// 		int textLength = hkString::strLen(textBlock);
// 		int textWidth = s_consoleWidth - startPosition;
// 		int chars = 0;
// 
// 		#ifdef HK_DEBUG
// 		for (int i = 0; i < textLength; i++)
// 		{
// 			OPTION_PARSER_ASSERT(textBlock[i] != '\n', "Text blocks cannot contain new line characters");
// 		}
// 		#endif
// 
// 		while (chars < textLength)
// 		{
// 			// Print left whitespace
// 			if (absolute || chars > 0)
// 			{
// 				printf("%*s", startPosition, "");
// 			}
// 
// 			char textLine[s_consoleWidth];
// 			int numChars = sprintf(textLine, "%.*s", textWidth, textBlock + chars);
// 
// 			int fix = 0;
// 			int start = chars;
// 			chars += numChars;
// 
// 			if (numChars == textWidth && textLine[textWidth - 1] != ' ')
// 			{
// 				// Don't break words, find the starting position of the last word
// 				int index = 1;
// 
// 				while ((textLine[numChars - index] != ' ') && (numChars - index > 0))
// 				{
// 					index++;
// 				}
// 
// 				// If word does not take up whole line
// 				if (numChars > index)
// 				{
// 					chars -= index - 1; 
// 				}
// 			}
// 			else if (chars < textLength)
// 			{
// 				// Remove end of line space
// 				fix = -1;
// 			}
// 
// 			printf("%.*s\n", chars - start + fix, textBlock + start);
// 		}
	}

	void outputAssertMessage(const char* format, ...)
	{
		va_list va;
		va_start(va, format);
		char errorMessage[s_maxErrorMessageLength];
		vsnprintf(errorMessage, s_maxErrorMessageLength, format, va);
		va_end(va);
		printf("ASSERT FAILED: ");
		printTextBlock(errorMessage, 15);
	}

	void outputErrorMessage(hkOptionParser* parser, const char* format, ...)
	{
		va_list va;
		va_start(va, format);
		char errorMessage[s_maxErrorMessageLength];
		vsnprintf(errorMessage, s_maxErrorMessageLength, format, va);
		va_end(va);
		printf("ERROR: ");
		printTextBlock(errorMessage, 7);
		printf("\n");
		parser->usage();
	}

	const char* getFlagName(OptionFlagInfo* flagInfo, hkOptionParser::Option* option)
	{
		return ((flagInfo->m_flags & FLAGINFO_LONG) ? option->m_longFlag : option->m_shortFlag);
	}

	// Parse a single argument and validate it as a flag
	hkResult parseFlag(const char* argv, OptionFlagInfo* flagInfo)
	{
		// Clear flagInfo struct
		flagInfo->m_name		= HK_NULL;
		flagInfo->m_nameLength	= 0;
		flagInfo->m_value		= HK_NULL;
		flagInfo->m_valueLength	= 0;
		flagInfo->m_flags		= 0;

		// All flags must start with a '-'
		if (argv[0] == '-')
		{
			flagInfo->m_name = argv + 1;

			if (argv[1] == '-')
			{
				// Flag is a long flag "--flag" as opposed to a short flag "-f"
				flagInfo->m_name++;
				flagInfo->m_flags |= FLAGINFO_LONG;
			}

			// Loop over characters
			while (true)
			{
				if (flagInfo->m_name[flagInfo->m_nameLength] == '=')
				{
					// Flag has a value
					flagInfo->m_flags |= FLAGINFO_EQUALS;
					break;
				}
				else if (flagInfo->m_name[flagInfo->m_nameLength] == '\0')
				{
					// Flag does not have a value
					break;
				}
				else if (!isLiteralChar(flagInfo->m_name[flagInfo->m_nameLength]))
				{
					// Flag is invalid
					flagInfo->m_nameLength = 0; // To trigger invalid
					break;
				}

				flagInfo->m_nameLength++;
			}

			// Assumes any flag which has a digit at the end is of type '--flagVALUE' and does not use '=' is a concatenated flag
			if (!(flagInfo->m_flags & FLAGINFO_EQUALS) &&
				flagInfo->m_nameLength > (flagInfo->m_flags & FLAGINFO_LONG ? 2 : 1) &&
				isDigitChar(flagInfo->m_name[flagInfo->m_nameLength - 1]))
			{
				flagInfo->m_flags |= FLAGINFO_CONCAT;
			}

			// if the flag name length is at least 1 character for short flags, and at least 2 for long flags
			if (flagInfo->m_nameLength > (flagInfo->m_flags & FLAGINFO_LONG ? 1 : 0))
			{
				if (flagInfo->m_flags & FLAGINFO_EQUALS)
				{
					// Parse value
					flagInfo->m_value = flagInfo->m_name + flagInfo->m_nameLength + 1;

					while (flagInfo->m_value[flagInfo->m_valueLength] != '\0')
					{
						flagInfo->m_valueLength++;
					}
				}
				else if (flagInfo->m_flags & FLAGINFO_CONCAT)
				{
					// Alter name length and value for a counter flag --flagVALUE
					while (isDigitChar(flagInfo->m_name[flagInfo->m_nameLength - 1]))
					{
						flagInfo->m_nameLength --;
						flagInfo->m_valueLength ++;
					}

					flagInfo->m_value = (flagInfo->m_name + flagInfo->m_nameLength);
				}
				return HK_SUCCESS;
			}
		}

		return HK_FAILURE;
	}

	// Search our stored options for the option in question
	hkResult findOption(const char* name, int nameLength, int flags, hkOptionParser::Option* options, int optionCount, hkOptionParser::Option* option = HK_NULL, int skipIndex = -1)
	{
		bool found = false;

		for (int i = 0; i < optionCount; i ++)
		{
			// Used when verifying options to make sure that there are no duplicates
			if (i == skipIndex)
			{
				continue;
			}

			hkOptionParser::Option* currentOption = (options + i);

			if (flags & FLAGINFO_LONG)
			{
				// Check that the length of both matches, this is not necessary for short flags as they are always 1
				if (nameLength == hkString::strLen(currentOption->m_longFlag))
				{
					found = (hkString::strNcasecmp(name, currentOption->m_longFlag, nameLength) == 0);
				}
			}
			else if (currentOption->m_shortFlag)
			{
				found = (hkString::strNcasecmp(name, currentOption->m_shortFlag, nameLength) == 0);
			}

			if (found)
			{
				if (option != HK_NULL)
				{
					option->m_shortFlag	= currentOption->m_shortFlag;
					option->m_longFlag	= currentOption->m_longFlag;
					option->m_help		= currentOption->m_help;
					option->m_type		= currentOption->m_type;
					option->m_value		= currentOption->m_value;
					option->m_default	= currentOption->m_default;
				}
				return HK_SUCCESS;
			}
		}

		return HK_FAILURE;
	}

	// Parses the string value to set the option's value in its native format
	hkResult setOptionValue(hkOptionParser::Option* option, const char* value, int valueLength = -1)
	{
		if (valueLength == -1)
		{
			valueLength = hkString::strLen(value);
		}

		switch (option->m_type)
		{
		case hkOptionParser::OPTION_BOOL:
			{
				if (valueLength == 0 || (valueLength == 1 && value[0] == '1'))
				{
					option->m_value->b = true;
				}
				else if (valueLength == 1 && value[0] == '0')
				{
					option->m_value->b = false;
				}
				else
				{
					return HK_FAILURE;
				}
				break;
			}
		case hkOptionParser::OPTION_COUNTER:
			{
				if (valueLength == 0)
				{
					option->m_value->i ++;
				}
				else if (isInteger(value, valueLength, false))
				{
					option->m_value->i += hkString::atoi(value);
				}
				else
				{
					return HK_FAILURE;
				}
			}
			break;
		case hkOptionParser::OPTION_INT:
			{
				if (valueLength > 0)
				{
					if (isInteger(value, valueLength))
					{
						option->m_value->i = hkString::atoi(value);
					}
					else
					{
						return HK_FAILURE;
					}
				}
				else
				{
					return HK_FAILURE;
				}
			}
			break;
		case hkOptionParser::OPTION_STRING:
			{
				option->m_value->s = value;
			}
			break;
		}

		return HK_SUCCESS;
	}

	// Reset all option's values to their defaults
	void defaultOptions(hkOptionParser::Option* options, int optionCount)
	{
		for (int i = 0; i < optionCount; i++)
		{
			hkOptionParser::Option* currentOption = (options + i);

			switch (currentOption->m_type)
			{
			case hkOptionParser::OPTION_BOOL:
				currentOption->m_value->b = currentOption->m_default.b;
				break;
			case hkOptionParser::OPTION_COUNTER:
				currentOption->m_value->u = currentOption->m_default.u;
				break;
			case hkOptionParser::OPTION_INT:
				currentOption->m_value->i = currentOption->m_default.i;
				break;
			case hkOptionParser::OPTION_STRING:
				currentOption->m_value->s = currentOption->m_default.s;
				break;
			}
		}
	}
}

hkOptionParser::hkOptionParser(const char* title, const char* desc) :
	m_title(title),
	m_desc(desc),
	m_arguments(HK_NULL),
	m_argumentsName(HK_NULL),
	m_argumentsHelp(HK_NULL)
{}

hkOptionParser::Option::Option(const char* shortFlag, const char* longFlag, const char* help, const char** value, const char* defaultValue) :
	m_shortFlag(shortFlag),
	m_longFlag(longFlag),
	m_help(help),
	m_type(hkOptionParser::OPTION_STRING)
{
	m_value = static_cast<hkOptionParser::OptionValue*>((void*)value);
	m_default.s = defaultValue;
}

hkOptionParser::Option::Option(const char* shortFlag, const char* longFlag, const char* help, bool* value, bool defaultValue) :
	m_shortFlag(shortFlag),
	m_longFlag(longFlag),
	m_help(help),
	m_type(hkOptionParser::OPTION_BOOL)
{
	m_value = static_cast<hkOptionParser::OptionValue*>((void*)value);
	m_default.b = defaultValue;
}

hkOptionParser::Option::Option(const char* shortFlag, const char* longFlag, const char* help, int* value, int defaultValue) :
	m_shortFlag(shortFlag),
	m_longFlag(longFlag),
	m_help(help),
	m_type(hkOptionParser::OPTION_INT)
{
	m_value = static_cast<hkOptionParser::OptionValue*>((void*)value);
	m_default.i = defaultValue;
}

hkOptionParser::Option::Option(const char* shortFlag, const char* longFlag, const char* help, unsigned int* value, unsigned int defaultValue) :
	m_shortFlag(shortFlag),
	m_longFlag(longFlag),
	m_help(help),
	m_type(hkOptionParser::OPTION_COUNTER)
{
	m_value = static_cast<hkOptionParser::OptionValue*>((void*)value);
	m_default.u = defaultValue;
}

bool hkOptionParser::setOptions(Option* options, int count)
{
	for (int i = 0; i < count; i++)
	{
		Option* option = &options[i];

		// Change the empty string to HK_NULL
		nullIfEmpty(option->m_shortFlag);
		nullIfEmpty(option->m_longFlag);
		nullIfEmpty(option->m_help);

		// Null checks
		OPTION_PARSER_ASSERT_RETURN(option->m_longFlag != HK_NULL, "Long flag must be specified\n");
		OPTION_PARSER_ASSERT_RETURN(option->m_help != HK_NULL, "Help text must beprovided\n");
		OPTION_PARSER_ASSERT_RETURN(option->m_value != HK_NULL, "Address to store option value must be provided\n");

		// Check option flags are valid
		OPTION_PARSER_ASSERT_RETURN(option->m_shortFlag == HK_NULL || hkString::strLen(option->m_shortFlag) == 1, "Length of short flag must be exactly 1\n");
		OPTION_PARSER_ASSERT_RETURN(option->m_shortFlag == HK_NULL || isLiteral(option->m_shortFlag), "Short flag not valid\n");
		OPTION_PARSER_ASSERT_RETURN(hkString::strLen(option->m_longFlag) >= 2, "Length of long flag must be at least 2\n");
		OPTION_PARSER_ASSERT_RETURN(isLiteral(option->m_longFlag), "Long flag not valid, must be alphanumeric\n");

		// Check if the flag of counters ends in a number (this is not allowed)
		OPTION_PARSER_ASSERT_RETURN(option->m_type != OPTION_COUNTER || (!isDigitChar(option->m_longFlag[hkString::strLen(option->m_longFlag) - 1])), "Flags for counters cannot end in a digit");
		OPTION_PARSER_ASSERT_RETURN(option->m_type != OPTION_COUNTER || option->m_shortFlag == HK_NULL || (!isDigitChar(option->m_shortFlag[0])), "Flags for counters cannot end in a digit");

		// Check if the help option isn't manually defined
		OPTION_PARSER_ASSERT_RETURN(option->m_shortFlag == HK_NULL || hkString::strNcasecmp(option->m_shortFlag, "h", 1) != 0, "The -h flag is already built in, and cannot be manually defined");
		OPTION_PARSER_ASSERT_RETURN(!(hkString::strNcasecmp(option->m_longFlag, "help", 4) == 0 && hkString::strLen(option->m_longFlag) == 4), "The -help flag is already built in, and cannot be manually defined");

		// Check if option flags are not already in use
		OPTION_PARSER_ASSERT_RETURN(!(option->m_shortFlag && (findOption(option->m_shortFlag, hkString::strLen(option->m_shortFlag), 0, options, count, HK_NULL, i) == HK_SUCCESS)), "Duplicate short flag definition\n");
		OPTION_PARSER_ASSERT_RETURN(!(option->m_longFlag && (findOption(option->m_longFlag, hkString::strLen(option->m_longFlag), FLAGINFO_LONG, options, count, HK_NULL, i) == HK_SUCCESS)), "Duplicate long flag definition\n");
	}

	m_options = options;
	m_optionCount = count;

	return true;
}

void hkOptionParser::setArguments(const char* name, const char* help, ArgumentsType argumentsType, const char** arguments, int count)
{
	// Change the empty string to HK_NULL
	nullIfEmpty(name);
	nullIfEmpty(help);

	OPTION_PARSER_ASSERT(name != HK_NULL, "Name of the arguments must be specified\n");

	// Store argument buffer
	m_arguments = arguments;
	m_argumentCount = count;

	// Store argument information
	m_argumentsName = name;
	m_argumentsHelp = help;
	m_argumentsType = argumentsType;

	// null arguments
	for (int i = 0; i < count; i ++)
	{
		arguments[i] = HK_NULL;
	}
}

hkOptionParser::ParseResult hkOptionParser::parse(int argc, const char** argv)
{
	bool doubleDash = false;
	bool counterDefaultOverride = false;
	int argumentIndex = 0;

	// Make sure options are at there default values
	defaultOptions(m_options, m_optionCount);

	// For each argument passed
	for (int i = 1; i < argc; i ++)
	{
		if (!doubleDash && hkString::strCmp(argv[i], "--") == 0)
		{
			// Stop processing if a '--' is found
			doubleDash = true;
			continue;
		}

		OptionFlagInfo flagInfo;
		Option option;

		// Check if option flag syntax is valid, and is in the list of accepted options
		if (!doubleDash && (parseFlag(argv[i], &flagInfo) == HK_SUCCESS))
		{
			if (findOption(flagInfo.m_name, flagInfo.m_nameLength, flagInfo.m_flags & FLAGINFO_LONG, m_options, m_optionCount, &option) == HK_SUCCESS)
			{
				const char* flagValue	= HK_NULL;
				int flagValueLength		= 0;

				// Get value
				if (flagInfo.m_flags & FLAGINFO_EQUALS)
				{
					if (flagInfo.m_valueLength > 0 || option.m_type == OPTION_STRING) // Can be the empty string
					{
						flagValue = flagInfo.m_value;
						flagValueLength = flagInfo.m_valueLength;
					}
					else if (option.m_type == OPTION_INT || option.m_type == OPTION_COUNTER || option.m_type == OPTION_BOOL)
					{
						outputErrorMessage(this, "Option '%s' requires a value", getFlagName(&flagInfo, &option));
						return PARSE_FAILURE;
					}
				}
				else if (flagInfo.m_flags & FLAGINFO_CONCAT)
				{
					if (option.m_type == OPTION_COUNTER)
					{
						flagValue = flagInfo.m_value;
						flagValueLength = flagInfo.m_valueLength;
					}
					else
					{
						const char* flagName = getFlagName(&flagInfo, &option);
						outputErrorMessage(this, "Option '%s' does not support %sVALUE syntax. Use %s=VALUE instead", flagName, flagName, flagName);
						return PARSE_FAILURE;
					}
				}
				else
				{
					// Use the next positional argument as the value for integers and strings 
					if (option.m_type == OPTION_INT || option.m_type == OPTION_STRING)
					{
						i++;

						if (i < argc)
						{
							flagValue = argv[i];
							flagValueLength = hkString::strLen(flagValue);
						}
						else
						{
							outputErrorMessage(this, "Option '%s' requires a value", getFlagName(&flagInfo, &option));
							return PARSE_FAILURE;
						}
					}
				}

				// Since counters increment a value and don't override it, we have to manually erase the default value if a counter is specified
				if (option.m_type == OPTION_COUNTER && (!counterDefaultOverride || flagInfo.m_flags & FLAGINFO_EQUALS))
				{
					option.m_value->u = 0u;
					counterDefaultOverride = true;
				}

				// Check value and parse
				if (setOptionValue(&option, flagValue, flagValueLength) != HK_SUCCESS)
				{
					outputErrorMessage(this, "Invalid value '%s' found for option '%s'", flagValue, getFlagName(&flagInfo, &option));
					return PARSE_FAILURE;
				}
			}
			else
			{
				// Check for the help flag
				if (hkString::strNcasecmp(flagInfo.m_name, "h", 1) == 0 && flagInfo.m_nameLength == 1 ||
					hkString::strNcasecmp(flagInfo.m_name, "help", 4) == 0 && flagInfo.m_nameLength == 4)
				{
					// Help was sought
					usage();
					return PARSE_HELP;
				}
				else
				{
					// Throw an error
					outputErrorMessage(this, "Cannot process '%s', unknown flag", flagInfo.m_name);
					return PARSE_FAILURE;
				}
			}
		}
		else
		{
			// Is a positional argument
			if (argumentIndex < m_argumentCount)
			{
				m_arguments[argumentIndex++] = argv[i];
			}
			else
			{
				outputErrorMessage(this, "The maximum number of positional arguments (%d) has been exceeded", m_argumentCount);
				return PARSE_FAILURE;
			}
		}
	}

	// Check number of arguments
	if (m_argumentsType == ARGUMENTS_ONE &&  argumentIndex != 1)
	{
		outputErrorMessage(this, "Incorrect number of positional arguments; Should be 1, found %d", argumentIndex);
		return PARSE_FAILURE;
	}
	else if (m_argumentsType == ARGUMENTS_ONE_OR_MORE &&  argumentIndex == 0)
	{
		outputErrorMessage(this, "Incorrect number of positional arguments; Should be 1 or more, found 0");
		return PARSE_FAILURE;
	}

	return PARSE_SUCCESS;
}

void hkOptionParser::usage(const char* extra)
{
	// Extra information
	if (extra)
	{
		printf("MESSAGE: ");
		printTextBlock(extra, 9);
		printf("\n");
	}
	
	// Program details
	printTextBlock(m_title);
	printf("Built with %s\n\n", HAVOK_SDK_VERSION_STRING);
	printTextBlock(m_desc);

	// Usage string
	int count = printf("\nUsage: ");

	// Usage options
	for (int i = 0; i < m_optionCount; i ++)
	{
		const char* typeString("");

		switch (m_options[i].m_type)
		{
		case OPTION_INT:
			typeString = "=INT";
			break;
		case OPTION_STRING:
			typeString = "=STR";
			break;
		}

		const char* optionStart("[--");
		const char* optionEnd("] ");

		int optionLength = hkString::strLen(optionStart) + hkString::strLen(m_options[i].m_longFlag) + hkString::strLen(typeString) + hkString::strLen(optionEnd);

		if (optionLength >= s_consoleWidth - count)
		{
			printf("\n%*s", 7, "");
			count = 7;
		}

		count += printf("%s%s%s%s", optionStart, m_options[i].m_longFlag, typeString, optionEnd);
	}

	// Usage positional argument
	if (m_arguments)
	{
		switch (m_argumentsType)
		{
		case ARGUMENTS_ONE:
			if (count + hkString::strLen(m_argumentsName) > s_consoleWidth)
			{
				printf("\n%*s", 7, "");
			}
			printf("%s", m_argumentsName);
			break;
		case ARGUMENTS_ONE_OR_MORE:
			if (count + 8 + hkString::strLen(m_argumentsName) + hkString::strLen(m_argumentsName) > s_consoleWidth)
			{
				printf("\n%*s", 7, "");
			}
			printf("%s [%s ...]", m_argumentsName, m_argumentsName);
			break;
		case ARGUMENTS_ZERO_OR_MORE:
			if (count + 6 + hkString::strLen(m_argumentsName) > s_consoleWidth)
			{
				printf("\n%*s", 7, "");
			}
			printf("[%s ...]", m_argumentsName);
			break;
		}
	}

	// Positional arguments
	printf("\n\nPositional arguments:\n\n");
	int posArgWidth = printf("    %s    ", m_argumentsName);
	printTextBlock(m_argumentsHelp, posArgWidth);

	// Optional arguments
	printf("\nOptional arguments:\n\n");

	// Get longest short and long flag
	int longestLongFlag = 0;

	for (int i = 0; i < m_optionCount; i ++)
	{
		int longFlagLength = hkString::strLen(m_options[i].m_longFlag);

		if (longestLongFlag < longFlagLength)
		{
			longestLongFlag = longFlagLength;
		}
	}

	// Print help option
	{
		int count = printf("    -h    --help");
		count += printf("%s%*s ", "       ", longestLongFlag - 4, "");
		printTextBlock("show this help message and exit", count);
		printf("\n");
	}

	// Print options
	for (int i = 0; i < m_optionCount; i ++)
	{
		int count = 0;

		if (m_options[i].m_shortFlag)
		{
			count += printf("    -%s    ", m_options[i].m_shortFlag);
		}
		else
		{
			count += printf("          ");
		}

		int num = 0;

		count += num = printf("--%s", m_options[i].m_longFlag);

		const char* typeString("");

		switch (m_options[i].m_type)
		{
		case OPTION_INT:
			typeString = "=INT   ";
			break;
		case OPTION_STRING:
			typeString = "=STR   ";
			break;
		case OPTION_BOOL:
			typeString = "[=BOOL]";
			break;
		case OPTION_COUNTER:
			typeString = "[NUM]  ";
			break;
		}

		count += printf("%s%*s ", typeString, longestLongFlag - num + 2, "");

		printTextBlock(m_options[i].m_help, count);

		{
			char defaultText[s_maxDefaultTextLength];

			switch (m_options[i].m_type)
			{

			case OPTION_INT:
				if (m_options[i].m_default.i != 0)
				{
					sprintf(defaultText, "(default: %d)", m_options[i].m_default.i);
					printTextBlock(defaultText, count, true);
				}
				break;
			case OPTION_STRING:
				if (m_options[i].m_default.s != HK_NULL)
				{
					sprintf(defaultText, "(default: %s)", m_options[i].m_default.s);
					printTextBlock(defaultText, count, true);
				}
				break;
			case OPTION_BOOL:
				if (m_options[i].m_default.b != false)
				{
					sprintf(defaultText, "(default: %s)", m_options[i].m_default.b ? "1" : "0");
					printTextBlock(defaultText, count, true);
				}
				break;
			case OPTION_COUNTER:
				if (m_options[i].m_default.u != 0)
				{
					sprintf(defaultText, "(default: %u)", m_options[i].m_default.u);
					printTextBlock(defaultText, count, true);
				}
				break;
			}
		}
		

		printf("\n");
	}

	// Print type information
	printf("\nType Information:\n");
	printf("\n    BOOL    ");
	printTextBlock("1 for True, 0 for False", 12);
	printf("\n    INT     ");
	printTextBlock("Signed integer value between -268435455 and 268435455. Also supports hexadecimal input (-0xFFFFFFF to 0xFFFFFFF)", 12);
	printf("\n    STR     ");
	printTextBlock("String of characters. Surround strings containing spaces with double quotes", 12);
	printf("\n    NUM     ");
	printTextBlock("Same as INT, but cannot be negative", 12);
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
