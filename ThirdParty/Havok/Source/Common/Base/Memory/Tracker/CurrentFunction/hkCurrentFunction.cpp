/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/CurrentFunction/hkCurrentFunction.h>
#include <Common/Base/Container/SubString/hkSubString.h>

static const char* findReverse(const char* start, const char* cur, char c)
{
	for (; cur >= start && *cur != c; cur--) ;
	return cur >= start ? cur : HK_NULL;
}

static int findIndex(const hkSubString* names, int numNames, const hkSubString& name)
{
	for (int i = 0; i < numNames; i++)
	{
		if (names[i] == name)
		{
			return i;
		}
	}

	return -1;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Gcc !!!!!!!!!!!!!!!!!!!!!!!

static int extractGccTemplateParams(const hkSubString& text, hkSubString* names, hkSubString* values)
{
	const char* cur = text.m_start;
	const char* end = text.m_end;

	int numParams = 0;
	while (cur < end)
	{
		// get rid of the white space
		while (*cur == ' ' && cur < end) cur++;

		// Get the parameter name
		const char* start = cur;

		while (*cur != ' ' && *cur != '=' && cur < end) cur++;

		// Set the param name
		names[numParams].set(start, cur);

		// Skip white space
		while (*cur == ' ' && cur < end) cur++;
		// Skip =
		if (*cur == '=') cur++;
		// Skip any white space
		while (*cur == ' ' && cur < end) cur++;

		start = cur;
		int depth = 0;
		
		while (cur < end)
		{
			const char c = *cur;
			if (c == '<')
			{
				depth++;
			} else if (c == '>')
			{
				depth--;
			} else if ( c == ',' && depth == 0)
			{
				break;
			}
			cur++;
		}

		// Remove white space at end
		while (cur > start && cur[-1] == ' ')
		{
			cur--;
		}

		// Set the value
		values[numParams].set(start, cur);
		// We added a param
		numParams++;

		// Skip white space
		while (*cur == ' ' && cur < end) cur++;
		// At the end
		if (cur >= end) break;

		// Must be a comma
		HK_ASSERT(0x2345242, *cur == ',');
		if (*cur == ',') cur++;
	}
	
	return numParams;
}

static void hkGetGccClassName(const char* type, char* typeOut)
{
	// A typical string would look like "A<T>::getTypeIdentifier() [with T = A<float>]"
	// Most of the complexity here is substituting the template parameters

	// Find the params
	const char* openBracket = hkString::strChr(type, '(');
	HK_ASSERT(0x32324324, openBracket);
	// Get the colon
	const char* lastColon = findReverse(type, openBracket, ':');
	HK_ASSERT(0x3242a423, lastColon && lastColon > type && lastColon[-1] == ':');
	const char* typeEnd = lastColon - 1;

	// Work backwards trying to determine the class type.
	const char* typeStart;
	{
		int depth = 0;
		const char* cur = typeEnd - 1;

		while (cur >= type)
		{
			const char c = *cur;
			if (c == '>')
			{
				depth++;
			} else if (c == '<')
			{
				depth--;
			} else if ((c == ' ' || c == ',' || c == '*') && depth == 0)
			{
				break;
			}
			cur--;
		}

		HK_ASSERT(0x3242a432, cur >= type);
		typeStart = cur + 1;
	}

	// Write out the type name
	const char* cur = typeStart;
	while (cur < typeEnd && *cur != '<')
	{
		*typeOut++ = *cur;
		cur++;
	}
	
	if (*cur == '<')
	{
		// Its a templated type
		const int maxParamNameLength = 64;
		const int maxParams = 16;

		// Copy to the output
		*typeOut++ = *cur++;

		// Find the template values

		hkSubString paramNames[maxParams];
		hkSubString values[maxParams];
		int numParams;

		{
			const char* templateValues = hkString::strStr(typeEnd, "[with ");
			HK_ASSERT(0x2432432a, templateValues);
			templateValues += 6;

			const char* templateValuesEnd = findReverse(templateValues, templateValues + hkString::strLen(templateValues) - 1, ']');
			numParams = extractGccTemplateParams(hkSubString(templateValues, templateValuesEnd), paramNames, values);
		}

	
		char paramName[maxParamNameLength];
		while (*cur != '>')
		{
			// Look for where the expansion is

			// Skip white space
			while (*cur == ' ') cur++;

			// Its templated... so insert the template substitutions
			{
				char* out = paramName;
				while (*cur != ' ' && *cur != ',' && *cur != '>') 
				{
					*out++ = *cur++;
				}
				
				int paramIndex = findIndex(paramNames, numParams, hkSubString(paramName, out));				

				HK_ASSERT(0x2432a423, paramIndex >= 0);

				const hkSubString& value = values[paramIndex];
				const int len = value.length();

				hkString::strNcpy(typeOut, value.m_start, len);
				typeOut += len;
			}

			// Skip any white space 
			while (*cur == ' ') cur++;

			// Add the comma if there is one
			if (*cur == ',') 
			{
				*typeOut++ = ',';
				cur++;
			}
		}

		// Mark end of template
		*typeOut++ = '>';
	}

	*typeOut = 0;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Microsoft Visual Studio !!!!!!!!!!!!!!!!!!!!!!!

static void hkGetMsvcClassName(const char* type, char* typeOut)
{
	// Work from the back - looking for first ::
	const char* cur = type + hkString::strLen(type) - 1;

	while (cur >= type && *cur != ':')
	{
		cur--;
	}
	HK_ASSERT(0x32442a32, cur > type && cur[-1] == ':');
	cur--;

	hkString::strNcpy(typeOut, type, int(cur - type));
	typeOut[cur - type] = 0;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Clang PS4 !!!!!!!!!!!!!!!!!!!!!!!

static void hkGetClangPs4ClassName(const char* type, char* typeOut)
{
	// Clang includes the return type
	// "static const char *A<float>::getTypeIdentifier()"

	const char retType[] = "static const char *";
	if( hkString::strNcmp(type, retType, sizeof(retType)-1) == 0 )
	{
		type += sizeof(retType)-1;
	}
	const char* cur = hkString::strStr(type, "::getTypeIdentifier");
	if( cur == HK_NULL ) // Failed? Work from the back - looking for first ::
	{
		// Work from the back - looking for first ::
	    cur = type + hkString::strLen(type) - 1;
    
	    while (cur >= type && *cur != ':')
	    {
		    cur--;
	    }
	    HK_ASSERT(0x32442a32, cur > type && cur[-1] == ':');
	    cur--;
	}
	hkString::strNcpy(typeOut, type, int(cur - type));
	typeOut[cur - type] = 0;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Microsoft Visual Studio !!!!!!!!!!!!!!!!!!!!!!!

/* static */void HK_CALL hkCurrentFunctionUtil::getClassName(const char* methodText, char* className)
{
	#if defined(_MSC_VER)
		hkGetMsvcClassName(methodText, className);
	#elif defined(HK_PLATFORM_PS4) // clang:ps4 includes the return type. strip it
		hkGetClangPs4ClassName(methodText, className);
	#else // other clangs seem to follow gcc conventions
		hkGetGccClassName(methodText, className);
	#endif
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
