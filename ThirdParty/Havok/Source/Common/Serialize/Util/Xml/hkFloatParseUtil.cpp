/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/Xml/hkFloatParseUtil.h>
#include <Common/Base/Container/String/hkStringBuf.h>

static const char*const s_textLut[] = 
{
	"-1.#INF00",		// TYPE_NEG_INF
	"1.#INF00",			// TYPE_INF
	"-1.#QNAN0",		// TYPE_NEG_NAN
	"1.#QNAN0",			// TYPE_NAN
	"-1.#IND00",		// TYPE_IND
	HK_NULL,			// TYPE_FINITE
};

HK_COMPILE_TIME_ASSERT(HK_COUNT_OF(s_textLut) == hkFloatParseUtil::TYPE_COUNT_OF);

#if defined(HK_REAL_IS_DOUBLE)
/* static */const hkUint64 hkFloatParseUtil::s_valuesLut[] = 
{
	0xfff0000000000000ull,			// TYPE_NEG_INF
	0x7ff0000000000000ull,			// TYPE_INF
	0xfff0000000000001ull,			// TYPE_NEG_NAN
	0x7ff0000000000001ull,			// TYPE_NAN
	0xfff8000000000000ull,			// TYPE_IND
	0x0000000000000000ull,			// TYPE_FINITE
};
#else
/* static */const hkUint32 hkFloatParseUtil::s_valuesLut[] = 
{
	0xff800000,			// TYPE_NEG_INF
	0x7f800000,			// TYPE_INF
	0xff800001,			// TYPE_NEG_NAN
	0x7f800001,			// TYPE_NAN
	0xffc00000,			// TYPE_IND
	0x00000000,			// TYPE_FINITE
};
#endif

/* static */hkFloatParseUtil::Type HK_CALL hkFloatParseUtil::getFloatType(hkFloat32 f)
{
	hkUint32 v = *(hkUint32*)&f;

	if ((v & 0x7f800000) != 0x7f800000)
	{
		return TYPE_FINITE;
	}

	if (v == 0x7f800000)
	{
		return TYPE_INF;
	}
	if (v == 0xff800000)
	{
		return TYPE_NEG_INF;
	}
	if (v == 0xffc00000)
	{
		return TYPE_IND;
	}

	const hkUint32 signMask = 0x80000000;
	return (v & signMask) ? TYPE_NEG_NAN : TYPE_NAN;
}

/* static */hkFloatParseUtil::Type HK_CALL hkFloatParseUtil::getFloatType(hkDouble64 f)
{
	hkUint64 v = *(hkUint64*)&f;

	if ((v & 0x7ff0000000000000ull) != 0x7ff0000000000000ull)
	{
		return TYPE_FINITE;
	}

	if (v == 0x7ff0000000000000ull)
	{
		return TYPE_INF;
	}
	if (v == 0xfff0000000000000ull)
	{
		return TYPE_NEG_INF;
	}
	if (v == 0xfff8000000000000ull)
	{
		return TYPE_IND;
	}

	const hkUint64 signMask = 0x8000000000000000ull;
	return (v & signMask) ? TYPE_NEG_NAN : TYPE_NAN;
}

/* static */bool HK_CALL hkFloatParseUtil::equals(hkFloat32 a, hkFloat32 b)
{
	Type typeA = getFloatType(a);
	Type typeB = getFloatType(b);

	if (typeA != typeB)
	{
		return false;
	}

	// If not finite then the type being equal is enough
	// else they must == too
	return (typeA != TYPE_FINITE) || a == b;
}

/* static */bool HK_CALL hkFloatParseUtil::equals(hkDouble64 a, hkDouble64 b)
{
	Type typeA = getFloatType(a);
	Type typeB = getFloatType(b);

	if (typeA != typeB)
	{
		return false;
	}

	// If not finite then the type being equal is enough
	// else they must == too
	return (typeA != TYPE_FINITE) || a == b;
}

/* static */void HK_CALL hkFloatParseUtil::calcFloatText(hkFloat32 a, hkStringBuf& buf)
{
	Type type = getFloatType(a);

	if (type == TYPE_FINITE)
	{
		buf.printf("%g", a);
		return;
	}
	
	buf = s_textLut[type];
}

/* static */void HK_CALL hkFloatParseUtil::calcFloatText(hkDouble64 a, hkStringBuf& buf)
{
	Type type = getFloatType(a);

	if (type == TYPE_FINITE)
	{
		buf.printf("%g", a);
		return;
	}

	buf = s_textLut[type];
}

/* static */void HK_CALL hkFloatParseUtil::calcFloatTextWithPoint(hkFloat32 a, hkStringBuf& buf)
{
	Type type = getFloatType(a);
	if (type == TYPE_FINITE)
	{
		buf.printf("%g", a);
		if( buf.indexOf('.') == -1 ) // make sure we have a decimal point
		{
			buf += ".0";
		}
		return;
	}
	buf = s_textLut[type];
}

/* static */void HK_CALL hkFloatParseUtil::calcFloatTextWithPoint(hkDouble64 a, hkStringBuf& buf)
{
	Type type = getFloatType(a);
	if (type == TYPE_FINITE)
	{
		buf.printf("%g", a);
		if( buf.indexOf('.') == -1 ) // make sure we have a decimal point
		{
			buf += ".0";
		}
		return;
	}
	buf = s_textLut[type];
}

/* static */hkResult hkFloatParseUtil::parseFloat(const hkSubString& str, hkReal& valueOut)
{
	HK_ASSERT(0x24243aaa, str.length() > 0);

	const int maxLen = 64;
	// Needs to be enough space to process
	if (str.length() > maxLen)
	{
		return HK_FAILURE;
	}

	const char* start = str.m_start;

	const char* cur = start;
	const char* end = str.m_end;

	for (; cur < end; cur++)
	{
		// Look for a # - means its a special number
		if (*cur == '#')
		{
			// Okay its not a regular number.. go look it up in the list
			for (int i = 0; i < TYPE_COUNT_OF; i++)
			{
				const char* text = s_textLut[i];

				if (str == text)
				{
					// save the value
					valueOut = getFloatFromType(Type(i));
					return HK_SUCCESS;
				}
			}

			return HK_FAILURE;
		}
	}
	
	// Looks good. Do the conversion
	char buffer[maxLen + 1];
	hkString::strNcpy(buffer, start, int(end - start));
	// Null terminate
	buffer[end - start] = 0;
	
	valueOut = hkString::atof(buffer); // note this actually returns a hkReal

	return HK_SUCCESS;
}

#if 0

// This code was used to confirm the ranges and values of non finite floats

/* static */ void hkFloatParseUtil::findTypeRanges()
{

	struct Value
	{
		hkUint32 m_min;
		hkUint32 m_max;
		const char* m_text;
	};

	// Just check 
	{
		for (int i = 0; i < TYPE_COUNT_OF; i++)
		{
			hkFloat32 f = getFloat32(Type(i));
			Type type = getFloat32Type(f);
			HK_ASSERT(0x24234, type == Type(i));
		}
	}

	hkStorageStringMap<int> map;
	hkArray<Value> values;

	//for (hkUint64 i = 0; i  < hkUint64(0x10000000); i++)
	for (int i = 0; i  < int(0x1000000); i++)
	{
		// Produces a non finate no
		hkUint32 ii = ((i & 0x00800000) << 8) | (i & 0x007fffff) | 0x7f800000;

		hkReal f = *(hkReal*)&ii;
		HK_ASSERT(0x324243, !hkMath::isFinite(f));

		char buf[80];
		hkString::sprintf(buf, "%f", f);

		int index = map.getWithDefault(buf, -1);
		if (!map.hasKey(buf))
		{	
			int index = values.getSize(); 
			const char* text = map.insert(buf, index);

			Value& value = values.expandOne();
			value.m_min = ii;
			value.m_max = ii;
			value.m_text = text;
		}
		else
		{
			const char* text = values[index].m_text;
			hkBool found = false;
			for (int j = index; j < values.getSize(); j++)
			{
				if (values[j].m_text == text && values[j].m_max + 1 == ii)
				{
					values[j].m_max = ii;
					found = true;
				}
			}
			// Wasn't found so add a new run
			if (!found)
			{
				Value& value = values.expandOne();
				value.m_text = text;
				value.m_min = ii;
				value.m_max = ii;
			}
		}
	}
}

#endif

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
