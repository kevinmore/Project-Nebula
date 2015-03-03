/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/String/hkStringBuf.h>

#if 0

//
//	Encodes a float into a custom 5:11 bits format to be stored in the table

static hkUint16 HK_CALL hkUFloat8_encodeFloat(const hkFloat32 f)
{
	const int intVal		= *((int*)&f);
	const int exponent		= (intVal >> hkUFloat8::FLOAT_MANTISSA_BITS) & 0xFF;
	const int mantissa		= intVal & hkUFloat8::FLOAT_MANTISSA_MASK;

	// Encode exponent on 5 bits
	const int encExpo		= (exponent - hkUFloat8::ENCODED_EXPONENT_BIAS) & hkUFloat8::ENCODED_EXPONENT_MASK;

	// Encode mantissa on 11 bits
	const int encMantissa	= (mantissa >> hkUFloat8::ENCODE_MANTISSA_SHIFT) & hkUFloat8::ENCODED_MANTISSA_MASK;

	return intVal ? (hkUint16)(encMantissa | (encExpo << hkUFloat8::ENCODED_MANTISSA_BITS)) : 0;
}

//
//	Builds the compressed hkUFloat8_shortToReal table

void hkUFloat8_buildCompressedTable()
{
	// the first 8 steps are used to evenly increase the value by eps
	int i;
	float lastValue = 0.0f;
	for (i =0; i < 8; i++)
	{
		lastValue = i * hkUFloat8_eps;
		hkUFloat8_shortToReal[i] = hkUFloat8_encodeFloat(lastValue);
	}

	// now we have to distribute the values evenly from i to max
	float numSteps = float( hkUFloat8::MAX_VALUE - i );
	float range = hkUFloat8_maxValue/lastValue;
	float factor = pow(range, 1.0f/numSteps);

	for ( ; i < hkUFloat8::MAX_VALUE; i++)
	{
		lastValue *= factor;
		hkUFloat8_shortToReal[i] = hkUFloat8_encodeFloat(lastValue);
	}

	{
		hkStringBuf strb;	strb.clear();
		for (int f= 0;f < hkUFloat8::MAX_VALUE; f++)
		{
			hkUint32 v = hkUFloat8_shortToReal[f];
			strb.appendPrintf("0x%08X, ", v);
			if ( (f & 15) == 15)
			{
				HK_REPORT(strb);
				strb.clear();
			}
		}
	}
}

#endif

hkUFloat8& hkUFloat8::operator=(const double& fv)
{
	return *this = float(fv);
}

hkUFloat8& hkUFloat8::operator=(const float& fv)
{
	const float minVal = decodeFloat(getEncodedFloat(1));
	if ( fv < minVal )
	{
		m_value = 0;
		return *this;
	}

	int minf = 0;
	int maxf = hkUFloat8::MAX_VALUE;
	int midf = hkUFloat8::MAX_VALUE >> 1;

	for (int i = 6; i >= 0; i--)
	{
		if (fv > decodeFloat(getEncodedFloat(hkUint8(midf))))	{	minf = midf;	}
		else							{	maxf = midf;	}
		midf = (minf + maxf) >> 1;
	}

	if ( fv > decodeFloat(getEncodedFloat(hkUint8(midf))) )
	{
		if ( midf < hkUFloat8::MAX_VALUE-1 )
		{
			midf++;
		}
	}

	m_value = hkUint8(midf);
	return *this;
}

const hkFloat32 hkUInt8ToFloat32[256] =
{
	0,		1,		2,		3,		4,		5,		6,		7,		8,		9,		10,		11,		12,		13,		14,		15,
	16,		17,		18,		19,		20,		21,		22,		23,		24,		25,		26,		27,		28,		29,		30,		31,
	32,		33,		34,		35,		36,		37,		38,		39,		40,		41,		42,		43,		44,		45,		46,		47,
	48,		49,		50,		51,		52,		53,		54,		55,		56,		57,		58,		59,		60,		61,		62,		63,
	64,		65,		66,		67,		68,		69,		70,		71,		72,		73,		74,		75,		76,		77,		78,		79,
	80,		81,		82,		83,		84,		85,		86,		87,		88,		89,		90,		91,		92,		93,		94,		95,
	96,		97,		98,		99,		100,	101,	102,	103,	104,	105,	106,	107,	108,	109,	110,	111,
	112,	113,	114,	115,	116,	117,	118,	119,	120,	121,	122,	123,	124,	125,	126,	127,
	128,	129,	130,	131,	132,	133,	134,	135,	136,	137,	138,	139,	140,	141,	142,	143,
	144,	145,	146,	147,	148,	149,	150,	151,	152,	153,	154,	155,	156,	157,	158,	159,
	160,	161,	162,	163,	164,	165,	166,	167,	168,	169,	170,	171,	172,	173,	174,	175,
	176,	177,	178,	179,	180,	181,	182,	183,	184,	185,	186,	187,	188,	189,	190,	191,
	192,	193,	194,	195,	196,	197,	198,	199,	200,	201,	202,	203,	204,	205,	206,	207,
	208,	209,	210,	211,	212,	213,	214,	215,	216,	217,	218,	219,	220,	221,	222,	223,
	224,	225,	226,	227,	228,	229,	230,	231,	232,	233,	234,	235,	236,	237,	238,	239,
	240,	241,	242,	243,	244,	245,	246,	247,	248,	249,	250,	251,	252,	253,	254,	255
};

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
