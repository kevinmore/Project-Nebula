/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/SceneData/hkSceneData.h>
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/SceneData/Skin/hkxSkinUtils.h>


int calcSum(const hkArray<hkUint8>& x)
{
	int sum = 0;
	for( int i=0; i<x.getSize(); i++)
	{
		sum += x[i];
	}

	return sum;
}

bool checkNormalizedValues(const hkArray<hkUint8>& x)
{
	for (int i=0; i<x.getSize(); i++)
	{
		const hkReal w = hkReal(x[i]) / 255.0f;

		if ( hkUint8(w*255.0f) != x[i] )
		{
			return false;
		}
	}

	return true;
}

#define SIZE 4

int calcSum(const hkUint8 x[SIZE])
{
	int sum = 0;
	for (int i=0; i<SIZE; i++)
	{
		sum += x[i];
	}

	return sum;
}

bool checkNormalizedValues(const hkUint8 x[SIZE])
{
	for (int i=0; i<SIZE; i++)
	{
		const hkReal w = hkReal(x[i]) / 255.0f;

		if ( hkUint8(w*255.0f) != x[i] )
		{
			return false;
		}
	}

	return true;
}

void test_normalization()
{
	// 2 weights summing up to 255
	{
		hkArray<hkReal> weights;
		weights.pushBack(0.2f);
		weights.pushBack(0.8f);
		hkArray<hkUint8> qWeights;
		
		hkxSkinUtils::quantizeWeights(weights, qWeights);

		HK_TEST( calcSum(qWeights) == 255 );
		HK_TEST( checkNormalizedValues(qWeights) );
	}	

	// 3 weights summing up to 254
	{
		hkArray<hkReal> weights;
		weights.pushBack(0.39f);
		weights.pushBack(0.39f);
		weights.pushBack(0.22f);
		hkArray<hkUint8> qWeights;

		hkxSkinUtils::quantizeWeights(weights, qWeights);

		HK_TEST( calcSum(qWeights) == 255 );
		HK_TEST( checkNormalizedValues(qWeights) );
	}

	// 4 weights summing up to 252
	{
		// use C arrays to also test the respective util function
		hkReal weights[4];
		weights[0] = 0.24999f;
		weights[1] = 0.24999f;
		weights[2] = 0.24999f;
		weights[3] = 0.25003f;
		hkUint8 qWeights[4];

		hkxSkinUtils::quantizeWeights(weights, qWeights);

		HK_TEST( calcSum(qWeights) == 255 );
		HK_TEST( checkNormalizedValues(qWeights) );
	}
}


int skin_main()
{
	test_normalization();

	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(skin_main, "Fast", "Common/Test/UnitTest/SceneData/", __FILE__     );

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
