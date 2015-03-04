/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Skin/hkxSkinUtils.h>


void hkxSkinUtils::quantizeWeights( const hkReal inputWeights[4], hkUint8 quantizedWeights[4] )
{
	hkArray<hkReal> tempWeights; tempWeights.setSize(4,0);
	hkArray<hkUint8> tempQWeights;

	for (int i=0; i<4; i++)
	{
		tempWeights[i] = inputWeights[i];
	}

	quantizeWeights(tempWeights,tempQWeights);

	for (int i=0; i<4; i++)
	{
		quantizedWeights[i] = tempQWeights[i];
	}
}

void hkxSkinUtils::quantizeWeights( const hkArray<hkReal>& weights, hkArray<hkUint8>& quantizedWeights )
{
	if (weights.getSize()==0) return;

	const int numWeights = weights.getSize();
	quantizedWeights.clear();
	quantizedWeights.setSize(numWeights);

	int totalQuantizedWeight = 0;
	for(int i=0; i < numWeights; i++)
	{
		quantizedWeights[i] = hkUint8(weights[i] * hkReal(255));
		totalQuantizedWeight += quantizedWeights[i];
	}

	const int quantizedWeightError = 255 - totalQuantizedWeight;
	if(quantizedWeightError > 0)
	{
		// spread the difference using a greedy approach to minimize the relative weight error
		{
			const hkReal oneOver255 = hkReal(1.0f/255.0f);
			for( int influenceIdxToChange = 0; influenceIdxToChange < quantizedWeightError; influenceIdxToChange++)
			{
				// modify weights by adding 1 and find weight with minimum relative error
				int minErrorIndex = 0;
				hkReal minError = (hkReal(quantizedWeights[minErrorIndex]+1)*oneOver255 - weights[minErrorIndex]) / weights[minErrorIndex];
				for( int i = 1; i < numWeights; i++)
				{
					const hkReal relError = (hkReal(quantizedWeights[i]+1)*oneOver255 - weights[i]) / weights[i];
					if (minError > relError)
					{
						minError = relError;
						minErrorIndex = i;
					}
				}

				quantizedWeights[minErrorIndex]++;
			}
		}
	}
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
