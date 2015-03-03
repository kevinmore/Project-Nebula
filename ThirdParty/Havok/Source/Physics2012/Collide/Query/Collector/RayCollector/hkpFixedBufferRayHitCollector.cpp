/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Query/Collector/RayCollector/hkpFixedBufferRayHitCollector.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastOutput.h>

// global callback function for hkpFixedBufferRayHitCollector::addCdPoint()
hkFixedBufferRayHitCollectorAddRayHitCallbackFunc g_FixedBufferRayHitCollectorAddRayHitCallbackFunc = HK_NULL;


void hkpFixedBufferRayHitCollector::registerDefaultAddRayHitFunction()
{
	registerFixedBufferRayHitCollectorAddRayHitCallbackFunction(addRayHitImplementation);
}


void hkpFixedBufferRayHitCollector::addRayHitImplementation(const hkpCdBody& cdBody, const hkpShapeRayCastCollectorOutput& hitInfo, hkpFixedBufferRayHitCollector* collector) 
{
	hkpWorldRayCastOutput_Storage* insertAt;

	//
	// if there's still room for a new closest point, simply append it to the list
	// else replace the list's furthest point with the new one
	//
	if ( collector->m_numOutputs < collector->m_capacity )
	{
		insertAt					= collector->m_nextFreeOutput;
		collector->m_numOutputs		= collector->m_numOutputs + 1;
		collector->m_nextFreeOutput	= hkAddByteOffset(collector->m_nextFreeOutput.val(), sizeof(hkpWorldRayCastOutput_Storage));
	}
	else
	{
		HK_WARN_ONCE(0xaf531e14, "Collector buffer full. Replacing raycast outputs from now on.");

		hkpWorldRayCastOutput_Storage* currentOutputInArray		= hkAddByteOffset(collector->m_rayCastOutputBase.val(), sizeof(hkpWorldRayCastOutput_Storage));
		hkpWorldRayCastOutput_Storage* furthestOutput			= collector->m_rayCastOutputBase;

		//
		// search for the furthest point
		//
		
		for (int i=1; i<collector->m_numOutputs; i++)
		{
			if ( currentOutputInArray->m_hitFraction > furthestOutput->m_hitFraction )
			{
				furthestOutput = currentOutputInArray;
			}
			currentOutputInArray = hkAddByteOffset(currentOutputInArray, sizeof(hkpWorldRayCastOutput_Storage));
		}

		if ( hitInfo.m_hitFraction < furthestOutput->m_hitFraction )
		{
			insertAt = furthestOutput;
		}
		else
		{
			return;
		}
	}

	//
	// write data to array
	//
		
	insertAt->m_hitFraction = hitInfo.m_hitFraction ;
	insertAt->m_normal = hitInfo.m_normal; // it's rotated already for WithCollector version of rayCast
	insertAt->m_extraInfo = hitInfo.m_extraInfo;

	if (collector->m_collidableOnPpu)
	{
		insertAt->m_rootCollidable = collector->m_collidableOnPpu;
	}
	else
	{
		insertAt->m_rootCollidable = cdBody.getRootCollidable();
	}
	int cdBodyHierarchyDepth = 0;
	const hkpCdBody* currBody = &cdBody;
	while(currBody->getParent())
	{
		cdBodyHierarchyDepth++;
		currBody = currBody->getParent();
	}

	currBody = &cdBody;
	insertAt->m_shapeKeys[cdBodyHierarchyDepth] = HK_INVALID_SHAPE_KEY;
	for (int i = cdBodyHierarchyDepth-1; i >= 0; i--)
	{
		insertAt->m_shapeKeys[i] = currBody->m_shapeKey;
		currBody = currBody->getParent();
	}
	
	// To collect all points we use no early outs
	//if ( hitInfo.m_hitFraction < collector->m_earlyOutHitFraction )
	//{
	//	collector->m_earlyOutHitFraction = hitInfo.m_hitFraction;
	//}
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
