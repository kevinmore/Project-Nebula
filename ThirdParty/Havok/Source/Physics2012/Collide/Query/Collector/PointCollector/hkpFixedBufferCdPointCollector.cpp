/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpFixedBufferCdPointCollector.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Algorithm/FindIndex/hkFindIndex.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>


// global callback function for hkpFixedBufferCdPointCollector::addCdPoint()
hkFixedBufferCdPointCollectorAddCdPointCallbackFunc g_FixedBufferCdPointCollectorAddCdPointCallbackFunc = HK_NULL;


#if defined(HK_PLATFORM_SPU)
	// on spu these global variables are used to pass the ppu collidable ptr from the job to the collector's addCdPoint() function
	// (as we are going through the standard hkpCdPointCollector interface where there is no way to pass additional parameters through)
	const hkpCollidable* g_spuFixedBufferCdPointCollectorCollidableAOnPpu;
	const hkpCollidable* g_spuFixedBufferCdPointCollectorCollidableBOnPpu;
#endif

void hkpFixedBufferCdPointCollector::sortHits( ) 
{	
	hkSort( m_pointsArrayBase.val(), m_numPoints );
}


void hkpFixedBufferCdPointCollector::registerDefaultAddCdPointFunction()
{
	registerFixedBufferCdPointCollectorAddCdPointCallbackFunction(addCdPointImplementation);
}

void hkpFixedBufferCdPointCollector::registerCustomAddCdPointFunction(hkFixedBufferCdPointCollectorAddCdPointCallbackFunc func)
{
	registerFixedBufferCdPointCollectorAddCdPointCallbackFunction(func);
}

static void HK_CALL _setHit(hkpRootCdPoint* insertAt, const hkpCdPoint& event)
{
	insertAt->m_contact			= event.getContact();
#if defined(HK_PLATFORM_SPU)
	insertAt->m_rootCollidableA	= g_spuFixedBufferCdPointCollectorCollidableAOnPpu;
	insertAt->m_rootCollidableB	= g_spuFixedBufferCdPointCollectorCollidableBOnPpu;
#else
	insertAt->m_rootCollidableA	= event.m_cdBodyA.getRootCollidable();
	insertAt->m_rootCollidableB	= event.m_cdBodyB.getRootCollidable();
#endif
	insertAt->m_shapeKeyA		= event.m_cdBodyA.getShapeKey();
	insertAt->m_shapeKeyB		= event.m_cdBodyB.getShapeKey();
}

void hkpFixedBufferCdPointCollector::addCdPointImplementation(const hkpCdPoint& event, hkpFixedBufferCdPointCollector* collector) 
{
	if (collector->m_capacity == 0)
	{
		return;
	}

	//
	// if there's still room for a new closest point, simply append it to the list
	// else replace the list's furthest point with the new one
	//
	if ( collector->m_numPoints < collector->m_capacity )
	{
		_setHit(collector->m_pointsArrayBase + collector->m_numPoints, event);		
		collector->m_numPoints = collector->m_numPoints + 1;
	}
	else 
	{
		HK_WARN_ONCE(0xaf531e14, "Collector buffer full. Replacing points from now on.");

		hkpRootCdPoint* currentPointInArray  = collector->m_pointsArrayBase;
		hkpRootCdPoint* endOfArray = currentPointInArray + collector->m_numPoints;

		// The furthest point of those in the array.
		hkpRootCdPoint* furthestPoint = currentPointInArray;

		// Traverse the array, updating the furthest point
		for ( ++currentPointInArray; currentPointInArray < endOfArray; ++currentPointInArray )
		{
			if ( currentPointInArray->m_contact.getDistanceSimdReal() > furthestPoint->m_contact.getDistanceSimdReal() )
			{
				furthestPoint = currentPointInArray;
			}
		}

		//
		// If the new point is not the furthest point, we insert the new point and find the new furthest point.
		//
		if ( event.getContact().getDistanceSimdReal() < furthestPoint->m_contact.getDistanceSimdReal() )
		{
			//
			// overwrite furthest point data with the new hit data
			//
			_setHit(furthestPoint, event);

			//
			// Now find the latest furthest point.
			//
			furthestPoint = currentPointInArray = collector->m_pointsArrayBase;
			for ( ++currentPointInArray; currentPointInArray < endOfArray; ++currentPointInArray )
			{
				if ( currentPointInArray->m_contact.getDistanceSimdReal() > furthestPoint->m_contact.getDistanceSimdReal() )
				{
					furthestPoint = currentPointInArray;
				}
			}
		}

		//
		//	Set the early-out distance to the furthest point distance
		//
		furthestPoint->m_contact.getDistanceSimdReal().store<1>(&collector->m_earlyOutDistance);
	}
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
