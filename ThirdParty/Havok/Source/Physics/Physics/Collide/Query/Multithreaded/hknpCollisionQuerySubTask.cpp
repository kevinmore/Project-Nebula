/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Query/Multithreaded/hknpCollisionQuerySubTask.h>
#include <Physics/Physics/Collide/Query/Multithreaded/hknpCollisionQueryTask.h>
#include <Physics/Physics/Collide/Query/Collector/hknpAllHitsCollector.h>
#include <Physics/Physics/Collide/Query/Collector/hknpClosestHitCollector.h>
#include <Physics/Physics/Collide/Shape/hknpShapeQueryInterface.h>
#include <Physics/Physics/Collide/Filter/Group/hknpGroupCollisionFilter.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>


#if !defined(HK_PLATFORM_SPU)

hknpPairGetClosestPointsSubTask::hknpPairGetClosestPointsSubTask() :
m_queryShape(HK_NULL),
m_targetShape(HK_NULL),
m_resultArray(HK_NULL),
m_resultArraySize(0),
m_numHits(0)
{

}

void hknpPairGetClosestPointsSubTask::initialize(const hknpBody& queryBody, const hknpBody& targetBody)
{
	m_queryShape = queryBody.m_shape;
	m_queryShapeSize = queryBody.m_shape->calcSize();
	m_queryShapeToWorld = queryBody.getTransform();

	m_targetShape = targetBody.m_shape;
	m_targetShapeSize = targetBody.m_shape->calcSize();
	m_targetShapeToWorld = targetBody.getTransform();
}

void hknpPairGetClosestPointsSubTask::process(hknpCollisionQueryTask* task)
{
	hknpAllHitsCollector collector;
	{
		hknpCollisionQueryContext queryContext(HK_NULL, HK_NULL);
		hknpClosestPointsQuery query;

		queryContext.m_dispatcher = task->m_querySharedData.m_pCollisionQueryDispatcher;
		query.m_shape = m_queryShape;

		hknpShapeQueryInterface::getClosestPoints( &queryContext, query, m_queryShapeToWorld, *m_targetShape, m_targetShapeToWorld, &collector );
	}

	m_numHits = 0;

	for(int hitIndex=0; hitIndex<collector.getNumHits(); hitIndex++)
	{
		if(hitIndex >= m_resultArraySize)
		{
			break;
		}

		m_resultArray[hitIndex] = collector.getHits()[hitIndex];
		m_numHits++;
	}
}

#else

void hknpPairGetClosestPointsSubTask::process(hknpCollisionQueryTask* task, const hknpPairGetClosestPointsSubTask* jobPpu, void* preloadBuffer, const hknpCollisionQueryDispatcherBase* pQueryDispatcher)
{
	// patch the pointers
	hknpShape* pQueryShape;
	hknpShape* pTargetShape;
	{
		hkUint8* pBuffer = (hkUint8*)preloadBuffer;

		hknpShapeVirtualTableUtil::patchVirtualTable((hknpShape*)pBuffer);
		pQueryShape = (hknpShape*)pBuffer;
		pBuffer += m_queryShapeSize;

		hknpShapeVirtualTableUtil::patchVirtualTable((hknpShape*)pBuffer);
		pTargetShape = (hknpShape*)pBuffer;
		pBuffer += m_targetShapeSize;
	}


	hknpClosestHitCollector collector;
	{
		hknpCollisionQueryContext queryContext(HK_NULL, HK_NULL);
		hknpClosestPointsQuery query;

		queryContext.m_dispatcher = pQueryDispatcher;
		query.m_shape = pQueryShape;

		hknpShapeQueryInterface::getClosestPoints( &queryContext, query, m_queryShapeToWorld, *pTargetShape, m_targetShapeToWorld, &collector );
	}

	hkPadSpu<int> numHits = collector.getNumHits();
	if(numHits > m_resultArraySize)
	{
		numHits = m_resultArraySize;
	}
	hkSpuDmaManager::putToMainMemorySmallAndWaitForCompletion(&jobPpu->m_numHits.ref(), &numHits, sizeof(int), hkSpuDmaManager::WRITE_NEW);

	if(numHits)
	{
		//collector.sortHits();
		hkSpuDmaManager::putToMainMemoryAndWaitForCompletion(m_resultArray, collector.getHits(), numHits * sizeof(hknpCollisionResult), hkSpuDmaManager::WRITE_NEW);
		hkSpuDmaManager::performFinalChecks(m_resultArray, collector.getHits(), numHits * sizeof(hknpCollisionResult));
	}

	hkSpuDmaManager::performFinalChecks(&jobPpu->m_numHits, &numHits, sizeof(int));
	hkSpuDmaManager::performFinalChecks(m_queryShape, pQueryShape, m_queryShapeSize);
	hkSpuDmaManager::performFinalChecks(m_targetShape, pTargetShape, m_targetShapeSize);
}

#endif

#if !defined(HK_PLATFORM_SPU)

hknpPairShapeCastSubTask::hknpPairShapeCastSubTask()
{
}

void hknpPairShapeCastSubTask::process(hknpCollisionQueryTask* task)
{
	hknpClosestHitCollector collector;
	{
		hknpCollisionQueryContext queryContext;

		queryContext.m_dispatcher = task->m_querySharedData.m_pCollisionQueryDispatcher;

		if(m_query.m_filter)
		{
			queryContext.m_shapeTagCodec = task->m_querySharedData.m_shapeTagCodec;
			hknpShapeQueryInterface::castShape( &queryContext, m_query, m_queryShapeInfo, *m_targetShape, m_targetShapeFilterData, m_targetShapeInfo, &collector );
		}
		else
		{
			hknpShapeQueryInterface::castShape( &queryContext, m_query, m_queryShapeOrientationInWorld, *m_targetShape, m_targetShapeToWorld, &collector );
		}
	}

	m_numHits = 0;

	if(collector.hasHit())
	{
		m_numHits = 1;
		m_resultArray[0] = collector.getHits()[0];
	}
}

#else

void hknpPairShapeCastSubTask::process(hknpCollisionQueryTask* task, const hknpPairShapeCastSubTask* jobPpu, void* preloadBuffer, const hknpCollisionQueryDispatcherBase* pQueryDispatcher)
{
	// patch the pointers
	const hknpShape* pShapePPU = m_query.m_shape;
	hknpShape* pQueryShape;
	hknpShape* pTargetShape;
	{
		hkUint8* pBuffer = (hkUint8*)preloadBuffer;

		hknpShapeVirtualTableUtil::patchVirtualTable((hknpShape*)pBuffer);
		pQueryShape = (hknpShape*)pBuffer;
		pBuffer += m_queryShapeSize;

		hknpShapeVirtualTableUtil::patchVirtualTable((hknpShape*)pBuffer);
		pTargetShape = (hknpShape*)pBuffer;
		pBuffer += m_targetShapeSize;
	}


	hknpClosestHitCollector collector;
	{
		hknpInplaceTriangleShape queryTriangle;
		hknpInplaceTriangleShape targetTriangle;

		hknpCollisionQueryContext queryContext(queryTriangle.getTriangleShape(), targetTriangle.getTriangleShape());

		m_query.m_shape = pQueryShape;
		queryContext.m_dispatcher = pQueryDispatcher;

		if(m_query.m_filter)
		{
			queryContext.m_shapeTagCodec = task->m_querySharedData.m_shapeTagCodec;
			m_query.m_filter = task->m_querySharedData.m_collisionFilter;

			m_queryShapeInfo.m_shapeToWorld = &m_queryShapeToWorld;
			m_targetShapeInfo.m_shapeToWorld = &m_targetShapeToWorld;

			m_queryShapeInfo.m_body = HK_NULL;
			m_targetShapeInfo.m_body = HK_NULL;

			hknpShapeQueryInterface::castShape( &queryContext, m_query, m_queryShapeInfo, *pTargetShape, m_targetShapeFilterData, m_targetShapeInfo, &collector );
		}
		else
		{
			hknpShapeQueryInterface::castShape( &queryContext, m_query, m_queryShapeOrientationInWorld, *pTargetShape, m_targetShapeToWorld, &collector );
		}
	}

	hkPadSpu<int> numHits = 0;
	if(collector.hasHit())
	{
		numHits = 1;
	}
	hkSpuDmaManager::putToMainMemoryAndWaitForCompletion(&jobPpu->m_numHits.ref(), &numHits, sizeof(hkPadSpu<int>), hkSpuDmaManager::WRITE_NEW);
	hkSpuDmaManager::performFinalChecks(&jobPpu->m_numHits, &numHits, sizeof(hkPadSpu<int>));

	if(numHits)
	{
		hkSpuDmaManager::putToMainMemoryAndWaitForCompletion(m_resultArray, collector.getHits(), sizeof(hknpCollisionResult), hkSpuDmaManager::WRITE_NEW);
		hkSpuDmaManager::performFinalChecks(m_resultArray, collector.getHits(), sizeof(hknpCollisionResult));
	}
	hkSpuDmaManager::performFinalChecks(pShapePPU, pQueryShape, m_queryShapeSize);
	hkSpuDmaManager::performFinalChecks(m_targetShape, pTargetShape, m_targetShapeSize);
}

#endif

#if !defined(HK_PLATFORM_SPU)

hknpAabbQuerySubTask::hknpAabbQuerySubTask() :
m_pShape(HK_NULL),
m_resultArray(HK_NULL),
m_resultArraySize(0),
m_numHits(0)
{

}

void hknpAabbQuerySubTask::initialize(const hknpShape* pShape, hkAabb& queryAABB)
{
	m_pShape = pShape;
	m_shapeSize = pShape->calcSize();
	m_aabb = queryAABB;
}

void hknpAabbQuerySubTask::process(hknpCollisionQueryTask* task)
{
	hknpAllHitsCollector collector;
	{
		hknpAabbQuery query;
		query.m_aabb = m_aabb;

		hknpCollisionQueryContext queryContext(HK_NULL, HK_NULL);
		hknpShapeQueryInfo queryShapeInfo;
		hknpQueryFilterData targetShapeFilterData;
		hknpShapeQueryInfo targetShapeInfo;

		hknpShapeQueryInterface::queryAabb( &queryContext, query, *m_pShape, targetShapeFilterData, targetShapeInfo, &collector );

	}

	//hknpShapeQueryInterface::queryAabb( query, *m_pShape, &collector );

	m_numHits = 0;

	for(int hitIndex=0; hitIndex<collector.getNumHits(); hitIndex++)
	{
		if(hitIndex >= m_resultArraySize)
		{
			break;
		}

		m_resultArray[hitIndex] = collector.getHits()[hitIndex];
		m_numHits++;
	}
}

#else

void hknpAabbQuerySubTask::process(hknpCollisionQueryTask* task, const hknpAabbQuerySubTask* jobPpu, void* preloadBuffer)
{
	hknpShape* pShape = (hknpShape*)preloadBuffer;
	hknpShapeVirtualTableUtil::patchVirtualTable(pShape);

	hknpAllHitsCollector collector;

	{
		hknpAabbQuery query;
		query.m_aabb = m_aabb;

		hknpCollisionQueryContext queryContext(HK_NULL, HK_NULL);
		hknpShapeQueryInfo queryShapeInfo;
		hknpQueryFilterData targetShapeFilterData;
		hknpShapeQueryInfo targetShapeInfo;

		hknpShapeQueryInterface::queryAabb( &queryContext, query, *pShape, targetShapeFilterData, targetShapeInfo, &collector );

	}

	hkPadSpu<int> numHits = collector.getNumHits();
	if(numHits > m_resultArraySize)
	{
		numHits = m_resultArraySize;
	}
	hkSpuDmaManager::putToMainMemorySmallAndWaitForCompletion(&jobPpu->m_numHits.ref(), &numHits, sizeof(int), hkSpuDmaManager::WRITE_NEW);

	if(numHits)
	{
		collector.sortHits();
		hkSpuDmaManager::putToMainMemoryAndWaitForCompletion(m_resultArray, collector.getHits(), numHits * sizeof(hknpCollisionResult), hkSpuDmaManager::WRITE_NEW);
		hkSpuDmaManager::performFinalChecks(m_resultArray, collector.getHits(), numHits * sizeof(hknpCollisionResult));
	}

	hkSpuDmaManager::performFinalChecks(&jobPpu->m_numHits, &numHits, sizeof(int));
	hkSpuDmaManager::performFinalChecks(m_pShape, pShape, m_shapeSize);
}

#endif

#if !defined(HK_PLATFORM_SPU)

hknpWorldGetClosestPointsSubTask::hknpWorldGetClosestPointsSubTask() :
m_pWorld(HK_NULL),
m_resultArray(HK_NULL),
m_resultArraySize(0),
m_numHits(0)
{

}

void hknpWorldGetClosestPointsSubTask::initialize(hknpWorld* pWorld, hknpClosestPointsQuery& query)
{
	m_query = query;
	m_pWorld = pWorld;
	m_queryShapeTransform = query.m_body->getTransform();
	m_queryShapeSize = query.m_body->m_shape->calcSize();
}

void hknpWorldGetClosestPointsSubTask::process(hknpCollisionQueryTask* task)
{
	hknpClosestHitCollector collector;
	m_pWorld->getClosestPoints(m_query, m_queryShapeTransform, &collector);

	m_numHits = 0;

	if(collector.hasHit())
	{
		m_numHits = 1;
		m_resultArray[0] = collector.getHits()[0];
	}
}

#else

void hknpWorldGetClosestPointsSubTask::process(hknpCollisionQueryTask* spuJob, const hknpWorldGetClosestPointsSubTask* jobPpu, void* preloadBuffer, const hknpCollisionQueryDispatcherBase* pQueryDispatcher)
{
	// patch the pointers
	const hknpShape* pShapePPU = m_query.m_shape;
	hknpShape* pQueryShape;
	hknpWorld* pWorld;
	{
		hkUint8* pBuffer = (hkUint8*)preloadBuffer;

		hknpShapeVirtualTableUtil::patchVirtualTable((hknpShape*)pBuffer);
		pQueryShape = (hknpShape*)pBuffer;
		pBuffer += m_queryShapeSize;

		pWorld = (hknpWorld*)pBuffer;
		pBuffer += sizeof(hknpWorld);
	}

	const hknpBody* bodyPPU = m_query.m_body;
	hknpBody bodySPU;

	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(&bodySPU, bodyPPU, sizeof(hknpBody), hkSpuDmaManager::READ_COPY);
	hkSpuDmaManager::performFinalChecks(bodyPPU, &bodySPU, sizeof(hknpBody));

	m_query.m_body = &bodySPU;

	hknpClosestHitCollector collector;
	{
		m_query.m_shape = pQueryShape;
		m_query.m_filter = spuJob->m_querySharedData.m_collisionFilter;

		pWorld->m_collisionQueryDispatcher = const_cast<hknpCollisionQueryDispatcherBase*>(pQueryDispatcher);
		pWorld->setShapeTagCodec(const_cast<hknpShapeTagCodec*>(spuJob->m_querySharedData.m_shapeTagCodec.val()));

		pWorld->getClosestPoints( m_query, m_queryShapeTransform, &collector);
	}

	hkPadSpu<int> numHits = 0;
	if(collector.hasHit())
	{
		numHits = 1;
	}
	hkSpuDmaManager::putToMainMemoryAndWaitForCompletion(&jobPpu->m_numHits.ref(), &numHits, sizeof(hkPadSpu<int>), hkSpuDmaManager::WRITE_NEW);
	hkSpuDmaManager::performFinalChecks(&jobPpu->m_numHits, &numHits, sizeof(hkPadSpu<int>));

	if(numHits)
	{
		hkSpuDmaManager::putToMainMemoryAndWaitForCompletion(m_resultArray, collector.getHits(), sizeof(hknpCollisionResult), hkSpuDmaManager::WRITE_NEW);
		hkSpuDmaManager::performFinalChecks(m_resultArray, collector.getHits(), sizeof(hknpCollisionResult));
	}
	hkSpuDmaManager::performFinalChecks(pShapePPU, pQueryShape, m_queryShapeSize);
	hkSpuDmaManager::performFinalChecks(m_pWorld, pWorld, sizeof(hknpWorld));
}

#endif

#ifndef HK_PLATFORM_SPU

hknpWorldShapeCastSubTask::hknpWorldShapeCastSubTask()
{
};

void hknpWorldShapeCastSubTask::process(hknpCollisionQueryTask* task)
{
	hknpClosestHitCollector collector;
	m_pWorld->castShape(m_query, m_queryShapeOrientationInWorld, &collector);

	m_numHits = 0;

	if(collector.hasHit())
	{
		m_numHits = 1;
		m_resultArray[0] = collector.getHits()[0];
	}
}

#else

void hknpWorldShapeCastSubTask::process(hknpCollisionQueryTask* spuJob, const hknpWorldShapeCastSubTask* jobPpu, void* preloadBuffer, const hknpCollisionQueryDispatcherBase* pQueryDispatcher)
{
	// patch the pointers
	hknpShape* pQueryShape;
	hknpWorld* pWorld;
	{
		hkUint8* pBuffer = (hkUint8*)preloadBuffer;

		hknpShapeVirtualTableUtil::patchVirtualTable((hknpShape*)pBuffer);
		pQueryShape = (hknpShape*)pBuffer;
		pBuffer += m_queryShapeSize;

		pWorld = (hknpWorld*)pBuffer;
		pBuffer += sizeof(hknpWorld);
	}

	const hknpBody* bodyPPU = m_query.m_body;
	hknpBody bodySPU;

	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(&bodySPU, bodyPPU, sizeof(hknpBody), hkSpuDmaManager::READ_COPY);
	hkSpuDmaManager::performFinalChecks(bodyPPU, &bodySPU, sizeof(hknpBody));

	m_query.m_body = &bodySPU;

	hknpClosestHitCollector collector;
	{
		m_query.m_shape = pQueryShape;
		m_query.m_filter = spuJob->m_querySharedData.m_collisionFilter;

		pWorld->m_collisionQueryDispatcher = const_cast<hknpCollisionQueryDispatcherBase*>(pQueryDispatcher);
		pWorld->setShapeTagCodec(const_cast<hknpShapeTagCodec*>(spuJob->m_querySharedData.m_shapeTagCodec.val()));

		pWorld->castShape( m_query, m_queryShapeOrientationInWorld, &collector);
	}

	hkPadSpu<int> numHits = 0;
	if(collector.hasHit())
	{
		numHits = 1;
	}
	hkSpuDmaManager::putToMainMemoryAndWaitForCompletion(&jobPpu->m_numHits.ref(), &numHits, sizeof(hkPadSpu<int>), hkSpuDmaManager::WRITE_NEW);
	hkSpuDmaManager::performFinalChecks(&jobPpu->m_numHits, &numHits, sizeof(hkPadSpu<int>));

	if(numHits)
	{
		hkSpuDmaManager::putToMainMemoryAndWaitForCompletion(m_resultArray, collector.getHits(), sizeof(hknpCollisionResult), hkSpuDmaManager::WRITE_NEW);
		hkSpuDmaManager::performFinalChecks(m_resultArray, collector.getHits(), sizeof(hknpCollisionResult));
	}
	hkSpuDmaManager::performFinalChecks(m_queryShape, pQueryShape, m_queryShapeSize);
	hkSpuDmaManager::performFinalChecks(m_pWorld, pWorld, sizeof(hknpWorld));
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
