/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Agent/CompoundAgent/List/hkpListAgent.h>
#include <Physics2012/Collide/Agent/CompoundAgent/ShapeCollection/hkpShapeCollectionAgent.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#include <Physics2012/Collide/Agent/Util/Null/hkpNullAgent.h>

#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>

#if defined(HK_PLATFORM_SPU)
#	include <Common/Base/Memory/PlatformUtils/Spu/SpuDmaCache/hkSpu4WayCache.h>
#	include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#	include <Physics2012/Collide/Query/Multithreaded/Spu/hkpSpuConfig.h>
#	include <Physics2012/Collide/Filter/Spu/hkpSpuCollisionFilterUtil.h>
	extern hkSpu4WayCache* g_SpuCollideUntypedCache;
#endif


#if !defined(HK_PLATFORM_SPU)

void HK_CALL hkpListAgent::initAgentFunc(hkpCollisionDispatcher::AgentFuncs& af)
{
	// This is symmetric (flipped) and the static query functions are not.
	af.m_createFunc          = createListAAgent;

	// fallback to hkpShapeCollection Agent implementations
	af.m_getPenetrationsFunc = hkpShapeCollectionAgent::staticGetPenetrations;
	af.m_getClosestPointFunc = hkpShapeCollectionAgent::staticGetClosestPoints;
	af.m_linearCastFunc      = hkpShapeCollectionAgent::staticLinearCast;
	af.m_isFlipped           = true;
	af.m_isPredictive		 = true;
}


void HK_CALL hkpListAgent::initAgentFuncInverse(hkpCollisionDispatcher::AgentFuncs& af)
{
	// This is not symmetric (not flipped) as are the static query functions.
	af.m_createFunc          = createListBAgent;

	// fallback to hkpShapeCollection Agent implementations
	af.m_getPenetrationsFunc = hkpSymmetricAgent<hkpShapeCollectionAgent>::staticGetPenetrations;
	af.m_getClosestPointFunc = hkpSymmetricAgent<hkpShapeCollectionAgent>::staticGetClosestPoints;
	af.m_linearCastFunc      = hkpSymmetricAgent<hkpShapeCollectionAgent>::staticLinearCast;
	af.m_isFlipped           = false;
	af.m_isPredictive		 = true;
}

#else

void HK_CALL hkpListAgent::initAgentFunc(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = hkpListAgent::staticGetClosestPoints;
	af.m_linearCastFunc	     = hkpListAgent::staticLinearCast;
}


void HK_CALL hkpListAgent::initAgentFuncInverse(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = hkpSymmetricAgent<hkpListAgent>::staticGetClosestPoints;
	af.m_linearCastFunc	     = hkpSymmetricAgent<hkpListAgent>::staticLinearCast;
}

#endif


#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkpListAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// symmetric
	{
		hkpCollisionDispatcher::AgentFuncs af;
		initAgentFunc(af);
		dispatcher->registerCollisionAgent(af, hkcdShapeType::LIST, hkcdShapeType::ALL_SHAPE_TYPES);
	}

	// direct
	{
		hkpCollisionDispatcher::AgentFuncs af;
		initAgentFuncInverse(af);
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::LIST);
	}
}


hkpCollisionAgent* HK_CALL hkpListAgent::createListAAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB,
														const hkpCollisionInput& input, hkpContactMgr* contactMgr)
{
	hkpListAgent* agent = new hkpSymmetricAgent<hkpListAgent>(bodyA, bodyB, input, contactMgr);
	return agent;
}

hkpCollisionAgent* HK_CALL hkpListAgent::createListBAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB,
									const hkpCollisionInput& input, hkpContactMgr* contactMgr)
{
	hkpListAgent* agent = new hkpListAgent(bodyA, bodyB, input, contactMgr);
	return agent;
}



hkpListAgent::hkpListAgent(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* contactMgr)
: hkpCollisionAgent( contactMgr )
{
	m_dispatcher = input.m_dispatcher;
	hkAgent1nMachine_Create( m_agentTrack );
}


void hkpListAgent::cleanup( hkCollisionConstraintOwner& constraintOwner )
{
	hkAgent1nMachine_Destroy( m_agentTrack, m_dispatcher, m_contactMgr, constraintOwner );
	delete this;
}

void hkpListAgent::invalidateTim( const hkpCollisionInput& input )
{
	hkAgent1nMachine_InvalidateTim(m_agentTrack, input);
}

void hkpListAgent::warpTime( hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	hkAgent1nMachine_WarpTime(m_agentTrack, oldTime, newTime, input);
}
#endif

void hkpListAgent::processCollision( const hkpCdBody& bodyA, const hkpCdBody& bodyB,
									const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& output)
{
	HK_ASSERT2(0x158d6356,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_INTERNAL_TIMER_BEGIN("list", this);

	const hkpListShape* lShapeB = static_cast<const hkpListShape*>(bodyB.getShape());

	//
	//	Set the input structure
	//
	hkpAgent3ProcessInput in3;
	{
		in3.m_bodyA = &bodyA;
		in3.m_bodyB = &bodyB;
#if !defined(HK_PLATFORM_SPU)
		in3.m_contactMgr = m_contactMgr;
#else
		in3.m_contactMgr = output.m_contactMgr;
#endif
		in3.m_input = &input;

		const hkMotionState* msA = bodyA.getMotionState();
		const hkMotionState* msB = bodyB.getMotionState();

		hkSweptTransformUtil::calcTimInfo( *msA, *msB, input.m_stepInfo.m_deltaTime, in3.m_linearTimInfo);

		in3.m_aTb.setMulInverseMul(msA->getTransform(), msB->getTransform());
	}

	int size = lShapeB->m_childInfo.getSize();
	hkLocalBuffer<hkpShapeKey> hitList( size+1 );
	{
		int d = 0;
		for ( int i = 0; i < size; i++ )
		{
			if ( lShapeB->hkpListShape::isChildEnabled(i) )	{	hitList[d++] = hkpShapeKey(i);	}
		}
		hitList[d] = HK_INVALID_SHAPE_KEY;
	}

	hkAgent1nMachine_Process( m_agentTrack, in3, lShapeB, hitList.begin(), output );

	HK_INTERNAL_TIMER_END();
}


#if defined(HK_PLATFORM_SPU)
	HK_COMPILE_TIME_ASSERT( HK_SPU_MAXIMUM_SHAPE_SIZE >= 256); // for now this value has to be of at least 256bytes as we use a memCpy256 to copy data from the cached shape into a temporary buffer
#endif


// hkpCollisionAgent interface implementation.
void hkpListAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
#if !defined(HK_PLATFORM_SPU)

	HK_INTERNAL_TIMER_BEGIN("ListAgent", this);
	hkpSymmetricAgent<hkpShapeCollectionAgent>::staticGetClosestPoints( bodyA, bodyB, input, collector );
	HK_INTERNAL_TIMER_END();

#else

	HK_TIMER_BEGIN( "ListShape", HK_NULL );

	// bodyB can be of any type except hkpListShape
	hkpShapeType typeB = bodyB.getShape()->getType();

	// bodyA is of type hkpListShape
	const hkpListShape* listShape = static_cast<const hkpListShape*>(bodyA.getShape());

	hkpCdBody newBodyA( &bodyA );

	HK_ASSERT( 0xaf234345, hkMath::isPower2(sizeof(hkpListShape::ChildInfo)));

	const unsigned numKeysPerCacheline = HK_SPU_COLLIDE_PAIR_GET_CLOSEST_POINTS_JOB_CHILD_INFO_BUFFER_SIZE/sizeof(hkpListShape::ChildInfo);
	const unsigned numKeysTotal = listShape->m_childInfo.getSize();
	void* arrayOnPPu = (void*) listShape->m_childInfo.begin();
	hkpListShape::ChildInfo childInfoBuffer[HK_SPU_COLLIDE_PAIR_GET_CLOSEST_POINTS_JOB_CHILD_INFO_BUFFER_SIZE];
	const hkpListShape::ChildInfo* childInfos = &childInfoBuffer[0];
	{
		for ( unsigned int i = 0; i < numKeysTotal; i+=numKeysPerCacheline)
		{
			int numKeysPerBatch = hkMath::min2(numKeysPerCacheline, numKeysTotal-i);
			int childInfoMemSize = numKeysPerBatch * sizeof(hkpListShape::ChildInfo);

			// for now we will not double-buffer this dma read... as a result we will have some code stall here
			hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(&childInfoBuffer[0], arrayOnPPu, childInfoMemSize, hkSpuDmaManager::READ_COPY, hkSpuPairGetClosestPointsDmaGroups::GET_CHILD_INFOS);
			HK_SPU_DMA_PERFORM_FINAL_CHECKS(arrayOnPPu, &childInfoBuffer[0], childInfoMemSize);

			//
			// process all children in this batch
			//
			{
				for ( int c = 0; c < numKeysPerBatch; c++ )
				{
					//
					// check if collision between both objects is enabled
					//

					const hkpShapeKey listShapeChildShapeKey = i + c;
					hkBool isCollisionEnabled = listShape->isChildEnabled(listShapeChildShapeKey) && hkpSpuCollisionFilterUtil::s_shapeContainerIsCollisionEnabled(static_cast<const hkpCollisionFilter*>(input.m_filter.val()), input, bodyB, bodyA, *listShape, listShapeChildShapeKey);

					//
					// call getClosestPoint function for both objects (if collision is enabled between them)
					//
					if ( isCollisionEnabled )
					{
						//
						// get shape from ppu (using the untyped cache)
						//
						const hkpShape* shape;
						{
							const void* shapeOnPpu = childInfos[c].m_shape;
							hkInt32     shapeSize  = childInfos[c].m_shapeSize;
							const void* shapeOnSpu = g_SpuCollideUntypedCache->getFromMainMemory(shapeOnPpu, shapeSize);
							shape = reinterpret_cast<const hkpShape*>(shapeOnSpu);
							HKP_PATCH_CONST_SHAPE_VTABLE(shape);
						}

						// we have to temporarily buffer the child shape before descending in the hierarchy, as the shape itself might be evicted from the untyped cache
						HK_ALIGN16( hkUchar shapeBuffer[HK_SPU_MAXIMUM_SHAPE_SIZE] );
						hkpShape* shapeCopy = reinterpret_cast<hkpShape*>(&shapeBuffer[0]);
						hkString::memCpy256(shapeCopy, shape);

						hkpShapeType typeA = shape->getType();

						hkpSpuCollisionQueryDispatcher::GetClosestPointsFunc getClosestPointsFunc = input.m_queryDispatcher->getGetClosestPointsFunc( typeA, typeB );

						newBodyA.setShape(shapeCopy, listShapeChildShapeKey);

						getClosestPointsFunc(newBodyA, bodyB, input, collector);
					}
				}
			}

			arrayOnPPu = hkAddByteOffset( arrayOnPPu, HK_SPU_COLLIDE_PAIR_GET_CLOSEST_POINTS_JOB_CHILD_INFO_BUFFER_SIZE );
		}
	}

	HK_TIMER_END();

#endif

}


// This is exact copy of hkpListAgent::staticGetClosestPoints except the few lines where we forward the call to child shapes.
void hkpListAgent::staticLinearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
#if !defined(HK_PLATFORM_SPU)
	HK_INTERNAL_TIMER_BEGIN("ListAgent", this);
	hkpSymmetricAgent<hkpShapeCollectionAgent>::staticLinearCast( bodyA, bodyB, input, collector, startCollector );
	HK_INTERNAL_TIMER_END();
#else
	HK_TIMER_BEGIN( "ListShape", HK_NULL );

	// bodyB can be of any type except hkpListShape
	hkpShapeType typeB = bodyB.getShape()->getType();

	// bodyA is of type hkpListShape
	const hkpListShape* listShape = static_cast<const hkpListShape*>(bodyA.getShape());

	hkpCdBody newBodyA( &bodyA );

	HK_ASSERT( 0xaf234345, hkMath::isPower2(sizeof(hkpListShape::ChildInfo)));

	const unsigned numKeysPerCacheline = HK_SPU_COLLIDE_PAIR_GET_CLOSEST_POINTS_JOB_CHILD_INFO_BUFFER_SIZE/sizeof(hkpListShape::ChildInfo);
	const unsigned numKeysTotal = listShape->m_childInfo.getSize();
	void* arrayOnPPu = (void*) listShape->m_childInfo.begin();
	hkpListShape::ChildInfo childInfoBuffer[HK_SPU_COLLIDE_PAIR_GET_CLOSEST_POINTS_JOB_CHILD_INFO_BUFFER_SIZE];
	const hkpListShape::ChildInfo* childInfos = &childInfoBuffer[0];
	{
		for ( unsigned int i = 0; i < numKeysTotal; i+=numKeysPerCacheline)
		{
			int numKeysPerBatch = hkMath::min2(numKeysPerCacheline, numKeysTotal-i);
			int childInfoMemSize = numKeysPerBatch * sizeof(hkpListShape::ChildInfo);

			// for now we will not double-buffer this dma read... as a result we will have some code stall here
			hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(&childInfoBuffer[0], arrayOnPPu, childInfoMemSize, hkSpuDmaManager::READ_COPY, hkSpuPairGetClosestPointsDmaGroups::GET_CHILD_INFOS);
			HK_SPU_DMA_PERFORM_FINAL_CHECKS(arrayOnPPu, &childInfoBuffer[0], childInfoMemSize);

			//
			// process all children in this batch
			//
			{
				for ( int c = 0; c < numKeysPerBatch; c++ )
				{
					//
					// check if collision between both objects is enabled
					//

					const hkpShapeKey listShapeChildShapeKey = i + c;
					hkBool isCollisionEnabled = listShape->isChildEnabled(listShapeChildShapeKey) && hkpSpuCollisionFilterUtil::s_shapeContainerIsCollisionEnabled(static_cast<const hkpCollisionFilter*>(input.m_filter.val()), input, bodyB, bodyA, *listShape, listShapeChildShapeKey);


					//
					// call getClosestPoint function for both objects (if collision is enabled between them)
					//
					if ( isCollisionEnabled )
					{
						//
						// get shape from ppu (using the untyped cache)
						//
						const hkpShape* shape;
						{
							const void* shapeOnPpu = childInfos[c].m_shape;
							hkInt32     shapeSize  = childInfos[c].m_shapeSize;
							const void* shapeOnSpu = g_SpuCollideUntypedCache->getFromMainMemory(shapeOnPpu, shapeSize);
							shape = reinterpret_cast<const hkpShape*>(shapeOnSpu);
							HKP_PATCH_CONST_SHAPE_VTABLE(shape);
						}

						// we have to temporarily buffer the child shape before descending in the hierarchy, as the shape itself might be evicted from the untyped cache
						HK_ALIGN16( hkUchar shapeBuffer[HK_SPU_MAXIMUM_SHAPE_SIZE] );
						hkpShape* shapeCopy = reinterpret_cast<hkpShape*>(&shapeBuffer[0]);
						hkString::memCpy256(shapeCopy, shape);

						hkpShapeType typeA = shape->getType();

						hkpSpuCollisionQueryDispatcher::LinearCastFunc getLinearCastFunc = input.m_queryDispatcher->getLinearCastFunc( typeA, typeB );

						newBodyA.setShape(shapeCopy, listShapeChildShapeKey);

						getLinearCastFunc(newBodyA, bodyB, input, collector, startCollector);
					}
				}
			}

			arrayOnPPu = hkAddByteOffset( arrayOnPPu, HK_SPU_COLLIDE_PAIR_GET_CLOSEST_POINTS_JOB_CHILD_INFO_BUFFER_SIZE );
		}
	}

	HK_TIMER_END();
#endif
}


#if !defined(HK_PLATFORM_SPU)
void hkpListAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector )
{
	hkpListAgent::staticGetClosestPoints( bodyA, bodyB, input, collector );
}

void hkpListAgent::linearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	hkpListAgent::staticLinearCast( bodyA, bodyB, input, collector, startCollector );
}

void hkpListAgent::getPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	hkpListAgent::staticGetPenetrations( bodyA, bodyB, input, collector);
}

void hkpListAgent::staticGetPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_INTERNAL_TIMER_BEGIN("ListAgent", this);
	hkpSymmetricAgent<hkpShapeCollectionAgent>::staticGetPenetrations( bodyA, bodyB, input, collector );
	HK_INTERNAL_TIMER_END();
}

void hkpListAgent::updateShapeCollectionFilter( const hkpCdBody& bodyA, const hkpCdBody& listShapeBodyB, const hkpCollisionInput& input, hkCollisionConstraintOwner& constraintOwner )
{
	hkpAgent1nMachine_VisitorInput vin;
	vin.m_bodyA = &bodyA;
	vin.m_collectionBodyB = &listShapeBodyB;
	vin.m_input = &input;
	vin.m_contactMgr = m_contactMgr;
	vin.m_constraintOwner = &constraintOwner;
	vin.m_containerShapeB = static_cast<const hkpShapeCollection*>(listShapeBodyB.getShape())->getContainer();

	hkAgent1nMachine_UpdateShapeCollectionFilter( m_agentTrack, vin );
}

#endif

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
