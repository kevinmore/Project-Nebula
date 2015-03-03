/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#if defined(HK_PLATFORM_HAS_SPU)
#	include <Common/Base/Monitor/Spu/hkSpuMonitorCache.h>
#endif
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Common/Base/Algorithm/Collide/1AxisSweep/hk1AxisSweep.h>

#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Filter/hkpShapeCollectionFilter.h>
#include <Physics2012/Collide/Agent/Util/Null/hkpNullAgent.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>
#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Physics2012/Collide/Agent3/BvTree3/hkpBvTreeAgent3.h>
#include <Physics2012/Collide/Agent3/CollectionCollection3/hkpCollectionCollectionAgent3.h>
#include <Physics2012/Collide/Agent3/List3/hkpListAgent3.h>

#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nTrack.h>

#include <Physics2012/Collide/Agent3/Machine/Midphase/hkpMidphaseAgentData.h>
#include <Physics2012/Collide/Agent3/Machine/Midphase/hkpShapeKeyTrack.h>

#define HK_THIS_AGENT_SIZE HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpMidphaseAgentData) )
HK_COMPILE_TIME_ASSERT(HK_THIS_AGENT_SIZE <= hkAgent3::MAX_NET_SIZE);

bool hkpCollectionCollectionAgent3::g_agentRegistered = false;

void hkpCollectionCollectionAgent3::initAgentFunc(hkpCollisionDispatcher::Agent3Funcs& f)
{
	f.m_createFunc   = hkListAgent3::create;
	f.m_processFunc  = process;
	f.m_sepNormalFunc = HK_NULL; 
	f.m_cleanupFunc  = HK_NULL;
#if !defined(HK_PLATFORM_SPU)
	f.m_removePointFunc  = HK_NULL;
	f.m_commitPotentialFunc  = HK_NULL;
	f.m_createZombieFunc  = HK_NULL;
	f.m_updateFilterFunc = updateFilter;
	f.m_invalidateTimFunc = invalidateTim;
	f.m_warpTimeFunc = warpTime;
#endif
	f.m_destroyFunc  = hkListAgent3::destroy;
	f.m_isPredictive = true;
}

#if !defined(HK_PLATFORM_SPU)
void hkpCollectionCollectionAgent3::registerAgent3(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::Agent3Funcs f;
	initAgentFunc(f);
	dispatcher->registerAgent3( f, hkcdShapeType::BV_TREE,    hkcdShapeType::BV_TREE );

	f.m_ignoreSymmetricVersion = true;
	f.m_reusePreviousEntry = true;
	dispatcher->registerAgent3( f, hkcdShapeType::COLLECTION, hkcdShapeType::BV_TREE );
	dispatcher->registerAgent3( f, hkcdShapeType::BV_TREE, hkcdShapeType::COLLECTION );
	dispatcher->registerAgent3( f, hkcdShapeType::COLLECTION, hkcdShapeType::COLLECTION );
	dispatcher->m_midphaseAgent3Registered = true;

	g_agentRegistered = true;
}
#endif

HK_COMPILE_TIME_ASSERT( sizeof(hk1AxisSweep::AabbInt) == sizeof(hkAabbUint32) );

static int hkCollectionBvTreeAgent3_extractCachedAabbsOrRecalculate(const hkpCdBody* cdBody, const hkpCdBody* overrideCdBody, const HK_SHAPE_CONTAINER* collection, const hkpProcessCollisionInput& input, hk1AxisSweep::AabbInt* aabbs, int aabbsCapacity, int aabbBufferSize_usedOnSpu)
{
	const hkBool32 useContinuousPhysics = input.m_collisionQualityInfo->m_useContinuousPhysics;

	int numChildAabbs = 0;

	HK_ON_CPU( const hkpCollidable   ::BoundingVolumeData* bvData = HK_NULL );
	HK_ON_SPU( const hkCollidablePpu::BoundingVolumeData* bvData = HK_NULL );

	if ( ( !cdBody->getParent() )
		&& HK_NULL != ( bvData = &static_cast<const hkpCollidable*>(cdBody)->m_boundingVolumeData )->m_childShapeAabbs
		&& ( bvData->isValid()) )
	{
		//
		// Extract the already calculated and cached AABBs.
		//

		if(bvData->m_numChildShapeAabbs == 0)
		{
			return 0;
		}

#if !defined(HK_PLATFORM_SPU)

		hkAabbUint32* childShapeAabbs = bvData->m_childShapeAabbs;
		hkpShapeKey* childShapeKeys = bvData->m_childShapeKeys;
		for (int i = 0; i < bvData->m_numChildShapeAabbs; i++)
		{
			hk1AxisSweep::AabbInt& aabbInt = aabbs[i];
			if (useContinuousPhysics)	{		hkAabbUtil::uncompressExpandedAabbUint32(*childShapeAabbs, static_cast<hkAabbUint32&>(aabbInt));		}
			else						{		static_cast<hkAabbUint32&>(aabbInt) = *childShapeAabbs;			}
			aabbInt.getKey() = *childShapeKeys;

			childShapeAabbs++;
			childShapeKeys++;

			numChildAabbs++; // we need to increment numChildAabbs as we make use of it later on for placing the end markers
		}

#else

		// Use the output buffer as input buffer for the DMA transfer and perform the operations inplace.
		{
			numChildAabbs = bvData->m_numChildShapeAabbs;
			int sizeOfchildShapeAabbs = numChildAabbs * sizeof(hkAabbUint32);
			hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(aabbs, bvData->m_childShapeAabbs, sizeOfchildShapeAabbs, hkSpuDmaManager::READ_COPY);
			HK_SPU_DMA_PERFORM_FINAL_CHECKS(bvData->m_childShapeAabbs, aabbs, sizeOfchildShapeAabbs);
			HK_ASSERT(0x38703bf7, numChildAabbs < 256 );
		}

		// expand AABB if it's continuous
		if (useContinuousPhysics)
		{
			hk1AxisSweep::AabbInt* d = aabbs;
			for (int i = 0; i < bvData->m_numChildShapeAabbs; i++)
			{
				hkAabbUtil::uncompressExpandedAabbUint32(*d, *d);
				d++;
			}
		}
		// put key into the AABB.
		{
			hk1AxisSweep::AabbInt* d = aabbs;
			for (int i = 0; i < bvData->m_numChildShapeAabbs; i++)
			{
				d->getKey() = hkpShapeKey(d->m_shapeKeyByte);
				d++;
			}
		}

#endif

		if ( !useContinuousPhysics )
		{
			goto DONT_SORT;	// already sorted
		}
		else
		{
			// do bubble sort and
			// goto DONT_SORT
		}
	}
	else if (bvData && bvData->isValid() && overrideCdBody)
	{
		HK_ASSERT2(0xad873433, bvData->m_childShapeAabbs == HK_NULL && aabbsCapacity == 1, "We're assuming this block is for a hkpCdBody coming from the hkpCollectionAgent3.");

		hk1AxisSweep::AabbInt& aabbInt = aabbs[0];

	#ifdef HK_ARCH_ARM
		// GGC on Arm etc will only align to native, will ignore the align16 if on stack, so will end up aligned to hkUint32 members etc
		HK_ASSERT2(0x46aefcee, (((hkUlong)bvData) & 0x3) == 0, "Unaligned bounding volume data!");
	#else
		HK_ASSERT2(0x46aefcee, (((hkUlong)bvData) & 0xF) == 0, "Unaligned bounding volume data!");
	#endif

		const hkAabbUint32& bvDataAabb = * reinterpret_cast<const hkAabbUint32*> (bvData);
		if (useContinuousPhysics)
		{	
			hkAabbUtil::uncompressExpandedAabbUint32(bvDataAabb, static_cast<hkAabbUint32&>(aabbInt));
		}
		else
		{
			static_cast<hkAabbUint32&>(aabbInt) = bvDataAabb;
		}
		aabbInt.getKey() = 0;
		numChildAabbs++;
		goto DONT_SORT;	// no need to sort a single element
	}
	else
	{
		//
		// Recalculate the children's AABBs.
		//

		hkAabbUtil::OffsetAabbInput sweepInput;
		if ( useContinuousPhysics )
		{
			hkAabbUtil::initOffsetAabbInput(cdBody->getRootCollidable()->getMotionState(), sweepInput); // this is safe for hkpCollectionAgent3 with virtual shape added
		}

#if !defined(HK_PLATFORM_SPU)

		hkpShapeBuffer buffer;
		for (hkpShapeKey key = HK_ACCESS_COLLECTION_METHOD(collection, getFirstKey()); key != HK_INVALID_SHAPE_KEY; key = HK_ACCESS_COLLECTION_METHOD(collection, getNextKey(key)))
		{
			const hkpShape* child = HK_ACCESS_COLLECTION_METHOD(collection, getChildShape(key, buffer));

			hkAabb aabb;
			child->getAabb(cdBody->getTransform(), input.m_tolerance, aabb);

			if (useContinuousPhysics)
			{
				// Expand AABB
				hkAabbUtil::sweepOffsetAabb(sweepInput, aabb, aabb);
			}

			hk1AxisSweep::AabbInt& aabbInt = aabbs[numChildAabbs++];

			// Convert AABB to integer space + increase precision
			hkAabbUtil::convertAabbToUint32(aabb, input.m_aabb32Info.m_bitOffsetLow, input.m_aabb32Info.m_bitOffsetHigh, input.m_aabb32Info.m_bitScale, aabbInt);

			aabbInt.getKey() = key;
		}

#else

		// Assume list shape on SPU
		const hkpListShape* listShape = reinterpret_cast<const hkpListShape*>( collection );
		hkAabb dummyRootAabb;

		if (!overrideCdBody)
		{
			numChildAabbs = listShape->getAabbWithChildShapesForAgent(input, sweepInput, useContinuousPhysics, cdBody->getTransform(), input.m_tolerance, dummyRootAabb, aabbs, aabbsCapacity, aabbBufferSize_usedOnSpu);
		}
		else
		{
			numChildAabbs = listShape->getAabbWithChildShapesForAgent_withNoDmas(input, sweepInput, useContinuousPhysics, cdBody->getTransform(), input.m_tolerance, dummyRootAabb, aabbs, aabbsCapacity);
		}

#endif
	}

	// Sort them for the 1-axis sweep
	//HK_TIMER_BEGIN("Sort", HK_NULL);
	hkSort(aabbs, numChildAabbs);
	//HK_TIMER_END();

DONT_SORT:
	HK_ASSERT2(0xad9755ba, numChildAabbs <= aabbsCapacity, "AABB array size & num elements don't match.");
	return numChildAabbs;
}

static HK_FORCE_INLINE void  hkCollectionBvTreeAgent3_calcAabbs( const hkpAgent3ProcessInput& input, const HK_SHAPE_CONTAINER* container, const hkpCdBody* cdBody, const hkpShapeKey* hitList, int numHits, hk1AxisSweep::AabbInt* HK_RESTRICT aabbsOut )
{
	hkAabbUtil::OffsetAabbInput sweepInput;
	const hkBool32 useContinuousPhysics = input.m_input->m_collisionQualityInfo->m_useContinuousPhysics.val();
	if ( useContinuousPhysics )
	{
		hkAabbUtil::initOffsetAabbInput(cdBody->getRootCollidable()->getMotionState(), sweepInput);
	}

	int idx;
	hkpShapeBuffer shapeBuffer;
	for (idx = 0; idx < numHits; idx++ )
	{
		hkpShapeKey key = hitList[idx];
		const hkpShape* child = container->getChildShape(key, shapeBuffer);

		hkAabb aabb;
		child->getAabb(cdBody->getTransform(), input.m_input->m_tolerance, aabb );
		if (useContinuousPhysics)
		{
			hkAabbUtil::sweepOffsetAabb(sweepInput, aabb, aabb);
		}

		hkAabbUtil::convertAabbToUint32(aabb, input.m_input->m_aabb32Info.m_bitOffsetLow, input.m_input->m_aabb32Info.m_bitOffsetHigh, input.m_input->m_aabb32Info.m_bitScale, aabbsOut[idx]);
		aabbsOut[idx].getKey() = key;
	}
}

#if defined (HK_PLATFORM_SPU)
#	define HK_ACCESS_COLLECTION_METHOD(obj, func) static_cast<const hkpListShape*>(obj)->hkpListShape::func
#else
#	define HK_ACCESS_COLLECTION_METHOD(obj, func) obj->func
#endif

#if 1 || !defined(HK_PLATFORM_SIM)
#	define WORST_CASE_BUFFER( SIZE, maxSize ) SIZE
#else
#	define WORST_CASE_BUFFER( SIZE, maxSize ) hkMath::max2(SIZE, int(maxSize) )
#endif

int hkpCollectionCollectionAgent3::process_gatherShapeKeys( const hkpAgent3ProcessInput& input, const HK_SHAPE_CONTAINER* shapeContainerA, const HK_SHAPE_CONTAINER* shapeContainerB, hkpShapeKeyPair* shapeKeyPairs )
{
	int numShapeKeyPairs;

	HK_TIMER_BEGIN_LIST( "CollColl3", "Init" );

	//
	// Extract the shape containers
	//

	const hkpShapeType	typeA = input.m_bodyA->getShape()->getType();
	const hkpShapeType	typeB = input.m_bodyB->getShape()->getType();
	
	const bool shapeAisCollection = 0 ==  input.m_input->m_dispatcher->hasAlternateType(typeA, hkcdShapeType::BV_TREE);
	const bool shapeBisCollection = 0 ==  input.m_input->m_dispatcher->hasAlternateType(typeB, hkcdShapeType::BV_TREE);

	bool treatAasCollection;

	if (shapeAisCollection | shapeBisCollection)
	{
		// now we have at least one collection colliding
		treatAasCollection = shapeAisCollection;

		if ( (shapeAisCollection && input.m_bodyA->getShape()->getType() != hkcdShapeType::LIST )
		  || (shapeBisCollection && input.m_bodyB->getShape()->getType() != hkcdShapeType::LIST))
		{
#		if !defined (HK_PLATFORM_SPU)
			HK_WARN_ONCE(0xad744aa3, "For perfomance reasons, it's advised to only use hkListShapes for compound moving bodies. Other collections are too slow. If you use a different collection be sure to wrap it with an hkpBvTreeShape.");
#		else
			HK_ASSERT2(0xad9755bb, false, "The only supported collection on Spu is hkpListShape.");
			HK_TIMER_END_LIST();
			return -1;
#		endif
		}
	}
	else
	{
		// So we have 2 bvTrees colliding here...

		const bool bodyAHasCachedChildShapeAabbs = (input.m_bodyA->m_parent == HK_NULL) && ( HK_NULL != static_cast<const hkpCollidable*>(input.m_bodyA.val())->m_boundingVolumeData.m_childShapeAabbs );
		const bool bodyBHasCachedChildShapeAabbs = (input.m_bodyB->m_parent == HK_NULL) && ( HK_NULL != static_cast<const hkpCollidable*>(input.m_bodyB.val())->m_boundingVolumeData.m_childShapeAabbs );

		// Force to always use the cached AABBs when they're available
		if ( bodyAHasCachedChildShapeAabbs == bodyBHasCachedChildShapeAabbs )
		{
			// If both trees are dynamic (i.e. they both have cached child shape AABBs), we try to find the collection. If we find a list, this should get preference
			hkpShapeBuffer tmpBufferA;
			hkpShapeBuffer tmpBufferB;

			const hkpShapeCollection* childCollectionA = hkBvTreeAgent3::getShapeCollectionIfBvTreeSupportsAabbQueries(input.m_bodyA, tmpBufferA);
			const hkpShapeCollection* childCollectionB = hkBvTreeAgent3::getShapeCollectionIfBvTreeSupportsAabbQueries(input.m_bodyB, tmpBufferB);

			const bool shapeAhasList = childCollectionA ? childCollectionA->getType() == hkcdShapeType::LIST : false;
			const bool shapeBhasList = childCollectionB ? childCollectionB->getType() == hkcdShapeType::LIST : false;

			if (shapeAhasList && shapeBhasList)
			{
				// 			// choose the one which has cached AABBs
				// 			if ( !input.m_bodyA->getParent() && !input.m_bodyB->getParent() )
				// 			{
				// 
				// 			}
				// choose smaller or the one which has cached AABB
				treatAasCollection = static_cast<const hkpListShape*>(childCollectionA)->hkpListShape::getNumChildShapes() <= static_cast<const hkpListShape*>(childCollectionB)->hkpListShape::getNumChildShapes();
			}
			else if (shapeAhasList | shapeBhasList)
			{
				treatAasCollection = shapeAhasList;
			}
			else // 2 meshes colliding, only for PPU
			{
#		if !defined (HK_PLATFORM_SPU)
				// choose smaller radius
				treatAasCollection = input.m_bodyA->getMotionState()->m_objectRadius <= input.m_bodyB->getMotionState()->m_objectRadius;
				HK_WARN_ONCE(0xad744aa3, "Colliding two hkBvTreeShapes where neither contains a hkpListShape collections. For perfomance reasons, it's advised for one to have a hkpListShape collection. Other collections are too slow.");
#		else	
				HK_ASSERT2(0xadbcd65d, false, "When supporting two hkBvTrees on Spu, at least one must have a hkpListShape as its collection.");
				HK_TIMER_END_LIST();
				return -2;
#		endif
			}
		}
		else
		{
			// If one of the two trees is fixed/keyframed (i.e. it doesn't have any child shape AABBs cached) we need to make sure that the cached AABBs are extracted from the dynamic tree.
			treatAasCollection = bodyAHasCachedChildShapeAabbs;
		}
	}


	{
		int numHitsFirst = 0;
		int numHitsSecond = 0;

		hk1AxisSweep::AabbInt* aabbsFirst;
		int aabbsFirstSize;
		hk1AxisSweep::AabbInt* aabbsSecond;
		int aabbsSecondSize;

		const hkpCdBody* firstCdBody;
		{
			const HK_SHAPE_CONTAINER* secondContainer;
			const hkpCdBody* secondBody;

			if ( !(shapeAisCollection & shapeBisCollection) )
			{
				const HK_SHAPE_CONTAINER* firstContainer;
				HK_TIMER_SPLIT_LIST("QueryTree");

#if !defined(HK_PLATFORM_XBOX360) && !defined(HK_PLATFORM_XBOX)			
				hkpShapeKey hitList[ HK_MAX_NUM_HITS_PER_AABB_QUERY ]; // we can't use a local array here, because we need to call hkAllocateStack further down
#else	// xbox has a very small stack so we have to use the localStack
				aabbsFirstSize = HK_MAX_NUM_HITS_PER_AABB_QUERY_USED+4;
				aabbsFirst = hkAllocateStack<hk1AxisSweep::AabbInt>(aabbsFirstSize, "aabbsFirst");
				hkpShapeKey* hitList = hkAllocateStack<hkpShapeKey>(HK_MAX_NUM_HITS_PER_AABB_QUERY);
#endif
				if ( treatAasCollection)
				{
					HK_ASSERT2(0xadb9a762, !shapeBisCollection, "Trying to perform AABB query on a non-bvtree shape.");
					// now shape b is a tree which gets queried
					hkTransform bTa;	bTa.setInverse( input.m_aTb );
					numHitsFirst = hkBvTreeAgent3::calcAabbAndQueryTree( *input.m_bodyA, *input.m_bodyB, bTa, input.m_linearTimInfo, *input.m_input, HK_NULL, hitList, HK_MAX_NUM_HITS_PER_AABB_QUERY );

					firstCdBody     = input.m_bodyB;
					firstContainer  = shapeContainerB;
					secondBody		= input.m_bodyA;
					secondContainer = shapeContainerA;
				}
				else
				{
					HK_ASSERT2(0xadb9a762, !shapeAisCollection, "Trying to perform AABB query on a non-bvtree shape.");
					// now shape a is a tree which gets queried
					hkVector4 negLinearTimInfo; negLinearTimInfo.setNeg<3>(input.m_linearTimInfo);
					numHitsFirst = hkBvTreeAgent3::calcAabbAndQueryTree( *input.m_bodyB, *input.m_bodyA, input.m_aTb, negLinearTimInfo, *input.m_input, HK_NULL, hitList, HK_MAX_NUM_HITS_PER_AABB_QUERY );

					firstCdBody     = input.m_bodyA;
					firstContainer  = shapeContainerA;
					secondBody		= input.m_bodyB;
					secondContainer = shapeContainerB;
				}
				HK_ON_SPU( numHitsFirst = hkMath::min2( numHitsFirst, int(HK_MAX_NUM_HITS_PER_AABB_QUERY_USED )));

				// convert hitlist into AABB
#if !defined(HK_PLATFORM_XBOX360) && !defined(HK_PLATFORM_XBOX)
				aabbsFirstSize = WORST_CASE_BUFFER(numHitsFirst + 4, hkpListShape::MAX_CHILDREN_FOR_SPU_MIDPHASE+4);
				aabbsFirst = hkAllocateStack<hk1AxisSweep::AabbInt>(aabbsFirstSize, "aabbsFirst");

				HK_SPU_STACK_POINTER_CHECK();
#endif
				HK_TIMER_SPLIT_LIST("CalcAABBs");
				hkCollectionBvTreeAgent3_calcAabbs( input, firstContainer, firstCdBody, hitList, numHitsFirst, aabbsFirst );

#if !defined(HK_PLATFORM_XBOX360) && !defined(HK_PLATFORM_XBOX)			
#else
				hkDeallocateStack(hitList, HK_MAX_NUM_HITS_PER_AABB_QUERY);
				if (hkShrinkAllocatedStack(aabbsFirst, numHitsFirst + 4))
				{
					aabbsFirstSize = numHitsFirst + 4;
				}
#endif

				// Sort them for the 1 axis sweep
				HK_TIMER_SPLIT_LIST("SortAABBs");
				hkSort(aabbsFirst, numHitsFirst);
			}
			else	// fully extract object A 
			{
				numHitsFirst = HK_ACCESS_COLLECTION_METHOD(shapeContainerA, getNumChildShapes());

				// we may have more cached AABBs (and though some may be invalid) and we need to accommodate memory for all of them
				{
					const hkpCollidable* coll = static_cast<const hkpCollidable*>(input.m_bodyA.val()); // this is an invalid cast, but the first condition of the following if statement checks for that.
					if (!coll->getParent() && coll->m_boundingVolumeData.hasAllocations() && coll->m_boundingVolumeData.isValid())
					{
						numHitsFirst = hkMath::max2(numHitsFirst, int(coll->m_boundingVolumeData.m_numChildShapeAabbs));
					}
				}

				// we also need to take into account the AABB buffer being used to store childInfos too
				int numHitsForAllocate = numHitsFirst;
#if defined (HK_PLATFORM_SPU)
				// on spu we know we're processing a list
				if (!input.m_overrideBodyA)
				{
					HK_ASSERT2(0xad808191, shapeContainerA->getType() == hkcdShapeType::LIST, "Non-list containers are not supported on Spu by the Collection-collection agent.");
					numHitsForAllocate = hkMath::max2(numHitsForAllocate, HK_ACCESS_COLLECTION_METHOD(shapeContainerA, getNumAabbsForSharedBufferForAabbsAndChildInfos()));
				}
#endif
 				
				aabbsFirstSize = WORST_CASE_BUFFER(numHitsForAllocate + 4, hkpListShape::MAX_CHILDREN_FOR_SPU_MIDPHASE+4);
				aabbsFirst = hkAllocateStack<hk1AxisSweep::AabbInt>( aabbsFirstSize, "aabbsFirst" );
				numHitsFirst = hkCollectionBvTreeAgent3_extractCachedAabbsOrRecalculate(input.m_bodyA, input.m_overrideBodyA, shapeContainerA, *input.m_input.val(), aabbsFirst, numHitsFirst, aabbsFirstSize * sizeof(hk1AxisSweep::AabbInt));
				firstCdBody     = input.m_bodyA;
				secondBody		= input.m_bodyB;
				secondContainer = shapeContainerB;
			}

			if (numHitsFirst)	// fully extract the second object
			{
				numHitsSecond = HK_ACCESS_COLLECTION_METHOD(secondContainer, getNumChildShapes());

				// we may have more cached AABBs (and although some may be invalid) and we need to accommodate memory for all of them
				{
					const hkpCollidable* coll = static_cast<const hkpCollidable*>(secondBody); // this is an invalid cast, but the first condition of the following if statement checks for that.
					if (!coll->getParent() && coll->m_boundingVolumeData.hasAllocations() && coll->m_boundingVolumeData.isValid())
					{
						numHitsSecond = hkMath::max2(numHitsSecond, int(coll->m_boundingVolumeData.m_numChildShapeAabbs));
					}
				}

				// we also need to take into account the AABB buffer being used to store childInfos too
				int numHitsForAllocate = numHitsSecond;
#if defined (HK_PLATFORM_SPU)
				// on SPU we know we're processing a list
				HK_ASSERT2(0xad808191, secondContainer->getType() == hkcdShapeType::LIST, "Non-list containers are not supported on Spu by the Collection-collection agent.");
				numHitsForAllocate = hkMath::max2(numHitsForAllocate, HK_ACCESS_COLLECTION_METHOD(secondContainer, getNumAabbsForSharedBufferForAabbsAndChildInfos()));
#endif
				
 				
				aabbsSecondSize = WORST_CASE_BUFFER(numHitsForAllocate + 4, hkpListShape::MAX_CHILDREN_FOR_SPU_MIDPHASE+4);
				HK_ON_CPU(aabbsSecond = hkAllocateStack<hk1AxisSweep::AabbInt>( aabbsSecondSize, "aabbsSecond" ));
				HK_ON_SPU(aabbsSecond = (hk1AxisSweep::AabbInt*)hkSpuMonitorCache::stealMonitorBuffer( sizeof(hk1AxisSweep::AabbInt)* (hkpListShape::MAX_CHILDREN_FOR_SPU_MIDPHASE+4) ) );
				// check for hkpCollectionAgent3
				HK_ASSERT2(0xad634323, (firstCdBody == input.m_bodyA) || (input.m_overrideBodyA == HK_NULL), "When called from the hkpCollectionAgent3::process(), this function can only handle list vs list shapes (no bvTrees).");
				numHitsSecond = hkCollectionBvTreeAgent3_extractCachedAabbsOrRecalculate(secondBody, HK_NULL, secondContainer, *input.m_input.val(), aabbsSecond, numHitsSecond, aabbsSecondSize * sizeof(hk1AxisSweep::AabbInt));
			}
			else
			{
				numHitsSecond = 0;
				aabbsSecondSize = 4;
				HK_ON_CPU(aabbsSecond = hkAllocateStack<hk1AxisSweep::AabbInt>(aabbsSecondSize, "aabbsSecond"));
				HK_ON_SPU(aabbsSecond = (hk1AxisSweep::AabbInt*)hkSpuMonitorCache::stealMonitorBuffer( sizeof(hk1AxisSweep::AabbInt)* (hkpListShape::MAX_CHILDREN_FOR_SPU_MIDPHASE+4) ) );
			}

			// place invalid elements at end
			{for (int i = numHitsFirst; i  < numHitsFirst + 4; i++)		{	aabbsFirst[i].m_min[0]  = hkUint32(-1);	}}
			{for (int i = numHitsSecond; i < numHitsSecond + 4; i++)	{	aabbsSecond[i].m_min[0] = hkUint32(-1);	}}
		}

		//
		// Do 1 axis sweep
		//
		{
			//{	for (int j = 0; j < numHitsFirst; j++ )	{	HK_SPU_DEBUG_PRINTF(("1st Key %i\n", aabbsFirst[j].getKey() ));		}	}
			//{	for (int j = 0; j < numHitsSecond; j++ ){	HK_SPU_DEBUG_PRINTF(("2nd Key %i\n", aabbsSecond[j].getKey() ));		}	}

			HK_ON_CPU(HK_TIMER_SPLIT_LIST("1-Axis"));	// no timers till we have given back the buffers to the hkSpuMonitorCache
			hkPadSpu<int> numPairsSkipped = 0;

				if ( firstCdBody == input.m_bodyA )
				{
					numShapeKeyPairs = hk1AxisSweep::collide(aabbsFirst, numHitsFirst, aabbsSecond, numHitsSecond, (hkKeyPair*)shapeKeyPairs, HK_MAX_NUM_HITS_PER_AABB_QUERY-1, numPairsSkipped);
				}
				else
				{
					numShapeKeyPairs = hk1AxisSweep::collide(aabbsSecond, numHitsSecond, aabbsFirst, numHitsFirst, (hkKeyPair*)shapeKeyPairs, HK_MAX_NUM_HITS_PER_AABB_QUERY-1, numPairsSkipped);
				}

			HK_ON_CPU(hkDeallocateStack(aabbsSecond, aabbsSecondSize));
			HK_ON_SPU(hkSpuMonitorCache::returnStolenMonitorBuffer());
			hkDeallocateStack(aabbsFirst, aabbsFirstSize);

			// Agent NM machine expects sorted pairs
			if ( numShapeKeyPairs )
			{
				HK_TIMER_SPLIT_LIST("SortKeyPairs");
				HK_MONITOR_ADD_VALUE("NumKeyPairs", float(numShapeKeyPairs), HK_MONITOR_TYPE_INT );
				hkSort(shapeKeyPairs, numShapeKeyPairs);
			}

			HK_ASSERT2(0xad9755bc, numShapeKeyPairs < HK_MAX_NUM_HITS_PER_AABB_QUERY, "Num hkpShapeKeyPairs exceeded");
			hkpShapeKeyPair& pair = shapeKeyPairs[numShapeKeyPairs];
			pair.m_shapeKeyA = HK_INVALID_SHAPE_KEY;
			pair.m_shapeKeyB = HK_INVALID_SHAPE_KEY;
		}

		//for (int j = 0; j <= numShapeKeyPairs; j++)	{	HK_SPU_DEBUG_PRINTF(("Pair-%i-%i\n",shapeKeyPairs[j].m_shapeKeyA, shapeKeyPairs[j].m_shapeKeyB));	}
	}
	HK_TIMER_END_LIST();
	return numShapeKeyPairs;
}

hkpAgentData* hkpCollectionCollectionAgent3::process( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4* separatingNormal, hkpProcessCollisionOutput& output)
{
	HK_ON_MONITORS_ENABLED( hkMonitorStream& mStream = hkMonitorStream::getInstance() );

	HK_TIMER_BEGIN_LIST2(mStream, "ProcessCollColl", "Init" );

	//
	// Query the BV tree for key pairs
	//

	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;

	hkpShapeBuffer bufferA;
	hkpShapeBuffer bufferB;
	const HK_SHAPE_CONTAINER* shapeContainerA = hkBvTreeAgent3::getShapeContainerFrom(input.m_bodyA, bufferA);
	const HK_SHAPE_CONTAINER* shapeContainerB = hkBvTreeAgent3::getShapeContainerFrom(input.m_bodyB, bufferB);

	int shapeKeyPairsSize = HK_MAX_NUM_HITS_PER_AABB_QUERY;
	hkpShapeKeyPair* shapeKeyPairs = hkAllocateStack<hkpShapeKeyPair>(shapeKeyPairsSize, "ShapeKeyPairs"); 
	HK_SPU_STACK_POINTER_CHECK();

	HK_TIMER_SPLIT_LIST2( mStream, "query");
	const int numShapeKeyPairs = process_gatherShapeKeys( input, shapeContainerA, shapeContainerB, shapeKeyPairs );

	if ( numShapeKeyPairs < 0 )
	{
		// Error, so skip this agent.
		hkDeallocateStack(shapeKeyPairs, shapeKeyPairsSize);
		HK_TIMER_END_LIST2(mStream);
		return hkAddByteOffset( agentData, HK_THIS_AGENT_SIZE );	// ignore collisions on spu
	}

	// Shrink stack usage for shape keys.
	{
		int total = numShapeKeyPairs + 1;
		
		//HK_ON_CPU(int total = numShapeKeyPairs + 1);	// readjust buffer size
		//HK_ON_SPU(int total = HK_HINT_SIZE16(numHitsFirst) * HK_HINT_SIZE16(numHitsSecond) + 1); // we are doing a worst case approximation here. We are not taking the real hits as this might delay a crash on SPU
		if ( total < HK_MAX_NUM_HITS_PER_AABB_QUERY)
		{
			// if the worst case is smaller than the allocated size, just shrink the used stack
			if (hkShrinkAllocatedStack(shapeKeyPairs, total))
			{
				shapeKeyPairsSize = total;
			}
		}
	}

	//
	// Process the key pairs
	//

	if (input.m_overrideBodyA == HK_NULL)
	{
		HK_TIMER_SPLIT_LIST2( mStream, "process");
#		if ! defined (HK_PLATFORM_SPU)
			hkAgentNmMachine_Process( *agent1nTrack, input, shapeContainerA, shapeContainerB, shapeKeyPairs, output );
#		else
			HK_SPU_STACK_POINTER_CHECK();
			hkAgentNmMachine_Process( HKP_AGENT_MACHINE_MODE_NM, *agent1nTrack, input, shapeContainerA, shapeContainerB, &shapeKeyPairs[0].m_shapeKeyA, output );
#		endif
	}
	else
	{
		HK_TIMER_SPLIT_LIST2( mStream, "shrink");
		// collapse hkpShapeKeyPair list into hkpShapeKey list. 
		hkpShapeKey* HK_RESTRICT shapeKeys = hkAllocateStack<hkpShapeKey>(numShapeKeyPairs+1, "ShapeKeyPairs"); 
		hkpShapeKeyPair* HK_RESTRICT fullShapeKeyPairs = shapeKeyPairs;
		for (int i = 0; i <= numShapeKeyPairs; i++) // include the last HK_INVALID_SHAPE_KEY entry
		{
			shapeKeys[i] = fullShapeKeyPairs[i].m_shapeKeyB;
		}

		hkpAgent3ProcessInput modInput = input;
		modInput.m_bodyA = input.m_overrideBodyA;
		modInput.m_overrideBodyA = HK_NULL;
		HK_TIMER_SPLIT_LIST2( mStream, "process");
		hkAgent1nMachine_Process( *agent1nTrack, modInput, shapeContainerB, shapeKeys, output );
		//HK_TIMER_SPLIT_LIST2( mStream, "dealloc");
		hkDeallocateStack(shapeKeys, numShapeKeyPairs + 1);
	}
	//HK_TIMER_SPLIT_LIST2( mStream, "dealloc");
	hkDeallocateStack(shapeKeyPairs, shapeKeyPairsSize);
	HK_TIMER_END_LIST2(mStream);
	return hkAddByteOffset( agentData, HK_THIS_AGENT_SIZE );
}


#if !defined(HK_PLATFORM_SPU)

void hkpCollectionCollectionAgent3::updateFilter( hkpAgentEntry* entry, hkpAgentData* agentData, const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner )
{
	const hkpShapeCollection* collectionA = static_cast<const hkpShapeCollection*>(bodyA.getShape());
	const hkpShapeCollection* collectionB = static_cast<const hkpShapeCollection*>(bodyB.getShape());

	hkpAgent1nMachine_VisitorInput vin;
	vin.m_bodyA = &bodyA;
	vin.m_collectionBodyB = &bodyB;
	vin.m_input = &input;
	vin.m_contactMgr = mgr;
	vin.m_constraintOwner = &constraintOwner;
	vin.m_containerShapeA = collectionA->getContainer();
	vin.m_containerShapeB = collectionB->getContainer();

	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;

	hkAgentNmMachine_UpdateShapeCollectionFilter( *agent1nTrack, vin );
}

void hkpCollectionCollectionAgent3::invalidateTim( hkpAgentEntry* entry, hkpAgentData* agentData, const hkpCollisionInput& input )
{
	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;
	hkAgent1nMachine_InvalidateTim(*agent1nTrack, input);
}

void hkpCollectionCollectionAgent3::warpTime( hkpAgentEntry* entry, hkpAgentData* agentData, hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;
	hkAgent1nMachine_WarpTime(*agent1nTrack, oldTime, newTime, input);
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
