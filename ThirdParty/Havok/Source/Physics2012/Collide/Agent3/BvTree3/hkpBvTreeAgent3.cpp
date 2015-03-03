/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>

#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>

#include <Physics2012/Collide/Agent3/BvTree3/hkpBvTreeAgent3.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nTrack.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Physics2012/Collide/Filter/hkpConvexListFilter.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#include <Physics2012/Collide/Agent/CompoundAgent/ShapeCollection/hkpShapeCollectionAgent.h>

#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/HeightField/TriSampledHeightField/hkpTriSampledHeightFieldBvTreeShape.h>
#include <Physics2012/Collide/Shape/HeightField/TriSampledHeightField/hkpTriSampledHeightFieldCollection.h>
#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>
#include <Physics2012/Collide/Agent3/List3/hkpListAgent3.h>

#include <Physics2012/Dynamics/Collide/Deprecated/Dispatch/hkpCollideCallbackDispatcher.h>

#include <Physics2012/Collide/Agent3/Machine/Midphase/hkpMidphaseAgentData.h>
#include <Physics2012/Collide/Agent3/Machine/Midphase/hkpShapeKeyTrack.h>

#if defined(HK_PLATFORM_SPU)
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>
#endif

#ifdef HK_DEBUG
//#	define HK_BV_TREE_DISPLAY_AABB
//#	define HK_DISPLAY_TRIANGLES
#endif

#if defined(HK_BV_TREE_DISPLAY_AABB) || defined( HK_DISPLAY_TRIANGLES )
#	include <Common/Visualize/hkDebugDisplay.h>
#endif

#define HK_THIS_AGENT_SIZE HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpMidphaseAgentData) )
HK_COMPILE_TIME_ASSERT(HK_THIS_AGENT_SIZE <= hkAgent3::MAX_NET_SIZE);

void hkBvTreeAgent3::initAgentFunc(hkpCollisionDispatcher::Agent3Funcs& f)
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
void hkBvTreeAgent3::registerAgent3(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::Agent3Funcs f;
	initAgentFunc(f);
	dispatcher->registerAgent3( f,  hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::BV_TREE );	
}
#endif


hkpAgentData* hkBvTreeAgent3::process( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4* separatingNormal, hkpProcessCollisionOutput& output)
{
	HK_TIMER_BEGIN_LIST( "BvTree3", "QueryTree" );

	hkLocalBuffer<hkpShapeKey> hitList( HK_MAX_NUM_HITS_PER_AABB_QUERY );

	HK_SPU_STACK_POINTER_CHECK(); // make sure that the allocation didn't cross into the program stack.

	int numHits;
	{
		hkTransform bTa;	bTa.setInverse( input.m_aTb );
		numHits = calcAabbAndQueryTree( *input.m_bodyA, *input.m_bodyB, bTa, input.m_linearTimInfo, *input.m_input, HK_NULL, hitList.begin(), HK_MAX_NUM_HITS_PER_AABB_QUERY);
		HK_ASSERT2 (0xf0432345, numHits<0 || hitList[ numHits ] == HK_INVALID_SHAPE_KEY, "Your result from queryAabb deleted the HK_INVALID_SHAPE_KEY entry" );
		hkSort( hitList.begin(), numHits );	// 1N machine expects sorted keys
	}
	
	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;
	hkpShapeBuffer buffer;
	const HK_SHAPE_CONTAINER* shapeContainer = hkBvTreeAgent3::getShapeContainerFrom(input.m_bodyB, buffer);

	HK_TIMER_SPLIT_LIST("Narrow");

	hkpShapeKey* hits = (numHits >= 0) ? hitList.begin() : HK_NULL;
	hkAgent1nMachine_Process( *agent1nTrack, input, shapeContainer, hits, output );
	
	HK_TIMER_END_LIST();

	return hkAddByteOffset( agentData, HK_THIS_AGENT_SIZE );
}


#if !defined(HK_PLATFORM_SPU)

void hkBvTreeAgent3::updateFilter( hkpAgentEntry* entry, hkpAgentData* agentData, const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner )
{
	// Check that it is a BV tree shape
	HK_ASSERT(0xf031ed45, bodyB.getShape()->isBvTree());

	const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>( bodyB.getShape() );
	const hkpShapeContainer* shapeContainerB = bvB->getContainer();

	hkpAgent1nMachine_VisitorInput vin;
	vin.m_bodyA = &bodyA;
	vin.m_collectionBodyB = &bodyB;
	vin.m_input = &input;
	vin.m_contactMgr = mgr;
	vin.m_constraintOwner = &constraintOwner;
	vin.m_containerShapeB = shapeContainerB;

	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;

	hkAgent1nMachine_UpdateShapeCollectionFilter( *agent1nTrack, vin );
}

void hkBvTreeAgent3::invalidateTim( hkpAgentEntry* entry, hkpAgentData* agentData, const hkpCollisionInput& input )
{
	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;
	hkAgent1nMachine_InvalidateTim(*agent1nTrack, input);
}

void hkBvTreeAgent3::warpTime( hkpAgentEntry* entry, hkpAgentData* agentData, hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;
	hkAgent1nMachine_WarpTime(*agent1nTrack, oldTime, newTime, input);
}


#endif
////////////////////////////////////////////////////////////////////////////
//// Calc the AABB extruded by the relative movement
////
//hkAabb aabb;

//// get the AABB from the collidable and orient it in local space
//hkAabbUint32& aabbUint32 = bodyA->getRootCollidable()->m_boundingVolumeData.getAabbUint32();
//if ( input.m_collisionQualityInfo->m_useContinuousPhysics )
//{
//	hkAabbUtil::uncompressExpandedAabbUint32(aabbUint32, aabbUint32);
//}
//hkVector4Util::convertAabbFromUint32(aabbUint32, input.m_aabb32Info.m_bitOffsetLow, input.m_aabb32Info.m_bitScale, aabb);

////
//// Get the AABB, into bodyB's local space ...we're lazy, so we'll just use what we have ...
//hkVector4 aabbCenter; aabbCenter.setAdd4(aabb.m_min, aabb.m_max); aabbCenter.mul4(0.5f);
//hkVector4 aabbHalfSize; aabbHalfSize.setSub4(aabb.m_max, aabb.m_min); aabbHalfSize.mul4(0.5f);

//hkTransform wTbox; 
//wTbox.getRotation().setIdentity();
//wTbox.getTranslation() = aabbCenter;
//hkTransform t; t.setMulInverseMul(bodyB->getTransform(), wTbox);

////
//// Calc AABB in MOPP's (bodyB) space
//hkpBoxShape boxShape(aabbHalfSize, 0.0f);
//boxShape->getAabb(t, 0.0f, aabb);

//hkVector4 aabbExtents; 
//aabbExtents.setSub4( aabb.m_max, aabb.m_min );

////////////////////////////////////////////////////////////////////////////
int hkBvTreeAgent3::calcAabbAndQueryTree( const hkpCdBody& bodyA,	const hkpCdBody& bodyB, const hkTransform& bTa,
											  const hkVector4& linearTimInfo, const hkpProcessCollisionInput& input,
											  hkAabb* cachedAabb, hkpShapeKey* hitListOut, int hitListCapacity )
{

	//
	// Calc the AABB extruded by the relative movement
	//
	hkAabb aabb;
#ifdef HK_BV_TREE_DISPLAY_AABB
	hkAabb baseAabb;
#endif
	{
		//added an early out so if the AABB is the same, don't query the MOPP and don't sort, nor call the dispatch/agent
		{
			hkSimdReal inputTol; inputTol.load<1>(&(input.m_tolerance.ref())); inputTol.mul(hkSimdReal_Half);
			const hkMotionState* msA = bodyA.getMotionState();
			const hkMotionState* msB = bodyB.getMotionState();

				// if using continuous physics, expand the AABB backwards
				// rotate tim into the bvTree space
			hkVector4 timInfo;	timInfo._setRotatedInverseDir( bodyB.getTransform().getRotation(), linearTimInfo );


			hkVector4 aabbExtents; 
			if ( input.m_collisionQualityInfo->m_useContinuousPhysics.val() )
			{
				hkSimdReal objectRadiusA; objectRadiusA.load<1>(&msA->m_objectRadius);
				hkSimdReal objectRadiusB; objectRadiusB.load<1>(&msB->m_objectRadius);
				const hkSimdReal deltaAngleB = msB->m_deltaAngle.getW();

					// object A rotates within object B with the diff of both angular velocities
				const hkSimdReal secOrdErrA = (msA->m_deltaAngle.getW() + deltaAngleB) * objectRadiusA;

					// The angular velocity gets correctly calculated into the trajectory of object A
					// we only need to calculate the maximum error. So we use the square of the error B
				const hkSimdReal secOrdErrB = deltaAngleB * deltaAngleB * objectRadiusB;

				const hkSimdReal checkEpsilon = inputTol + secOrdErrA + secOrdErrB;
				bodyA.getShape()->getAabb( bTa, checkEpsilon.getReal(), aabb );

				// restrict the size of the AABB to the worst case radius size
				hkVector4 massCenterAinB;
				{
					hkVector4 radius4; radius4.setAll( objectRadiusA + inputTol + secOrdErrB ); radius4.zeroComponent<3>();
					massCenterAinB._setTransformedInversePos(bodyB.getTransform(), msA->getSweptTransform().m_centerOfMass1 );
					hkVector4 maxR; maxR.setAdd( massCenterAinB, radius4 );
					hkVector4 minR; minR.setSub( massCenterAinB, radius4 );
					aabb.m_min.setMax( aabb.m_min, minR );
					aabb.m_max.setMin( aabb.m_max, maxR );
				}

				// export the size of the base AABB
				aabbExtents.setSub( aabb.m_max, aabb.m_min );

				// expand the AABB backwards
				{
					// correct the timInfo if we have a rotating tree
					if (msB->m_deltaAngle.getComponent<3>().isGreaterZero())
					{
						hkVector4 relPos; relPos.setSub( massCenterAinB, msB->getSweptTransform().m_centerOfMassLocal );
						hkVector4 offsetOut; offsetOut.setCross( relPos, msB->m_deltaAngle );
						hkSimdReal f; f.load<1>(&(input.m_stepInfo.m_deltaTime.ref())); f.mul(msB->getSweptTransform().getInvDeltaTimeSr());
						timInfo.addMul( f, offsetOut );
					}

					hkVector4 zero;		zero.setZero();
					hkVector4 minPath; 	minPath.setMin( zero, timInfo );
					hkVector4 maxPath;	maxPath.setMax( zero, timInfo );

#ifdef HK_BV_TREE_DISPLAY_AABB
					baseAabb = aabb;
					//baseAabb.m_min.add( timInfo );
					//baseAabb.m_max.add( timInfo );
#endif
					
					aabb.m_min.add( minPath );
					aabb.m_max.add( maxPath );
				}
			}
			else
			{
				const hkReal checkEpsilon = input.m_tolerance * 0.5f;
				bodyA.getShape()->getAabb( bTa, checkEpsilon, aabb );
				aabbExtents.setSub( aabb.m_max, aabb.m_min );
#ifdef HK_BV_TREE_DISPLAY_AABB
				baseAabb = aabb;
#endif
			}

			//
			//	Try to do some AABB caching to reduce the number of calls to the bounding volume structure
			//
			if (cachedAabb)
			{
				if ( cachedAabb->contains( aabb ))
				{
					return -1;
				}

				hkVector4 zero; zero.setZero();
				hkVector4 minPath;minPath.setMin( zero, timInfo );
				hkVector4 maxPath;maxPath.setMax( zero, timInfo );


				// expand AABB so we have a higher chance of a hit next frame
				// we expand it by half of our tolerance
				hkVector4 expand4; expand4.setAll( inputTol ); expand4.zeroComponent<3>();
				aabb.m_min.sub( expand4 );
				aabb.m_max.add( expand4 );

				// expand along our path linearly at least 2 frames ahead
				// but a maximum of 40% of the original AABB
				const hkSimdReal maxExpand = hkSimdReal::fromFloat(0.4f);
				const hkSimdReal framesLookAhead = -hkSimdReal_2;

				hkVector4 minExtentPath; minExtentPath.setMul( framesLookAhead, maxPath );
				hkVector4 maxExtentPath; maxExtentPath.setMul( framesLookAhead, minPath );

				hkVector4 maxExpand4; maxExpand4.setMul( maxExpand, aabbExtents );
				maxExtentPath.setMin( maxExtentPath, maxExpand4 );
				hkVector4 minExpand4; minExpand4.setNeg<4>(maxExpand4);
				minExtentPath.setMax( minExtentPath, minExpand4 );

				aabb.m_min.add( minExtentPath );
				aabb.m_max.add( maxExtentPath );
				*cachedAabb = aabb;
			}
		}
	}

	//
	// display the AABB and the cached AABB
	//
#ifdef HK_BV_TREE_DISPLAY_AABB
	{
		hkAabb* bb = &baseAabb; 
		hkColor::Argb color = hkColor::YELLOW;
		for ( int a = 0; a < 2; a ++)
		{
			for ( int x = 0; x < 2; x ++ )
			{	for ( int y = 0; y < 2; y ++ )
				{	for ( int z = 0; z < 2; z ++ )
					{
						hkVector4 a; a.set( (&bb->m_min)[x](0), (&bb->m_min)[y](1), (&bb->m_min)[z](2) );
						a.setTransformedPos( bodyB.getTransform(), a );
						hkVector4 b;

						b.set( (&bb->m_min)[1-x](0), (&bb->m_min)[y](1), (&bb->m_min)[z](2) );
						b.setTransformedPos( bodyB.getTransform(), b );
						HK_DISPLAY_LINE( a, b, color );
						b.set( (&bb->m_min)[x](0), (&bb->m_min)[1-y](1), (&bb->m_min)[z](2) );
						b.setTransformedPos( bodyB.getTransform(), b );
						HK_DISPLAY_LINE( a, b, color );
						b.set( (&bb->m_min)[x](0), (&bb->m_min)[y](1), (&bb->m_min)[1-z](2) );
						b.setTransformedPos( bodyB.getTransform(), b );
						HK_DISPLAY_LINE( a, b, color );
			}	}	}
			color = hkColor::BLUE;
			bb = cachedAabb;
			if (!bb) 
			{
				break;
			}
		}
	}
#endif

	//
	// Query the BV tree shape using the calculated AABB
	//

	int numHits;
	{
		const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>(bodyB.getShape());
		numHits = bvB->queryAabb( aabb, hitListOut, hitListCapacity );	
	}

#if defined(HK_PLATFORM_SPU)
	HK_ASSERT2(0xad675aa, numHits <= hitListCapacity, "Output buffer overflown.");
#endif

#if defined(HK_DEBUG)
	if ( numHits > 256)
	{
		HK_WARN(0xad345a23, "Peformance warning: hkpBvTreeShape::queryAabb() returned more than 256 hkpShapeKey hits.");
	}
#endif

	if (numHits >= hitListCapacity)
	{
		const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>( bodyB.getShape() );
#	if !defined (HK_PLATFORM_SPU)
		numHits = input.m_filter->numShapeKeyHitsLimitBreached(input, bodyA, bodyB, bvB, aabb, hitListOut, hitListCapacity);
#	else
		numHits = g_NumShapeKeyHitsLimitBreached(&input, &bodyA, &bodyB, bvB, aabb, hitListOut, hitListCapacity);
#	endif
		numHits = hkMath::min2(numHits, hitListCapacity-1);
	}
	
	// Set end marker
	hitListOut[ numHits ] = HK_INVALID_SHAPE_KEY;

	return numHits;
}



const hkpShapeCollection* HK_CALL hkBvTreeAgent3::getShapeCollectionIfBvTreeSupportsAabbQueries(const hkpCdBody* body, hkpShapeBuffer& buffer)
{
	const hkpBvTreeShape* bvt = static_cast<const hkpBvTreeShape*>( body->getShape() );

	switch ( bvt->getType() )
	{
	case hkcdShapeType::MOPP:
		{	
			const hkpMoppBvTreeShape* moppBvTree = (const hkpMoppBvTreeShape*)(bvt);
#if !defined (HK_PLATFORM_SPU)
			return moppBvTree->hkpMoppBvTreeShape::getShapeCollection();
#else
			return moppBvTree->hkpMoppBvTreeShape::getShapeCollectionFromPpu(buffer);
#endif
		}
	case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE:
		{
			const hkpTriSampledHeightFieldBvTreeShape* triSampled = (const hkpTriSampledHeightFieldBvTreeShape*)(bvt);
#if !defined (HK_PLATFORM_SPU)
			return triSampled->hkpTriSampledHeightFieldBvTreeShape::getShapeCollection();
#else
			return triSampled->hkpTriSampledHeightFieldBvTreeShape::getShapeCollectionFromPpu(buffer);
#endif
		}
	default:
		{	
			return HK_NULL;
		}	
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
