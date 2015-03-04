/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/CompositeComposite/hknpCompositeCompositeCollisionDetector.h>

#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/ConvexComposite/hknpConvexCompositeCollisionDetector.h>
#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>
#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>
#include <Physics/Physics/Dynamics/Solver/MxJacobianSorter/hknpMxJacobianSorter.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>


void hknpCompositeCompositeCollisionDetector::collideWithChildren(
	const hknpSimulationThreadContext& tl, const hknpInternalCollideSharedData& sharedData,
	hknpCdBody* HK_RESTRICT cdBodyA, const hknpShape* shapeA,
	hknpCdBody* HK_RESTRICT cdBodyB, const hknpShape* shapeB,
	hknpCompositeCollisionCache* HK_RESTRICT compositeCdCache,
	hknpCdCacheStream* HK_RESTRICT childCdCacheStream, hknpCdCacheStream* HK_RESTRICT childCdCacheStreamPpu,	// needed by the consumer
	hknpCdCacheWriter* HK_RESTRICT childCdCacheWriter, hknpMxJacobianSorter* HK_RESTRICT jacMxSorter,
	hknpLiveJacobianInfoWriter* liveJacInfoWriter )
{
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& monitorStream = hkMonitorStream::getInstance();
#endif
	HK_TIMER_BEGIN_LIST2( monitorStream, "CompositeVsComposite", "Init" );

	const hknpModifierFlags enabledModifiers = tl.m_modifierManager->getCombinedBodyRootFlags( tl.m_materials, *cdBodyA, *cdBodyB );

	/*
	
	
	if( 0 && compositeCdCache->m_childCdCacheRange.isEmpty() )
	{
		// we had no hits, so check root level AABB for early out

		HK_TIMER_SPLIT_LIST2( monitorStream, "CheckRootLevelAabb" );

		hkAabb localSpaceAabbBodyA;
		hknpConvexCompositeCollisionDetector::buildExpandedLocalSpaceAabb( sharedData, *cdBodyA, *cdBodyB, &localSpaceAabbBodyA );

		// Check NMP
		if( hknpQueryAabbNmpUtil::checkNmpStillValid( localSpaceAabbBodyA, *compositeCdCache->getQueryAabbNmp(), &compositeCdCache->m_nmpTimeToLive ) )
		{
			BUILD_TIM_AND_EXIT:
			// we had and have no hits, use TIM.
			hkSimdReal dist = hknpQueryAabbNmpUtil::calcTIMWithNmp( localSpaceAabbBodyA, *compositeCdCache->getQueryAabbNmp() );
			hknpMotionUtil::convertDistanceToLinearTIM( *sharedData.m_solverInfo, dist, compositeCdCache->m_linearTim );
			goto END_OF_FUNCTION;
		}

		//
		//	Query AABB
		//

		hknpCollisionQueryContext aabbQueryContext(HK_NULL, HK_NULL);
		aabbQueryContext.m_shapeTagCodec = tl.m_shapeTagCodec;

		hknpAabbQuery aabbQuery;
		aabbQuery.m_aabb = localSpaceAabbBodyA;
		aabbQuery.m_filterData.m_collisionFilterInfo = cdBodyA->m_body->m_collisionFilterInfo;
		aabbQuery.m_filterData.m_materialId          = cdBodyA->m_body->m_materialId;
		aabbQuery.m_filterData.m_userData            = cdBodyA->m_body->m_userData;
		aabbQuery.m_filter                           = tl.m_modifierManager->getCollisionFilter();

		// Shape A and B are both root-level shapes, so no parent shapes available to set here.
		hknpShapeQueryInfo queryShapeInfo;
		queryShapeInfo.m_body		= cdBodyA->m_body;
		queryShapeInfo.m_rootShape	= cdBodyA->m_rootShape;
		queryShapeInfo.m_shapeToWorld = &cdBodyA->m_body->getTransform();

		hknpShapeQueryInfo targetShapeInfo;
		targetShapeInfo.m_body		= cdBodyB->m_body;
		targetShapeInfo.m_rootShape	= cdBodyB->m_rootShape;
		targetShapeInfo.m_shapeToWorld = &cdBodyB->m_body->getTransform();

		hknpQueryFilterData targetShapeFilterData(*cdBodyB->m_body);

		// Check for any hits
		
		hkInplaceArray<hknpShapeKey, 1> hits;
		int numHits = hknpConvexCompositeCollisionDetector::queryAabbWithNmp(
			&aabbQueryContext, aabbQuery, queryShapeInfo,
			*cdBodyB->m_rootShape, targetShapeInfo, targetShapeFilterData,
			&hits, compositeCdCache->getQueryAabbNmp() );

		
		if ( tl.m_modifierManager->isFunctionRegistered( hknpModifier::FUNCTION_POST_COMPOSITE_QUERY_AABB, enabledModifiers ))
		{
			int newSize = hits.getSize();
			HKNP_FIRE_MODIFIER( tl.m_modifierManager, hknpModifier::FUNCTION_POST_COMPOSITE_QUERY_AABB, enabledModifiers,
				newSize = modifier->postCompositeQueryAabb( tl, sharedData, aabbQuery, queryShapeInfo, targetShapeInfo, hits.begin(), hits.getSize(), hits.getCapacity() ) );
			hits.setSize(newSize);
		}

		hits.clearAndDeallocate();
		if ( numHits == 0 )
		{
			goto BUILD_TIM_AND_EXIT;
		}

		HK_TIMER_SPLIT_LIST2( monitorStream, "Init" );
		compositeCdCache->m_linearTim = 0;
	}
	*/

	// Calculate expansion parameters
	hkVector4 linExpansion0inB;	// velocity of object A in B-space
	hkVector4 linExpansion1inB;	// velocity of object A in B-space
	hkSimdReal angExpansion;
	hkAabb expansionAabb;		// used if B is static
	{
		hkSimdReal collisionTolerance = sharedData.m_solverInfo->m_collisionTolerance * hkSimdReal_Inv2;

		hkSimdReal linExpansion;
		hknpMotionUtil::calcSweepExpansion(
			cdBodyA->m_body, *cdBodyA->m_motion, collisionTolerance, *sharedData.m_solverInfo,
			&linExpansion0inB, &linExpansion1inB, &angExpansion, &linExpansion );

		linExpansion0inB._setRotatedInverseDir( cdBodyB->m_body->getTransform().getRotation(), linExpansion0inB );
		linExpansion1inB._setRotatedInverseDir( cdBodyB->m_body->getTransform().getRotation(), linExpansion1inB );

		expansionAabb.m_min.setZero();
		expansionAabb.m_max.setZero();
		if( cdBodyA->m_quality->m_requestedFlags.anyIsSet( hknpBodyQuality::ENABLE_NEIGHBOR_WELDING ) )
		{
			hkAabbUtil::expandAabbByMotionCircle( expansionAabb, linExpansion0inB, linExpansion1inB, angExpansion, expansionAabb );
		}
		else
		{
			hkAabbUtil::expandAabbByMotion( expansionAabb, linExpansion0inB, linExpansion1inB, angExpansion, expansionAabb );
		}
	}

	//
	// Collide
	//

	const hkBool32 BisDynamic = cdBodyB->m_motion->isDynamic();

	hknpCollisionQueryContext aabbQueryContext( HK_NULL, HK_NULL );
	aabbQueryContext.m_shapeTagCodec = tl.m_shapeTagCodec;

	hknpCdCacheConsumer childCdCacheConsumer;
	childCdCacheConsumer.initSpu( HKNP_SPU_DMA_GROUP_STALL, 1, "CompositeCompositeCacheConsumer" );
	childCdCacheConsumer.setToRange( tl.m_heapAllocator, childCdCacheStream, childCdCacheStreamPpu, &compositeCdCache->m_childCdCacheRange );

	const hknpCollisionCache* HK_RESTRICT srcCache = childCdCacheConsumer.access();
	HK_ASSERT( 0xf03dcaad, !srcCache || srcCache->m_type == hknpCollisionCacheType::SET_SHAPE_KEY_A );

	// Get all shape keys of A (up to a hard limit)
	HK_ON_CPU( hkFixedCapacityArray<hknpShapeKeyPath> shapeKeyPathsA( 1024, "shapeKeyPathsA" ) );
	HK_ON_SPU( hkFixedCapacityArray<hknpShapeKeyPath> shapeKeyPathsA(  256, "shapeKeyPathsA" ) );
	{
		hknpShapeKeyPath root;
#if !defined(HK_PLATFORM_SPU)
		cdBodyA->m_rootShape->getAllShapeKeys( root, HK_NULL, &shapeKeyPathsA );
#else
		const int shapeBufferSize = HKNP_SHAPE_BUFFER_SIZE + HKNP_COMPOUND_HIERARCHY_BUFFER;
		HK_ALIGN16( hkUint8 ) shapeBuffer[ shapeBufferSize ];
		cdBodyA->m_rootShape->getAllShapeKeys( root, HK_NULL, shapeBuffer, shapeBufferSize, &shapeKeyPathsA );
#endif
	}

	hknpShapeCollector leafShapeCollector( tl.m_triangleShapePrototypes[0] );

	HK_TIMER_SPLIT_LIST2( monitorStream, "Loop" );

	// Allocate space for queryAabb() results.
	// Use an inplace array for the hits so that they are stack allocated, but can use the heap if more capacity is needed.
	// On SPU, queryAabb() implementations should never try to use the heap, dropping hits instead.
	hkInplaceArray< hknpShapeKey, 1024 > hits;

	// Allocate space for the manifolds that will be constructed
	hknpManifold* manifolds = hkAllocateStack<hknpManifold>( HKNP_MAX_NUM_MANIFOLDS_PER_BATCH, "manifoldsBuffer" );
#if defined (HK_ENABLE_DETERMINISM_CHECKS) /* || defined (HK_DEBUG) */
	hkString::memSet( manifolds, 0xcd, HKNP_MAX_NUM_MANIFOLDS_PER_BATCH * sizeof(hknpManifold) );
#endif

	//
	// Narrow phase collision detection.
	// Iterate over all pairs by grouping the pairs using the shape key A.
	//
	for( int sAi = 0; sAi < shapeKeyPathsA.getSize(); sAi++ )
	{
		const hknpShapeKeyPath& hitShapeKeyPath = shapeKeyPathsA[sAi];
		const hknpShapeKey hitShapeKey = hitShapeKeyPath.getKey();
		const int shapeKeyLengthA = hitShapeKeyPath.getKeySize();

		hknpShapeKey shapeKeyA = hitShapeKey;
		hknpSetShapeKeyACollisionCache* HK_RESTRICT dstCache;
		while( srcCache )	// search a source cache which matches hitShapeKey
		{
			HK_ASSERT( 0xf0793dd, srcCache->m_type == hknpCollisionCacheType::SET_SHAPE_KEY_A );
			shapeKeyA = srcCache->getShapeKey();
			if( shapeKeyA == hitShapeKey )
			{
				// reuse existing cache
				goto REUSE_EXISTING_CHILD_CACHE;
			}
			else if( shapeKeyA < hitShapeKey )
			{
				// key no longer present, so skip this and its child caches
				const hknpSetShapeKeyACollisionCache* srcCCCCache = static_cast<const hknpSetShapeKeyACollisionCache*>(srcCache);	// already checked
				int numCaches = srcCCCCache->m_numHitsShapeKeyA;

				// consume hknpChildCompositeCompositeCdCache
				srcCache = childCdCacheConsumer.consumeAndAccessNext( srcCache->getSizeInBytes() );

				// consume children
				for( int cvxI = 0; cvxI < numCaches; cvxI++ )
				{
					HK_ASSERT( 0xf034df46, srcCache->m_type == hknpCollisionCacheType::CONVEX_CONVEX );
					const hknpConvexConvexCollisionCache* srcCvxCache = static_cast<const hknpConvexConvexCollisionCache*>(srcCache);
					srcCvxCache->_destructCdCacheImpl( tl, sharedData, *cdBodyA, *cdBodyB, hknpCdCacheDestructReason::AABBS_DONT_OVERLAP );
					srcCache = childCdCacheConsumer.consumeAndAccessNext( srcCache->getSizeInBytes() );	// consume hknpConvexConvexCollisionCache
				}
				HK_ASSERT( 0xf034df47, !srcCache || srcCache->m_type == hknpCollisionCacheType::SET_SHAPE_KEY_A );

				continue;	// while ( srcCache )
			}
			// else falls through to // create new collision cache

			break;
		}	// while(srcCache)

		// Create a new collision cache
		{
			dstCache = (hknpSetShapeKeyACollisionCache*)childCdCacheWriter->reserve( sizeof(hknpSetShapeKeyACollisionCache) );
			dstCache->init( shapeKeyA );
			dstCache->m_collisionFilterInfo	= cdBodyA->m_body->m_collisionFilterInfo;
			dstCache->m_materialIdA			= cdBodyA->m_body->m_materialId;
			dstCache->m_tim					= 0;

			// Call the shape tag codec.
			
			{
				const hknpShape* compShapeA = cdBodyA->m_rootShape;
				leafShapeCollector.reset( cdBodyA->m_body->getTransform() );
				compShapeA->getLeafShape( shapeKeyA, &leafShapeCollector );

				hknpShapeTagCodec::Context targetShapeTagContext;
				{
					targetShapeTagContext.m_queryType			= hknpCollisionQueryType::QUERY_AABB;
					targetShapeTagContext.m_body				= cdBodyA->m_body;
					targetShapeTagContext.m_rootShape			= cdBodyA->m_rootShape;
					targetShapeTagContext.m_parentShape			= leafShapeCollector.m_parentShape;
					targetShapeTagContext.m_shapeKey			= shapeKeyA;
					targetShapeTagContext.m_shape				= leafShapeCollector.m_shapeOut;
					// At this stage we haven't resolved shapeB yet. Hence we can only pass in the root level shape.
					targetShapeTagContext.m_partnerBody			= cdBodyB->m_body;
					targetShapeTagContext.m_partnerRootShape	= cdBodyB->m_rootShape;
					targetShapeTagContext.m_partnerShapeKey		= HKNP_INVALID_SHAPE_KEY;
					targetShapeTagContext.m_partnerShape		= HK_NULL;
				}

				hkUint64 dummyUserData;
				tl.m_shapeTagCodec->decode(
					leafShapeCollector.m_shapeTagPath.begin(), leafShapeCollector.m_shapeTagPath.getSize(),
					leafShapeCollector.m_shapeTagOut, &targetShapeTagContext,
					&dstCache->m_collisionFilterInfo, &dstCache->m_materialIdA, &dummyUserData );
			}

			goto COLLIDE_CVX_COMPOSITE_PAIR;
		}

REUSE_EXISTING_CHILD_CACHE:
		// Copy source cache to destination cache
		{
			dstCache = (hknpSetShapeKeyACollisionCache*)childCdCacheWriter->reserve( sizeof(hknpSetShapeKeyACollisionCache) );
			hkString::memCpy16<sizeof(hknpSetShapeKeyACollisionCache)>( dstCache, srcCache );
			childCdCacheConsumer.consumeAndAccessNext( sizeof(hknpSetShapeKeyACollisionCache) );

			const hknpShape* compShapeA = cdBodyA->m_rootShape;
			leafShapeCollector.reset( cdBodyA->m_body->getTransform() );
			compShapeA->getLeafShape( shapeKeyA, &leafShapeCollector );

			HK_ON_DEBUG( srcCache = HK_NULL );
		}

COLLIDE_CVX_COMPOSITE_PAIR:
		// Collide a single convex shape with a composite shape
		{
			const hknpShape* cvxChildShapeA = leafShapeCollector.m_shapeOut;

			HK_ASSERT2( 0x6b562430,
				leafShapeCollector.m_scaleOut.equal( hkVector4::getConstant<HK_QUADREAL_1>() ).allAreSet<hkVector4ComparisonMask::MASK_XYZ>(),
				"Non-identity scale returned from getLeafShape(). This is not supported by collision detection." );

			int numKeys = dstCache->m_numHitsShapeKeyA;
			hknpShapeKey* keys = HK_NULL;
			int keysStriding = 0;

			const int deltaDistCombined = cdBodyA->m_body->m_maxTimDistance + cdBodyB->m_body->m_maxTimDistance;
			const int newLinearTim = dstCache->m_linearTim - deltaDistCombined;
			if( newLinearTim > deltaDistCombined )		// use current velocity twice
			{
				dstCache->m_linearTim = hknpLinearTimType(newLinearTim);
				if( numKeys == 0 )	// now we had no and have no collisions, we can simply skip recursive calls.
				{
					//HK_TIMER_SPLIT_LIST2( monitorStream, "TimEarlyOut");
					childCdCacheWriter->advance( sizeof(hknpSetShapeKeyACollisionCache) );
					goto CONTINUE_WITH_NEXT_SRC_CACHE;
				}
			}
			else
			{
				HK_TIMER_SPLIT_LIST2( monitorStream, "QueryAabbNmp" );

				hknpAabbQuery aabbQuery;

				// Calculate expanded AABB
				{
					hkAabb aabb;
					hkTransform bTa_child; bTa_child.setMulInverseMul( cdBodyB->m_body->getTransform(), leafShapeCollector.m_transformOut );
					cvxChildShapeA->calcAabb( bTa_child, aabb );
					HK_ASSERT( 0xbf2534a2, aabb.isValid() );

					if( !BisDynamic )
					{
						aabbQuery.m_aabb.m_min.setAdd( aabb.m_min, expansionAabb.m_min );
						aabbQuery.m_aabb.m_max.setAdd( aabb.m_max, expansionAabb.m_max );
					}
					else
					{
						hkVector4 comA = cdBodyA->m_motion->getCenterOfMassInWorld();
						hkVector4 velB; cdBodyB->m_motion->_getPointVelocity( comA, velB );
						hkVector4 velBinB; velBinB._setRotatedInverseDir( cdBodyB->m_body->getTransform().getRotation(), velB );
						hkVector4 movedDistBinB;movedDistBinB.setMul( sharedData.m_solverInfo->m_deltaTime, velBinB );

						hkVector4 relLin0; relLin0.setSub( linExpansion0inB, movedDistBinB );
						hkVector4 relLin1; relLin1.setSub( linExpansion1inB, movedDistBinB );

						hkAabbUtil::expandAabbByMotion( aabb, relLin0, relLin1, angExpansion, aabbQuery.m_aabb );
					}
				}

				// Do the query
				if( hknpQueryAabbNmpUtil::checkNmpStillValid( aabbQuery.m_aabb, *dstCache->getQueryAabbNmp(), &dstCache->m_nmpTimeToLive) )
				{
					if( numKeys == 0 )	// now we had no and have no collisions, we can simply skip recursive calls.
					{
						childCdCacheWriter->advance( sizeof(hknpSetShapeKeyACollisionCache) );
						goto CONTINUE_WITH_NEXT_SRC_CACHE;
					}
				}
				else
				{
					aabbQuery.m_filter = tl.m_modifierManager->getCollisionFilter();

					
					aabbQuery.m_filterData.setFromBody( *cdBodyA->m_body );

					hknpQueryFilterData targetShapeFilterData( *cdBodyB->m_body );

					hknpShapeQueryInfo queryShapeInfo;
					{
						queryShapeInfo.m_body			= cdBodyA->m_body;
						queryShapeInfo.m_rootShape		= cdBodyA->m_rootShape;
						queryShapeInfo.m_parentShape	= leafShapeCollector.m_parentShape;
						queryShapeInfo.m_shapeKeyPath	. setFromKey(shapeKeyA, shapeKeyLengthA);
						queryShapeInfo.m_shapeToWorld	= &leafShapeCollector.m_transformOut;
					}

					hknpShapeQueryInfo targetShapeInfo;
					{
						targetShapeInfo.m_body			= cdBodyB->m_body;
						targetShapeInfo.m_rootShape		= cdBodyB->m_rootShape;
						targetShapeInfo.m_shapeToWorld	= &cdBodyB->m_body->getTransform();
					}

					// Reset the hits array (since it can contain previous results from queryAabbWithNmp)
					hits.setSize(0);

					// Do the query
					numKeys = hknpConvexCompositeCollisionDetector::queryAabbWithNmp(
						&aabbQueryContext, aabbQuery, queryShapeInfo,
						*cdBodyB->m_rootShape, targetShapeInfo, targetShapeFilterData,
						&hits, dstCache->getQueryAabbNmp() );

					
					if( tl.m_modifierManager->isFunctionRegistered( hknpModifier::FUNCTION_POST_COMPOSITE_QUERY_AABB, enabledModifiers ) )
					{
						int newSize = hits.getSize();
						HKNP_FIRE_MODIFIER( tl.m_modifierManager, hknpModifier::FUNCTION_POST_COMPOSITE_QUERY_AABB, enabledModifiers,
							newSize = modifier->postCompositeQueryAabb( tl, sharedData, aabbQuery, queryShapeInfo, targetShapeInfo, hits.begin(), hits.getSize(), hits.getCapacity() ) );
						hits.setSize( newSize );
					}

					keys = hits.begin();
					keysStriding = sizeof(hknpShapeKey);
					dstCache->m_numHitsShapeKeyA = numKeys;

					// Update TIM
					hkSimdReal dist = hknpQueryAabbNmpUtil::calcTIMWithNmp( aabbQuery.m_aabb, *dstCache->getQueryAabbNmp() );
					hknpMotionUtil::convertDistanceToLinearTIM( *sharedData.m_solverInfo, dist, dstCache->m_linearTim );
				}
			}

			cdBodyA->m_collisionFilterInfo	= dstCache->m_collisionFilterInfo;
			cdBodyA->m_material				= &tl.m_materials[ dstCache->m_materialIdA.value() ];
			cdBodyA->m_shapeKey				= shapeKeyA;
			cdBodyA->m_leafShape			= leafShapeCollector.m_shapeOut;
			cdBodyA->m_transform			= &leafShapeCollector.m_transformOut;

			// Now we are finished using our dstCache, so we can advance the writer.
			// As a result the childCdCacheWriter can be used to place the child cvxCvx caches.
			childCdCacheWriter->advance( sizeof(hknpSetShapeKeyACollisionCache) );
			HK_ON_DEBUG( dstCache = HK_NULL );

			// This call will collideWithChildren the CURRENT leaf shape of body A with ALL leaf shape candidates from body B
			hknpConvexCompositeCollisionDetector::collideConvexWithCompositeKeys(
				tl, sharedData,
				*cdBodyA, cdBodyB,
				keys, keysStriding, numKeys,
				&childCdCacheConsumer, childCdCacheWriter, manifolds, jacMxSorter, liveJacInfoWriter
				);
		}

CONTINUE_WITH_NEXT_SRC_CACHE:
		srcCache = childCdCacheConsumer.access();
	}	// for sAi

	hkDeallocateStack( manifolds, HKNP_MAX_NUM_MANIFOLDS_PER_BATCH );
	hits.clearAndDeallocate();
	shapeKeyPathsA.clearAndDeallocate();
	childCdCacheConsumer.exitSpu();

#if 0
	END_OF_FUNCTION:
#endif

	jacMxSorter->resetHintUsingSameBodies();

	HK_TIMER_END_LIST2( monitorStream );
}


void hknpSetShapeKeyACdDetector::destructCollisionCache(
		const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
		hknpCollisionCache* cacheToDestruct,
		hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,
		hknpCdBodyBase* HK_RESTRICT cdBodyA, hknpCdBodyBase* HK_RESTRICT cdBodyB,
		hknpCdCacheDestructReason::Enum reason )
{
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
