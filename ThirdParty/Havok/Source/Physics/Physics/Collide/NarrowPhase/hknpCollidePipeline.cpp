/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/NarrowPhase/hknpCollidePipeline.h>

#include <Common/Visualize/Container/CommandStream/DebugCommands/hkDebugCommands.h>

#if defined(HK_PLATFORM_SPU)
#	include <Common/Base/Monitor/Spu/hkSpuMonitorCache.h>
#endif

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactSolverSetup.h>
#if defined(HKNP_CVX_COLLIDE_INLINE)
#	include <Physics/Internal/Dynamics/Solver/Contact/hknpContactSolverSetup.inl>	
#endif

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>
#include <Physics/Physics/Dynamics/Solver/MxJacobianSorter/hknpMxJacobianSorter.h>
#include <Physics/Physics/Dynamics/World/Deactivation/hknpCollisionPair.h>
#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>

#include <Physics/Physics/Collide/NarrowPhase/Detector/SignedDistanceField/hknpSignedDistanceFieldCollisionDetector.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/ConvexConvex/hknpConvexConvexCollisionDetector.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/ConvexConvex/hknpConvexConvexCollisionDetector.inl>
#include <Physics/Physics/Collide/NarrowPhase/Detector/ConvexComposite/hknpConvexCompositeCollisionDetector.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/CompositeComposite/hknpCompositeCompositeCollisionDetector.h>


#if defined(HK_PLATFORM_HAS_SPU)
HK_COMPILE_TIME_ASSERT( HKNP_NUM_MXJACOBIANS_PER_BLOCK == hkBlockStreamBase::Block::BLOCK_DATA_SIZE / sizeof(hknpMxContactJacobian));
#endif


#if defined(HK_PLATFORM_SPU)
struct CompositeBodiesMotionsShapesBuffers
{
	HK_ALIGN16(hknpBody   m_bodyABuffer									);
	HK_ALIGN16(hknpBody   m_bodyBBuffer									);
	HK_ALIGN16(hknpMotion m_motionABuffer								);
	HK_ALIGN16(hknpMotion m_motionBBuffer								);
	HK_ALIGN16(hkUint8    m_shapeABuffer[HKNP_MAX_SHAPE_SIZE_ON_SPU]	);
	HK_ALIGN16(hkUint8    m_shapeBBuffer[HKNP_MAX_SHAPE_SIZE_ON_SPU]	);
};
#endif


void hknpCollidePipeline::mergeAndCollide2Streams(
	const hknpSimulationThreadContext& threadContext, const hknpInternalCollideSharedData& sharedData, int currentLinkIndex,

	hknpCdCacheConsumer& cdCacheConsumer1, hknpCdCacheStream& childCdCacheStreamIn1, hknpCdCacheStream* childCdCacheStreamIn1OnPpu,
	hknpCdCacheConsumer* cdCacheConsumer2, hknpCdCacheStream* childCdCacheStreamIn2, hknpCdCacheStream* childCdCacheStreamIn2OnPpu,

	// output
	hknpCdCacheWriter& cdCacheWriter,			hknpCdCacheWriter& childCdCacheWriter,
	hknpCdCacheWriter* inactiveCdCacheWriter,   hknpCdCacheWriter* inactiveChildCdCacheWriter,
	hknpCdCacheWriter* crossGridCdCacheWriter,  hknpCdCacheWriter* crossGridChildCdCacheWriter, //optional

	hknpCdPairWriter& activePairWriter, hknpLiveJacobianInfoWriter* HK_RESTRICT liveJacInfoWriter,
	hknpMxJacobianSorter* HK_RESTRICT jacMovingMxSorter, hknpMxJacobianSorter* HK_RESTRICT fixedJacMxSorter
	)
{
#if defined(HK_PLATFORM_SPU)

	// Allocate buffer on SPU stack for bodies, motions and shapes
	CompositeBodiesMotionsShapesBuffers* bmsBuffer = (CompositeBodiesMotionsShapesBuffers*)hkSpuStack::getInstance().allocateStack(
		HK_NEXT_MULTIPLE_OF(128, sizeof(CompositeBodiesMotionsShapesBuffers)), "processCollisions_CompositeBodiesMotionsShapesBuffer" );

	const hknpBody* HK_RESTRICT bodyA = &bmsBuffer->m_bodyABuffer;
	const hknpBody* HK_RESTRICT bodyB = &bmsBuffer->m_bodyBBuffer;

	const hknpMotion* HK_RESTRICT motionA = &bmsBuffer->m_motionABuffer;
	const hknpMotion* HK_RESTRICT motionB = &bmsBuffer->m_motionBBuffer;

	hknpShape* shapeA = (hknpShape*)&bmsBuffer->m_shapeABuffer;
	hknpShape* shapeB = (hknpShape*)&bmsBuffer->m_shapeBBuffer;

#endif

	hkMonitorStream& mStream = hkMonitorStream::getInstance();
	HK_TIMER_BEGIN_LIST2(mStream, "MergeAndCollideStreams", "GetData");

	hknpCdCacheConsumer emptyConsumer;
	if ( !cdCacheConsumer2 )
	{
		emptyConsumer.setEmpty();
		cdCacheConsumer2 = &emptyConsumer;
	}

	while(1)
	{
		HK_TIMER_SPLIT_LIST2( mStream, "GetData");
		hknpCollisionCache* cacheOut;	// the output collision cache we will be writing to.
		hknpCdCacheStream* childCacheStreamIn;
		hknpCdCacheStream* childCacheStreamInPpu;
		hkBool32 userCacheDeleteRequest = 0;
		hkUint32 enableRebuildCaches = sharedData.m_enableRebuildCdCaches1;

		// Obtain next cache to process and copy it to cacheOut
		{
			const hknpCollisionCache* cacheIn1 = cdCacheConsumer1.access();
			const hknpCollisionCache* cacheIn2 = cdCacheConsumer2->access();

			// Select next input stream, optimistically select stream 1 (more likely to have most of data)
			const hknpCollisionCache* HK_RESTRICT cacheIn = cacheIn1;
			hknpCdCacheConsumer* HK_RESTRICT cacheConsumer = &cdCacheConsumer1;
			childCacheStreamIn = &childCdCacheStreamIn1;
			childCacheStreamInPpu = childCdCacheStreamIn1OnPpu;

			if (cacheIn1)
			{
				if (cacheIn2)
				{
					if (cacheIn1->getBodyIdPair() < cacheIn2->getBodyIdPair())
					{
						goto COPY_INPUT_CACHE;
					}
					else if (cacheIn2->getBodyIdPair() < cacheIn1->getBodyIdPair())
					{
						goto SELECT_STREAM_2;
					}
					else
					{
						// Both stream have the same body pairs, this can happen if we have 'resent' new pairs to the
						// stream, in this case we simply filter kill the first cache
						userCacheDeleteRequest = true;
						goto COPY_INPUT_CACHE;
					}
				}
				goto COPY_INPUT_CACHE;
			}

			if (cacheIn2)
			{
				goto SELECT_STREAM_2;
			}

			// Both streams empty, done
			break;

SELECT_STREAM_2:

			{
				cacheIn = cacheIn2;
				cacheConsumer = cdCacheConsumer2;
				childCacheStreamIn = childCdCacheStreamIn2;
				childCacheStreamInPpu = childCdCacheStreamIn2OnPpu;
				enableRebuildCaches = sharedData.m_enableRebuildCdCaches2;
			}

COPY_INPUT_CACHE:

		#if defined(HK_PLATFORM_SPU)
			HK_ASSERT2(0x230283a1, !cacheIn->m_spuFlags.anyIsSet(hknpBody::FORCE_NARROW_PHASE_PPU), "Found PPU only cache on SPU");
		#endif

			// Optimistically copy the source cache to the destination cache
			{
				HK_ASSERT(0xfd21ea, (int)cacheIn->getSizeInBytes() <= HKNP_MAX_COLLISION_CACHE_SIZE );
				cacheOut = cdCacheWriter.reserve( HKNP_MAX_COLLISION_CACHE_SIZE );
				hkMath::prefetch128( hkAddByteOffsetConst( cacheOut, 128 ));
				hkMath::prefetch128( hkAddByteOffsetConst( cacheOut, 256 ));
				int cacheSize = cacheIn->getSizeInBytes();

				// Copy input cache
#if defined(HK_PLATFORM_XBOX360)
				// Use int vector load/store operations in Xbox360 instead of memcpy
				{
					int offset = 0;
					hkIntVector a; a.load<4>((const hkUint32*)hkAddByteOffsetConst(cacheIn,offset+00) );
					hkIntVector b; b.load<4>((const hkUint32*)hkAddByteOffsetConst(cacheIn,offset+16) );
					hkIntVector c; c.load<4>((const hkUint32*)hkAddByteOffsetConst(cacheIn,offset+32) );
					hkIntVector d; d.load<4>((const hkUint32*)hkAddByteOffsetConst(cacheIn,offset+48) );

					a.store<4>( (hkUint32*)hkAddByteOffset(cacheOut,offset+00) );
					b.store<4>( (hkUint32*)hkAddByteOffset(cacheOut,offset+16) );
					c.store<4>( (hkUint32*)hkAddByteOffset(cacheOut,offset+32) );
					d.store<4>( (hkUint32*)hkAddByteOffset(cacheOut,offset+48) );
				}

				for (int offset = cacheSize-64; offset>=16; offset-= 64)
				{
					hkIntVector a; a.load<4>((const hkUint32*)hkAddByteOffsetConst(cacheIn,offset+00) );
					hkIntVector b; b.load<4>((const hkUint32*)hkAddByteOffsetConst(cacheIn,offset+16) );
					hkIntVector c; c.load<4>((const hkUint32*)hkAddByteOffsetConst(cacheIn,offset+32) );
					hkIntVector d; d.load<4>((const hkUint32*)hkAddByteOffsetConst(cacheIn,offset+48) );

					a.store<4>( (hkUint32*)hkAddByteOffset(cacheOut,offset+00) );
					b.store<4>( (hkUint32*)hkAddByteOffset(cacheOut,offset+16) );
					c.store<4>( (hkUint32*)hkAddByteOffset(cacheOut,offset+32) );
					d.store<4>( (hkUint32*)hkAddByteOffset(cacheOut,offset+48) );
				}
#else
				hkString::memCpy16( cacheOut, cacheIn, cacheIn->m_sizeInQuads );
#endif

				// Consume input cache
#if defined(HK_PLATFORM_XBOX360)
				const hknpCollisionCache* nextCache =
#endif
				cacheConsumer->consumeAndAccessNext( cacheSize );

#if defined(HK_PLATFORM_XBOX360)
				// Obtain second next cache if possible and prefetch it
				if ( cacheConsumer->getNumUnreadElementsInThisBlock() >=2 )
				{
					const hknpCollisionCache* nextNextCache = hkAddByteOffsetConst( nextCache, nextCache->getSizeInBytes() );
					const hknpBody*   HK_RESTRICT bodies  = sharedData.m_bodies;
					{
						hknpBodyId bodyIdA = nextNextCache->m_bodyA;
						hknpBodyId bodyIdB = nextNextCache->m_bodyB;
						hkMath::prefetch128( &bodies[bodyIdA.value()] );
						hkMath::prefetch128( &bodies[bodyIdB.value()] );
					}
					hkMath::prefetch128( hkAddByteOffsetConst( nextNextCache, 128 ));
					hkMath::prefetch128( hkAddByteOffsetConst( nextNextCache, 256 ));

					hknpBodyId bodyIdA = nextCache->m_bodyA;
					hknpBodyId bodyIdB = nextCache->m_bodyB;
					const hknpBody* HK_RESTRICT bodyA = &bodies[bodyIdA.value()];
					const hknpBody* HK_RESTRICT bodyB = &bodies[bodyIdB.value()];

					const hknpMotion* HK_RESTRICT motions = sharedData.m_motions;

					const hknpMotion* HK_RESTRICT motionA = &motions[ bodyA->m_motionId.valueUnchecked() ];
					const hknpMotion* HK_RESTRICT motionB = &motions[ bodyB->m_motionId.valueUnchecked() ];
					hkMath::prefetch128( motionA );		// needed for solver setup, not needed for collision detection
					hkMath::prefetch128( motionB );
				}
#endif

				HK_ON_DEBUG( cacheIn = HK_NULL );
			}
		}

#if defined (HK_ENABLE_DETERMINISM_CHECKS)
		{
			hknpCollisionCacheType::Enum cacheType = cacheOut->m_type;
			if ( cacheType == hknpCollisionCacheType::CONVEX_CONVEX )
			{
				if ( ((hknpConvexConvexCollisionCache*) cacheOut)->hasManifoldData( ) )
				{
					int exclude[] = {	HK_OFFSET_OF( hknpManifoldCollisionCache, m_gskCache ), sizeof( hkcdGsk::Cache ), // see hkcdGsk::Cache::pack
						HK_OFFSET_OF( hknpManifoldCollisionCache, m_manifoldSolverInfo.m_contactJacobian ), sizeof( hknpMxContactJacobian* ),
						-1
					};

					hkCheckDeterminismUtil::checkMt( 0xf0002e45, (const char*)cacheOut, cacheOut->getSizeInBytes(), exclude );
				}
				else
				{
					int exclude[] = {	HK_OFFSET_OF( hknpConvexConvexCollisionCache, m_gskCache ), sizeof( hkcdGsk::Cache ), // see hkcdGsk::Cache::pack
						-1
					};

					hkCheckDeterminismUtil::checkMt( 0xf0002e46, (const char*)cacheOut, cacheOut->getSizeInBytes(), exclude );
				}
			}
			else if ( cacheType == hknpCollisionCacheType::CONVEX_COMPOSITE || cacheType == hknpCollisionCacheType::COMPOSITE_COMPOSITE || cacheOut->m_type == hknpCollisionCacheType::DISTANCE_FIELD )
			{
				int exclude[] = {	HK_OFFSET_OF( hknpCompositeCollisionCache, m_childCdCacheRange), sizeof( hkBlockStreamBase::Range ),
					-1
				};

				hkCheckDeterminismUtil::checkMt( 0xf0002e47, (const char*)cacheOut, cacheOut->getSizeInBytes(), exclude );
				hkCheckDeterminismUtil::checkMt( 0xf0002e48, (const char*) &(((hknpCompositeCollisionCache*)cacheOut)->m_childCdCacheRange.m_numElements), sizeof( hkBlockStreamBase::Block::CountType ) );
			}
			else
			{
				hkCheckDeterminismUtil::checkMt( 0xf0002e51, (const char*)cacheOut, cacheOut->getSizeInBytes() );
			}
		}
#endif

		const hknpBody* HK_RESTRICT bodies  = sharedData.m_bodies;
		const hknpMotion* HK_RESTRICT motions = sharedData.m_motions;
		hknpBodyId bodyIdA = cacheOut->m_bodyA;
		hknpBodyId bodyIdB = cacheOut->m_bodyB;

//		HK_TIMER_SPLIT_LIST2( mStream, "AccessBodies");
#if !defined(HK_PLATFORM_SPU)
		const hknpBody* HK_RESTRICT bodyA = &bodies[bodyIdA.value()];
		const hknpBody* HK_RESTRICT bodyB = &bodies[bodyIdB.value()];
#else

		// Transfer the bodies from PPU and wait for their arrival.
		hkSpuDmaManager::getFromMainMemory(&bmsBuffer->m_bodyABuffer, &bodies[bodyIdA.value()], sizeof(hknpBody), hkSpuDmaManager::READ_COPY); 
		hkSpuDmaManager::getFromMainMemory(&bmsBuffer->m_bodyBBuffer, &bodies[bodyIdB.value()], sizeof(hknpBody), hkSpuDmaManager::READ_COPY); 
		hkSpuDmaManager::waitForDmaCompletion();
		hkSpuDmaManager::performFinalChecks(&bodies[bodyIdA.value()], &bmsBuffer->m_bodyABuffer, sizeof(hknpBody));
		hkSpuDmaManager::performFinalChecks(&bodies[bodyIdB.value()], &bmsBuffer->m_bodyBBuffer, sizeof(hknpBody));

		// As soon as we have the bodies we can start transferring the motions and the shapes.
		if ( bodyA->m_motionId.isValid() )
		{
			hkSpuDmaManager::getFromMainMemory(&bmsBuffer->m_motionABuffer, &motions[bodyA->m_motionId.value()], sizeof(hknpMotion), hkSpuDmaManager::READ_COPY, hknpCollidePipeline::DMA_ID_MESH_GET_MOTIONS);
			hkSpuDmaManager::deferFinalChecksUntilWait(&motions[bodyA->m_motionId.value()], &bmsBuffer->m_motionABuffer, sizeof(hknpMotion));
		}
		if ( bodyB->m_motionId.isValid() )
		{
			hkSpuDmaManager::getFromMainMemory(&bmsBuffer->m_motionBBuffer, &motions[bodyB->m_motionId.value()], sizeof(hknpMotion), hkSpuDmaManager::READ_COPY, hknpCollidePipeline::DMA_ID_MESH_GET_MOTIONS);
			hkSpuDmaManager::deferFinalChecksUntilWait(&motions[bodyB->m_motionId.value()], &bmsBuffer->m_motionBBuffer, sizeof(hknpMotion));
		}

		
		int shapeASizeInBytes = bodyA->m_shapeSizeDiv16 << 4;
		int shapeBSizeInBytes = bodyB->m_shapeSizeDiv16 << 4;

		// Start transfer of shapes
		hkSpuDmaManager::getFromMainMemory(&bmsBuffer->m_shapeABuffer, bodyA->m_shape, shapeASizeInBytes, hkSpuDmaManager::READ_COPY, hknpCollidePipeline::DMA_ID_MESH_GET_SHAPES);
		hkSpuDmaManager::getFromMainMemory(&bmsBuffer->m_shapeBBuffer, bodyB->m_shape, shapeBSizeInBytes, hkSpuDmaManager::READ_COPY, hknpCollidePipeline::DMA_ID_MESH_GET_SHAPES);
		hkSpuDmaManager::deferFinalChecksUntilWait(bodyA->m_shape, &bmsBuffer->m_shapeABuffer, shapeASizeInBytes);
		hkSpuDmaManager::deferFinalChecksUntilWait(bodyB->m_shape, &bmsBuffer->m_shapeBBuffer, shapeBSizeInBytes);

#endif

#if !defined(HK_PLATFORM_SPU)
		const hknpMotion* HK_RESTRICT motionA = &motions[ bodyA->m_motionId.valueUnchecked() ];
		const hknpMotion* HK_RESTRICT motionB = &motions[ bodyB->m_motionId.valueUnchecked() ];
		hkMath::prefetch128( motionA );		// needed for solver setup, not needed for collision detection
		hkMath::prefetch128( motionB );
#endif

		HK_ASSERT2(0x6fbea51a, (bodyA->m_qualityId.value() < threadContext.m_numQualities) &&
							   (bodyB->m_qualityId.value() < threadContext.m_numQualities), "Invalid quality Id");
		const hknpBodyQuality* qualityA = &threadContext.m_qualities[ bodyA->m_qualityId.value() ];
		const hknpBodyQuality* qualityB = &threadContext.m_qualities[ bodyB->m_qualityId.value() ];

		HK_ASSERT2(0x6fbea51b, (bodyA->m_materialId.value() < threadContext.m_numMaterials) &&
							   (bodyB->m_materialId.value() < threadContext.m_numMaterials), "Invalid material Id");
		const hknpMaterial* materialA = &threadContext.m_materials[ bodyA->m_materialId.value() ];
		const hknpMaterial* materialB = &threadContext.m_materials[ bodyB->m_materialId.value() ];

		hknpCdBody cdBodyA;
		{
			cdBodyA.m_body					= bodyA;
			cdBodyA.m_rootShape				= HK_NULL;
			cdBodyA.m_motion				= HK_NULL;
			cdBodyA.m_quality				= qualityA;
			cdBodyA.m_material				= materialA;
			cdBodyA.m_collisionFilterInfo	= bodyA->m_collisionFilterInfo;
			cdBodyA.m_shapeKey				= HKNP_INVALID_SHAPE_KEY;
		}

		hknpCdBody cdBodyB;
		{
			cdBodyB.m_body					= bodyB;
			cdBodyB.m_rootShape				= HK_NULL;
			cdBodyB.m_motion				= HK_NULL;
			cdBodyB.m_quality				= qualityB;
			cdBodyB.m_material				= materialB;
			cdBodyB.m_collisionFilterInfo	= bodyB->m_collisionFilterInfo;
			cdBodyB.m_shapeKey				= HKNP_INVALID_SHAPE_KEY;
		}

		//
		//	Drop non overlapping pairs
		//
		hknpBodyFlags oredBodyFlags  = ( bodyA->m_flags.get() | bodyB->m_flags.get() );
		hknpBodyFlags andedBodyFlags = ( bodyA->m_flags.get() & bodyB->m_flags.get() );

		hkBool32 disjoint = bodyA->m_aabb.disjoint( bodyB->m_aabb );
		userCacheDeleteRequest |= enableRebuildCaches & (oredBodyFlags & hknpBody::TEMP_REBUILD_COLLISION_CACHES);
		HK_ASSERT( 0xf0fced12, enableRebuildCaches == 0 || enableRebuildCaches == ~(hkUint32)0 );
		if( disjoint | userCacheDeleteRequest )
		{
			hknpCdCacheDestructReason::Enum reason;
			if ( disjoint )
			{
				reason = hknpCdCacheDestructReason::AABBS_DONT_OVERLAP;
			}
			else
			{
				reason = hknpCdCacheDestructReason::CACHE_DELETED_BY_USER;
				if ( !bodyA->isAddedToWorld() || !bodyB->isAddedToWorld() )
				{
					reason = hknpCdCacheDestructReason::BODY_IS_INVALID;
				}
			}
			cacheOut->destruct( threadContext, sharedData, childCacheStreamIn, childCacheStreamInPpu, &cdBodyA, &cdBodyB, reason );

			HK_ON_SPU(hkSpuDmaManager::waitForAllDmaCompletion());
			continue;	// discard this pair
		}

		// No links to fixed bodies.
		hknpMxJacobianSorter* HK_RESTRICT jacMxSorter = fixedJacMxSorter;

		if ( andedBodyFlags & hknpBody::IS_DYNAMIC )
		{
			// now both bodies are dynamic,
			//	check for deactivation
			if ( 0 == (andedBodyFlags & hknpBody::IS_ACTIVE) )	// at least one body is not active
			{
				DEACTIVATE_CACHE:
				cacheOut->moveAndConsumeChildCaches( threadContext, childCacheStreamIn, childCacheStreamInPpu, inactiveChildCdCacheWriter );
				inactiveCdCacheWriter->write16( cacheOut, cacheOut->getSizeInBytes() );
				HK_ON_SPU(hkSpuDmaManager::waitForAllDmaCompletion());
				continue;
			}
			{
				// Wait for the motions to arrive on SPU.
				HK_ON_SPU(hkSpuDmaManager::waitForDmaCompletion(hknpCollidePipeline::DMA_ID_MESH_GET_MOTIONS));

				int cellIdxA = motionA->m_cellIndex;
				int cellIdxB = motionB->m_cellIndex;

				jacMxSorter = jacMovingMxSorter;
				hknpCollisionPair* HK_RESTRICT activePair = activePairWriter.reserve(sizeof(hknpCollisionPair)) ;
				activePair->m_cell[0] = hknpCellIndex(cellIdxA);
				activePair->m_cell[1] = hknpCellIndex(cellIdxB);
				activePair->m_id[0] = motionA->m_solverId;
				activePair->m_id[1] = motionB->m_solverId;
				activePairWriter.advance(sizeof(hknpCollisionPair));
			}
		}
		else
		{
			//	deactivate if both bodies are deactive (at least one should be fixed, so we effectively only checking the other body)
			if( 0 == (oredBodyFlags & hknpBody::IS_ACTIVE) )
			{
				goto DEACTIVATE_CACHE;
			}
			HK_ON_SPU(hkSpuDmaManager::waitForDmaCompletion(hknpCollidePipeline::DMA_ID_MESH_GET_MOTIONS));
		}


		//
		// check if we crossed a grid cell (after deactivation)
		//
		//	HK_TIMER_SPLIT_LIST2( mStream, "CrossGrid");
		if ( crossGridCdCacheWriter )
		{
			int cellIdxA = motionA->m_cellIndex;
			int cellIdxB = motionB->m_cellIndex;
			int linkIdx = sharedData.m_spaceSplitter->getLinkIdx( cellIdxA, cellIdxB );	// expensive !!!
			if ( linkIdx != currentLinkIndex )
			{
				cacheOut->moveAndConsumeChildCaches( threadContext, childCacheStreamIn, childCacheStreamInPpu, crossGridChildCdCacheWriter );
				crossGridCdCacheWriter->write16( cacheOut, cacheOut->getSizeInBytes() );
				HK_ON_SPU(hkSpuDmaManager::waitForAllDmaCompletion());
				continue;
			}
		}

		int deltaDistCombined = bodyA->m_maxTimDistance + bodyB->m_maxTimDistance;
		int newLinearTim = cacheOut->m_linearTim - deltaDistCombined;

		// Wait for shapes
		HK_ON_SPU(hkSpuDmaManager::waitForDmaCompletion(hknpCollidePipeline::DMA_ID_MESH_GET_SHAPES));

		if( newLinearTim > deltaDistCombined )		// use current velocity twice
		{
			cacheOut->m_linearTim = hknpLinearTimType(newLinearTim);
		}
		else
		{
		#if !defined(HK_PLATFORM_SPU)
			const hknpShape* shapeA = bodyA->m_shape;
			const hknpShape* shapeB = bodyB->m_shape;
		#else
			hknpShapeVirtualTableUtil::patchVirtualTable(shapeA);
			hknpShapeVirtualTableUtil::patchVirtualTable(shapeB);
		#endif

			cdBodyA.m_rootShape	= shapeA;
			cdBodyA.m_motion	= motionA;
			cdBodyA.m_transform	= &bodyA->getTransform();

			cdBodyB.m_rootShape	= shapeB;
			cdBodyB.m_motion	= motionB;
			cdBodyB.m_transform	= &bodyB->getTransform();

			hknpCollisionCacheType::Enum cacheType = cacheOut->m_type;
			if ( cacheType == hknpCollisionCacheType::CONVEX_CONVEX )
			{
				hknpConvexConvexCollisionCache* HK_RESTRICT cvxCvxCache = static_cast<hknpConvexConvexCollisionCache*>(cacheOut);
				HK_TIMER_SPLIT_LIST2( mStream, "CvxVsCvx");

				cdBodyA.m_leafShape = shapeA;	// Body A is convex.
				cdBodyB.m_leafShape = shapeB;	// Body B is convex.

				HK_ON_CPU( hknpConvexConvexCollisionCache* cvxCvxCachePpu = cvxCvxCache );
				HK_ON_SPU( hknpConvexConvexCollisionCache* cvxCvxCachePpu = cdCacheWriter.spuToPpu(cvxCvxCache) );

				hknpConvexConvexCollisionDetector::collideConvexConvex(
					threadContext, sharedData, &mStream,
					cdBodyA, cdBodyB,
					cvxCvxCache, cvxCvxCachePpu, jacMxSorter, liveJacInfoWriter );
			}
			else
			{
				hknpCollisionDetector* detector = threadContext.m_modifierManager->m_collisionDetectors[ cacheType ];
				HK_ASSERT2( 0x12950381, detector, "No collision detector registered for this cache type");
				if ( detector->m_useChildCaches )
				{
					HK_TIMER_SPLIT_LIST2( mStream, "Composite");
					cdBodyB.m_leafShape	= HK_NULL;
					cdBodyB.m_transform	= HK_NULL;

					// remember the start of the output caches (using a separate variable as cacheOut->m_caches still is in use)
					hknpCdCacheRange childCdCacheOutputRange;
					childCdCacheOutputRange.setStartPoint( &childCdCacheWriter );
					hknpCompositeCollisionCache* HK_RESTRICT compCache = static_cast<hknpCompositeCollisionCache*>(cacheOut);

					// Collide convex and composite shape.
					hknpCompositeCollisionDetector* compDetector = static_cast<hknpCompositeCollisionDetector*>(detector);
					compDetector->collideWithChildren(
						threadContext, sharedData,
						&cdBodyA, shapeA, &cdBodyB, shapeB,
						compCache, childCacheStreamIn, childCacheStreamInPpu,
						&childCdCacheWriter, jacMxSorter, liveJacInfoWriter );

					// export cvx caches
					compCache->m_childCdCacheRange = childCdCacheOutputRange;
					compCache->m_childCdCacheRange.setEndPoint( &childCdCacheWriter );	// remember the caches used for this pair
				}
				else
				{
					HK_TIMER_SPLIT_LIST2( mStream, "SDF");
					hknpConvexConvexCollisionCache* HK_RESTRICT cvxCache = static_cast<hknpConvexConvexCollisionCache*>(cacheOut);

					detector->collide(threadContext, sharedData,
						&cdBodyA, shapeA, &cdBodyB, shapeB,
						cvxCache, jacMxSorter, liveJacInfoWriter );
				}
			}
		} // if TIM

		cdCacheWriter.advance( cacheOut->getSizeInBytes() );
		HK_ON_SPU(hkSpuMonitorCache::dmaMonitorDataToMainMemorySpu());
	}	// while(1)

	HK_TIMER_END_LIST2(mStream);
	HK_ON_SPU(hkSpuStack::getInstance().deallocateStack(HK_NEXT_MULTIPLE_OF(128, sizeof(CompositeBodiesMotionsShapesBuffers))));
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
