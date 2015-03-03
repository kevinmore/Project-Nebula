/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/NarrowPhase/Detector/SignedDistanceField/hknpSignedDistanceFieldCollisionDetector.h>

#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/SignedDistanceField/hknpSignedDistanceCollisionCache.h>
#include <Physics/Physics/Collide/Shape/TagCodec/hknpShapeTagCodec.h>
#include <Physics/Physics/Dynamics/Solver/MxJacobianSorter/hknpMxJacobianSorter.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactSolverSetup.h>


#if !defined(HK_PLATFORM_SPU)
	typedef hkBlockStream<hknpShape::SdfContactPoint>::Reader hknpSdfContactPointReader;
#else
#	include <Physics/Physics/Collide/NarrowPhase/Detector/SignedDistanceField/hknpSpuSdfContactPointReader.h>
	typedef hknpSpuSdfContactPointReader hknpSdfContactPointReader;
#endif


class hknpCachedVertexReader
{
	public:

		HK_FORCE_INLINE hknpCachedVertexReader( hknpCdCacheConsumer* srcChildCdCacheConsumer )
		{
			m_srcChildCdCacheConsumer = srcChildCdCacheConsumer;
			m_nextVertexIndex = 0;
			m_cacheWasCopied = false;
			m_currentCache = (hknpSignedDistanceFieldChildCollisionCache*)m_srcChildCdCacheConsumer->access();
		}

		HK_FORCE_INLINE void readContactPointCache(
			const hknpSimulationThreadContext &tl, const hknpModifierSharedData &sharedData,
			hknpCdBody* HK_RESTRICT cdBodyA, hknpCdBody* HK_RESTRICT cdBodyB,
			hkUint16 vertexId, hknpShapeTag shapeTagB, hknpShapeKey shapeKeyB,
			hkSimdReal& lastFrameImpulseOut, hkSimdReal& distanceOffsetOut, hkSimdReal& prevFrameRhsOut )
		{
			int nvi = m_nextVertexIndex;
			hknpSignedDistanceFieldChildCollisionCache* HK_RESTRICT cache = m_currentCache;
			while ( cache )
			{
				// consume cache if finished
				if ( nvi >= cache->m_numContactPoints )
				{
					if ( !m_cacheWasCopied )
					{
						const hknpShape* distanceFieldShape = cdBodyB->m_rootShape;

						hknpShapeTagCodec::Context targetShapeTagContext;
						targetShapeTagContext.m_queryType			= hknpCollisionQueryType::QUERY_AABB;
						targetShapeTagContext.m_body				= cdBodyB->m_body;
						targetShapeTagContext.m_rootShape			= cdBodyB->m_rootShape;
						targetShapeTagContext.m_parentShape			= distanceFieldShape;
						targetShapeTagContext.m_shapeKey			= shapeKeyB;
						targetShapeTagContext.m_shape				= HK_NULL;
						targetShapeTagContext.m_partnerBody			= cdBodyA->m_body;
						targetShapeTagContext.m_partnerRootShape	= cdBodyA->m_rootShape;
						targetShapeTagContext.m_partnerShapeKey		= cdBodyA->m_shapeKey;
						targetShapeTagContext.m_partnerShape		= HK_NULL;

						// Pre-fill collision filter info, material and user data (in case they are not overwritten in decode()).
						hknpQueryFilterData decodedShapeTagDataB(*cdBodyB->m_body);

						tl.m_shapeTagCodec->decode(shapeTagB, &targetShapeTagContext,
												   &decodedShapeTagDataB.m_collisionFilterInfo.ref(),
												   &decodedShapeTagDataB.m_materialId, &decodedShapeTagDataB.m_userData.ref());

						cdBodyB->m_collisionFilterInfo	= decodedShapeTagDataB.m_collisionFilterInfo;
						cdBodyB->m_material				= &tl.m_materials[decodedShapeTagDataB.m_materialId.value()];

						cache->fireManifoldDestroyed( tl, sharedData, *cdBodyA, *cdBodyB, hknpCdCacheDestructReason::SHAPE_TAG_CHANGED );
					}
					m_cacheWasCopied = false;
					cache = (hknpSignedDistanceFieldChildCollisionCache*)m_srcChildCdCacheConsumer->consumeAndAccessNext( m_currentCache->getSizeInBytes() );
					m_currentCache = cache;
					m_nextVertexIndex = nvi = 0;
					continue;
				}

				int cachedVertexId = m_currentCache->m_vertexIndices[nvi];
				if ( vertexId < cachedVertexId )
				{
					break;	// new vertex
				}

				if ( vertexId == cachedVertexId )
				{
					// cache found, use
					lastFrameImpulseOut.load<1>( &cache->m_manifoldSolverInfo.m_impulses[nvi] );
					distanceOffsetOut.load<1>( &cache->m_distanceOffset(nvi) );
					prevFrameRhsOut.load<1>( &cache->m_prevFrameRhs(nvi) );
					m_nextVertexIndex = nvi+1;
					return;		// return, we found the point
				}

				// consume vertex
				nvi++;
			}
			lastFrameImpulseOut.setZero();
			distanceOffsetOut.setZero();
			prevFrameRhsOut.setZero();
		}

		HK_FORCE_INLINE void consumeRemaining(
			const hknpSimulationThreadContext &tl, const hknpModifierSharedData &sharedData,
			hknpCdBody* HK_RESTRICT cdBodyA, hknpCdBody* HK_RESTRICT cdBodyB, hknpShapeKey shapeKeyB, hknpShapeTag shapeTagB)
		{
			while ( m_currentCache )
			{
				if ( !m_cacheWasCopied )
				{
					const hknpShape* distanceFieldShape = cdBodyB->m_rootShape;

					hknpShapeTagCodec::Context targetShapeTagContext;
					targetShapeTagContext.m_queryType			= hknpCollisionQueryType::QUERY_AABB;
					targetShapeTagContext.m_body				= cdBodyB->m_body;
					targetShapeTagContext.m_rootShape			= cdBodyB->m_rootShape;
					targetShapeTagContext.m_parentShape			= distanceFieldShape;
					targetShapeTagContext.m_shapeKey			= shapeKeyB;
					targetShapeTagContext.m_shape				= HK_NULL;
					targetShapeTagContext.m_partnerBody			= cdBodyA->m_body;
					targetShapeTagContext.m_partnerRootShape	= cdBodyA->m_rootShape;
					targetShapeTagContext.m_partnerShapeKey		= cdBodyA->m_shapeKey;
					targetShapeTagContext.m_partnerShape		= HK_NULL;

					// Pre-fill collision filter info, material and user data (in case they are not overwritten in decode()).
					hknpQueryFilterData decodedShapeTagDataB(*cdBodyB->m_body);

					tl.m_shapeTagCodec->decode(shapeTagB, &targetShapeTagContext,
											   &decodedShapeTagDataB.m_collisionFilterInfo.ref(),
											   &decodedShapeTagDataB.m_materialId, &decodedShapeTagDataB.m_userData.ref());

					cdBodyB->m_collisionFilterInfo	= decodedShapeTagDataB.m_collisionFilterInfo;
					cdBodyB->m_material				= &tl.m_materials[decodedShapeTagDataB.m_materialId.value()];

					m_currentCache->fireManifoldDestroyed( tl, sharedData, *cdBodyA, *cdBodyB, hknpCdCacheDestructReason::SHAPE_TAG_CHANGED );
				}
				m_cacheWasCopied = false;
				m_currentCache = (hknpSignedDistanceFieldChildCollisionCache*)m_srcChildCdCacheConsumer->consumeAndAccessNext( m_currentCache->getSizeInBytes() );
			}
		}

	public:

		hknpSignedDistanceFieldChildCollisionCache* m_currentCache;
		hkBool m_cacheWasCopied;	// set to true if we have copied the last cache.
		int m_nextVertexIndex;
		hknpCdCacheConsumer* m_srcChildCdCacheConsumer;
};


void collideSignedDistances( const hknpSimulationThreadContext &tl, const hknpModifierSharedData &sharedData,
	hknpCdBody* HK_RESTRICT cdBodyA, hknpCdBody* HK_RESTRICT cdBodyB,
	hknpSdfContactPointReader* HK_RESTRICT contactReader,
	hknpCdCacheConsumer* srcChildCdCacheConsumer, hknpCdCacheWriter* HK_RESTRICT childCdCacheWriter,
	hknpMxJacobianSorter* HK_RESTRICT jacMxSorter )
{
	hkUint32 bodyIdHashCode = hknpMxJacobianSorter::calcBodyIdsHashCode( *cdBodyA->m_body, *cdBodyB->m_body );

	hknpManifold manifold;
	manifold.m_distances.setZero();

	hknpCachedVertexReader cacheReader ( srcChildCdCacheConsumer );

	// iterate over all points
	const hknpShape::SdfContactPoint* vert = contactReader->access();
	if ( vert == HK_NULL )
	{
		cacheReader.consumeRemaining( tl , sharedData, cdBodyA, cdBodyB, HKNP_INVALID_SHAPE_KEY, HKNP_INVALID_SHAPE_TAG);
		return;
	}
	hknpContactSolverSetup::BuildConfig buildConfig; buildConfig.init( cdBodyA->m_motion, cdBodyB->m_motion );

	int numPointsInManifold = 0;
	hknpSignedDistanceFieldChildCollisionCache* /*HK_RESTRICT*/ dstCache = HK_NULL;
	hkVector4 avgNormal; avgNormal.setZero();
	int isNewManifold = 0;
	hknpShapeTag shapeTagOfMaterial = HKNP_INVALID_SHAPE_TAG;
	bool contactPointDisabledByFilter = false;

	hknpShapeTag shapeTag = HKNP_INVALID_SHAPE_TAG;
	hknpVertexId vertexId   = 0;
	hknpShapeKey shapeKeyB  = HKNP_INVALID_SHAPE_KEY;
	bool initializeOutputCache = false;
	bool buildJacobian = false;

	for ( ;; vert = contactReader->advanceAndAccessNext( sizeof(hknpShape::SdfContactPoint) ) )
	{
		// We need to declare and initialize these variables here to stop SPU complaining about the goto entering
		// the scope of non-POD
		hkSimdReal lastFrameImpulse;
		hkSimdReal distanceOffset;
		hkSimdReal prevFrameRhs;
#if defined ( HK_PLATFORM_PS3 )
		lastFrameImpulse.setZero();
		distanceOffset.setZero();
		prevFrameRhs.setZero();
#endif

		hknpShapeKeyPath shapeKeyPathB;

		if ( !vert )
		{
			if ( numPointsInManifold )
			{
				goto BUILD_JACOBIAN;	// cannot happen the first iteration
			}
			break;	// done
		}

		shapeTag  = vert->m_shapeTag;
		vertexId  = vert->m_vertexId;
		shapeKeyB = vert->m_shapeKey;

		shapeKeyPathB.appendSubKey(shapeKeyB, cdBodyB->m_rootShape->getNumShapeKeyBits());

		//
		//	Read contact point cache
		//
		{
			cacheReader.readContactPointCache( tl, sharedData, cdBodyA, cdBodyB,
				vertexId, shapeTag, shapeKeyPathB.getKey(),
				lastFrameImpulse, distanceOffset, prevFrameRhs );
		}

		// material management
		if ( shapeTag != shapeTagOfMaterial)
		{
			const hknpShape* distanceFieldShape = cdBodyB->m_rootShape;

			hknpShapeTagCodec::Context targetShapeTagContext;
			targetShapeTagContext.m_queryType			= hknpCollisionQueryType::QUERY_AABB;
			targetShapeTagContext.m_body				= cdBodyB->m_body;
			targetShapeTagContext.m_rootShape			= cdBodyB->m_rootShape;
			targetShapeTagContext.m_parentShape			= distanceFieldShape;
			targetShapeTagContext.m_shapeKey			= shapeKeyPathB.getKey();
			targetShapeTagContext.m_shape				= HK_NULL;
			targetShapeTagContext.m_partnerBody			= cdBodyA->m_body;
			targetShapeTagContext.m_partnerRootShape	= cdBodyA->m_rootShape;
			targetShapeTagContext.m_partnerShapeKey		= cdBodyA->m_shapeKey;
			targetShapeTagContext.m_partnerShape		= HK_NULL;

			// Pre-fill collision filter info, material and user data (in case they are not overwritten in decode()).
			hknpQueryFilterData decodedShapeTagDataB(*cdBodyB->m_body);

			tl.m_shapeTagCodec->decode(shapeTag, &targetShapeTagContext, &decodedShapeTagDataB.m_collisionFilterInfo.ref(),
									   &decodedShapeTagDataB.m_materialId, &decodedShapeTagDataB.m_userData.ref());

			cdBodyB->m_collisionFilterInfo	= decodedShapeTagDataB.m_collisionFilterInfo;
			cdBodyB->m_material				= &tl.m_materials[decodedShapeTagDataB.m_materialId.value()];

			shapeTagOfMaterial = shapeTag;

			// Filtering
			{
				contactPointDisabledByFilter = false;

				hknpCollisionFilter::FilterInput shapeFilterInputA;
				shapeFilterInputA.m_filterData.m_materialId				= cdBodyA->m_body->m_materialId;
				shapeFilterInputA.m_filterData.m_collisionFilterInfo	= cdBodyA->m_collisionFilterInfo;
				shapeFilterInputA.m_filterData.m_userData				= cdBodyA->m_body->m_userData;
				shapeFilterInputA.m_body								= cdBodyA->m_body;
				shapeFilterInputA.m_rootShape							= cdBodyA->m_rootShape;
				shapeFilterInputA.m_parentShape							= HK_NULL;
				shapeFilterInputA.m_shapeKey							= cdBodyA->m_shapeKey;
				shapeFilterInputA.m_shape								= HK_NULL;

				hknpCollisionFilter::FilterInput shapeFilterInputB;
				shapeFilterInputB.m_filterData							= decodedShapeTagDataB;
				shapeFilterInputB.m_body								= cdBodyB->m_body;
				shapeFilterInputB.m_rootShape							= cdBodyB->m_rootShape;
				shapeFilterInputB.m_parentShape							= distanceFieldShape;
				shapeFilterInputB.m_shapeKey							= shapeKeyPathB.getKey();
				shapeFilterInputB.m_shape								= HK_NULL;

				// Test filtering for the most important vertex.
				if ( !tl.m_modifierManager->getCollisionFilter()->isCollisionEnabled( hknpCollisionQueryType::QUERY_AABB, true, shapeFilterInputA, shapeFilterInputB ) )
				{
					// Collision is disabled for that vertex so we disable all collision.
					contactPointDisabledByFilter = true;
				}
			}
		}
		if ( contactPointDisabledByFilter )
		{
			continue;
		}


		//
		// decide whether we need a new jacobian/manifold and output cache
		//
		initializeOutputCache = false;
		buildJacobian = false;


		if ( numPointsInManifold == 0  )
		{
			initializeOutputCache = true;	// only happens the very first iteration
		}
		else
		{
			if (numPointsInManifold>=4 ||
				dstCache->m_shapeTag != shapeTag ||
				avgNormal.dot<3>(vert->m_normal) < hkSimdReal::fromFloat(0.95f) * hkSimdReal::getConstant( hkVectorConstant(HK_QUADREAL_0 + numPointsInManifold))  )
			{
				initializeOutputCache = true;
				buildJacobian = true;
			}
		}

		//
		// If a manifold if already full, sent it to the solver.
		// Note that this manifold will never include the current vertex.
		//
		if (buildJacobian)
		{
BUILD_JACOBIAN:
			manifold.m_collisionCache = dstCache;
			HK_ON_CPU( manifold.m_collisionCacheInMainMemory = dstCache; )
			HK_ON_SPU( manifold.m_collisionCacheInMainMemory = (hknpConvexConvexManifoldCollisionCache*)childCdCacheWriter->spuToPpu(dstCache); )
			manifold.m_isNewSurface = (hkUint8)isNewManifold;
			manifold.m_isNewManifold = (hkUint8)isNewManifold;
			avgNormal.normalize<3,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
			manifold.m_normal._setRotatedDir( cdBodyB->m_body->getTransform().getRotation(), avgNormal );
			manifold.m_gskPosition = manifold.m_positions[0];
			manifold.m_massChangerData = 0.0f;
			manifold.m_useIncreasedIterations = (hkUint8)( hkUint32(dstCache->m_qualityFlags.get() & hknpBodyQuality::USE_HIGHER_QUALITY_CONTACT_SOLVING) / hknpBodyQuality::USE_HIGHER_QUALITY_CONTACT_SOLVING );
			manifold.m_manifoldType = hknpManifold::TYPE_NORMAL;
			manifold.m_materialB = cdBodyB->m_material;
			manifold.m_numPoints = numPointsInManifold;
			dstCache->m_numContactPoints = hkUint8(numPointsInManifold);
			manifold.m_shapeKeyA = HKNP_INVALID_SHAPE_KEY;
			manifold.m_shapeKeyB = shapeKeyB;
			for (int i = numPointsInManifold; i < 4; i++)
			{
				hkVector4 pos = manifold.m_positions[0];
				manifold.m_distances(i) = manifold.m_distances(0);
				manifold.m_positions[i] = pos;
			}

			//
			//	fire process callbacks
			//
			{
				HK_ON_DEBUG( int extraDataSize =dstCache->getPropertyBufferSize() );
				hknpBodyFlags enabledModifiers = dstCache->m_bodyAndMaterialFlags;
				if ( tl.m_modifierManager->isFunctionRegistered( hknpModifier::FUNCTION_MANIFOLD_PROCESS, enabledModifiers ))
				{
					HKNP_FIRE_MODIFIER( tl.m_modifierManager, hknpModifier::FUNCTION_MANIFOLD_PROCESS, enabledModifiers,
						modifier->manifoldProcessCallback( tl, sharedData, *cdBodyA, *cdBodyB, &manifold ) );
				}
				HK_ASSERT2( 0xf0345454, extraDataSize == dstCache->getPropertyBufferSize(),
						"You can't add more properties after the manifold created callback." );
			}

			if ( !(dstCache->m_bodyAndMaterialFlags & hknpBody::DONT_BUILD_CONTACT_JACOBIANS) )
			{
				HK_PAD_ON_SPU(hknpMxContactJacobian*) mxJac = HK_NULL;
				HK_PAD_ON_SPU(hknpMxContactJacobian*) mxJacOnPpu = HK_NULL;
				HK_ON_SPU( int mxJacIdx = jacMxSorter-> getJacobianLocation( bodyIdHashCode, &mxJac, &mxJacOnPpu) );
				HK_ON_CPU( int mxJacIdx = jacMxSorter->_getJacobianLocation( bodyIdHashCode, &mxJac, &mxJacOnPpu) );

				const hknpSolverInfo* solverInfo = sharedData.m_solverInfo;
				hknpContactSolverSetup::buildContactJacobianForSingleManifold<hknpContactSolverSetup::BUILD_USING_CACHE>(
					&tl, *solverInfo, HK_NULL, &manifold, &buildConfig,
					*cdBodyA, HK_NULL,
					*cdBodyB, HK_NULL,
					mxJac, mxJacOnPpu, mxJacIdx
					);
				jacMxSorter->hintUsingSameBodies();
			}
			else
			{
				dstCache->m_manifoldSolverInfo.m_contactJacobian = HK_NULL;
			}

			numPointsInManifold = 0;	// manifold is sent to the jacobian stream
			avgNormal.setZero();
			if ( vert == HK_NULL )
			{
				if ( dstCache )
				{
					childCdCacheWriter->advance(  dstCache->getSizeInBytes() );
				}
				cacheReader.consumeRemaining( tl , sharedData, cdBodyA, cdBodyB, shapeKeyPathB.getKey(), shapeTag );
				break;	// last iteration
			}
		}

		//
		// Try to get an output cache
		//
		if ( initializeOutputCache )
		{
			if ( dstCache )
			{
				childCdCacheWriter->advance(  dstCache->getSizeInBytes() );
			}
			dstCache = (hknpSignedDistanceFieldChildCollisionCache*)childCdCacheWriter->reserve( HKNP_MAX_SDF_CHILD_CACHE_SIZE );

			// try to reuse input cache data if possible
			if ( cacheReader.m_currentCache &&
				(!cacheReader.m_cacheWasCopied) &&
				(cacheReader.m_currentCache->m_shapeTag == shapeTag ) )
			{
				hkString::memCpy16( dstCache, cacheReader.m_currentCache, cacheReader.m_currentCache->m_sizeInQuads );
				cacheReader.m_cacheWasCopied = true;
				isNewManifold = 0;
			}
			else
			{
				dstCache->init();
				dstCache->m_shapeTag = shapeTag;
				dstCache->setFrictionAndRestitution( *cdBodyA->m_material, *cdBodyB->m_material );

				hknpBodyFlags combinedBodyFlags = tl.m_modifierManager->getCombinedBodyFlags( *cdBodyA, *cdBodyB );
				dstCache->m_bodyAndMaterialFlags = combinedBodyFlags;

				hkUint32 totalCacheSize = sizeof(hknpSignedDistanceFieldChildCollisionCache);
				dstCache->m_sizeInQuads = hkUint8(HK_NEXT_MULTIPLE_OF(1<<4, totalCacheSize)>>4);
				if ( tl.m_modifierManager->isFunctionRegistered( hknpModifier::FUNCTION_MANIFOLD_CREATED_OR_DESTROYED, combinedBodyFlags ) )
				{
					hknpModifier::ManifoldCreatedCallbackInput callbackInput;
					{
						callbackInput.m_collisionCache	= dstCache;
						callbackInput.m_manifold = HK_NULL;
						HK_ON_CPU( callbackInput.m_collisionCacheInMainMemory = dstCache; )
						HK_ON_SPU( callbackInput.m_collisionCacheInMainMemory = (hknpConvexConvexManifoldCollisionCache*)childCdCacheWriter->spuToPpu(dstCache); )
					}

					HKNP_FIRE_MODIFIER( tl.m_modifierManager, hknpModifier::FUNCTION_MANIFOLD_CREATED_OR_DESTROYED, combinedBodyFlags,
						 modifier->manifoldCreatedCallback( tl, sharedData, *cdBodyA, *cdBodyB, &callbackInput) );
					totalCacheSize += dstCache->getPropertyBufferSize();
					HK_ASSERT(0xf0dedf44, HK_NEXT_MULTIPLE_OF(1<<4, totalCacheSize) == hkUint32(dstCache->getSizeInBytes()));
					HK_ASSERT2( 0xf0dedf45, totalCacheSize <= HKNP_MAX_SDF_CHILD_CACHE_SIZE, "Too much user data added in manifoldCreatedCallback()" );
				}
				isNewManifold = 1;
			}
		}

		// store point in manifold and cache
		{
			avgNormal.add( vert->m_normal );
			manifold.m_distances( numPointsInManifold ) = vert->m_distance;
			manifold.m_positions[ numPointsInManifold ]._setTransformedPos( cdBodyB->m_body->getTransform(), vert->m_position );
			dstCache->m_vertexIndices[ numPointsInManifold] = vert->m_vertexId;
			lastFrameImpulse.store<1>( &dstCache->m_manifoldSolverInfo.m_impulses[ numPointsInManifold ] );
			prevFrameRhs.store<1>( &dstCache->m_prevFrameRhs( numPointsInManifold ) );
			distanceOffset.store<1>( &dstCache->m_distanceOffset( numPointsInManifold ) );
		}
		numPointsInManifold++;
	}
}

void hknpSignedDistanceFieldCollisionDetector::collideWithChildren(
	const hknpSimulationThreadContext& tl, const hknpInternalCollideSharedData& sharedData,
	hknpCdBody* HK_RESTRICT cdBodyA, const hknpShape* shapeA,
	hknpCdBody* HK_RESTRICT cdBodyB, const hknpShape* shapeB,
	hknpCompositeCollisionCache* cdCache,
	hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu, // needed by the consumer
	hknpCdCacheWriter* childCdCacheWriter, hknpMxJacobianSorter* jacMxSorter,
	hknpLiveJacobianInfoWriter* liveJacInfoWriter )
{
	HK_TIMER_SPLIT_LIST( "CvxVsSdf");

	cdBodyA->m_leafShape = HK_NULL;
	cdBodyB->m_leafShape = HK_NULL;
	// Body B is a signed distance field

#if !defined(HK_PLATFORM_SPU)
	hkBlockStream<hknpShape::SdfContactPoint>::Stream contactStream;
	contactStream.initBlockStream( tl.m_tempAllocator, false );
#else
	const int capacity = 32; // The maximum number of contact points supported.
	int contactStreamBufferSize = HK_NEXT_MULTIPLE_OF( 128, sizeof(hknpShape::SdfContactPoint) * capacity );
	hknpShape::SdfContactPoint* contactStreamBuffer = (hknpShape::SdfContactPoint*)hkSpuStack::getInstance().allocateStack( contactStreamBufferSize, "hknpSpuSdfContactPointWriterBuffer" );
	hknpShape::SdfContactPoint* contactStreamDataEnd = contactStreamBuffer;
#endif

	// Get signed distances
	{
#if !defined(HK_PLATFORM_SPU)
		hkBlockStream<hknpShape::SdfContactPoint>::Writer contactWriter;
		contactWriter.setToStartOfStream( tl.m_tempAllocator, &contactStream );
#else
		hknpSpuSdfContactPointWriter contactWriter( contactStreamBuffer, contactStreamBufferSize );
#endif

		// Calculate a maximum distance for accepting contacts
		hkSimdReal maxPointDistance;
		{
			hkSimdReal linearTimToDistance = sharedData.m_solverInfo->m_linearTimToDistance;
			hkSimdReal maxPointDistanceA; maxPointDistanceA.setFromUint16( cdBodyA->m_body->m_maxContactDistance );
			hkSimdReal maxPointDistanceB; maxPointDistanceB.setFromUint16( cdBodyB->m_body->m_maxContactDistance );
			maxPointDistance = linearTimToDistance * (maxPointDistanceA + maxPointDistanceB) + hkSimdReal::getConstant<HK_QUADREAL_INV_255>();
		}
		hkTransform bTa; bTa.setMulInverseMul( cdBodyB->m_body->getTransform(), cdBodyA->m_body->getTransform() );

		hkReal maxDistance; maxPointDistance.store<1>(&maxDistance);
		shapeB->getSignedDistanceContacts( tl, shapeA, bTa, maxDistance, 0, contactWriter );
#if !defined(HK_PLATFORM_SPU)
		contactWriter.finalize();
#else
		contactStreamDataEnd = contactWriter.m_currentByteLocation;
#endif
	}

	// Convert to Jacobians
	{
#if !defined(HK_PLATFORM_SPU)
		hkBlockStream<hknpShape::SdfContactPoint>::Reader contactReader;
		contactReader.setToStartOfStream( &contactStream );
#else
		hknpSdfContactPointReader contactReader( contactStreamBuffer, contactStreamDataEnd );
#endif

		hknpCdCacheConsumer cvxCachesReader;
		cvxCachesReader.initSpu( HKNP_SPU_DMA_GROUP_STALL, 1, "CvxVsSdfCacheConsumer" );
		cvxCachesReader.setToRange( tl.m_heapAllocator, childCdCacheStream, childCdCacheStreamPpu, &cdCache->m_childCdCacheRange );

		collideSignedDistances( tl, sharedData, cdBodyA, cdBodyB,
			&contactReader, &cvxCachesReader, childCdCacheWriter, jacMxSorter);

		cvxCachesReader.exitSpu();
		jacMxSorter->resetHintUsingSameBodies();
	}

#if !defined(HK_PLATFORM_SPU)
	contactStream.clear( tl.m_tempAllocator );
#else
	hkSpuStack::getInstance().deallocateStack( contactStreamBuffer, contactStreamBufferSize );
#endif
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
