/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/NarrowPhase/Detector/ConvexComposite/hknpConvexCompositeCollisionDetector.h>

#if defined(HK_PLATFORM_HAS_SPU)
#	include <Common/Base/Monitor/Spu/hkSpuMonitorCache.h>
#endif

#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/NarrowPhase/LiveJacobian/hknpLiveJacobianInfo.h>
#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>
#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>
#include <Physics/Physics/Dynamics/Solver/MxJacobianSorter/hknpMxJacobianSorter.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactSolverSetup.h>
#include <Physics/Internal/Collide/Agent/ProcessCollision2DFast/hknpCollision2DFastProcessUtil.h>
#if defined(HKNP_CVX_COLLIDE_INLINE)
#	include <Physics/Internal/Collide/Agent/ProcessCollision2DFast/hknpCollision2DFastProcessUtil.inl>	
#endif


//#define DEBUG_TRIANGLES
#if defined(DEBUG_TRIANGLES)
#	include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>
#	include <Common/Visualize/hkDebugDisplay.h>
#endif

//#define COUNT_NMP

namespace
{
#if defined(DEBUG_TRIANGLES)
	HK_FORCE_INLINE void _debugDisplayTriangle(
		const hknpSimulationThreadContext &tl, const hkTransform& transformB, const hknpTriangleShape* triangleShape )
	{
		{
			const hkVector4*	vertices = triangleShape->getVertices();
			hkVector4			tVertices[4];
			const int			nv = triangleShape->getNumberOfVertices();
			for(int i=0; i<nv; ++i){ tVertices[i]._setTransformedPos(transformB, vertices[i]); }
			tl.execCommand( hkDebugLineCommand( tVertices[0], tVertices[1], 0xffffffff ) );
			tl.execCommand( hkDebugLineCommand( tVertices[1], tVertices[2], 0xffffffff ) );
			tl.execCommand( hkDebugLineCommand( tVertices[2], tVertices[0], 0xffffffff ) );
			if(nv==4)
			{
				tl.execCommand( hkDebugLineCommand( tVertices[0], tVertices[3], 0xffffffff ) );
				tl.execCommand( hkDebugLineCommand( tVertices[2], tVertices[3], 0xffffffff ) );
			}
		}
	}
#endif

	HK_FORCE_INLINE static void _buildJacobiansForManifold(
		const hknpSimulationThreadContext &tl, const hknpModifierSharedData& sharedData,
		hkBool32 delayProcessCallback,
		hknpManifold* manifolds, int numManifolds, hkMonitorStream& mStream,
		const hknpCdBody& cdBodyA, hknpCdBody* HK_RESTRICT cdBodyB,
		hknpMxJacobianSorter* HK_RESTRICT jacMxSorter, hknpLiveJacobianInfoWriter* HK_RESTRICT liveJacInfoWriter )
	{
		HK_ASSERT( 0x519cc960, numManifolds > 0 );

		const hknpBodyQuality* qA = cdBodyA.m_quality;
		const hknpBodyQuality* qB = cdBodyB->m_quality;
		hknpBodyQuality::Flags combinedBodyQuality; hknpBodyQuality::combineBodyQualities( qA, qB, &combinedBodyQuality );

		//
		// Fire modifier callbacks
		//

		if( delayProcessCallback )
		{
			hknpModifierManager* modifierMgr = tl.m_modifierManager;

			const hknpBodyFlags enabledModifiers = modifierMgr->getCombinedBodyRootFlags( tl.m_materials, cdBodyA, *cdBodyB );

#if !defined(HKNP_DISABLE_WELDING_MODIFIER_ON_SPU) || !defined( HK_PLATFORM_SPU )
			// Welding (not really a modifier)
			if( combinedBodyQuality.anyIsSet( hknpBodyQuality::ANY_WELDING ) )
			{
				HK_TIMER_SPLIT_LIST2(mStream, "WeldingModifier");
				modifierMgr->fireWeldingModifier( tl, sharedData, combinedBodyQuality, cdBodyA, cdBodyB, manifolds, numManifolds);
			}
#endif

			// Manifold processed
			if( tl.m_modifierManager->isFunctionRegistered( hknpModifier::FUNCTION_MANIFOLD_PROCESS, enabledModifiers ))
			{
				HK_TIMER_SPLIT_LIST2(mStream, "ProcessModifier");
				int i = 0;
				do
				{
					hknpManifold* HK_RESTRICT manifold = &manifolds[i];

					// Set child shape data.
					// we know that the manifold has a cache because of 0xf0fg5677
					HK_ON_DEBUG( hknpManifoldCollisionCache* manifoldCache = manifold->m_collisionCache );
					cdBodyB->m_shapeKey = manifold->m_shapeKeyB;
					cdBodyB->m_material = manifold->m_materialB;
					HK_ON_DEBUG(int extraDataSize = manifoldCache->getPropertyBufferSize());

					HKNP_FIRE_MODIFIER( tl.m_modifierManager, hknpModifier::FUNCTION_MANIFOLD_PROCESS, enabledModifiers,
						modifier->manifoldProcessCallback( tl, sharedData,cdBodyA, *cdBodyB,	&manifolds[i] );
					);

					HK_ASSERT2( 0xf0345455, extraDataSize == manifoldCache->getPropertyBufferSize(),
						"You can't add more properties after the manifold created callback." );
				}
				while( ++i < numManifolds );
			}
		}

		//
		// Build Jacobians
		//

		HK_TIMER_SPLIT_LIST2( mStream, "BuildJac" );

		const hkUint32 bodyIdHashCode = hknpMxJacobianSorter::calcBodyIdsHashCode( *cdBodyA.m_body, *cdBodyB->m_body );

		const hknpSolverInfo* solverInfo = sharedData.m_solverInfo;

		hknpContactSolverSetup::BuildConfig buildConfig;
		buildConfig.init( cdBodyA.m_motion, cdBodyB->m_motion );

		cdBodyB->m_material = manifolds[0].m_materialB;

		for( int im = 0; im < numManifolds; im++ )
		{
			hknpManifold* HK_RESTRICT manifold = &manifolds[im];
			hknpManifoldCollisionCache* manifoldCache = manifold->m_collisionCache;

			// Skip if we don't want to build it
			
			if( manifoldCache && (manifoldCache->m_bodyAndMaterialFlags & hknpBody::DONT_BUILD_CONTACT_JACOBIANS) )
			{
				manifoldCache->m_manifoldSolverInfo.m_contactJacobian = HK_NULL;
				continue;
			}

			// Check if we should use merged friction
			if( manifold->m_materialB == cdBodyB->m_material )	// only combine friction if the material is identical
			{
				if( combinedBodyQuality.anyIsSet( hknpBodyQuality::MERGE_FRICTION_JACOBIANS ) )
				{
					buildConfig.m_mergeFriction = numManifolds-1-im;	// if not the last manifold
				}
			}

			HK_PAD_ON_SPU(hknpMxContactJacobian*) mxJac = HK_NULL;
			HK_PAD_ON_SPU(hknpMxContactJacobian*) mxJacOnPpu = HK_NULL;
			HK_ON_SPU( int mxJacIdx = jacMxSorter->getJacobianLocation( bodyIdHashCode,  &mxJac, &mxJacOnPpu) );
			HK_ON_CPU( int mxJacIdx = jacMxSorter->_getJacobianLocation( bodyIdHashCode, &mxJac, &mxJacOnPpu) );
			jacMxSorter->hintUsingSameBodies();

			cdBodyB->m_material = manifold->m_materialB;
			cdBodyB->m_shapeKey = manifold->m_shapeKeyB;

			hknpContactSolverSetup::buildContactJacobianForSingleManifold<hknpContactSolverSetup::BUILD_USING_CACHE>(
				&tl, *solverInfo, HK_NULL, manifold, &buildConfig,
				cdBodyA, HK_NULL,
				*cdBodyB, HK_NULL,
				mxJac, mxJacOnPpu, mxJacIdx );

		#if !defined(HK_PLATFORM_SPU)
			// Write live Jacobian info if we want live Jacobians
			if( manifoldCache && manifoldCache->m_qualityFlags.anyIsSet(hknpBodyQuality::ENABLE_LIVE_JACOBIANS) )
			{
				hknpLiveJacobianInfo liveJacInfo;
				liveJacInfo.m_numManifolds = 1;
				liveJacInfo.m_jacobian[0] = mxJacOnPpu;
				liveJacInfo.m_indexOfManifoldInJacobian[0] = hkUint8(mxJacIdx);

				hknpLiveJacobianInfo::Type type= (im == numManifolds-1) ?
					hknpLiveJacobianInfo::CHILD_CVX_CVX_LAST_IN_BATCH : hknpLiveJacobianInfo::CHILD_CVX_CVX;
				liveJacInfo.initLiveJacobian( manifold, cdBodyA, *cdBodyB, type );

				liveJacInfoWriter->write16( &liveJacInfo, sizeof(liveJacInfo) );
			}
		#endif
		}
	}

}	// anonymous namespace


void hknpConvexCompositeCollisionDetector::collideConvexWithCompositeKeys(
	const hknpSimulationThreadContext& tl, const hknpInternalCollideSharedData& sharedData,
	const hknpCdBody& cdBodyA, hknpCdBody* HK_RESTRICT cdBodyB,
	const hknpShapeKey* keys, int keysStriding, int numKeys,
	hknpCdCacheConsumer* HK_RESTRICT srcChildCdCacheConsumer, hknpCdCacheWriter* HK_RESTRICT childCdCacheWriter,
	hknpManifold* HK_RESTRICT manifoldBuffer, hknpMxJacobianSorter* HK_RESTRICT jacMxSorter,
	hknpLiveJacobianInfoWriter* HK_RESTRICT liveJacInfoWriter )
{
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& mStream = hkMonitorStream::getInstance();	
#endif

	//tl.m_commandStream->exec( hkDebugLineCommand( hkVector4::getZero(), hkVector4::getConstant<HK_QUADREAL_1248>(), 0xff00ff00 ) );

	HK_ASSERT( 0x6a130770, cdBodyB->m_rootShape->getFlags().get( hknpShape::IS_COMPOSITE_SHAPE ) );
	const hknpCompositeShape* compShape = static_cast<const hknpCompositeShape*>( HK_PADSPU_REF( cdBodyB->m_rootShape ) );

	const hknpCollisionCache* HK_RESTRICT srcCache = srcChildCdCacheConsumer->access();
	hknpConvexConvexCollisionCache* HK_RESTRICT dstCache = static_cast<hknpConvexConvexCollisionCache*>(childCdCacheWriter->reserve( HKNP_MAX_CVX_CVX_CACHE_SIZE ));

	hknpManifold* manifoldStart = manifoldBuffer;

	const int deltaDistCombined = cdBodyA.m_body->m_maxTimDistance + cdBodyB->m_body->m_maxTimDistance;


	// If we have welding, don't fire processCallbacks during collisions, do this after welding
	hkBool32 delayProcessCallback;
	{
		hknpBodyQuality::Flags combinedQualityFlags;
		hknpBodyQuality::combineBodyQualities( cdBodyA.m_quality, cdBodyB->m_quality, &combinedQualityFlags );

		delayProcessCallback = combinedQualityFlags.anyIsSet( hknpBodyQuality::ANY_WELDING );

		// But don't delay if we have no caches. This is important because of 0xf0fg5677.
		if( HK_VERY_UNLIKELY( combinedQualityFlags.anyIsSet(hknpBodyQuality::DROP_MANIFOLD_CACHE) ) )
		{
			delayProcessCallback = 0;
		}
	}

	int numManifoldsInBuffer = 0;

	hknpShapeCollector leafShapeCollector( tl.m_triangleShapePrototypes[1] );

	hkTransform aTb;
	hkBool32 recalcATB = false;	// if set, a getLeafShape() call modified the atb, so we need to reset
	if( numKeys )
	{
		hkMath::prefetch128( srcCache );	// _setMulInverseMul takes a while, get the caches
		hkMath::prefetch128( dstCache );
		aTb._setMulInverseMul( *cdBodyA.m_transform, cdBodyB->m_body->getTransform() );
		//hkVector4Util::_setMulInverseMul( *cdBodyA.m_childTransform, cdBodyB->m_body->getTransform(), &aTb );
	}

	while( numKeys )
	{
		HK_TIMER_SPLIT_LIST2( mStream, "GetData" );

		hknpShapeKey shapeKey;
		hknpQueryFilterData decodedShapeTagDataB;

		// srcCache becomes zero when we have convex-composite collisions,
		// srcCache->m_type gets != TYPE_CONVEX_CONVEX when we reach a hknpChildCompositeCompositeCdCache for composite-composite collisions
		if( srcCache && srcCache->m_type == hknpCollisionCacheType::CONVEX_CONVEX )
		{
			shapeKey = srcCache->getShapeKey();
			if( !keys || shapeKey == *keys )
			{
				// reuse this cache

//#if defined(HK_PLATFORM_XBOX360) || defined(HK_PLATFORM_PS3)
//				hkString::memCpy16<HKNP_MAX_CVX_CVX_CACHE_SIZE>(dstCache, srcCache);
//#else
				hkString::memCpy16NonEmpty( dstCache, srcCache, srcCache->m_sizeInQuads );
//#endif
				const hknpConvexConvexCollisionCache* srcCvxCache = static_cast<const hknpConvexConvexCollisionCache*>(srcCache);
				const int newLinearTim = srcCvxCache->m_linearTim - deltaDistCombined;
				srcCache = srcChildCdCacheConsumer->consumeAndAccessNext( srcCache->getSizeInBytes() );

				hkMath::prefetch128( hkAddByteOffsetConst(srcCache, 256) );

				if( newLinearTim <= deltaDistCombined )		// use current velocity twice
				{
					goto GET_LEAF_SHAPE_AND_CREATE_MANIFOLD;
				}
				dstCache->m_linearTim = hknpLinearTimType(newLinearTim);
				goto ADVANCE_PROCESS_NEXT;
			}
			else if( shapeKey < *keys )
			{
				// key no longer present, so skip this cache
				cdBodyB->m_shapeKey = shapeKey;
				const hknpConvexConvexCollisionCache* srcCvxCache = static_cast<const hknpConvexConvexCollisionCache*>(srcCache);
				srcCvxCache->_destructCdCacheImpl( tl, sharedData, cdBodyA, *cdBodyB, hknpCdCacheDestructReason::AABBS_DONT_OVERLAP );
				srcCache = srcChildCdCacheConsumer->consumeAndAccessNext( srcCache->getSizeInBytes() );
				continue;
			}
		}

		// create new convex convex collision cache
		{
			dstCache->::hknpConvexConvexCollisionCache::init();
			shapeKey = *keys;
			dstCache->setShapeKey( shapeKey );
			dstCache->m_edgeWeldingInfo = (hkUint8)( compShape->getEdgeWeldingInfo(shapeKey) );

			// get leaf shape
			{
				HK_TIMER_SPLIT_LIST2( mStream, "GetLeafShape" );
				leafShapeCollector.reset( cdBodyB->m_body->getTransform() );
				compShape->getLeafShape( shapeKey, &leafShapeCollector );

				HK_ASSERT2( 0x559cc900,
					leafShapeCollector.m_scaleOut.equal( hkVector4::getConstant<HK_QUADREAL_1>() ).allAreSet<hkVector4ComparisonMask::MASK_XYZ>(),
					"Non-identity scale returned from getLeafShape(). This is not supported by collision detection." );

				recalcATB |= leafShapeCollector.m_transformModifiedFlag;
				if( recalcATB )
				{
					aTb._setMulInverseMul( *cdBodyA.m_transform, leafShapeCollector.m_transformOut );
				}
			}

			// decode shape tag
			{
				const hknpBody* bodyB = cdBodyB->m_body;

				hknpShapeTagCodec::Context targetShapeTagContext;
				{
					targetShapeTagContext.m_queryType			= hknpCollisionQueryType::QUERY_AABB;
					targetShapeTagContext.m_body				= bodyB;
					targetShapeTagContext.m_rootShape			= cdBodyB->m_rootShape;
					targetShapeTagContext.m_parentShape			= leafShapeCollector.m_parentShape;
					targetShapeTagContext.m_shapeKey			= shapeKey;
					targetShapeTagContext.m_shape				= leafShapeCollector.m_shapeOut;
					targetShapeTagContext.m_partnerBody			= cdBodyA.m_body;
					targetShapeTagContext.m_partnerRootShape	= cdBodyA.m_rootShape;
					targetShapeTagContext.m_partnerShapeKey		= cdBodyA.m_shapeKey;
					targetShapeTagContext.m_partnerShape		= cdBodyA.m_leafShape;
				}

				// Pre-fill collision filter info, material and user data (in case they are not overwritten in decode()).
				decodedShapeTagDataB.setFromBody( *bodyB );

				tl.m_shapeTagCodec->decode(
					leafShapeCollector.m_shapeTagPath.begin(), leafShapeCollector.m_shapeTagPath.getSize(),
					leafShapeCollector.m_shapeTagOut, &targetShapeTagContext,
					&decodedShapeTagDataB.m_collisionFilterInfo.ref(), &decodedShapeTagDataB.m_materialId, &decodedShapeTagDataB.m_userData.ref() );
			}

			// Set leaf shape data
			{
				cdBodyB->m_shapeKey				= shapeKey;
				cdBodyB->m_leafShape			= leafShapeCollector.m_shapeOut;
				cdBodyB->m_transform			= &leafShapeCollector.m_transformOut;
				cdBodyB->m_collisionFilterInfo	= decodedShapeTagDataB.m_collisionFilterInfo;
				cdBodyB->m_material				= &tl.m_materials[ decodedShapeTagDataB.m_materialId.value() ];
			}

			// Set cache quality
			dstCache->setQuality( *sharedData.m_solverInfo,
				*cdBodyA.m_body, cdBodyA.m_quality, *cdBodyA.m_material, *cdBodyA.m_leafShape,
				*cdBodyB->m_body, cdBodyB->m_quality, *cdBodyB->m_material, *leafShapeCollector.m_shapeOut );

			goto CREATE_MANIFOLD;
		}

GET_LEAF_SHAPE_AND_CREATE_MANIFOLD:
		

		// Get leaf shape
		{
			HK_TIMER_SPLIT_LIST2( mStream, "GetLeafShape" );
			leafShapeCollector.reset( cdBodyB->m_body->getTransform() );
			compShape->getLeafShape( shapeKey, &leafShapeCollector );

			HK_ASSERT2( 0x6a130780,
				leafShapeCollector.m_scaleOut.equal( hkVector4::getConstant<HK_QUADREAL_1>() ).allAreSet<hkVector4ComparisonMask::MASK_XYZ>(),
				"Non-identity scale returned from getLeafShape(). This is not supported by collision detection." );

			recalcATB |= leafShapeCollector.m_transformModifiedFlag;
			if( recalcATB )
			{
				aTb._setMulInverseMul( *cdBodyA.m_transform, leafShapeCollector.m_transformOut );
			}
		}

		// decode shape tag
		{
			const hknpBody* bodyB = cdBodyB->m_body;

			hknpShapeTagCodec::Context targetShapeTagContext;
			{
				targetShapeTagContext.m_queryType			= hknpCollisionQueryType::QUERY_AABB;
				targetShapeTagContext.m_body				= bodyB;
				targetShapeTagContext.m_rootShape			= cdBodyB->m_rootShape;
				targetShapeTagContext.m_parentShape			= leafShapeCollector.m_parentShape;
				targetShapeTagContext.m_shapeKey			= shapeKey;
				targetShapeTagContext.m_shape				= leafShapeCollector.m_shapeOut;
				targetShapeTagContext.m_partnerBody			= cdBodyA.m_body;
				targetShapeTagContext.m_partnerRootShape	= cdBodyA.m_rootShape;
				targetShapeTagContext.m_partnerShapeKey		= cdBodyA.m_shapeKey;
				targetShapeTagContext.m_partnerShape		= cdBodyA.m_leafShape;
			}

			// Pre-fill collision filter info, material and user data (in case they are not overwritten in decode()).
			decodedShapeTagDataB.setFromBody( *cdBodyB->m_body );

			tl.m_shapeTagCodec->decode(
				leafShapeCollector.m_shapeTagPath.begin(), leafShapeCollector.m_shapeTagPath.getSize(),
				leafShapeCollector.m_shapeTagOut, &targetShapeTagContext,
				&decodedShapeTagDataB.m_collisionFilterInfo.ref(), &decodedShapeTagDataB.m_materialId, &decodedShapeTagDataB.m_userData.ref() );
		}

		// Set leaf shape data
		{
			cdBodyB->m_shapeKey				= shapeKey;
			cdBodyB->m_leafShape			= leafShapeCollector.m_shapeOut;
			cdBodyB->m_transform			= &leafShapeCollector.m_transformOut;
			cdBodyB->m_collisionFilterInfo	= decodedShapeTagDataB.m_collisionFilterInfo;
			cdBodyB->m_material				= &tl.m_materials[ decodedShapeTagDataB.m_materialId.value() ];
		}

CREATE_MANIFOLD:
		{
			/*
			if( getChildOutput.m_shape == tl.m_triangleShapePrototype )
			{
				_debugDisplayTriangle( tl, transformB, tl.m_triangleShapePrototype );
			}
			*/

			HK_ON_CPU( hknpConvexConvexCollisionCache* dstCachePpu = dstCache );
			HK_ON_SPU( hknpConvexConvexCollisionCache* dstCachePpu = childCdCacheWriter->spuToPpu(dstCache) );

			int numManifolds = 0;

#if !defined(HK_PLATFORM_SPU)
			// Check if we can do quick convex-plane collision detection (unlikely)
			if( HK_VERY_UNLIKELY( delayProcessCallback &&
				dstCache->m_qualityFlags.anyIsSet( hknpBodyQuality::ALLOW_CONCAVE_TRIANGLE_COLLISIONS ) &&
				cdBodyA.m_leafShape->getFlags().get( hknpShape::SUPPORTS_BPLANE_COLLISIONS ) &&
				cdBodyB->m_leafShape->getFlags().get( hknpShape::IS_TRIANGLE_OR_QUAD_NO_EDGES ) ) )
			{
				HK_TIMER_SPLIT_LIST2( mStream, "CvxVsPlane" );
				numManifolds = hknpCollision2DFastProcessUtil::collideFlat(
					tl, sharedData, aTb, delayProcessCallback,
					cdBodyA, *cdBodyB, dstCache, dstCachePpu, manifoldBuffer );
			}
#else
			if( HK_VERY_UNLIKELY( delayProcessCallback &&
				dstCache->m_qualityFlags.anyIsSet( hknpBodyQuality::ALLOW_CONCAVE_TRIANGLE_COLLISIONS ) ) )
			{
				HK_WARN_ONCE(0x1d0c7900, "Ignoring body quality flag hknpBodyQuality::ALLOW_CONCAVE_TRIANGLE_COLLISIONS. Not supported on SPU");
			}
#endif
			else
			{
				// Otherwise do normal convex-convex collision detection
				HK_TIMER_SPLIT_LIST2( mStream, "CvxVsCvx" );
#if defined(HKNP_CVX_COLLIDE_INLINE)
				numManifolds = hknpCollision2DFastProcessUtil_convexConvexCollideAndGenerateManifold(
					tl, sharedData, mStream, aTb, delayProcessCallback,
					cdBodyA, *cdBodyB, dstCache, dstCachePpu, manifoldBuffer );
#else
				numManifolds = hknpCollision2DFastProcessUtil::collide(
					tl, sharedData, aTb, delayProcessCallback,
					cdBodyA, *cdBodyB, dstCache, dstCachePpu, manifoldBuffer );
#endif
			}

			manifoldBuffer			+= numManifolds;
			numManifoldsInBuffer	+= numManifolds;
		}

ADVANCE_PROCESS_NEXT:
		{
			// Build Jacobians
			if( ( numManifoldsInBuffer >= HKNP_MAX_NUM_MANIFOLDS_PER_BATCH-1 ) ||	// -1 because of silhouette manifolds
				( numKeys == 1 && numManifoldsInBuffer > 0 ) )						// last key
			{
				HK_ASSERT( 0xf034de45, numManifoldsInBuffer <= HKNP_MAX_NUM_MANIFOLDS_PER_BATCH );

				// Set leaf shape data
				cdBodyB->m_material = manifoldStart->m_materialB;
				cdBodyB->m_leafShape = HK_NULL;	// leaf shapes no longer valid

				_buildJacobiansForManifold(
					tl, sharedData,
					delayProcessCallback, // if we fired processCallbacks already, we have to ignore postMeshCallbacks
					manifoldStart, numManifoldsInBuffer, mStream,
					cdBodyA, cdBodyB,
					jacMxSorter, liveJacInfoWriter );

				manifoldBuffer = manifoldStart;
				numManifoldsInBuffer = 0;

				HK_ON_SPU( hkSpuMonitorCache::dmaMonitorDataToMainMemorySpu() );
			}

			// Advance dst cache
			dstCache = static_cast<hknpConvexConvexCollisionCache*>(
				childCdCacheWriter->advanceAndReserveNext( dstCache->getSizeInBytes(), HKNP_MAX_CVX_CVX_CACHE_SIZE ) );
		}

		// Advance to next hit
		keys = hkAddByteOffsetConst( keys, keysStriding );	
		--numKeys;
	}

	// skip any leftover caches
	while( srcCache && srcCache->m_type == hknpCollisionCacheType::CONVEX_CONVEX )
	{
		const hknpConvexConvexCollisionCache* srcCvxCache = static_cast<const hknpConvexConvexCollisionCache*>(srcCache);
		srcCvxCache->_destructCdCacheImpl( tl, sharedData, cdBodyA, *cdBodyB, hknpCdCacheDestructReason::AABBS_DONT_OVERLAP );
		srcCache = srcChildCdCacheConsumer->consumeAndAccessNext( srcCache->getSizeInBytes() );
	}

	HK_ASSERT( 0xf045dfed, numManifoldsInBuffer == 0 );
}


int hknpConvexCompositeCollisionDetector::queryAabbWithNmp(
	hknpCollisionQueryContext* queryContext,
	const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpShape& targetShape, const hknpShapeQueryInfo& targetShapeInfo, const hknpQueryFilterData& targetShapeFilterData,
	hkArray<hknpShapeKey>* hitLeafs, hknpQueryAabbNmp* HK_RESTRICT nmpInOut )
{
	{
#if defined(COUNT_NMP)
		HK_MONITOR_ADD_VALUE( "Queries", 1.0f, HK_MONITOR_TYPE_INT);
#endif

		hknpQueryAabbNmpUtil::resetNmp( nmpInOut );

		HK_ON_DEBUG( const int hitsCapacity = hitLeafs->getCapacity(); );

		// We use the shape implementation directly here to be able to pass in the queryShapeInfo.
		targetShape.queryAabbImpl( queryContext, query, queryShapeInfo, targetShapeFilterData, targetShapeInfo, hitLeafs, nmpInOut );

		HK_WARN_ON_DEBUG_IF( hitLeafs->getCapacity() > hitsCapacity, 0x45AABB24,
			"Reallocated queryAabb() results array from " << hitsCapacity << " to " << hitLeafs->getCapacity() << ". " <<
			"This could be due to highly detailed composite shapes.");
	}

	hknpShapeKey* keys = hitLeafs->begin();
	int numKeys = hitLeafs->getSize();
	HK_MONITOR_ADD_VALUE( "NumHits", hkFloat32(numKeys), HK_MONITOR_TYPE_INT );

	HK_TIMER_SPLIT_LIST( "Sort");
	hkAlgorithm::quickSort( keys, numKeys );

	//HK_ASSERT(0x561acf5a, numKeys < 250 );

	//HK_ON_DEBUG( HK_DISPLAY_BOUNDING_BOX( *(hkAabb*)nmpInOut, hkColor::GREEN ) );

#if !defined(HK_PLATFORM_SPU) && defined(DEBUG_TRIANGLES)
	{
		for ( int i =0; i < hitLeafs->getSize(); i++)
		{
			hknpShapeCollectorWithInplaceTriangle leafShapeCollector;
			targetShape.getLeafShape( (*hitLeafs)[i], &leafShapeCollector );
			hkTransform t = cdBodyB->m_body->getTransform();
			t.getTranslation()(1) += 0.01f;
			hkVector4 v0; v0.setTransformedPos( t, leafShapeCollector.m_triangleShapePrototype->getVertex(0));
			hkVector4 v1; v1.setTransformedPos( t, leafShapeCollector.m_triangleShapePrototype->getVertex(1));
			hkVector4 v2; v2.setTransformedPos( t, leafShapeCollector.m_triangleShapePrototype->getVertex(2));

			HK_DISPLAY_TRIANGLE( v0,v1,v2, 0x80ff80ff + 0x2000 * i );
		}
	}
#endif

	return numKeys;
}


void hknpConvexCompositeCollisionDetector::collideWithChildren(
	const hknpSimulationThreadContext& tl, const hknpInternalCollideSharedData& sharedData,
	hknpCdBody* HK_RESTRICT cdBodyA, const hknpShape* shapeA,
	hknpCdBody* HK_RESTRICT cdBodyB, const hknpShape* shapeB,
	hknpCompositeCollisionCache* HK_RESTRICT compositeCdCache,
	hknpCdCacheStream* HK_RESTRICT childCdCacheStream, hknpCdCacheStream* HK_RESTRICT childCdCacheStreamPpu,
	hknpCdCacheWriter* HK_RESTRICT childCdCachesOut,   hknpMxJacobianSorter* jacMxSorter,
	hknpLiveJacobianInfoWriter* liveJacInfoWriter )
{
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& monitorStream = hkMonitorStream::getInstance();
#endif

	HK_TIMER_BEGIN_LIST2( monitorStream, "CvxVsComposite", "QueryAabb" );

	{
		// Body A is convex.
		cdBodyA->m_leafShape = shapeA;

		
		cdBodyB->m_leafShape = HK_NULL;
		cdBodyB->m_transform = HK_NULL;
	}

	//
	//	Mid phase (query AABB)
	//

	// Calculate expanded AABB of A in B-space
	hkAabb localSpaceAabbBodyA;
	buildExpandedLocalSpaceAabb( sharedData, *cdBodyA, *cdBodyB, &localSpaceAabbBodyA );

	// Allocate space for queryAabb() results.
	// Use an inplace array for the hits so that they are stack allocated, but can use the heap if more capacity is needed.
	// On SPU, queryAabb() implementations should never try to use the heap, dropping hits instead.
	hkInplaceArray< hknpShapeKey, 1024 > hits;

	hknpShapeKey* keys;
	int keysStriding;		
	int numKeys;

	// Check if we should discard the NMP. Probably caused by a shape change (see hknpShapeManager)
	const bool discardNmp = ( cdBodyA->m_body->m_maxTimDistance | cdBodyB->m_body->m_maxTimDistance ) == 0xffff;
	if( !discardNmp &&
		hknpQueryAabbNmpUtil::checkNmpStillValid( localSpaceAabbBodyA, *compositeCdCache->getQueryAabbNmp(), &compositeCdCache->m_nmpTimeToLive ) )
	{
		keys = HK_NULL;
		keysStriding = 0;
		numKeys = compositeCdCache->m_numHits;

		if( numKeys == 0 )
		{
			// we had and have no hits, use TIM.
			hkSimdReal dist = hknpQueryAabbNmpUtil::calcTIMWithNmp( localSpaceAabbBodyA, *compositeCdCache->getQueryAabbNmp() );
			hknpMotionUtil::convertDistanceToLinearTIM( *sharedData.m_solverInfo, dist, compositeCdCache->m_linearTim );
			HK_ASSERT( 0xf03fde34, compositeCdCache->m_childCdCacheRange.isEmpty() );
			goto END_OF_FUNCTION;
		}
	}
	else
	{
		hknpCollisionQueryContext aabbQueryContext( HK_NULL, HK_NULL );
		aabbQueryContext.m_shapeTagCodec = tl.m_shapeTagCodec;

		hknpAabbQuery aabbQuery;
		{
			aabbQuery.m_aabb = localSpaceAabbBodyA;
			aabbQuery.m_filterData.m_collisionFilterInfo = cdBodyA->m_body->m_collisionFilterInfo;
			aabbQuery.m_filterData.m_materialId = cdBodyA->m_body->m_materialId;
			aabbQuery.m_filterData.m_userData = cdBodyA->m_body->m_userData;
			aabbQuery.m_filter = tl.m_modifierManager->getCollisionFilter();
		}

		// Shape A and B are both root-level shapes, so no parent shapes available to set here.
		hknpShapeQueryInfo queryShapeInfo;
		{
			queryShapeInfo.m_body = cdBodyA->m_body;
			queryShapeInfo.m_rootShape = cdBodyA->m_rootShape;
			queryShapeInfo.m_shapeToWorld = &cdBodyA->m_body->getTransform();
		}

		hknpShapeQueryInfo targetShapeInfo;
		{
			targetShapeInfo.m_body = cdBodyB->m_body;
			targetShapeInfo.m_rootShape = cdBodyB->m_rootShape;
			targetShapeInfo.m_shapeToWorld = &cdBodyB->m_body->getTransform();
		}

		hknpQueryFilterData targetShapeFilterData( *cdBodyB->m_body );

		// Do the query
		numKeys = queryAabbWithNmp(
			&aabbQueryContext, aabbQuery, queryShapeInfo, *cdBodyB->m_rootShape,
			targetShapeInfo, targetShapeFilterData,
			&hits, compositeCdCache->getQueryAabbNmp() );

		
		{
			hknpBodyFlags enabledModifiers = tl.m_modifierManager->getCombinedBodyRootFlags(tl.m_materials, *cdBodyA, *cdBodyB);
			if ( tl.m_modifierManager->isFunctionRegistered( hknpModifier::FUNCTION_POST_COMPOSITE_QUERY_AABB, enabledModifiers ))
			{
				int newSize = hits.getSize();
				HKNP_FIRE_MODIFIER( tl.m_modifierManager, hknpModifier::FUNCTION_POST_COMPOSITE_QUERY_AABB, enabledModifiers,
					newSize = modifier->postCompositeQueryAabb( tl, sharedData, aabbQuery, queryShapeInfo, targetShapeInfo, hits.begin(), hits.getSize(), hits.getCapacity() ) );
				hits.setSize(newSize);
			}
		}

		keys = hits.begin();
		keysStriding = sizeof(hknpShapeKey);
		compositeCdCache->m_numHits = numKeys;
	}

	//
	//	Narrow phase
	//

	if( numKeys > 0 || !compositeCdCache->m_childCdCacheRange.isEmpty() )
	{
		// Allocate space for the manifolds that will be created
		hknpManifold* manifolds = hkAllocateStack<hknpManifold>( HKNP_MAX_NUM_MANIFOLDS_PER_BATCH, "manifoldsBuffer" );
#if defined (HK_WANT_DETERMINISM_CHECKS) /* || defined (HK_DEBUG) */
		hkString::memSet( manifolds, 0xcd, HKNP_MAX_NUM_MANIFOLDS_PER_BATCH * sizeof(hknpManifold) );
#endif

		compositeCdCache->m_linearTim = 0;

		hknpCdCacheConsumer cvxCachesReader;
		cvxCachesReader.initSpu( HKNP_SPU_DMA_GROUP_STALL, 1, "CvxVsCompositeCacheConsumer" );
		HK_SPU_STACK_POINTER_CHECK();
		cvxCachesReader.setToRange( tl.m_heapAllocator, childCdCacheStream, childCdCacheStreamPpu, &compositeCdCache->m_childCdCacheRange );

		hknpConvexCompositeCollisionDetector::collideConvexWithCompositeKeys(
			tl, sharedData,
			*cdBodyA, cdBodyB,
			keys, keysStriding, numKeys,
			&cvxCachesReader, childCdCachesOut, manifolds, jacMxSorter, liveJacInfoWriter );

		cvxCachesReader.exitSpu();
		hkDeallocateStack( manifolds, HKNP_MAX_NUM_MANIFOLDS_PER_BATCH );
	}

END_OF_FUNCTION:
	hits.clearAndDeallocate();
	jacMxSorter->resetHintUsingSameBodies();

	HK_TIMER_END_LIST2( monitorStream );
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
