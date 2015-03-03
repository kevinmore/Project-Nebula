/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Collide/NarrowPhase/LiveJacobian/hknpLiveJacobianInfo.h>

#include <Physics/Internal/Collide/Agent/ProcessCollision2DFast/hknpCollision2DFastProcessUtil.h>
#if defined(HKNP_CVX_COLLIDE_INLINE)
#	include <Physics/Internal/Collide/Agent/ProcessCollision2DFast/hknpCollision2DFastProcessUtil.inl>	
#endif


static HK_FORCE_INLINE void hknpCvxCvxCdDetector_buildJacobians(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, hkMonitorStream* HK_RESTRICT mStream,
	const hknpCdBody& cdBodyA, hknpCdBody* HK_RESTRICT cdBodyB,
	hknpManifold* HK_RESTRICT manifolds, int numManifoldCreated,
	hknpConvexConvexManifoldCollisionCache* HK_RESTRICT manifoldCdCache, hknpMxJacobianSorter* HK_RESTRICT jacMxSorter,
	hknpLiveJacobianInfoWriter* HK_RESTRICT liveJacInfoWriter )
{
	HK_TIMER_SPLIT_LIST2( (*mStream), "BuildJac");
	hkUint32 bodyIdHashCode = hknpMxJacobianSorter::calcBodyIdsHashCode( *cdBodyA.m_body, *cdBodyB->m_body );

	hknpContactSolverSetup::BuildConfig buildConfig; buildConfig.init( cdBodyA.m_motion, cdBodyB->m_motion );

	int i = 0;
	hknpLiveJacobianInfo liveJacInfo;
	liveJacInfo.m_cache	= manifoldCdCache;
	liveJacInfo.m_numManifolds = hkUint8(numManifoldCreated);
	do
	{
		HK_PAD_ON_SPU(hknpMxContactJacobian*) mxJac = HK_NULL;
		HK_PAD_ON_SPU(hknpMxContactJacobian*) mxJacOnPpu = HK_NULL;
		HK_ON_SPU(int mxJacIdx = jacMxSorter-> getJacobianLocation( bodyIdHashCode, &mxJac, &mxJacOnPpu));
		HK_ON_CPU(int mxJacIdx = jacMxSorter->_getJacobianLocation( bodyIdHashCode, &mxJac, &mxJacOnPpu));
		HK_ASSERT2( 0xf043dedf, (mxJacIdx != 0) || mxJac->m_manifoldData[0].m_solverVelIdA == 0 , "Your jacobians seem not to be zeroed.");
		hknpManifold* HK_RESTRICT manifold = &manifolds[i];

		cdBodyB->m_material = manifold->m_materialB;

		// Store collisionCache-jacobian locations to use to generate live jacobians
		liveJacInfo.m_jacobian[i] = mxJac;
		liveJacInfo.m_indexOfManifoldInJacobian[i] = hkUint8(mxJacIdx);
#if defined(HKNP_CVX_COLLIDE_INLINE)
		(hknpContactSolverSetup::_buildContactJacobianForSingleManifold<hknpContactSolverSetup::BUILD_USING_CACHE>)
#else
		(hknpContactSolverSetup::buildContactJacobianForSingleManifold<hknpContactSolverSetup::BUILD_USING_CACHE>)
#endif
			( &tl, *sharedData.m_solverInfo, manifoldCdCache, manifold, &buildConfig,
			cdBodyA, HK_NULL,
			*cdBodyB, HK_NULL,
			mxJac, mxJacOnPpu,  mxJacIdx  );

	}
	while(++i < numManifoldCreated );

#if !defined(HK_PLATFORM_SPU)
	if (manifoldCdCache->m_qualityFlags.anyIsSet(hknpBodyQuality::ENABLE_LIVE_JACOBIANS) )
	{
		liveJacInfo.initLiveJacobian( &manifolds[0], cdBodyA, *cdBodyB );
		liveJacInfoWriter->write16(&liveJacInfo, sizeof(liveJacInfo));
	}
#endif
}


HK_FORCE_INLINE void HK_CALL hknpConvexConvexCollisionDetector::collideConvexConvex(
	const hknpSimulationThreadContext& tl,  const hknpModifierSharedData& sharedData, hkMonitorStream* mStream,
	const hknpCdBody& cdBodyA, hknpCdBody& cdBodyB,
	hknpConvexConvexCollisionCache* HK_RESTRICT cvxCvxCache, hknpConvexConvexCollisionCache* HK_RESTRICT cvxCvxCachePpu,
	hknpMxJacobianSorter* jacMxSorter, hknpLiveJacobianInfoWriter* liveJacInfoWriter )
{
	//
	// Create manifolds for a pair of convex objects
	//

	hknpManifold manifolds[2];
	int numManifoldsCreated;
	{
		const bool delayProcessCallback = false;
		hkTransform aTb; aTb._setMulInverseMul( *cdBodyA.m_transform, *cdBodyB.m_transform );

#if defined(HKNP_CVX_COLLIDE_INLINE)
		numManifoldsCreated = hknpCollision2DFastProcessUtil_convexConvexCollideAndGenerateManifold(
			tl, sharedData, *mStream, aTb, delayProcessCallback,
			cdBodyA, cdBodyB, cvxCvxCache, cvxCvxCachePpu, manifolds );
#else
		numManifoldsCreated = hknpCollision2DFastProcessUtil::collide(
			tl, sharedData, aTb, /*mStream, */delayProcessCallback,
			cdBodyA, cdBodyB,	cvxCvxCache, cvxCvxCachePpu, manifolds );
#endif

		HK_ASSERT( 0xf00045fe, numManifoldsCreated <= 2 );
	}

	//
	// Build Jacobians
	//

	if( numManifoldsCreated )
	{
		
		//HK_ASSERT( 0xf00045ff, cvxCvxCache->hasManifoldData() );
		hknpConvexConvexManifoldCollisionCache* cvxCvxManifoldCache = static_cast<hknpConvexConvexManifoldCollisionCache*>(cvxCvxCache);

		if( !(cvxCvxManifoldCache->m_bodyAndMaterialFlags & hknpBody::DONT_BUILD_CONTACT_JACOBIANS ) )
		{
			hknpCvxCvxCdDetector_buildJacobians(
				tl, sharedData, mStream,
				cdBodyA, &cdBodyB,
				manifolds, numManifoldsCreated, cvxCvxManifoldCache, jacMxSorter, liveJacInfoWriter );
		}
		else
		{
			cvxCvxManifoldCache->m_manifoldSolverInfo.m_contactJacobian = HK_NULL;
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
