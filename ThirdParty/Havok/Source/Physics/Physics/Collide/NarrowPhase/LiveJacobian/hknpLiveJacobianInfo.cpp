/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Collide/NarrowPhase/LiveJacobian/hknpLiveJacobianInfo.h>
#include <Physics/Physics/Collide/hknpCdBody.h>


void hknpLiveJacobianInfo::initLiveJacobian( const hknpManifold* manifold, const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB, Type type)
{
	hkSimdReal maxLinearMovement;
	{
		hkSimdReal maxLinearMovementA; maxLinearMovementA.setFromFloat( cdBodyA.m_quality->m_liveJacobianDistanceThreshold );
		hkSimdReal maxLinearMovementB; maxLinearMovementB.setFromFloat( cdBodyB.m_quality->m_liveJacobianDistanceThreshold );
		maxLinearMovement.setMin( maxLinearMovementA, maxLinearMovementB );
	}

	hkSimdReal maxAngularMovement;
	{
		hkSimdReal maxAngularMovementA; maxAngularMovementA.setFromFloat( cdBodyA.m_quality->m_liveJacobianAngleThreshold );
		hkSimdReal maxAngularMovementB; maxAngularMovementB.setFromFloat( cdBodyB.m_quality->m_liveJacobianAngleThreshold );
		maxAngularMovement.setMin( maxAngularMovementA, maxAngularMovementB );
	}

	hkSimdReal currentDistance = manifold->m_distances.horizontalMin<4>();

	maxLinearMovement.store<1>( &m_maxLinearMovement );
	maxAngularMovement.store<1>( &m_maxAngularMovement );
	currentDistance.store<1>( &m_currentDistance );

	this->m_substepOfLastBuildJac = 0;
	this->m_type = type;
	this->m_cache = (const hknpConvexConvexManifoldCollisionCache*) HK_PADSPU_REF(manifold->m_collisionCache);
}

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
