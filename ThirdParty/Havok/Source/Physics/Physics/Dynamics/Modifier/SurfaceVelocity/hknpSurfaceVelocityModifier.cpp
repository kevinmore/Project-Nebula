/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/Modifier/SurfaceVelocity/hknpSurfaceVelocityModifier.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobianUtil.h>
#include <Physics/Physics/Dynamics/Modifier/SurfaceVelocity/hknpSurfaceVelocity.h>
#include <Physics/Physics/Collide/hknpCdBody.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>


hknpSurfaceVelocityModifier::hknpSurfaceVelocityModifier(  ) : hknpModifier()
{
}


void hknpSurfaceVelocityModifier::postContactJacobianSetup(
	const hknpSimulationThreadContext& tl,
	const hknpSolverInfo& solverInfo,
	const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
	const hknpManifoldCollisionCache* cacheConst, const hknpManifold* manifold,
	hknpMxContactJacobian* HK_RESTRICT mxJac, int mxJacIdx
	)
{
	hkVector4 pivot = manifold->m_positions[0];
	for (int i=1; i < manifold->m_numPoints; i++)
	{
		pivot.add( manifold->m_positions[i] );
	}
	pivot.mul( hkVector4::getConstant( hkVectorConstant(HK_QUADREAL_INV_0 + manifold->m_numPoints) ));

	hkVector4 linVel4; linVel4.setZero();
	hkVector4 angVel4; angVel4.setZero();

	if ( cdBodyB.m_material->m_surfaceVelocity )
	{
		HK_ASSERT( 0xf0df12dc, cdBodyB.m_material->m_surfaceVelocity );
		cdBodyB.m_material->m_surfaceVelocity->calcSurfaceVelocity( pivot, manifold->m_normal, cdBodyB.m_body->getTransform(), &linVel4, &angVel4 );
		linVel4.setNeg<4>(linVel4);
		angVel4.setNeg<4>(angVel4);
	}

	if ( cdBodyA.m_material->m_surfaceVelocity )
	{
		hkVector4 l,a;
		cdBodyA.m_material->m_surfaceVelocity->calcSurfaceVelocity( pivot, manifold->m_normal, cdBodyA.m_body->getTransform(), &l, &a );
		linVel4.add(l);
		angVel4.add(a);
	}

	hknpManifoldCollisionCache* HK_RESTRICT cache = const_cast<hknpManifoldCollisionCache*>(cacheConst);
	hknpContactJacobianUtil::setSurfaceVelocity( solverInfo, cache, mxJac, mxJacIdx, linVel4, angVel4 );
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
