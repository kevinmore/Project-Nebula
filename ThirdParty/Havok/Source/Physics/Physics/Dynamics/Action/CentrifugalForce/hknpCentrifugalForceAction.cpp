/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Action/CentrifugalForce/hknpCentrifugalForceAction.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>

hknpCentrifugalForceAction::hknpCentrifugalForceAction( hknpBodyId idA )
: hknpUnaryAction( idA  )
{
}

hknpCentrifugalForceAction::ApplyActionResult hknpCentrifugalForceAction::applyAction( const hknpSimulationThreadContext& tl, const hknpSolverInfo& stepInfo, hknpCdPairWriter* HK_RESTRICT pairWriter )
{
	hknpWorld* world = tl.m_world;

	const hknpBody* bodyA;
	ApplyActionResult result = getAndCheckBodies( world, bodyA );
	if ( result != RESULT_OK )
	{
		return result;
	}

	// apply centrifugal forces
	{
		hknpMotion* HK_RESTRICT motionA = getMotion( world, *bodyA );

		hkVector4 invInertia; motionA->getInverseInertiaLocal(invInertia);

		hkVector4 one = hkVector4::getConstant<HK_QUADREAL_1>();
		hkVector4 inertia; inertia.setDiv<HK_ACC_12_BIT, HK_DIV_SET_ZERO>( one, invInertia );

		hkVector4 angVelSqrd;

		hkVector4 angVel = motionA->m_angularVelocity;
		angVelSqrd.setMul( angVel, angVel );
		hkSimdReal energy = angVelSqrd.dot<3>( inertia );

		hkVector4 cIn; cIn.setCross( inertia, one );
		cIn.mul( invInertia );

		hkVector4 maxAng; maxAng.setConstant<HK_QUADREAL_256>();
		hkVector4 angForCentrifugal; angForCentrifugal.setMin( angVel, maxAng );
		hkVector4 angP1; angP1.setPermutation<hkVectorPermutation::YZXZ>( angForCentrifugal );
		hkVector4 angP2; angP2.setPermutation<hkVectorPermutation::ZXYZ>( angForCentrifugal );
		hkVector4 angPSqrd; angPSqrd.setMul( angP1, angP2 );
		hkVector4 hp; hp.setMul( cIn, angPSqrd );

		hkSimdReal dt = stepInfo.m_deltaTime;
		angVel.addMul( dt, hp );

		angVelSqrd.setMul( angVel, angVel );
		hkSimdReal energy2;		 energy2.setDot<3>( angVelSqrd, inertia );
		hkSimdReal angVelFactor; angVelFactor.setDiv<HK_ACC_23_BIT, HK_DIV_SET_ZERO>( energy, energy2 );
		angVelFactor = angVelFactor.sqrt<HK_ACC_23_BIT, HK_SQRT_SET_ZERO>();
		angVel.mul( angVelFactor );
		motionA->m_angularVelocity = angVel;
	}
	return hknpAction::RESULT_OK;
}
//
// // apply centrifugal forces
// {
// 	const int inertiaOffset = HK_OFFSET_OF(hknpMotion,m_invInertia);
// 	hkMxVector<MXLENGTH> invInertia; invInertia.hkMxVector<MXLENGTH>::template gatherHalfsWithOffset<inertiaOffset>((const void**)&s.m_motions[0]);
//
// 	hkMxVector<MXLENGTH> one; one.setConstant<HK_QUADREAL_1>();
// 	hkMxVector<MXLENGTH> inertia; inertia.setDiv<HK_ACC_12_BIT, HK_DIV_SET_ZERO>( one, invInertia );
//
// 	hkMxVector<MXLENGTH> angVelSqrd;
//
// 	angVelSqrd.setMul( angVelClipped, angVelClipped );
// 	hkMxReal<MXLENGTH> energy; energy.setDot<3>( angVelSqrd, inertia );
//
// 	hkMxVector<MXLENGTH> cIn; cIn.setCross( inertia, one );
// 	cIn.mul( invInertia );
//
// 	hkMxVector<MXLENGTH> maxAng; maxAng.setConstant<HK_QUADREAL_256>();
// 	hkMxVector<MXLENGTH> angForCentrifugal; angForCentrifugal.setMin( angVelClipped, maxAng );
// 	hkMxVector<MXLENGTH> angP1; angP1.setComponentPermutation<hkVectorPermutation::YZXZ>( angForCentrifugal );
// 	hkMxVector<MXLENGTH> angP2; angP2.setComponentPermutation<hkVectorPermutation::ZXYZ>( angForCentrifugal );
// 	hkMxVector<MXLENGTH> angPSqrd; angPSqrd.setMul( angP1, angP2 );
// 	hkMxVector<MXLENGTH> hp; hp.setMul( cIn, angPSqrd );
//
// 	hkMxReal<MXLENGTH> dt; dt.setBroadcast( solverInfo->m_deltaTime );
// 	angVelClipped.addMul( dt, hp );
//
// 	angVelSqrd.setMul( angVelClipped, angVelClipped );
// 	hkMxReal<MXLENGTH> energy2; energy2.setDot<3>( angVelSqrd, inertia );
// 	hkMxReal<MXLENGTH> angVelFactor; angVelFactor.setDiv<HK_ACC_23_BIT, HK_DIV_SET_ZERO>( energy, energy2 );
// 	angVelFactor.sqrt<HK_ACC_23_BIT, HK_SQRT_SET_ZERO>();
// 	angVelClipped.mul( angVelFactor );
// }

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
