/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/DebugUtil/TraceStream/hkTraceStream.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Types/Physics/ContactPoint/hkContactPoint.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Internal/Solver/SimpleConstraints/hkpSimpleConstraintUtil.h>
#include <Physics2012/Dynamics/Constraint/Contact/hkpContactPointProperties.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>

#include <Physics2012/Dynamics/Constraint/Response/hkpSimpleCollisionResponse.h>
#include <Physics2012/Dynamics/Motion/hkpMotion.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Collide/hkpDynamicsContactMgr.h>
#include <Physics2012/Dynamics/Constraint/Contact/hkpSimpleContactConstraintData.h>


void hkpSimpleCollisionResponse::solveSingleContact( const hkContactPoint& cp, hkTime time, hkpSimpleConstraintUtilCollideParams& params, 
												    hkpMotion* bodyA, hkpMotion* bodyB, hkpDynamicsContactMgr* contactMgr, SolveSingleOutput& output )
{
	hkSimdReal timeSr; timeSr.setFromFloat(time);
	hkpSimpleConstraintInfoInitInput inA;
	{
		inA.m_invMasses = bodyA->m_inertiaAndMassInv;
		hkVector4 massCenter; hkSweptTransformUtil::calcCenterOfMassAt( bodyA->m_motionState, timeSr, massCenter );
		inA.m_massRelPos.setSub( cp.getPosition(), massCenter );
		bodyA->getInertiaInvWorld( inA.m_invInertia );
		inA.m_transform = &bodyA->getTransform();
	}

	hkpSimpleConstraintInfoInitInput inB;
	{
		inB.m_invMasses = bodyB->m_inertiaAndMassInv;
		hkVector4 massCenter; hkSweptTransformUtil::calcCenterOfMassAt( bodyB->m_motionState, timeSr, massCenter );
		inB.m_massRelPos.setSub( cp.getPosition(), massCenter );
		bodyB->getInertiaInvWorld( inB.m_invInertia );
		inB.m_transform = &bodyB->getTransform();
	}

	hkpBodyVelocity velA;
	hkpBodyVelocity velB;
	{
		velA.m_linear  = bodyA->getLinearVelocity();
		velA.m_angular = bodyA->getAngularVelocity();
		velB.m_linear  = bodyB->getLinearVelocity();
		velB.m_angular = bodyB->getAngularVelocity();
	}

	// use collision-related velocities for calculation of m_velocityKeyframedA/B
	
	
	hkpBodyVelocity origVelA; hkSweptTransformUtil::getVelocity( bodyA->m_motionState, origVelA.m_linear, origVelA.m_angular);
	hkpBodyVelocity origVelB; hkSweptTransformUtil::getVelocity( bodyB->m_motionState, origVelB.m_linear, origVelB.m_angular);

	hkpSimpleConstraintInfo info;
	hkRotation directions;
	hkVector4Util::buildOrthonormal( cp.getNormal(), directions );

	// insert callback here
	contactMgr->toiCollisionResponseBeginCallback( cp, inA, velA, inB, velB );

	hkSimpleConstraintUtil_InitInfo( inA, inB, directions, info );


	// This takes the m_extarnalSeparatingVelocity param as the current collision-detection velocities 
	output.m_impulse = hkSimpleConstraintUtil_Collide( info, params, velA, velB );

	contactMgr->toiCollisionResponseEndCallback( cp, output.m_impulse, inA, velA, inB, velB );


	if ( hkDebugToi)
	{
		hkVector4 pV; hkSimpleConstraintUtil_getPointVelocity(info, velA, velB, pV );
		hkToiPrintf( "Post", "#         Post v:%2.4f  i:%4.4f\n", pV(0), output.m_impulse );
	}

	//
	//	Check for delaying one impulse
	//
	{
		hkVector4 pA; hkSimpleConstraintUtil_getPointVelocity(info, origVelA, velB, pA );
		hkVector4 pB; hkSimpleConstraintUtil_getPointVelocity(info, velA, origVelB, pB );
#ifdef HK_DEBUG
		hkVector4 finalVelocityWhenBothBodiesAreReintegrated; hkSimpleConstraintUtil_getPointVelocity(info, velA, velB, finalVelocityWhenBothBodiesAreReintegrated );
		if (finalVelocityWhenBothBodiesAreReintegrated(0) < -HK_REAL_EPSILON && !params.m_contactImpulseLimitBreached )
		{
			HK_WARN(0xad45d441, "Internal warning. SCR generated invalid velocities");
		}
#endif

		output.m_velocityKeyframedA = pA(0);
		output.m_velocityKeyframedB = pB(0);
	}

	//HK_ASSERT(0xAD000002, output.m_velocityKeyframedA < 0.0f || output.m_velocityKeyframedB < 0.0f);

	//
	//	Write back the results
	//
	{
		if (bodyA->m_type != hkpMotion::MOTION_FIXED)
		{
			bodyA->setLinearVelocity( velA.m_linear );
			bodyA->setAngularVelocity( velA.m_angular );
		}

		if (bodyB->m_type != hkpMotion::MOTION_FIXED)
		{
			bodyB->setLinearVelocity( velB.m_linear );
			bodyB->setAngularVelocity( velB.m_angular );
		}
	}
}

void hkpSimpleCollisionResponse::solveSingleContact2( class hkpSimpleContactConstraintData* constraintData,
													 const hkContactPoint& cp, hkpSimpleConstraintUtilCollideParams& params,
													 hkpRigidBody* rigidBodyA, hkpRigidBody* rigidBodyB, 
													 hkpVelocityAccumulator* bodyA, hkpVelocityAccumulator* bodyB,
													 SolveSingleOutput2& output )
{
	//hkReal sumEnergy = 0.0f;
	hkpBodyVelocity vela[2];
	hkpSimpleConstraintInfoInitInput ina[2];
	{
		hkpVelocityAccumulator* body = bodyA;
		hkpRigidBody* rigidBody = rigidBodyA;
		for ( int i = 0; i< 2; i++ )
		{
			hkpSimpleConstraintInfoInitInput& in = ina[i];
			hkpBodyVelocity& vel = vela[i];
			{
				in.m_invMasses = body->m_invMasses;
				const hkVector4& massCenter = body->getCenterOfMassInWorld();
				const hkVector4& cpPos = cp.getPosition();
				in.m_massRelPos.setSub( cpPos, massCenter ); 

				const hkVector4& iD = body->m_invMasses;
				vel.m_linear = body->m_linearVel;

				{
					hkRotation t = body->getCoreFromWorldMatrix();
					vel.m_angular._setRotatedInverseDir( t, body->m_angularVel );

// 					{	// energy calculation
// 						hkVector4 invMasses; invMasses.setMax4( body->m_invMasses, hkVector4::getConstant(HK_QUADREAL_EPS));
// 						hkVector4 mass; mass.setReciprocal4( invMasses );
// 						hkVector4 angVelSqrd; angVelSqrd.setMul4( body->m_angularVel, body->m_angularVel );
// 						hkVector4 linVelSqrd; linVelSqrd.setMul4( body->m_linearVel, body->m_linearVel );
// 						angVelSqrd.mul4( mass );
// 						linVelSqrd.mul4( mass.getSimdAt(3));
// 						angVelSqrd.add4(linVelSqrd );
// 						sumEnergy += angVelSqrd.horizontalAdd3();
// 					}

					// this is the correct inertia calculation, see HVK-4983
					{
					    t.transpose();
					    hkMatrix3 x;	
					    x.getColumn(0).setMul( iD.getComponent<0>(), t.getColumn<0>() );
					    x.getColumn(1).setMul( iD.getComponent<1>(), t.getColumn<1>() );
					    x.getColumn(2).setMul( iD.getComponent<2>(), t.getColumn<2>() );
					    in.m_invInertia.setMul( x, body->getCoreFromWorldMatrix() );
					}
				}
				in.m_transform = &rigidBody->getTransform();
			}
			body = bodyB;
			rigidBody = rigidBodyB;
		}
	}

	HK_ALIGN_REAL( hkpSimpleConstraintInfo info );
	hkRotation directions;
	hkVector4Util::buildOrthonormal( cp.getNormal(), directions );

	constraintData->collisionResponseBeginCallback( cp, ina[0], vela[0], ina[1], vela[1] );

	hkSimpleConstraintUtil_InitInfo( ina[0], ina[1], directions, info );
	output.m_impulse = hkSimpleConstraintUtil_Collide( info, params, vela[0], vela[1] );

	constraintData->collisionResponseEndCallback( cp, output.m_impulse, ina[0], vela[0], ina[1], vela[1] );

	//
	//	Energy calculations
	//
// 	{
// 		hkReal sumEnergyAfter = 0.0f;
// 
// 		hkpVelocityAccumulator* body = bodyA;
// 		for (int j=0; j < 2; body=bodyB,j++)
// 		{
// 			hkpBodyVelocity& vel = vela[j];
// 			hkVector4 angVel; angVel._setRotatedDir( body->getCoreFromWorldMatrix(), vel.m_angular );
// 
// 			hkVector4 invMasses; invMasses.setAdd4( body->m_invMasses, hkVector4::getConstant(HK_QUADREAL_EPS));
// 			hkVector4 mass; mass.setReciprocal4( invMasses );
// 			hkVector4 angVelSqrd; angVelSqrd.setMul4( angVel, angVel );
// 			hkVector4 linVelSqrd; linVelSqrd.setMul4( vel.m_linear, vel.m_linear );
// 			angVelSqrd.mul4( mass );
// 			linVelSqrd.mul4( mass.getSimdAt(3));
// 			angVelSqrd.add4(linVelSqrd );
// 			sumEnergyAfter += angVelSqrd.horizontalAdd3();
// 		}
// 		HK_REPORT( "Energy " << sumEnergy << " " << sumEnergyAfter );
// 	}



	//
	//	Write back the results
	//
	{
		bodyA->m_linearVel  = vela[0].m_linear;

		if(rigidBodyA->getRigidMotion()->m_type != hkpMotion::MOTION_FIXED)
		{	
			rigidBodyA->getRigidMotion()->m_linearVelocity  = vela[0].m_linear;
			rigidBodyA->getRigidMotion()->m_angularVelocity = vela[0].m_angular;
		}

		bodyB->m_linearVel  = vela[1].m_linear;

		if(rigidBodyB->getRigidMotion()->m_type != hkpMotion::MOTION_FIXED)
		{	
			rigidBodyB->getRigidMotion()->m_linearVelocity  = vela[1].m_linear;
			rigidBodyB->getRigidMotion()->m_angularVelocity = vela[1].m_angular;
		}

		hkpVelocityAccumulator* body = bodyA;
		for (int j=0; j < 2; body=bodyB,j++)
		{
			hkpBodyVelocity& vel = vela[j];
			body->m_angularVel._setRotatedDir( body->getCoreFromWorldMatrix(), vel.m_angular );
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
