/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/Solver/Event/hknpSolverEvents.h>
#include <Common/Base/Math/Vector/Mx/hkMxVectorUtil.h>
#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>
#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverUtil.h>


void hknpContactSolverEvent::calculateContactPointPositions( hknpWorld* world, hkcdManifold4* HK_RESTRICT manifoldOut ) const
{
	const hknpMxContactJacobian* mxJac = m_contactJacobian;
	const int mxJacIdx = m_manifoldIndex;
	const hknpContactJacobianTypes::HeaderData& mfData = getJacobianHeader();

	// Find the smaller body
	int indexOfSmallerBody = 1;
	{
		hkMxVector<2> quadSum; quadSum.setZero();
		const int numPoints = getNumContactPoints();
		for( int i = 0; i < numPoints; i++ )
		{
			hkMxVector<2> angJac;
			hkMxVectorUtil::loadUnpack( &mxJac->m_contactPointData[i].m_angular[mxJacIdx][0], angJac );
			quadSum.addMul( angJac, angJac );
		}
		hkVector4 dots; quadSum.horizontalAdd<3>( dots );

		if ( dots.getComponent<0>() < dots.getComponent<1>())
		{
			indexOfSmallerBody = 0;
		}
	}

	const hknpBody& bodyA = world->getBody( m_bodyIds[0] );
	const hknpBody& body  = world->getBody( m_bodyIds[indexOfSmallerBody] );

	const hknpMotion& motionA = world->getMotion( bodyA.m_motionId );
	const hknpMotion& motion  = world->getMotion( body.m_motionId );

	hkVector4 normal = m_contactJacobian->m_linear0[m_manifoldIndex];
	manifoldOut->m_normal = normal;

	// We need to reconstruct the contact point from the Jacobians. As the angular part of the Jacobian is stored with
	// lower precision, we need to take the Jacobians from the smaller body (this->m_indexOfSmallerBody)

	hkVector4 shift;
	{
		hkVector4 comMinusComA; comMinusComA.setSub( motion.getCenterOfMassInWorld(), motionA.getCenterOfMassInWorld() );
		hkSimdReal projected = comMinusComA.dot<3>( normal );
		hkSimdReal contactPointMinusComA_DotNormal; contactPointMinusComA_DotNormal.setFromFloat( mfData.m_contactPointMinusComA_DotNormal );
		hkSimdReal contactPointMinusCom_DotNormal = contactPointMinusComA_DotNormal - comMinusComA.dot<3>( normal );
		shift.setAddMul( motion.getCenterOfMassInWorld(), normal, contactPointMinusCom_DotNormal );
	}

	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> allAngJac;
	if ( indexOfSmallerBody == 0 )
	{
		hkMxVectorUtil::gatherUnpackFirst<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD, sizeof(hknpMxContactJacobian::ContactPointData)>( &mxJac->m_contactPointData[0].m_angular[mxJacIdx][0], allAngJac );
	}
	else
	{
		hkMxVectorUtil::gatherUnpackSecond<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD, sizeof(hknpMxContactJacobian::ContactPointData)>( &mxJac->m_contactPointData[0].m_angular[mxJacIdx][0], allAngJac );
	}
	hkMxQuaternion<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> rotation; rotation.gather<0>( &motion.m_orientation );
	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> allAngJacWs; hkMxVectorUtil::rotateDirection( rotation, allAngJac, allAngJacWs );

	normal.setFlipSign( normal, hkVector4::getConstant( hkVectorConstant( HK_QUADREAL_0 - indexOfSmallerBody ) ) );
	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> mxNormal; mxNormal.setVector( normal );
	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> arm; arm.setCross( mxNormal, allAngJacWs );
	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> mxShift; mxShift.setVector(shift);
	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> positions; positions.setAdd( arm, mxShift );

	hkVector4 rhs;
	rhs.set(
		mxJac->m_contactPointData[0].m_rhs[mxJacIdx],
		mxJac->m_contactPointData[1].m_rhs[mxJacIdx],
		mxJac->m_contactPointData[2].m_rhs[mxJacIdx],
		mxJac->m_contactPointData[3].m_rhs[mxJacIdx]
	);

	positions.store( &manifoldOut->m_positions[0](0) );
	positions.getVector<0>().store<4>( &manifoldOut->m_gskPosition(0) );
	hkSimdReal f = -world->m_solverInfo.getRhsFactorInv();
	hkVector4 distances; distances.setMul( rhs, f );
	manifoldOut->m_distances = distances;
	distances.store<1>( &manifoldOut->m_gskPosition(3) );
	manifoldOut->m_distances.setMul( rhs, f );
}

void hknpContactSolverEvent::predictCurrentContactPointDistances( hknpWorld* world, const hkcdManifold4& manifold, hkVector4* contactPointDistances ) const
{
	const hknpBody&   bodyA = world->getBody( m_bodyIds[0]);
	const hknpBody&   bodyB = world->getBody( m_bodyIds[1]);
	const hknpMotion* motionA = &world->getMotion( bodyA.m_motionId );
	const hknpMotion* motionB = &world->getMotion( bodyB.m_motionId );
	hkVector4 projectedPointVelocities; hknpSolverUtil::calculateProjectedPointVelocitiesUsingIntegratedVelocities( manifold, motionA, motionB, &projectedPointVelocities );
	contactPointDistances->setAddMul( manifold.m_distances, projectedPointVelocities, world->m_solverInfo.m_deltaTime);
}

hkSimdReal hknpContactSolverEvent::calculateToi(hknpWorld* world, hkSimdRealParameter maxExtraPenetration ) const
{
	hkcdManifold4 manifold; calculateContactPointPositions( world, &manifold );
	hkVector4 currentDist;  predictCurrentContactPointDistances( world, manifold, &currentDist );

	hkVector4 allowedExtraPenetration; allowedExtraPenetration.setAll( maxExtraPenetration );
	hkVector4 allowedDistance; allowedDistance.setSub( manifold.m_distances, allowedExtraPenetration );
	allowedDistance.setMin( allowedDistance, hkVector4::getZero() );
	hkVector4 allowedTravelDistance; allowedTravelDistance.setSub( manifold.m_distances, allowedDistance );
	hkVector4 totalTravelDistance; totalTravelDistance.setSub( manifold.m_distances, currentDist );
	totalTravelDistance.setMax( totalTravelDistance, hkVector4::getConstant<HK_QUADREAL_EPS>() );
	hkVector4 toi4; toi4.setDiv( allowedTravelDistance, totalTravelDistance );
	hkSimdReal toi = toi4.horizontalMin<4>();
	return toi;
}

void hknpContactImpulseEvent::printCommand( hknpWorld* world, hkOstream& out ) const
{
	hkcdManifold4 manifold;
	const char* status[] = { "NONE", "STARTED", "FINISHED", "CONTINUED" };
	out << "hknpContactImpulseEvent " << status[m_status]
		<< " bodyIds=" << m_bodyIds[0].value() << "," << m_bodyIds[1].value();
	if( m_contactJacobian )
	{
		calculateContactPointPositions( world, &manifold );
		out << " normal=" << manifold.m_normal
		<< " pos0=" << manifold.m_positions[0]
		<< " pos1=" << manifold.m_positions[1]
		<< " pos2=" << manifold.m_positions[2]
		<< " pos3=" << manifold.m_positions[3]
		<< " impulses=" << *reinterpret_cast<const hkVector4*>(&m_contactImpulses[0]);
	}
	else
	{
		HK_ASSERT( 0x8ffc015e, m_status == STATUS_FINISHED );
		out << " (manifold destroyed)";
	}
}

void hknpContactImpulseClippedEvent::printCommand( hknpWorld* world, hkOstream& out ) const
{
	hkcdManifold4 manifold;
	calculateContactPointPositions( world, &manifold );
	out << "hknpContactImpulseClippedEvent bodyIds=" << m_bodyIds[0].value() << "," << m_bodyIds[1].value()
		<< " normal=" << manifold.m_normal
		<< " pos0=" << manifold.m_positions[0]
		<< " pos1=" << manifold.m_positions[1]
		<< " pos2=" << manifold.m_positions[2]
		<< " pos3=" << manifold.m_positions[3]
	;
}

void hknpConstraintForceEvent::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "ConstraintForceEvent";
}

void hknpConstraintForceExceededEvent::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "ConstraintForceExceededEvent";
}

void hknpLinearIntegrationClippedEvent::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "LinearIntegrationClippedCommand Id=" << m_bodyId.value() << " StolenVel =" << m_stolenVelocity;
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
