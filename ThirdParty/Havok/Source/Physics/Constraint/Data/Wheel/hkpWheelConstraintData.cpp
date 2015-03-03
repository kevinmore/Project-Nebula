/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/Wheel/hkpWheelConstraintData.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>


hkpWheelConstraintData::hkpWheelConstraintData( )
{
	m_atoms.m_lin0Limit.m_axisIndex = 0;
	m_atoms.m_lin0Soft.m_axisIndex = 0;
	m_atoms.m_lin1.m_axisIndex = 1;
	m_atoms.m_lin2.m_axisIndex = 2;

	m_atoms.m_lin0Limit.m_min = hkReal(-0.5f);
	m_atoms.m_lin0Limit.m_max = hkReal( 0.5f);

	m_atoms.m_lin0Soft.m_tau = hkReal(0);
	m_atoms.m_lin0Soft.m_damping = hkReal(0);

	m_atoms.m_steeringBase.m_rotationA.setIdentity();
	m_atoms.m_steeringBase.m_rotationB.setIdentity();
	m_atoms.m_suspensionBase.m_transformA.setIdentity();
	m_atoms.m_suspensionBase.m_transformB.setIdentity();

	m_atoms.m_2dAng.m_freeRotationAxis = 0;
}


void hkpWheelConstraintData::setInWorldSpace( const hkTransform& bodyATransform, const hkTransform& bodyBTransform,
										const hkVector4& pivot, const hkVector4& axleTmp,
										const hkVector4& suspensionAxis, const hkVector4& steeringAxisTmp )
{
	hkVector4 axle = axleTmp; axle.normalize<3>();
	hkVector4 steeringAxis = steeringAxisTmp; steeringAxis.normalize<3>();
	HK_ASSERT2(0x4ac33b29, hkMath::equal( axle.dot<3>(steeringAxis).getReal(), hkReal(0)), "Wheel axle and steering axis must be perpendicular.");
		// pivots
	{
		hkVector4 pivotAinW; pivotAinW.setSub(pivot, bodyATransform.getTranslation());
		m_atoms.m_suspensionBase.m_transformA.getTranslation().setRotatedInverseDir(bodyATransform.getRotation(), pivotAinW);
		hkVector4 pivotBinW; pivotBinW.setSub(pivot, bodyBTransform.getTranslation());
		m_atoms.m_suspensionBase.m_transformB.getTranslation().setRotatedInverseDir(bodyBTransform.getRotation(), pivotBinW);
	}

		// orientations

	m_atoms.m_suspensionBase.m_transformA.getRotation().setIdentity();

	m_atoms.m_suspensionBase.m_transformB.getColumn(0).setRotatedInverseDir(bodyBTransform.getRotation(), suspensionAxis);
	hkVector4Util::calculatePerpendicularVector( m_atoms.m_suspensionBase.m_transformB.getColumn<0>(), m_atoms.m_suspensionBase.m_transformB.getColumn(1) ); m_atoms.m_suspensionBase.m_transformB.getColumn(1).normalize<3>();
	m_atoms.m_suspensionBase.m_transformB.getColumn(2).setCross( m_atoms.m_suspensionBase.m_transformB.getColumn<0>(), m_atoms.m_suspensionBase.m_transformB.getColumn<1>());

		// Angular bits

	//0 axle
	//1 steering
	hkVector4* baseA = &m_atoms.m_steeringBase.m_rotationA.getColumn(0);
	hkVector4* baseB = &m_atoms.m_steeringBase.m_rotationB.getColumn(0);

	baseA[0].setRotatedInverseDir(bodyATransform.getRotation(), axle);
	baseB[0].setRotatedInverseDir(bodyBTransform.getRotation(), axle);
	m_initialAxleInB = baseB[0];

	baseA[1].setRotatedInverseDir(bodyATransform.getRotation(), steeringAxis);
	baseB[1].setRotatedInverseDir(bodyBTransform.getRotation(), steeringAxis);
	m_initialSteeringAxisInB = baseB[1];

	baseA[2].setCross( baseA[0], baseA[1] );
	baseB[2].setCross( baseB[0], baseB[1] );

	HK_ASSERT2(0xad8b7aaa, isValid(), "Members of wheel constraint inconsistent.");
}


void hkpWheelConstraintData::setInBodySpace(const hkVector4& pivotA,const hkVector4& pivotB,
									   const hkVector4& axleA,const hkVector4& axleB,
									   const hkVector4& suspensionAxisB, const hkVector4& steeringAxisB )
{
	m_atoms.m_suspensionBase.m_transformA.getTranslation() = pivotA;
	m_atoms.m_suspensionBase.m_transformB.getTranslation() = pivotB;

	// suspension -- only bodyB is important
	m_atoms.m_suspensionBase.m_transformA.getRotation().setIdentity();

	hkVector4* suspBaseB = &m_atoms.m_suspensionBase.m_transformB.getColumn(0);
	suspBaseB[0] = suspensionAxisB;
	hkVector4Util::calculatePerpendicularVector( suspBaseB[0], suspBaseB[1] ); suspBaseB[1].normalize<3>();
	suspBaseB[2].setCross( suspBaseB[0], suspBaseB[1] );

	// steering orientations

	//0 axle
	//1 steering
	hkVector4* baseA = &m_atoms.m_steeringBase.m_rotationA.getColumn(0);
	hkVector4* baseB = &m_atoms.m_steeringBase.m_rotationB.getColumn(0);

	baseA[0] = axleA; baseA[0].normalize<3>();
	baseB[0] = axleB; baseB[0].normalize<3>();
	m_initialAxleInB = baseB[0];

	hkVector4Util::calculatePerpendicularVector( baseA[0], baseA[1] ); baseA[1].normalize<3>();
	baseB[1] = steeringAxisB; baseB[1].normalize<3>();
	m_initialSteeringAxisInB = baseB[1];

	HK_ASSERT2(0xad78dd9a, hkMath::equal( baseB[0].dot<3>(baseB[1]).getReal(), hkReal(0)), "Wheel axle and steering axis must be perpendicular.");

	baseA[2].setCross( baseA[0], baseA[1] );
	baseB[2].setCross( baseB[0], baseB[1] );

	HK_ASSERT2(0xad8b7aab, isValid(), "Members of wheel constraint inconsistent.");
}

void hkpWheelConstraintData::setSteeringAngle( hkReal angle )
{
	hkReal sinAlpha = hkMath::sin(angle);
	hkReal cosAlpha = hkMath::sqrt( hkReal(1) - sinAlpha * sinAlpha );

	hkVector4 sinAxis; sinAxis.setCross( m_initialSteeringAxisInB, m_initialAxleInB );

	hkVector4 newAxle;
	newAxle.setMul( hkSimdReal::fromFloat(cosAlpha), m_initialAxleInB );
	newAxle.addMul( hkSimdReal::fromFloat(sinAlpha), sinAxis );

	m_atoms.m_steeringBase.m_rotationB.setColumn<0>(newAxle);
	m_atoms.m_steeringBase.m_rotationB.getColumn(2).setCross(newAxle, m_initialSteeringAxisInB);

}

void hkpWheelConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpWheelConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	if ( wantRuntime)
	{
		infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
		infoOut.m_sizeOfExternalRuntime = sizeof( Runtime );
	}
	else
	{
		infoOut.m_numSolverResults = 0;
		infoOut.m_sizeOfExternalRuntime = 0;
	}
}


hkBool hkpWheelConstraintData::isValid() const
{
	hkTransform steeringTransformA; steeringTransformA.set( m_atoms.m_steeringBase.m_rotationA, hkVector4::getZero() );
	hkTransform steeringTransformB; steeringTransformB.set( m_atoms.m_steeringBase.m_rotationB, hkVector4::getZero() );

	return m_atoms.m_suspensionBase.m_transformA.getRotation().isOrthonormal() && m_atoms.m_suspensionBase.m_transformB.getRotation().isOrthonormal()
		&& steeringTransformA.getRotation().isOrthonormal() && steeringTransformB.getRotation().isOrthonormal()
		&& m_atoms.m_lin0Limit.m_min <= m_atoms.m_lin0Limit.m_max;
}


int hkpWheelConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_WHEEL;
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
