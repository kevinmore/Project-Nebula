/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/AngularFriction/hkpAngularFrictionConstraintData.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>


hkpAngularFrictionConstraintData::hkpAngularFrictionConstraintData()
{
	m_atoms.m_transforms.m_transformA.setIdentity();
	m_atoms.m_transforms.m_transformB.setIdentity();

	m_atoms.m_angFriction.m_firstFrictionAxis = 0;
	m_atoms.m_angFriction.m_numFrictionAxes = 3;
	m_atoms.m_angFriction.m_maxFrictionTorque = 0.0f;
}

hkpAngularFrictionConstraintData::~hkpAngularFrictionConstraintData()
{

}

void hkpAngularFrictionConstraintData::setInWorldSpace(const hkTransform& bodyATransform, const hkTransform& bodyBTransform,
											   const hkVector4& pivot, const hkVector4& twistAxisW,
											   const hkVector4& planeAxisW)
{
	HK_ASSERT2(0x77ad7e93, hkMath::equal( twistAxisW.dot<3>(planeAxisW).getReal(), 0.0f), "twistAxisW && planeAxisW should be perpendicular");

	// Set relative orientation
	{
		hkVector4 constraintBaseInW[3];

		constraintBaseInW[Atoms::AXIS_TWIST] = twistAxisW; constraintBaseInW[Atoms::AXIS_TWIST].normalize<3>();
		constraintBaseInW[Atoms::AXIS_PLANES] = planeAxisW; constraintBaseInW[Atoms::AXIS_PLANES].normalize<3>();
		constraintBaseInW[Atoms::AXIS_CROSS_PRODUCT].setCross( constraintBaseInW[Atoms::AXIS_TWIST], constraintBaseInW[Atoms::AXIS_PLANES] );

		hkVector4Util::rotateInversePoints( bodyATransform.getRotation(), constraintBaseInW, 3, &m_atoms.m_transforms.m_transformA.getRotation().getColumn(0) );
		hkVector4Util::rotateInversePoints( bodyBTransform.getRotation(), constraintBaseInW, 3, &m_atoms.m_transforms.m_transformB.getRotation().getColumn(0) );
	}

	// Set pivot points
	m_atoms.m_transforms.m_transformA.getTranslation().setTransformedInversePos( bodyATransform, pivot );
	m_atoms.m_transforms.m_transformB.getTranslation().setTransformedInversePos( bodyBTransform, pivot );

	HK_ASSERT2(0xadea65ee, isValid(), "Members of ragdoll constraint inconsistent.");
}


void hkpAngularFrictionConstraintData::setInBodySpace( const hkVector4& pivotA,const hkVector4& pivotB,
											  const hkVector4& planeAxisA,const hkVector4& planeAxisB,
											  const hkVector4& twistAxisA, const hkVector4& twistAxisB)
{
	hkVector4* baseA = &m_atoms.m_transforms.m_transformA.getColumn(0);
	baseA[Atoms::AXIS_TWIST] = twistAxisA; baseA[Atoms::AXIS_TWIST].normalize<3>();
	baseA[Atoms::AXIS_PLANES] = planeAxisA; baseA[Atoms::AXIS_PLANES].normalize<3>();
	baseA[Atoms::AXIS_CROSS_PRODUCT].setCross( baseA[Atoms::AXIS_TWIST], baseA[Atoms::AXIS_PLANES] );
	m_atoms.m_transforms.m_transformA.getTranslation() = pivotA;

	hkVector4* baseB = &m_atoms.m_transforms.m_transformB.getColumn(0);
	baseB[Atoms::AXIS_TWIST] = twistAxisB; baseB[Atoms::AXIS_TWIST].normalize<3>();
	baseB[Atoms::AXIS_PLANES] = planeAxisB; baseB[Atoms::AXIS_PLANES].normalize<3>();
	baseB[Atoms::AXIS_CROSS_PRODUCT].setCross( baseB[Atoms::AXIS_TWIST], baseB[Atoms::AXIS_PLANES] );
	m_atoms.m_transforms.m_transformB.getTranslation() = pivotB;

	HK_ASSERT2(0xadea65ef, isValid(), "Members of ragdoll constraint inconsistent.");
}

void hkpAngularFrictionConstraintData::setBodyToNotify(int bodyIdx)
{
	//HK_ASSERT2(0xad808071, notifyBodyA >= 0 && notifyBodyA <= 1 && notifyBodyB >= 0 && notifyBodyB <= 1, "Notify parameters must be 0 or 1.");
	//m_atoms.m_ballSocket.m_bodiesToNotify = hkUint8(0x01 * notifyBodyA + 0x02 * notifyBodyB);
	HK_ASSERT2(0xad808071, bodyIdx >= 0 && bodyIdx <= 1, "Notify body index must be 0 or 1.");
}

hkUint8 hkpAngularFrictionConstraintData::getNotifiedBodyIndex() const
{
	//return m_atoms.m_ballSocket.m_bodiesToNotify;
	return 0;
}

void hkpAngularFrictionConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpAngularFrictionConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	// we need runtime data to be able to support lastAngle and friction
	infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
	infoOut.m_sizeOfExternalRuntime = sizeof( Runtime );
}



void hkpAngularFrictionConstraintData::setMaxFrictionTorque(hkReal tmag)
{
	m_atoms.m_angFriction.m_maxFrictionTorque = tmag;
}




hkBool hkpAngularFrictionConstraintData::isValid() const
{
	// In stable mode, we need the setupStabilization atom enabled!
	hkBool valid = true;
	valid = valid && m_atoms.m_transforms.m_transformA.getRotation().isOrthonormal();
	valid = valid && m_atoms.m_transforms.m_transformB.getRotation().isOrthonormal();
	return valid;
}

void hkpAngularFrictionConstraintData::getConstraintFrameA( hkMatrix3& constraintFrameA ) const
{
	constraintFrameA = m_atoms.m_transforms.m_transformA.getRotation();
}


void hkpAngularFrictionConstraintData::getConstraintFrameB( hkMatrix3& constraintFrameB ) const
{
	constraintFrameB = m_atoms.m_transforms.m_transformB.getRotation();
}

int hkpAngularFrictionConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL;
}

//
//	Sets the solving method for this constraint. Use one of the hkpConstraintAtom::SolvingMethod as a value for method.

void hkpAngularFrictionConstraintData::setSolvingMethod(hkpConstraintAtom::SolvingMethod method)
{
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
