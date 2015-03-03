/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/Hinge/hkpHingeConstraintData.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>


hkpHingeConstraintData::hkpHingeConstraintData()
{
	m_atoms.m_transforms.m_transformA.setIdentity();
	m_atoms.m_transforms.m_transformB.setIdentity();

	m_atoms.m_2dAng.m_freeRotationAxis = 0;
}

//
// hkHingeCinfo initialization methods
//

void hkpHingeConstraintData::setInWorldSpace( const hkTransform& bodyATransform, const hkTransform& bodyBTransform,
										const hkVector4& pivot, const hkVector4& axis )
{
	hkVector4 perpToAxle1;
	hkVector4 perpToAxle2;
	hkVector4Util::calculatePerpendicularVector( axis, perpToAxle1 ); perpToAxle1.normalize<3>();
	perpToAxle2.setCross(axis, perpToAxle1);

	m_atoms.m_transforms.m_transformA.getColumn(0).setRotatedInverseDir(bodyATransform.getRotation(), axis);
	m_atoms.m_transforms.m_transformA.getColumn(1).setRotatedInverseDir(bodyATransform.getRotation(), perpToAxle1);
	m_atoms.m_transforms.m_transformA.getColumn(2).setRotatedInverseDir(bodyATransform.getRotation(), perpToAxle2);
	m_atoms.m_transforms.m_transformA.getColumn(3).setTransformedInversePos(bodyATransform, pivot);

	m_atoms.m_transforms.m_transformB.getColumn(0).setRotatedInverseDir(bodyBTransform.getRotation(), axis);
	m_atoms.m_transforms.m_transformB.getColumn(1).setRotatedInverseDir(bodyBTransform.getRotation(), perpToAxle1);
	m_atoms.m_transforms.m_transformB.getColumn(2).setRotatedInverseDir(bodyBTransform.getRotation(), perpToAxle2);
	m_atoms.m_transforms.m_transformB.getColumn(3).setTransformedInversePos(bodyBTransform, pivot);

	HK_ASSERT2(0x3a0a5292, isValid(), "Members of Hinge constraint inconsistent after World Space constructor.");
}


void hkpHingeConstraintData::setInBodySpace( const hkVector4& pivotA,const hkVector4& pivotB,
									  const hkVector4& axisA,const hkVector4& axisB)
{
	m_atoms.m_transforms.m_transformA.getTranslation() = pivotA;
	m_atoms.m_transforms.m_transformB.getTranslation() = pivotB;

	hkVector4* baseA = &m_atoms.m_transforms.m_transformA.getColumn(0);
	baseA[0] = axisA; baseA[0].normalize<3>();
	hkVector4Util::calculatePerpendicularVector( baseA[0], baseA[1] ); baseA[1].normalize<3>();
	baseA[2].setCross( baseA[0], baseA[1] );

	hkVector4* baseB = &m_atoms.m_transforms.m_transformB.getColumn(0);
	baseB[0] = axisB; baseB[0].normalize<3>();
	hkVector4Util::calculatePerpendicularVector( baseB[0], baseB[1] ); baseB[1].normalize<3>();
	baseB[2].setCross( baseB[0], baseB[1] );

	HK_ASSERT2(0x3a0a5292, isValid(), "Members of Hinge constraint inconsistent after Body Space constructor..");
}

void hkpHingeConstraintData::setBreachImpulse(hkReal breachImpulse)
{
	m_atoms.m_ballSocket.setBreachImpulse(breachImpulse);
}

hkReal hkpHingeConstraintData::getBreachImpulse() const
{
	return m_atoms.m_ballSocket.getBreachImpulse();
}

void hkpHingeConstraintData::setBodyToNotify(int bodyIdx)
{
	//HK_ASSERT2(0xad808071, notifyBodyA >= 0 && notifyBodyA <= 1 && notifyBodyB >= 0 && notifyBodyB <= 1, "Notify parameters must be 0 or 1.");
	//m_atoms.m_ballSocket.m_bodiesToNotify = hkUint8(0x01 * notifyBodyA + 0x02 * notifyBodyB);
	HK_ASSERT2(0xad808071, bodyIdx >= 0 && bodyIdx <= 1, "Notify body index must be 0 or 1.");
	m_atoms.m_ballSocket.m_bodiesToNotify = hkUint8(0x01 << bodyIdx);
}

hkUint8 hkpHingeConstraintData::getNotifiedBodyIndex() const
{
	//return m_atoms.m_ballSocket.m_bodiesToNotify;
	HK_ASSERT2(0xad809032, m_atoms.m_ballSocket.m_bodiesToNotify & 0x3, "Body to be notified not set.");
	return m_atoms.m_ballSocket.m_bodiesToNotify >> 1;
}

void hkpHingeConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	if ( wantRuntime )
	{
		infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
		infoOut.m_sizeOfExternalRuntime = sizeof( Runtime);
	}
	else
	{
		infoOut.m_numSolverResults = 0;
		infoOut.m_sizeOfExternalRuntime = 0;
	}
}

void hkpHingeConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}



hkBool hkpHingeConstraintData::isValid() const
{
	// In stable mode, we need the setupStabilization atom enabled!
	if ( (m_atoms.m_ballSocket.m_solvingMethod == hkpConstraintAtom::METHOD_STABILIZED) &&
		!m_atoms.m_setupStabilization.m_enabled )
	{
		return false;
	}

	return m_atoms.m_transforms.m_transformA.getRotation().isOrthonormal() && m_atoms.m_transforms.m_transformB.getRotation().isOrthonormal();
}


int hkpHingeConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_HINGE;
}

//
//	Sets the solving method for this constraint. Use one of the hkpConstraintAtom::SolvingMethod as a value for method.

void hkpHingeConstraintData::setSolvingMethod(hkpConstraintAtom::SolvingMethod method)
{
	switch ( method )
	{
	case hkpConstraintAtom::METHOD_STABILIZED:
		{
			m_atoms.m_setupStabilization.m_enabled	= true;
			m_atoms.m_ballSocket.m_solvingMethod	= hkpConstraintAtom::METHOD_STABILIZED;
		}
		break;

	case hkpConstraintAtom::METHOD_OLD:
		{
			m_atoms.m_setupStabilization.m_enabled	= false;
			m_atoms.m_ballSocket.m_solvingMethod	= hkpConstraintAtom::METHOD_OLD;
		}
		break;

	default:
		{
			HK_ASSERT2(0x3ce72932, false, "Unknown solving method! Please use one of the values in hkpConstraintAtom::SolvingMethod!");
		}
		break;
	}
}

//
//	Gets the inertia stabilization factor, returns HK_FAILURE if the factor is not defined for the given constraint.

hkResult hkpHingeConstraintData::getInertiaStabilizationFactor(hkReal& inertiaStabilizationFactorOut) const
{
	inertiaStabilizationFactorOut = m_atoms.m_ballSocket.getInertiaStabilizationFactor();
	return HK_SUCCESS;
}

//
//	Sets the inertia stabilization factor, return HK_FAILURE if the factor is not defined for the given constraint.

hkResult hkpHingeConstraintData::setInertiaStabilizationFactor(const hkReal inertiaStabilizationFactorIn)
{
	m_atoms.m_ballSocket.setInertiaStabilizationFactor(inertiaStabilizationFactorIn);
	return HK_SUCCESS;
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
