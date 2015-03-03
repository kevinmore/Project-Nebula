/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>


hkpBallAndSocketConstraintData::hkpBallAndSocketConstraintData()
{
	m_atoms.m_pivots.m_translationA.setZero();
	m_atoms.m_pivots.m_translationB.setZero();
}

void hkpBallAndSocketConstraintData::setInWorldSpace(const hkTransform& bodyATransform,
												const hkTransform& bodyBTransform,
												const hkVector4& pivot)
{
	m_atoms.m_pivots.m_translationA._setTransformedInversePos( bodyATransform, pivot );
	m_atoms.m_pivots.m_translationB._setTransformedInversePos( bodyBTransform, pivot );
}

void hkpBallAndSocketConstraintData::setInBodySpace( const hkVector4& pivotA,
												const hkVector4& pivotB)
{
	m_atoms.m_pivots.m_translationA = pivotA;
	m_atoms.m_pivots.m_translationB = pivotB;
}

void hkpBallAndSocketConstraintData::setBreachImpulse(hkReal breachImpulse)
{
	m_atoms.m_ballSocket.setBreachImpulse(breachImpulse);
}

hkReal hkpBallAndSocketConstraintData::getBreachImpulse() const
{
	return m_atoms.m_ballSocket.getBreachImpulse();
}

void hkpBallAndSocketConstraintData::setMaximumLinearImpulse(hkReal maxLinearImpulse)
{
	m_atoms.m_setupStabilization.setMaximumLinearImpulse(maxLinearImpulse);

	m_atoms.m_ballSocket.m_enableLinearImpulseLimit = (maxLinearImpulse < HK_REAL_MAX);
}

hkReal hkpBallAndSocketConstraintData::getMaximumLinearImpulse() const
{
	return m_atoms.m_setupStabilization.getMaximumLinearImpulse();
}

void hkpBallAndSocketConstraintData::setBodyToNotify(int bodyIdx)
{
	//HK_ASSERT2(0xad808071, notifyBodyA >= 0 && notifyBodyA <= 1 && notifyBodyB >= 0 && notifyBodyB <= 1, "Notify parameters must be 0 or 1.");
	//m_atoms.m_ballSocket.m_bodiesToNotify = hkUint8(0x01 * notifyBodyA + 0x02 * notifyBodyB);
	HK_ASSERT2(0xad808071, bodyIdx >= 0 && bodyIdx <= 1, "Notify body index must be 0 or 1.");
	m_atoms.m_ballSocket.m_bodiesToNotify = hkUint8(0x01 << bodyIdx);
}

hkUint8 hkpBallAndSocketConstraintData::getNotifiedBodyIndex() const
{
	//return m_atoms.m_ballSocket.m_bodiesToNotify;
	HK_ASSERT2(0xad809032, m_atoms.m_ballSocket.m_bodiesToNotify & 0x3, "Body to be notified not set.");
	return m_atoms.m_ballSocket.m_bodiesToNotify >> 1;
}

void hkpBallAndSocketConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpBallAndSocketConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	if ( wantRuntime || m_atoms.m_ballSocket.getBreachImpulse() != HK_REAL_MAX )
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


hkBool hkpBallAndSocketConstraintData::isValid() const
{
	// In stable mode, we need the setupStabilization atom enabled!
	if ( (m_atoms.m_ballSocket.m_solvingMethod == hkpConstraintAtom::METHOD_STABILIZED) &&
		!m_atoms.m_setupStabilization.m_enabled )
	{
		return false;
	}

	return true;
}


int hkpBallAndSocketConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET;
}

//
//	Sets the solving method for this constraint. Use one of the hkpConstraintAtom::SolvingMethod as a value for method.

void hkpBallAndSocketConstraintData::setSolvingMethod(hkpConstraintAtom::SolvingMethod method)
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

hkResult hkpBallAndSocketConstraintData::getInertiaStabilizationFactor(hkReal& inertiaStabilizationFactorOut) const
{
	inertiaStabilizationFactorOut = m_atoms.m_ballSocket.getInertiaStabilizationFactor();
	return HK_SUCCESS;
}

//
//	Sets the inertia stabilization factor, return HK_FAILURE if the factor is not defined for the given constraint.

hkResult hkpBallAndSocketConstraintData::setInertiaStabilizationFactor(const hkReal inertiaStabilizationFactorIn)
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
