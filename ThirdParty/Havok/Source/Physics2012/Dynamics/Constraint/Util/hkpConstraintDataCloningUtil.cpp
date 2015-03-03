/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintDataCloningUtil.h>
#include <Physics/Constraint/Data/hkpConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>

#include <Physics/Constraint/Data/HingeLimits/hkpHingeLimitsData.h>
#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>
#include <Physics/Constraint/Data/Prismatic/hkpPrismaticConstraintData.h>
#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics/Constraint/Data/Fixed/hkpFixedConstraintData.h>
#include <Physics/Constraint/Data/DeformableFixed/hkpDeformableFixedConstraintData.h>
#include <Physics/Constraint/Data/RagdollLimits/hkpRagdollLimitsData.h>
#include <Physics/Constraint/Motor/hkpConstraintMotor.h>

#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>
#include <Physics/Constraint/Data/hkpConstraintData.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintInstance.h>
#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>
#include <Physics/Constraint/Data/PointToPath/hkpPointToPathConstraintData.h>
#include <Physics/Constraint/Data/StiffSpring/hkpStiffSpringConstraintData.h>

#include <Physics/Constraint/Data/Pulley/hkpPulleyConstraintData.h>
#include <Physics/Constraint/Data/Hinge/hkpHingeConstraintData.h>
#include <Physics/ConstraintSolver/Constraint/Bilateral/hkp1dBilateralConstraintInfo.h>
#include <Physics2012/Dynamics/Constraint/Malleable/hkpMalleableConstraintData.h>
#include <Physics/Constraint/Data/Wheel/hkpWheelConstraintData.h>
#include <Physics/Constraint/Data/PointToPlane/hkpPointToPlaneConstraintData.h>
#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics/Constraint/Data/hkpConstraintDataUtils.h>

hkpConstraintData* hkpConstraintDataCloningUtil::deepClone(const hkpConstraintData* data)
{
	HK_ASSERT2(0x38dbcef2, data, "Constraint data is null");

	switch ( data->getType() )
	{
		// Breakable & malleable
	case hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE:			
		{
			const hkpBreakableConstraintData* oldBreakable = static_cast<const hkpBreakableConstraintData*>(data);

			hkpConstraintData* newChild = hkpConstraintDataCloningUtil::deepClone(oldBreakable->getWrappedConstraintData());
			HK_ASSERT2(0XAD76BA32, newChild, "Wrapped constraintData of a hkpBreakableConstraintData is not clonable.");
			if (!newChild)
			{
				return HK_NULL;
			}

			hkpBreakableConstraintData* newBreakable = new hkpBreakableConstraintData(newChild); 
			newChild->removeReference();

			newBreakable->m_solverResultLimit = oldBreakable->m_solverResultLimit;
			newBreakable->m_removeWhenBroken = oldBreakable->m_removeWhenBroken;
			newBreakable->m_revertBackVelocityOnBreak = oldBreakable->m_revertBackVelocityOnBreak;
			newBreakable->m_userData = oldBreakable->m_userData;

			return newBreakable;
		}

	case hkpConstraintData::CONSTRAINT_TYPE_MALLEABLE:			
		{
			const hkpMalleableConstraintData* oldMalleable = static_cast<const hkpMalleableConstraintData*>(data);
			hkpConstraintData* newChild = hkpConstraintDataCloningUtil::deepClone(oldMalleable->getWrappedConstraintData());
			HK_ASSERT2(0XAD76BA32, newChild, "Wrapped constraintData of a hkpMalleableConstraintData is not clonable.");
			if (!newChild)
			{
				return HK_NULL;
			}

			hkpMalleableConstraintData* newMalleable = new hkpMalleableConstraintData(newChild); 
			newChild->removeReference();

			newMalleable->m_strength = oldMalleable->m_strength;
			newMalleable->m_userData = oldMalleable->m_userData;

			return newMalleable;
		}

	default:
		return hkpConstraintDataUtils::deepClone(data);
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
