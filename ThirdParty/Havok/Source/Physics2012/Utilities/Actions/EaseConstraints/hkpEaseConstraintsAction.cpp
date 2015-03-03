/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Actions/EaseConstraints/hkpEaseConstraintsAction.h>

// Dynamics includes
#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>
#include <Physics/Constraint/Data/hkpConstraintDataUtils.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

//#define CHECK_TIMINGS

// TODO: if we explore the Tau option, we'll need to store the original tau
//       in order to properly restore it.
//#define ALSO_USE_TAU // <-- just a hardcoded tau for testing, doesn't restore original
//#include <stdio.h>
//#define ALSO_USE_ANG_DAMPING // <-- not implemented yet

#ifdef CHECK_TIMINGS
#include <stdio.h>
using namespace std;
#include <Common/Base/System/Stopwatch/hkStopwatch.h>
hkStopwatch loosenConstraintsTimer;
hkReal loosenConstraintsTimerMax;
hkStopwatch restoreConstraintsTimer;HK_LERFP
hkReal restoreConstraintsTimerMax;
#define HK_UPDATE_MAX( __timer__, __timer_p__ ) {##__timer__##Max = hkMath::max2(##__timer__##Max, __timer__.getElapsedSeconds() - __timer_p__); }
#endif

#define hkpEaseConstraintsAction_LERP( __a__, __b__, __t__ ) (__a__ + __t__ * (__b__ - __a__))

hkpEaseConstraintsAction::hkpEaseConstraintsAction(const hkArray<hkpEntity*>& entities, hkUlong userData)
:	hkpArrayAction(entities, userData),
	m_duration(-1),
	m_timePassed(0.0f)
{
	// gather original constraints from entities
	CollectSupportedConstraints enabledConstraintsFilter;
	hkpConstraintUtils::collectConstraints( entities, m_originalConstraints, &enabledConstraintsFilter );

	_saveLimits( m_originalLimits, m_originalConstraints );

#ifdef CHECK_TIMINGS
	loosenConstraintsTimer.reset();
	restoreConstraintsTimer.reset();
#endif
}

void hkpEaseConstraintsAction::loosenConstraints()
{
	// effectively turn "off" the restoring action
	m_duration = -1;
	m_timePassed = 0;

#ifdef CHECK_TIMINGS
	hkReal lastTime = loosenConstraintsTimer.getElapsedSeconds();
	loosenConstraintsTimer.start();
#endif

	_loosenConstraints(m_originalConstraints);

#ifdef CHECK_TIMINGS
	loosenConstraintsTimer.stop();
	HK_UPDATE_MAX(loosenConstraintsTimer, lastTime);
#endif
}

void hkpEaseConstraintsAction::restoreConstraints(hkReal duration)
{
	HK_ASSERT2(0x2cd23f56, duration >= 0, "duration must be positive or zero.");

	m_duration = duration;
	m_timePassed = 0;

	// hard-restore old constraints if duration is 0
	if (m_duration == 0)
	{
		// fully restore constraint data
		_restoreLimits( m_originalConstraints, m_originalLimits );

#ifdef CHECK_TIMINGS
		printf("hkpEaseConstraintsAction STATS:\n");
		printf("loosen constraints time: (%f, %f)\n", loosenConstraintsTimer.getElapsedSeconds()/loosenConstraintsTimer.getNumTimings(), loosenConstraintsTimerMax);
		printf("restore constraints time: (%f, %f)\n\n\n", restoreConstraintsTimer.getElapsedSeconds()/restoreConstraintsTimer.getNumTimings(), restoreConstraintsTimerMax);
		loosenConstraintsTimer.reset();
		restoreConstraintsTimer.reset();
#endif
	}
}

// hkpAction implementation.
void hkpEaseConstraintsAction::applyAction( const hkStepInfo& stepInfo )
{
	// action has been disabled or is flagged for removal
	if (m_duration <= 0)
	{

		if (m_duration == 0)
		{
			// indicates restoreConstraints(0) has been called
			// outside of applyAction.
			getWorld()->removeAction(this);
		}

		// -1 indicates loosenConstraints has been called, but restoreConstraints(0) has not
		// so we have nothing to do in this case.
		return;
	}

	// define our restoration weight in terms of remaining time.
	// we do it this way because we don't store the loosened limits
	// we only have the current limit and the original limit.
	// NOTE: divide by zero will not happen because if
	// m_timePassed >= m_duration, we remove the action immediately.
	hkReal restorationWeight = stepInfo.m_deltaTime / (m_duration - m_timePassed);

	// accumulate our time
	m_timePassed += stepInfo.m_deltaTime;

	// if we've reached our duration, do a hard-restore on the constraints
	// and remove this action from the world.
	if (m_timePassed >= m_duration)
	{
		restoreConstraints(0);
		m_entities[0]->getWorld()->removeAction(this);
		return;
	}

#ifdef CHECK_TIMINGS
	hkReal lastTime = restoreConstraintsTimer.getElapsedSeconds();
	restoreConstraintsTimer.start();
#endif

	// iterate over constraints and linearly interpolate
	// based on restorationWeight and original constraint limit values
	int limitsIter = 0;
	for (int i = 0; i < m_originalConstraints.getSize(); i++)
	{
		int limitsUsed = _partiallyRestoreConstraint( m_originalConstraints[i], &m_originalLimits[limitsIter], restorationWeight );
		limitsIter += limitsUsed;
	}

#ifdef CHECK_TIMINGS
	restoreConstraintsTimer.stop();
	HK_UPDATE_MAX(restoreConstraintsTimer, lastTime);
#endif
}

hkpAction* hkpEaseConstraintsAction::clone( const hkArray<hkpEntity*>& newEntities, const hkArray<hkpPhantom*>& newPhantoms ) const 
{
	HK_ERROR(0x196baf1, "This action does not support cloning().");
	return HK_NULL;
}

hkReal hkpEaseConstraintsAction::getDuration() const
{
	return m_duration;
}

//////////////////////////////////////////////////////////////////////////
// Aux functions
//////////////////////////////////////////////////////////////////////////
/*static*/ void hkpEaseConstraintsAction::_saveLimits(hkArray<hkReal>& dst, const hkArray<hkpConstraintInstance*>& src)
{
	for ( int i = 0; i < src.getSize(); i++)
	{
		hkpConstraintData* constraintData = const_cast<hkpConstraintData*>(src[i]->getData());

		switch(constraintData->getType())
		{
		case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
			{
				hkpRagdollConstraintData* ragdollData = static_cast<hkpRagdollConstraintData*>(constraintData);
				hkpRagdollConstraintData::Atoms& atoms = ragdollData->m_atoms;

				hkReal* dstLimits = dst.expandBy( 5 );
				dstLimits[0] = atoms.m_coneLimit.m_maxAngle;
				dstLimits[1] = atoms.m_planesLimit.m_maxAngle;
				dstLimits[2] = atoms.m_planesLimit.m_minAngle;
				dstLimits[3] = atoms.m_twistLimit.m_maxAngle;
				dstLimits[4] = atoms.m_twistLimit.m_minAngle;
			}
			break;
		case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
			{
				hkpLimitedHingeConstraintData* hingeData = static_cast<hkpLimitedHingeConstraintData*>(constraintData);
				hkpLimitedHingeConstraintData::Atoms& atoms = hingeData->m_atoms;

				hkReal* dstLimits = dst.expandBy( 2 );
				dstLimits[0] = atoms.m_angLimit.m_maxAngle;
				dstLimits[1] = atoms.m_angLimit.m_minAngle;
			}
		default:
			break;
		}
	}
}

/*static*/ void hkpEaseConstraintsAction::_restoreLimits(const hkArray<hkpConstraintInstance*>& dst, const hkArray<hkReal>& src)
{
	int limitsIter = 0;
	for ( int i = 0; i < dst.getSize(); i++)
	{
		hkpConstraintData* constraintData = const_cast<hkpConstraintData*>(dst[i]->getData());

		switch(constraintData->getType())
		{
		case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
			{
				hkpRagdollConstraintData* ragdollData = static_cast<hkpRagdollConstraintData*>(constraintData);

#ifdef ALSO_USE_TAU
				ragdollData->setAngularLimitsTauFactor(0.8f);
#endif

				hkpRagdollConstraintData::Atoms& atoms = ragdollData->m_atoms;

				atoms.m_coneLimit.m_maxAngle = src[limitsIter++];
				atoms.m_planesLimit.m_maxAngle = src[limitsIter++];
				atoms.m_planesLimit.m_minAngle = src[limitsIter++];
				atoms.m_twistLimit.m_maxAngle = src[limitsIter++];
				atoms.m_twistLimit.m_minAngle = src[limitsIter++];
			}
			break;
		case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
			{
				hkpLimitedHingeConstraintData* hingeData = static_cast<hkpLimitedHingeConstraintData*>(constraintData);

#ifdef ALSO_USE_TAU
				hingeData->setAngularLimitsTauFactor(0.8f);
#endif

				hkpLimitedHingeConstraintData::Atoms& atoms = hingeData->m_atoms;

				atoms.m_angLimit.m_maxAngle = src[limitsIter++];
				atoms.m_angLimit.m_minAngle = src[limitsIter++];
			}
		default:
			break;
		}
	}
}

/*static*/ void hkpEaseConstraintsAction::_loosenConstraints(const hkArray<hkpConstraintInstance*>& constraints)
{
	for (int i = constraints.getSize()-1; i >= 0; i--)
	{
#ifdef ALSO_USE_TAU
		hkpConstraintData* constraintData = const_cast<hkpConstraintData*>(constraints[i]->getData());

		switch(constraintData->getType())
		{
		case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
			{
				hkpRagdollConstraintData* ragdollData = static_cast<hkpRagdollConstraintData*>(constraintData);
				ragdollData->setAngularLimitsTauFactor(0.0f);
			}
			break;
		case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
			{
				hkpLimitedHingeConstraintData* hingeData = static_cast<hkpLimitedHingeConstraintData*>(constraintData);
				hingeData->setAngularLimitsTauFactor(0.0f);
			}
		default:
			break;
		}
#endif

		hkpConstraintDataUtils::loosenConstraintLimits(constraints[i]->getDataRw(), constraints[i]->getRigidBodyA()->getTransform(), constraints[i]->getRigidBodyB()->getTransform());
	}
}

/*static*/ int hkpEaseConstraintsAction::_partiallyRestoreConstraint(hkpConstraintInstance* runtimeConstraint, hkReal* originalLimitsPtr, hkReal restorationWeight)
{
	HK_ASSERT2(0x5e8da05f, restorationWeight != 0 && restorationWeight != 1, "restorationWeight should always be between 0 and 1.");

	hkpConstraintData* rtConstraintData = const_cast<hkpConstraintData*>(runtimeConstraint->getData());

	switch(rtConstraintData->getType())
	{
	case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
		{
			hkpRagdollConstraintData* rtRagdollData = static_cast<hkpRagdollConstraintData*>(rtConstraintData);

#ifdef ALSO_USE_TAU
			hkReal tau = hkpEaseConstraintsAction_LERP(rtRagdollData->getAngularLimitsTauFactor(), 1.0f, restorationWeight);
			rtRagdollData->setAngularLimitsTauFactor(tau);
			printf("r_tau: %f\n", tau);
#endif

			hkpRagdollConstraintData::Atoms& rtAtoms = rtRagdollData->m_atoms;

			// TODO: should we add conditional for when atoms are equal?
			// check if this is more costly then doing the LERP

			rtAtoms.m_coneLimit.m_maxAngle = hkpEaseConstraintsAction_LERP(rtAtoms.m_coneLimit.m_maxAngle, originalLimitsPtr[0], restorationWeight);
			//rtAtoms.m_coneLimit.m_minAngle = hkpEaseConstraintsAction_LERP(rtAtoms.m_coneLimit.m_minAngle, origAtoms.m_coneLimit.m_minAngle, restorationWeight);

			rtAtoms.m_planesLimit.m_maxAngle = hkpEaseConstraintsAction_LERP(rtAtoms.m_planesLimit.m_maxAngle, originalLimitsPtr[1], restorationWeight);
			rtAtoms.m_planesLimit.m_minAngle = hkpEaseConstraintsAction_LERP(rtAtoms.m_planesLimit.m_minAngle, originalLimitsPtr[2], restorationWeight);


			rtAtoms.m_twistLimit.m_maxAngle = hkpEaseConstraintsAction_LERP(rtAtoms.m_twistLimit.m_maxAngle, originalLimitsPtr[3], restorationWeight);
			rtAtoms.m_twistLimit.m_minAngle = hkpEaseConstraintsAction_LERP(rtAtoms.m_twistLimit.m_minAngle, originalLimitsPtr[4], restorationWeight);
		}
		return 5;
	case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
		{
			hkpLimitedHingeConstraintData* rtHingeData = static_cast<hkpLimitedHingeConstraintData*>(rtConstraintData);

#ifdef ALSO_USE_TAU
			hkReal tau = hkpEaseConstraintsAction_LERP(rtHingeData->getAngularLimitsTauFactor(), 1.0f, restorationWeight);
			rtHingeData->setAngularLimitsTauFactor(tau);
			printf("h_tau: %f\n", tau);
#endif

			hkpLimitedHingeConstraintData::Atoms& rtAtoms = rtHingeData->m_atoms;

			// TODO: should we add conditional for when atoms are equal?
			// check if this is more costly then doing the LERP

			rtAtoms.m_angLimit.m_maxAngle = hkpEaseConstraintsAction_LERP(rtAtoms.m_angLimit.m_maxAngle, originalLimitsPtr[0], restorationWeight);
			rtAtoms.m_angLimit.m_minAngle = hkpEaseConstraintsAction_LERP(rtAtoms.m_angLimit.m_minAngle, originalLimitsPtr[1], restorationWeight);
		}
		return 2;
	default:
		HK_ERROR(0x2e61bf0e, "hkpEaseConstraintsAction does not handle this type of constraint.");
	}

	return 0;
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
