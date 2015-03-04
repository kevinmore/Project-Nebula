/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintChainLengthUtil.h>

#include <Common/Base/Math/Quaternion/hkQuaternionUtil.h>

#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstance.h>
#include <Physics2012/Dynamics/Constraint/Chain/BallSocket/hkpBallSocketChainData.h>
#include <Physics2012/Utilities/Constraint/Chain/hkpConstraintChainUtil.h>
#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>
#include <Physics2012/Collide/Filter/Group/hkpGroupFilter.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#ifdef HK_PLATFORM_RVL
#include <Common/Base/Fwd/hkcstring.h>
#endif


hkpConstraintChainLengthUtil::hkpConstraintChainLengthUtil(const RopeInfo& info, hkpRigidBody* craneBody, hkVector4Parameter pivotOnCrane, hkVector4Parameter ropeEndInWorld)
{
#ifdef HK_PLATFORM_RVL
	memcpy (&m_ropeInfo, &info, sizeof(RopeInfo));
#else
	m_ropeInfo = info;
#endif
	m_pivotOnCrane = pivotOnCrane;
	m_numLoadBodies = 0;

	// Create segment template.
	{
		const hkSimdReal ropeRadius = (info.getCableCrossSection() / hkSimdReal_Pi).sqrt();
		const hkSimdReal segmentMass = info.getSegmentLength() * info.getCableCrossSection() * info.getMaterialDensity();

		m_segmentCinfo.m_shape = new hkpCapsuleShape(getSegmentBPivot(), getSegmentAPivot(), ropeRadius.getReal());
		hkpInertiaTensorComputer::setShapeVolumeMassProperties(m_segmentCinfo.m_shape, segmentMass.getReal(), m_segmentCinfo);

		// Multiply inertia to increase stability
		m_segmentCinfo.m_inertiaTensor.mul(info.m_inertiaMultiplier);
		// Greatly multiply inertia around rope axis to prevent rotation
		m_segmentCinfo.m_inertiaTensor(2,2) *= 1000.0f;
		// Dampen angular velocity to eliminate rotation around rope axis.
		// And also to dampen oscillation along the rope.
		m_segmentCinfo.m_angularDamping = 0.1f;
		m_segmentCinfo.m_linearDamping = 0.0f;
		// Disable solver deactivation, which may freeze slwo moving bodies that are not in prefect balance.
		m_segmentCinfo.m_solverDeactivation = hkpRigidBodyCinfo::SOLVER_DEACTIVATION_OFF;

		m_segmentCinfo.m_collisionFilterInfo = hkpGroupFilter::calcFilterInfo(1);
	}

	// Calculate world pivots.
	hkVector4 cranePivotInW; cranePivotInW._setTransformedPos(craneBody->getTransform(), pivotOnCrane);

	hkVector4 diff; diff.setSub(cranePivotInW, ropeEndInWorld);
	hkVector4 dir = diff; dir.normalize<3>();
	m_currentUnstretchedLength = diff.length<3>();

	// Create rope bodies and the chain constraint.
	hkpBallSocketChainData* chainData = new hkpBallSocketChainData();
	m_instance = new hkpConstraintChainInstance( chainData );
	chainData->removeReference();

	hkQuaternionUtil::_computeShortestRotation(hkVector4::getConstant<HK_QUADREAL_0010>(), dir, m_segmentCinfo.m_rotation);
	m_segmentCinfo.m_position.setAddMul(ropeEndInWorld, dir, hkSimdReal_Inv2*info.getSegmentLength());

	int numSegments = getNumSegments(hkSimdReal_Minus1); // round down
	while (numSegments < 2)
	{
		HK_WARN(0xad38244, "At least two segments must exist. Initialize the rope with the ropeEndInWorld further away from the crane pivot.");
		// Lengthen the rope to keep the util going. // untested.
		m_currentUnstretchedLength.add(info.getSegmentLength());
		numSegments = getNumSegments(hkSimdReal_Minus1);
	}
	for (int si = 0; si < numSegments; si++)
	{
		m_segmentCinfo.m_position.addMul(info.getSegmentLength(), dir);
		hkpRigidBody* body = new hkpRigidBody(m_segmentCinfo);
		m_instance->addEntity(body);
		body->removeReference();
	}
	m_instance->addEntity(craneBody);

	for (int si = 0; si < numSegments-1; si++)
	{
		getChainData()->addConstraintInfoInBodySpace(getSegmentAPivot(), getSegmentBPivot());
	}
	getChainData()->addConstraintInfoInBodySpace(getLastSegmentAPivot(), pivotOnCrane);

	// Run to setup cfm, etc.
	updatePivotOnStretchedRope();
}

hkpConstraintChainLengthUtil::~hkpConstraintChainLengthUtil()
{
	m_instance->removeReference();
	m_segmentCinfo.m_shape->removeReference();
}

void hkpConstraintChainLengthUtil::attachHook(hkpRigidBody* hookBody, hkVector4Parameter pivotOnHook)
{
	if (getNumSegments(hkSimdReal_Minus1) + m_numLoadBodies + 1 + 1 > 107)
	{
		HK_WARN(0xad123414, "Max size of chain constraint reached.");
		// abort
		return;
	}

	hkpWorld* world = m_instance->getOwner() ? m_instance->m_chainedEntities[0]->getWorld() : HK_NULL;

	if (world) { world->removeConstraint(m_instance); }

	HK_ASSERT2(0xad843244, hookBody->getWorld() == world, "New body must be in or out of the world, just like the chain is.");
	m_instance->m_chainedEntities.insertAt(0, hookBody);
	hookBody->addReference();

	hkpBallSocketChainData::ConstraintInfo info;
	info.m_pivotInA = pivotOnHook;
	info.m_pivotInB = getSegmentBPivot();
	getChainData()->m_infos.insertAt(0, info);
	m_numLoadBodies++;

	if (world) { world->addConstraint(m_instance); }
}

void hkpConstraintChainLengthUtil::attachLoad(hkpRigidBody* loadBody, hkVector4Parameter pivotOnLoad, hkVector4Parameter pivotOnHook)
{
	if (getNumSegments(hkSimdReal_Minus1) + m_numLoadBodies + 1 + 1 > 107)
	{
		HK_WARN(0xad123414, "Max size of chain constraint reached.");
		// abort
		return;
	}

	hkpWorld* world = m_instance->getOwner() ? m_instance->m_chainedEntities[0]->getWorld() : HK_NULL;

	if (world) { world->removeConstraint(m_instance); }

	HK_ASSERT2(0xad843244, loadBody->getWorld() == world, "New body must be in or out of the world, just like the chain is.");
	m_instance->m_chainedEntities.insertAt(0, loadBody);
	loadBody->addReference();
	updateConstraintInstanceBaseEntities(m_instance);

	hkpBallSocketChainData::ConstraintInfo info;
	info.m_pivotInA = pivotOnLoad;
	info.m_pivotInB = pivotOnHook;
	getChainData()->m_infos.insertAt(0, info);
	m_numLoadBodies++;

	if (world) { world->addConstraint(m_instance); }
}

void hkpConstraintChainLengthUtil::detachHookOrLoad()
{
	HK_ASSERT2(0xad343431, m_numLoadBodies > 0, "Cannot remove non-existing loads.");

	hkpWorld* world = m_instance->getOwner() ? m_instance->m_chainedEntities[0]->getWorld() : HK_NULL;

	if (world) { world->removeConstraint(m_instance); }

	m_instance->m_chainedEntities[0]->removeReference();
	m_instance->m_chainedEntities.removeAtAndCopy(0);
	updateConstraintInstanceBaseEntities(m_instance);
	getChainData()->m_infos.removeAtAndCopy(0);
	m_numLoadBodies--;

	if (world) { world->addConstraint(m_instance); }
}

void hkpConstraintChainLengthUtil::setUnstretchedLength(hkSimdRealParameter length)
{
	hkpWorld* world = m_instance->getOwner() ? m_instance->m_chainedEntities[0]->getWorld() : HK_NULL;

	// add/remove segments .....
	int numSegments = getNumSegments(hkSimdReal_Minus1);
	const int targetNumSegments = getNumSegments(length);

	if (targetNumSegments < 2)
	{
		HK_WARN(0xad123413, "The util needs at least 2 rope segments.");
		// abort
		return; 
	}

	if (targetNumSegments + m_numLoadBodies + 1 > 107)
	{
		HK_WARN(0xad123414, "Max size of chain constraint reached.");
		// abort
		return;
	}

	if (world) { world->removeConstraint(m_instance); }

	while (targetNumSegments < numSegments)
	{
		// Remove segments
		int lastRopeBodyIdx = m_instance->m_chainedEntities.getSize()-2;
		if (world) { world->removeEntity(m_instance->m_chainedEntities[lastRopeBodyIdx]); }
		m_instance->m_chainedEntities[lastRopeBodyIdx]->removeReference();
		m_instance->m_chainedEntities.removeAtAndCopy(lastRopeBodyIdx);
		updateConstraintInstanceBaseEntities(m_instance);
		getChainData()->m_infos.popBack(); // We'll update the last info below
		numSegments--;
	}

	while (numSegments < targetNumSegments)
	{
		// Add segments
		int lastRopeBodyIdx = m_instance->m_chainedEntities.getSize()-2;
		hkpRigidBody* lastBody  = (hkpRigidBody*)m_instance->m_chainedEntities[lastRopeBodyIdx];
		hkVector4 dir; dir._setRotatedDir(lastBody->getRotation(), hkVector4::getConstant<HK_QUADREAL_0010>());
		m_segmentCinfo.m_position.setAddMul(lastBody->getPosition(), dir, m_ropeInfo.getSegmentLength());
		m_segmentCinfo.m_rotation = lastBody->getRotation();

		hkpRigidBody* body = new hkpRigidBody(m_segmentCinfo);

		if (world) { world->addEntity(body); }
		int newRopeBodyIdx = m_instance->m_chainedEntities.getSize()-1;
		m_instance->m_chainedEntities.insertAt(newRopeBodyIdx, body); body->addReference(); // cannot use m_instance->addEntity()
		updateConstraintInstanceBaseEntities(m_instance);
		body->removeReference();

		hkpBallSocketChainData::ConstraintInfo info;
		info.m_pivotInA = getSegmentAPivot();
		info.m_pivotInB = getSegmentBPivot();
		getChainData()->m_infos.back() = info; // reset the one before the last
		getChainData()->m_infos.pushBack(info); // add a new one, it will be set properly below

		numSegments++;
	}

	m_currentUnstretchedLength = length;
	updatePivotOnStretchedRope();

	if (world) { world->addConstraint(m_instance); }
}

void hkpConstraintChainLengthUtil::updatePivotOnStretchedRope() // just use the stretch above. simple. // consider also multiplying the inertia of the topmost element to compensate for the longer arm.
{
	hkpBallSocketChainData::ConstraintInfo& info = getChainData()->m_infos.back();
	info.m_pivotInA = getLastSegmentAPivot();
	info.m_pivotInB = m_pivotOnCrane;
}

void hkpConstraintChainLengthUtil::updateChainProperties(hkSimdRealParameter solverStepDeltaTime, int numSolverMicroSteps, hkSimdRealParameter solverTau) 
{
	// Spring constant for the rope segment
	const hkSimdReal springConstant = m_ropeInfo.getMaterialStretching() / m_currentUnstretchedLength;
	int numLinks = m_instance->m_chainedEntities.getSize() - m_numLoadBodies;

	const hkSimdReal cfm = hkSimdReal::fromFloat(hkReal(25)) * (hkSimdReal::fromInt32(numSolverMicroSteps) * solverTau) / ( springConstant * solverStepDeltaTime * hkSimdReal::fromInt32(numLinks));

	getChainData()->m_tau = solverTau.getReal(); 
	getChainData()->m_cfm = cfm.getReal();
	getChainData()->m_maxErrorDistance = hkReal(10000);
}

void hkpConstraintChainLengthUtil::getRopeBodies(hkArray<hkpRigidBody*>& ropeBodies)
{
	// Reject load bodies at the beginning & the crane body at the end.
	for (int bi = m_numLoadBodies; bi < m_instance->m_chainedEntities.getSize()-1; bi++)
	{
		ropeBodies.pushBack((hkpRigidBody*)m_instance->m_chainedEntities[bi]);
	}
}

int hkpConstraintChainLengthUtil::getNumSegments(hkSimdRealParameter length) const
{
	hkSimdReal len;
	len.setSelect(length.lessZero(), m_currentUnstretchedLength, length);

	// only set this to control when segments are removed

	const hkSimdReal fractionPast = hkSimdReal::fromFloat(0.8f);
	const hkSimdReal l = m_ropeInfo.getSegmentLength();

	// Remove when 'fractionPast' of it's length is past the crane pivot
	const hkSimdReal s = (len+fractionPast*l) / l;

	int num;
	s.storeSaturateInt32(&num);
	return num;
}

const hkVector4 hkpConstraintChainLengthUtil::getLastSegmentAPivot() const 
{
	const int numSegments = getNumSegments(hkSimdReal_Minus1); // round down
	const hkSimdReal remainingLength = m_currentUnstretchedLength - (hkSimdReal::fromInt32(numSegments)*m_ropeInfo.getSegmentLength());
	const hkSimdReal stretch = getStretchAtCraneLess1();
	const hkSimdReal remainingStretch = remainingLength / m_ropeInfo.getSegmentLength() * stretch;
	hkVector4 p; 
	p.setMul(hkVector4::getConstant<HK_QUADREAL_0010>(), hkSimdReal_Inv2 * m_ropeInfo.getSegmentLength() + remainingLength + remainingStretch ); 
	return p;
}

const hkSimdReal hkpConstraintChainLengthUtil::getStretchAtCraneLess1() const
{
	const hkpBallSocketChainData::ConstraintInfo& link = getChainData()->m_infos[getChainData()->m_infos.getSize()-2];
	hkpRigidBody* bodyA = (hkpRigidBody*)m_instance->m_chainedEntities[m_instance->m_chainedEntities.getSize()-3];
	hkpRigidBody* bodyB = (hkpRigidBody*)m_instance->m_chainedEntities[m_instance->m_chainedEntities.getSize()-2]; 

	hkVector4 pivotAInWorld; pivotAInWorld._setTransformedPos(bodyA->getTransform(), link.m_pivotInA);
	hkVector4 pivotBInWorld; pivotBInWorld._setTransformedPos(bodyB->getTransform(), link.m_pivotInB);
	hkVector4 stretch; stretch.setSub(pivotBInWorld, pivotAInWorld);

	return stretch.length<3>();
}

const hkSimdReal hkpConstraintChainLengthUtil::getStretchAtCrane() const
{
	const hkpBallSocketChainData::ConstraintInfo& link = getChainData()->m_infos[getChainData()->m_infos.getSize()-1];
	hkpRigidBody* bodyA = (hkpRigidBody*)m_instance->m_chainedEntities[m_instance->m_chainedEntities.getSize()-2];
	hkpRigidBody* bodyB = (hkpRigidBody*)m_instance->m_chainedEntities[m_instance->m_chainedEntities.getSize()-1]; 

	hkVector4 pivotAInWorld; pivotAInWorld._setTransformedPos(bodyA->getTransform(), link.m_pivotInA);
	hkVector4 pivotBInWorld; pivotBInWorld._setTransformedPos(bodyB->getTransform(), link.m_pivotInB);
	hkVector4 stretch; stretch.setSub(pivotBInWorld, pivotAInWorld);

	return stretch.length<3>();
}

void hkpConstraintChainLengthUtil::updateConstraintInstanceBaseEntities(hkpConstraintChainInstance* chain)
{
	if (chain->m_entities[0]) { chain->m_entities[0]->removeReference(); }
	if (chain->m_entities[1]) { chain->m_entities[1]->removeReference(); }
	chain->m_entities[0] = chain->m_chainedEntities.getSize() > 0 ? chain->m_chainedEntities[0] : HK_NULL;
	chain->m_entities[1] = chain->m_chainedEntities.getSize() > 1 ? chain->m_chainedEntities[1] : HK_NULL;
	if (chain->m_entities[0]) { chain->m_entities[0]->addReference(); }
	if (chain->m_entities[1]) { chain->m_entities[1]->addReference(); }
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
