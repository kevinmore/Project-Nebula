/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

// this
#include <Physics2012/Utilities/Dynamics/RigidBodyReset/hkpRigidBodyResetUtil.h>
#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>

hkpRigidBodyResetUtil::hkpRigidBodyResetUtil(hkpRigidBody* rigidBody)
	: m_mainRB(rigidBody)
{
	HK_ASSERT(0x234fed4d, m_mainRB);
	HK_ASSERT2(0x234fed4e, m_mainRB->m_breakableBody == HK_NULL, "Don't use this reset utility with a breakable body. Use hkdBreakableBodyResetUtil instead.");

	m_mainRB->addReference();

	// If the body is already in the world
	m_originalBodyWasActive = true;
 	if ( rigidBody->getWorld() )
 	{
 		m_originalBodyWasActive = rigidBody->isActive();
 	}

	// Store the transform, velocities and motion type
	{
		m_originalTransform = rigidBody->getTransform();
		m_originalLinearVelocity = rigidBody->getLinearVelocity();
		m_originalAngularVelocity = rigidBody->getAngularVelocity();
		m_originalMotionType = rigidBody->getMotionType();
	}
}

hkpRigidBodyResetUtil::~hkpRigidBodyResetUtil()
{
	if ( m_mainRB )
	{
		m_mainRB->removeReference();
	}
}

void hkpRigidBodyResetUtil::resetRB(hkpWorld* world)
{
	// Reset the position
	{
		m_mainRB->setTransform(m_originalTransform);
		m_mainRB->setMotionType(m_originalMotionType);
		if ( !m_mainRB->isFixed() )
		{
			m_mainRB->setLinearVelocity( m_originalLinearVelocity );
			m_mainRB->setAngularVelocity( m_originalAngularVelocity );
		}
	}

	if (!m_originalBodyWasActive && !m_mainRB->isFixedOrKeyframed())
	{
		m_mainRB->deactivate();
	}

	hkInplaceArray<hkpConstraintInstance*,16> constraints;
	m_mainRB->getAllConstraints(constraints);
	{
		for (int c = constraints.getSize()-1; c>=0; c--)
		{
			hkpConstraintInstance* instance = constraints[c];
			hkpConstraintData* data = const_cast<hkpConstraintData*>(instance->getData());
			if (data->getType() == hkpConstraintData::CONSTRAINT_TYPE_CONTACT)
			{
				constraints.removeAt(c);
				continue;
			}
		}
		hkReferencedObject::addReferences( constraints.begin(), constraints.getSize() );
	}

	// For each constraint, find the new closets body, and reassign the constraint to the new body.
	for (int c = 0; c < constraints.getSize(); c++)
	{
		hkpConstraintInstance* instance = constraints[c];
		hkpConstraintData* data = const_cast<hkpConstraintData*>(instance->getData());

		if ( data->getType() == hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE)
		{
			hkpBreakableConstraintData* bd = (hkpBreakableConstraintData*)data;
			bd->setBroken( instance, false );
		}
	}
	hkReferencedObject::removeReferences( constraints.begin(), constraints.getSize() );

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
