/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>

// this
#include <Physics2012/Utilities/Weapons/hkpBallGun.h>


hkpBallGun::hkpBallGun( int numBullets )
:	hkpFirstPersonGun()
,	m_bulletRadius(0.2f)
,	m_bulletVelocity(40.0f)
,	m_bulletMass(50.0f)
,	m_damageMultiplier(50.0f)
,	m_maxBulletsInWorld(numBullets)
{
	m_bulletOffsetFromCenter.setZero();
	m_type = WEAPON_TYPE_BALLGUN;
	m_name = "BallGun";
	m_addedBodies = new hkQueue<class hkpRigidBody*>;
	m_addedBodies->setCapacity(numBullets);
}

hkpBallGun::~hkpBallGun()
{
	// remove from any world
	while (!m_addedBodies->isEmpty() )
	{
		hkpRigidBody* rb;
		m_addedBodies->dequeue( rb );

		for (int l=m_listeners.getSize()-1; l>=0; --l)
		{
			m_listeners[l]->bulletDeletedCallback(rb);
		}

		if ( rb->getWorld() )
		{
			rb->getWorld()->removeEntity( rb );
		}
		rb->removeReference();
	}

	delete m_addedBodies;
}


void hkpBallGun::fireGun( hkpWorld* world, const hkTransform& viewTransform )
{
	hkVector4 forwardVector = viewTransform.getRotation().getColumn<0>();
	hkVector4 from = viewTransform.getTranslation();
	hkVector4 sweepStart; sweepStart.setAdd(from, forwardVector);
	hkVector4 sweepEnd; sweepEnd.setAddMul(sweepStart, forwardVector, hkSimdReal::fromFloat(200.0f));

	hkpRigidBodyCinfo ci;
	{
		ci.m_shape                   = new hkpSphereShape( m_bulletRadius );
		ci.m_mass                    = m_bulletMass;
		ci.m_qualityType             = HK_COLLIDABLE_QUALITY_BULLET;
		ci.m_allowedPenetrationDepth = 0.01f;
		ci.m_restitution             = 0.2f;
		ci.m_friction                = 1.0f;
		ci.m_motionType              = hkpMotion::MOTION_SPHERE_INERTIA;
		hkpInertiaTensorComputer::setShapeVolumeMassProperties( ci.m_shape, m_bulletMass*2.0f, ci );
		ci.m_mass                    = m_bulletMass;
		ci.m_position                . setAdd(sweepStart, m_bulletOffsetFromCenter);
		ci.m_linearDamping           = 0;
		ci.m_angularDamping          = 0.4f;
	}

	hkpFirstPersonGun::SweepSphereOut target;
	if (HK_SUCCESS == hkpFirstPersonGun::sweepSphere(world, sweepStart, m_bulletRadius, sweepEnd, target))
	{
		hkpFirstPersonGun::calcVelocityToTarget(ci.m_position, target.m_contactPoint.getPosition(), world->getGravity(), m_bulletVelocity, ci.m_linearVelocity);
	}
	else
	{
		ci.m_linearVelocity.setMul( hkSimdReal::fromFloat(m_bulletVelocity), forwardVector );
	}

	hkVector4 targetPos = target.m_contactPoint.getPosition();
	for (int l=0; l<m_listeners.getSize(); ++l)
	{
		m_listeners[l]->gunFiredCallback(&from, &targetPos);
	}

	hkpRigidBody* bullet = new hkpRigidBody( ci );
	ci.m_shape->removeReference();

	bullet->m_damageMultiplier = m_damageMultiplier;

	for (int l=0; l<m_listeners.getSize(); ++l)
	{
		m_listeners[l]->bulletCreatedCallback(bullet);
	}

	world->addEntity( bullet );

	m_addedBodies->enqueue( bullet );
	if (m_addedBodies->getSize() > m_maxBulletsInWorld)
	{
		hkpRigidBody* rb;
		m_addedBodies->dequeue( rb );

		for (int l=m_listeners.getSize()-1; l>=0; --l)
		{
			m_listeners[l]->bulletDeletedCallback(rb);
		}

		if ( rb->getWorld() )
		{
			world->removeEntity( rb );
		}
		rb->removeReference();
	}
}

void hkpBallGun::reset( hkpWorld* world )
{
	// reset bodies in given world
	while (!m_addedBodies->isEmpty() )
	{
		hkpRigidBody* rb;
		m_addedBodies->dequeue( rb );

		for (int l=m_listeners.getSize()-1; l>=0; --l)
		{
			m_listeners[l]->bulletDeletedCallback(rb);
		}

		if ( rb->getWorld() == world )
		{
			world->removeEntity( rb );
		}
		rb->removeReference();
	}
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
