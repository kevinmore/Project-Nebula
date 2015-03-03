/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

// This
#include <Physics2012/Utilities/Weapons/hkpProjectileGun.h>


hkpGunProjectile::hkpGunProjectile(hkpProjectileGun* gun, hkpRigidBody* body)
: m_flags(0)
, m_body(body)
, m_existanceTime(0.0f)
, m_gun(gun)
{
	body->addReference();
}

hkpGunProjectile::~hkpGunProjectile()
{
	_destroyBody();
}

void hkpGunProjectile::destroy()
{
    if (!isDestroyed())
    {
        onDestroy();
        m_flags.orWith(DESTROYED);
    }
}

void hkpGunProjectile::addToWorld(hkpWorld* world)
{
    HK_ASSERT(0x0278baee7, m_body && !isInWorld());
    if (isHitListening())
    {
        m_body->addContactListener(this);
    }
    world->addEntity(m_body);
}

void hkpGunProjectile::removeFromWorld()
{
    if (m_body == HK_NULL) return;

    HK_ASSERT(0x24323423, isInWorld());

    hkpWorld* world = m_body->getWorld();
    world->removeEntity(m_body);
    if (isHitListening())
    {
        m_body->removeContactListener(this);
    }
}

void hkpGunProjectile::onDestroy()
{
	_destroyBody();
}

void hkpGunProjectile::_destroyBody()
{
	if (m_body)
	{
		if (isInWorld())
		{
			removeFromWorld();
		}
		m_body->removeReference();
		m_body = HK_NULL;
	}
}

void hkpGunProjectile::update(hkReal timeStep)
{
	if (!isDestroyed())
	{
		m_existanceTime += timeStep;
		onUpdate(timeStep);
	}
}

void hkpGunProjectile::setHitListening(bool enable)
{
    if (enable == isHitListening())
    {
        return;
    }

    if (isInWorld())
    {
        if (enable)
        {
            m_body->addContactListener(this);
        }
        else
        {
            m_body->removeContactListener(this);
        }
    }

    // Set the flag
    if (enable)
	{
        m_flags.orWith(HIT_LISTENING);
	}
    else
	{
        m_flags.andWith(hkUint8(~HIT_LISTENING));
	}
}

/* static */void HK_CALL hkpGunProjectile::flyTrue(hkpRigidBody* body, hkReal minVelocity, hkReal timeStep)
{
    hkTransform trans = body->getTransform();
    const hkVector4& velocity = body->getLinearVelocity();

    const hkVector4& direction = trans.getColumn<0>();
	if (velocity.lengthSquared<3>() > hkSimdReal::fromFloat(minVelocity*minVelocity))
    {
        hkVector4 velDir = velocity; velDir.normalize<3>();
        hkVector4 cross; cross.setCross(velDir, direction);

        const hkSimdReal length = cross.length<3>();
		if (length > hkSimdReal::fromFloat(1e-5f))
        {
            cross.normalize<3>();
            // Rotate by some tiny amount
            hkRotation rot, finalRot;
            rot.setAxisAngle(cross, length.getReal() * timeStep * -5.0f);

            finalRot.setMul( rot, trans.getRotation());
            trans.getRotation() = finalRot;

            body->setTransform(trans);
        }
    }
}


hkpProjectileGun::hkpProjectileGun(hkpWorld* world, hkdWorld* destructionWorld)
: hkpFirstPersonGun()
, m_maxProjectiles(5)
, m_reloadTime(0.3f)
, m_reload(0.0f)
, m_world(world)
, m_destructionWorld(destructionWorld)
{
	// this is an invalid gun, so no name or type set
}

void hkpProjectileGun::clearProjectiles()
{
    const int numProjectiles = m_projectiles.getSize();
    for (int i = 0; i < numProjectiles; i++)
    {
        hkpGunProjectile* projectile = m_projectiles[i];
        projectile->destroy();
        projectile->removeReference();
    }
    m_projectiles.clear();
}

void hkpProjectileGun::destroyAgedProjectiles(hkReal time)
{
    const int numProjectiles = m_projectiles.getSize();
    for (int i = 0; i < numProjectiles; i++)
    {
        hkpGunProjectile* projectile = m_projectiles[i];
        if (projectile->getExistanceTime() > time)
        {
            projectile->destroy();
        }
    }
}

void hkpProjectileGun::updateProjectiles(hkReal timeStep)
{
    const int numProjectiles = m_projectiles.getSize();
    for (int i = 0; i < numProjectiles; i++)
    {
        hkpGunProjectile* projectile = m_projectiles[i];
        projectile->update(timeStep);
    }
}

void hkpProjectileGun::clearHitProjectiles()
{
    const int numProjectiles = m_projectiles.getSize();
    for (int i = 0; i < numProjectiles; i++)
    {
        m_projectiles[i]->clearHit();
    }
}

void hkpProjectileGun::removeDestroyedProjectiles()
{
    int numProjectiles = m_projectiles.getSize();
    for (int i = 0; i < numProjectiles; i++)
    {
        hkpGunProjectile* projectile = m_projectiles[i];

        if (projectile->isDestroyed())
        {
            projectile->removeReference();
            m_projectiles.removeAt(i);
            numProjectiles--;
            i--;
        }
    }
}

void hkpProjectileGun::addProjectile(hkpGunProjectile* proj)
{
    HK_ASSERT(0xdd82729f,proj->getGun() == this);
	HK_ASSERT(0x2747ba74,m_world);

    proj->addReference();

    m_projectiles.pushBack(proj);
    proj->addToWorld(m_world);
}

void hkpProjectileGun::fireGun( hkpWorld* world, const hkTransform& viewTransform )
{
    if (m_reload > 0.0f)
    {
        return;
    }

	// Set up the reload
	m_reload = m_reloadTime;
    onGunFired(world, viewTransform);
}

void hkpProjectileGun::stepGun( hkReal timeStep, hkpWorld* world, const hkpRigidBody* characterBody, const hkTransform& viewTransform, bool fire, bool fireRmb )
{
    if (m_reload > 0.0f)
    {
        m_reload -= timeStep;
    }
    onUpdate(timeStep, world, characterBody, viewTransform, fire, fireRmb);
}

void hkpProjectileGun::onUpdate( hkReal timeStep, hkpWorld* world, const hkpRigidBody* characterBody, const hkTransform& viewTransform, bool fire, bool fireRmb )
{
    updateProjectiles(timeStep);
    clearHitProjectiles();
    removeDestroyedProjectiles();
}

hkpGunProjectile* hkpProjectileGun::getFirstActiveProjectile() const
{
    int numProjectiles = m_projectiles.getSize();
    for (int i = 0; i < numProjectiles; i++)
    {
        hkpGunProjectile* projectile = m_projectiles[i];
        if (!projectile->isDestroyed())
        {
            return projectile;
        }
    }
    return HK_NULL;
}

void hkpProjectileGun::reset( hkpWorld* world )
{
	clearProjectiles();
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
