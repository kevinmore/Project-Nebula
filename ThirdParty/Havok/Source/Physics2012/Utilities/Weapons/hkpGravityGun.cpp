/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Collide/Query/Collector/RayCollector/hkpClosestRayHitCollector.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastInput.h>
#include <Physics2012/Utilities/CharacterControl/CharacterRigidBody/hkpCharacterRigidBody.h>

// this
#include <Physics2012/Utilities/Weapons/hkpGravityGun.h>


hkpGravityGun::hkpGravityGun()
:	hkpFirstPersonGun()
,	m_maxNumObjectsPicked(10)
,	m_maxMassOfObjectPicked(200.0f)
,	m_maxDistOfObjectPicked(50.0f)
,	m_impulseAppliedWhenObjectNotPicked(100.0f)
,   m_throwVelocity(50.0f)
{
	m_type = WEAPON_TYPE_GRAVITYGUN;
	m_name = "GravityGun";
	m_capturedObjectPosition.set(2.5f, 0.6f, 0.0f);
	m_capturedObjectsOffset.set(0.0f, 1.0f, 0.0f);
}

hkpGravityGun::hkpGravityGun(class hkFinishLoadedObjectFlag flag)
:	hkpFirstPersonGun(flag)
{
	if (flag.m_finishing)
	{
		m_type = WEAPON_TYPE_GRAVITYGUN;
	}
}


hkpRigidBody* hkpGravityGun::pickBody( const hkTransform& viewTransform, const hkpRigidBody* characterBody, const hkpWorld* world ) const
{
	hkpRigidBody* nearestBody = HK_NULL;
	{
		hkSimdReal bestCost = hkSimdReal::fromFloat(0.35f);
		const hkArray<hkpSimulationIsland*>* islands = &world->getActiveSimulationIslands();
		for ( int i = 0; i < 2; i++)
		{
			for (int ii = 0; ii < islands->getSize(); ii++)
			{
				hkpSimulationIsland* island = islands[0][ii];
				hkSimdReal dist2, cost; // for goto
				for (int e = 0; e < island->getEntities().getSize(); e++ )
				{
					hkpRigidBody* body = static_cast<hkpRigidBody*>(island->getEntities()[e]);
					hkVector4 inplayerSpace; inplayerSpace.setTransformedInversePos( viewTransform, body->getCenterOfMassInWorld());
					const hkSimdReal inplayerSpace0 = inplayerSpace.getComponent<0>();
					if ( inplayerSpace0.isLess( hkSimdReal::fromFloat(0.1f) ) | inplayerSpace0.isGreater( hkSimdReal::fromFloat(m_maxDistOfObjectPicked) ) )
					{
						goto tryNextBody; // too close
					}

					// don't grab myself
					if ( body == characterBody)
					{
						continue;
					}

					// check how well we are aiming at this body
					{
						hkVector4 absInplayerSpace; absInplayerSpace.setAbs(inplayerSpace);
						dist2 = absInplayerSpace.getComponent<2>() + absInplayerSpace.getComponent<1>();
					}
					cost = dist2 / (inplayerSpace0 + hkSimdReal_1);

					// heavy objects are more difficult to pick
					if ( body->getMass() > m_maxMassOfObjectPicked )
					{
						cost.add(hkSimdReal::fromFloat(0.1f));
					}

					if ( cost > bestCost )
					{
						goto tryNextBody;
					}

					{
						for ( int co = 0; co < m_grabbedBodies.getSize(); co++ )
						{
							if ( m_grabbedBodies[co] == body )	{		goto tryNextBody;			}
						}
					}


					// check if we can see this body
					{
						hkpClosestRayHitCollector collector;
						hkpWorldRayCastInput ray;
						ray.m_from = viewTransform.getTranslation();
						ray.m_to = body->getCenterOfMassInWorld();
						world->castRay( ray, collector );
						if ( collector.hasHit() && collector.getHit().m_rootCollidable != body->getCollidable())
						{
							goto tryNextBody;
						}
					}
					bestCost = cost;
					nearestBody = body;
tryNextBody:;
				}
			}

			islands = &world->getInactiveSimulationIslands();
		}
	}
	return nearestBody;
}

void hkpGravityGun::takeObject( hkpRigidBody* body )
{
	body->addReference();
	m_grabbedBodies.pushBack( body );
	body->setMassInv( 100.0f * body->getMassInv() );
}

hkpRigidBody* hkpGravityGun::dropObject( int index )
{
	hkpRigidBody* body = m_grabbedBodies[index];
	m_grabbedBodies.removeAtAndCopy(index);
	body->setMassInv( 0.01f * body->getMassInv() );
	body->removeReference();
	return body;
}

void hkpGravityGun::gravityGunPull(const hkTransform& viewTransform, const hkpRigidBody* characterBody, const hkpWorld* world)
{
	hkpRigidBody* nearestBody = pickBody( viewTransform, characterBody, world );
	if ( nearestBody )
	{
		if ( nearestBody->getMassInv() * m_maxMassOfObjectPicked < 1.0f || m_grabbedBodies.getSize() >= m_maxNumObjectsPicked )
		{
			// heave body just push
			hkVector4 impulse = viewTransform.getColumn<0>();
			impulse.mul( -hkSimdReal::fromFloat(m_impulseAppliedWhenObjectNotPicked) );
			nearestBody->applyLinearImpulse( impulse );
		}
		else
		{
			takeObject( nearestBody );
		}
	}
}

void hkpGravityGun::gravityGunPush(const hkTransform& viewTransform, const hkpRigidBody* characterBody, const hkpWorld* world)
{
	if ( m_grabbedBodies.getSize() )
	{
		hkpRigidBody* bullet = dropObject( 0 );
		hkVector4 newVel; newVel.setMul( hkSimdReal::fromFloat(m_throwVelocity), viewTransform.getColumn<0>() );
		newVel.addMul(hkSimdReal_Inv2, hkVector4::getConstant<HK_QUADREAL_0100>());
		newVel.add( characterBody->getLinearVelocity());
		bullet->setLinearVelocity( newVel );
		bullet->setQualityType( HK_COLLIDABLE_QUALITY_BULLET );
	}
	else
	{
		hkpRigidBody* nearestBody = pickBody( viewTransform, characterBody, world );
		if ( nearestBody )
		{
			// heave body just push
			hkVector4 impulse = viewTransform.getColumn<0>();
			impulse.mul( hkSimdReal::fromFloat(m_impulseAppliedWhenObjectNotPicked) );
			nearestBody->applyLinearImpulse( impulse );
		}
	}
}

void hkpGravityGun::gravityGunPutDown(const hkTransform& viewTransform, const hkpRigidBody* characterBody, const hkpWorld* world)
{
	if ( m_grabbedBodies.getSize() )
	{
		dropObject( 0 );
	}
}

void hkpGravityGun::gravityGunStep(hkReal timeStep, const hkTransform& viewTransform, const hkpRigidBody* characterBody)
{
	for (int i = 0; i < m_grabbedBodies.getSize(); i++)
	{
		hkVector4 posLocal; posLocal.setAddMul(m_capturedObjectPosition, m_capturedObjectsOffset, hkSimdReal::fromInt32(i));
		hkVector4 pos; pos.setTransformedPos( viewTransform, posLocal );

		hkpRigidBody* body = m_grabbedBodies[i];

		hkVector4 vel = body->getLinearVelocity();
		vel.setInterpolate( body->getLinearVelocity(), characterBody->getLinearVelocity(), hkSimdReal::fromFloat(0.8f) );

		hkVector4 centerOfMass = body->getCenterOfMassInWorld();
		hkVector4 delta; delta.setSub( pos, centerOfMass );
		delta.mul( hkSimdReal::fromFloat(0.4f / timeStep) );
		vel.add( delta );

		// clamp to max linear velocity
		hkVector4 velDir = vel;
		const hkSimdReal calcBodyVel = velDir.normalizeWithLength<3>();
		const hkSimdReal maxBodyVel = hkSimdReal::fromFloat(body->getMaxLinearVelocity());

		if (calcBodyVel >= maxBodyVel)
		{
			velDir.mul(maxBodyVel - hkSimdReal::fromFloat(0.01f)); // equality case not handled consistently by physics
			body->setLinearVelocity( velDir );
		}
		else
		{
			body->setLinearVelocity(vel);
		}

		//damp angular
		hkVector4 angVel; angVel.setMul( hkSimdReal::fromFloat(0.8f), body->getAngularVelocity());
		if ( angVel.lengthSquared<3>() > hkSimdReal_1 )
		{
			body->setAngularVelocity(angVel);
		}
	}

}

void hkpGravityGun::stepGun( hkReal timeStep, hkpWorld* world, const hkpRigidBody* body, const hkTransform& viewTransform, bool fire, bool fireRmb )
{
	if (fire)
	{
		// throw object
		gravityGunPush(viewTransform, body, world);

		hkVector4 from = body->getPosition();
		for (int l=0; l<m_listeners.getSize(); ++l)
		{
			m_listeners[l]->gunFiredCallback(&from, HK_NULL);
		}
	}
	else if (fireRmb)
	{
		if (m_grabbedBodies.getSize() != m_maxNumObjectsPicked)
		{
			// grab object
			gravityGunPull(viewTransform, body, world);

			hkVector4 from = viewTransform.getTranslation();
			hkVector4 targetPos = body->getPosition();
			for (int l=0; l<m_listeners.getSize(); ++l)
			{
				m_listeners[l]->gunFiredCallback(&from, &targetPos);
			}
		}
		else
		{
			// put object down
			gravityGunPutDown(viewTransform, body, world);

			hkVector4 from = body->getPosition();
			for (int l=0; l<m_listeners.getSize(); ++l)
			{
				m_listeners[l]->gunFiredCallback(&from, HK_NULL);
			}
		}
	}

	// update objects's positions?
	gravityGunStep(timeStep, viewTransform, body);
}

void hkpGravityGun::reset(hkpWorld* world)
{
	while (m_grabbedBodies.getSize() )
	{
		dropObject(0);
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
