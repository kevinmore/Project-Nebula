#include "PhysicsWorld.h"
#include "PhysicsWorldObject.inl"
#include <Physicis/Collider/BoxCollider.h>
#include <Physicis/Collider/SphereCollider.h>
#include <Physicis/Collider/CollisionFeedback.h>
#include <Physicis/Entity/RigidBody.h>
#include <Primitives/GameObject.h>

PhysicsWorld::PhysicsWorld(const PhysicsWorldConfig& config)
	: m_config(config),
	  m_time(0.0f),
	  m_locked(false)
{
}

PhysicsWorld::~PhysicsWorld()
{
}

PhysicsWorldConfig PhysicsWorld::getConfig() const
{
	return m_config;
}

void PhysicsWorld::simulate(const float deltaTime)
{
	// don't update the physics world when it is locked
	if (m_locked)
	{
		return;
	}

	// update the physics objects
	foreach(PhysicsWorldObject* ent, m_entityList)
	{
		ent->update(deltaTime);

		// sync the center position for the collider
		ComponentPtr comp = ent->gameObject()->getComponent("Collider");
 		ColliderPtr col = comp.dynamicCast<AbstractCollider>();
		RigidBody* rb = dynamic_cast<RigidBody*>(ent);
		col->setCenter(rb->getPosition());
	}

	// handle the collision detection
	handleCollisions();
}

bool PhysicsWorld::isLocked()
{
	return m_locked;
}

void PhysicsWorld::lock()
{
	m_locked = true;
}

void PhysicsWorld::unlock()
{
	m_locked = false;
}

void PhysicsWorld::addEntity( PhysicsWorldObject* entity )
{
	lock();
	// add the rigid body and its collider
	m_entityList << entity;
	m_colliderList << dynamic_cast<RigidBody*>(entity)->getCollider();
	entity->setWorld(this);
	unlock();
}

void PhysicsWorld::removeEntity( PhysicsWorldObject* entity )
{
	lock();
	// remove the rigid body and its collider
	RigidBody* rb = dynamic_cast<RigidBody*>(entity);
	AbstractCollider* col = rb->getCollider();
	m_colliderList.removeAt(m_colliderList.indexOf(col));
	m_entityList.removeOne(entity);
	unlock();
}

int PhysicsWorld::entitiesCount()
{
	return m_entityList.count();
}

void PhysicsWorld::handleCollisions()
{
	// pair to pair
	for (int i = 0; i < m_colliderList.size() - 1; ++i)
	{
		for (int j = i + 1; j < m_colliderList.size(); ++j)
		{
			SphereCollider* c1 = dynamic_cast<SphereCollider*>(m_colliderList[i]);
			SphereCollider* c2 = dynamic_cast<SphereCollider*>(m_colliderList[j]);

			CollisionFeedback result = c1->intersect(c2);
			// if they are colliding, reverse their linear velocity
			if (result.isColliding())
			{
				RigidBody* rb1 = c1->getRigidBody();
				RigidBody* rb2 = c2->getRigidBody();

// 				rb1->setLinearVelocity(-rb1->getLinearVelocity());
// 				rb2->setLinearVelocity(-rb2->getLinearVelocity());
				rb1->setLinearVelocity(vec3(0,0,0));
				rb2->setLinearVelocity(vec3(0,0,0));
			}
		}
	}
}
