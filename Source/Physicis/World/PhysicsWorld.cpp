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
	m_colliderList << dynamic_cast<RigidBody*>(entity)->getCollider().data();
	entity->setWorld(this);
	unlock();
}

void PhysicsWorld::removeEntity( PhysicsWorldObject* entity )
{
	lock();
	// remove the rigid body and its collider
	m_entityList.removeOne(entity);
	m_colliderList.removeAt(m_colliderList.indexOf(dynamic_cast<RigidBody*>(entity)->getCollider().data()));
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
			AbstractCollider* c1 = m_colliderList[i];
			AbstractCollider* c2 = m_colliderList[j];

			CollisionFeedback result = c1->intersect(c2);

			if (result.isColliding())
			{
				RigidBody* rb1 = c1->getRigidBody();
				RigidBody* rb2 = c2->getRigidBody();

				float m1 = rb1->getMass();
				float m2 = rb2->getMass();
				vec3 v1 = rb1->getLinearVelocity();
				vec3 v2 = rb2->getLinearVelocity();

				// Momentum Conservation Principle
				// in this case, the system does not lose kinematics energy
				vec3 v1Prime = v1*(m1-m2)/(m1+m2) + v2*(2*m2)/(m1+m2);
				vec3 v2Prime = v1*(2*m1)/(m1+m2) - v2*(m1-m2)/(m1+m2);

 				rb1->setLinearVelocity(v1Prime);
 				rb2->setLinearVelocity(v2Prime);
			}
		}
	}
}
