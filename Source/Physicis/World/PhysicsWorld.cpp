#include "PhysicsWorld.h"
#include "PhysicsWorldObject.inl"
#include <Physicis/Collision/Collider/BoxCollider.h>
#include <Physicis/Collision/Collider/SphereCollider.h>
#include <Physicis/Collision/Collider/ConvexHullCollider.h>
#include <Physicis/Collision/BroadPhase/BroadPhaseCollisionFeedback.h>
#include <Physicis/Collision/NarrowPhase/NarrowPhaseCollisionDetection.h>
#include <Physicis/Collision/NarrowPhase/GJKSolver.h>
#include <Physicis/Entity/RigidBody.h>
#include <Primitives/GameObject.h>
#include <Scene/IModel.h>

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
	foreach(PhysicsWorldObject* entity, m_entityList)
	{
		RigidBody* rb = dynamic_cast<RigidBody*>(entity);
		if(rb->getMotionType() != RigidBody::MOTION_FIXED)
		{
			rb->update(deltaTime);
		}
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
	bool lockState = m_locked;

	lock();
	// add the rigid body and its collider
	m_entityList << entity;
	m_colliderList << dynamic_cast<RigidBody*>(entity)->getCollider().data();
	entity->setWorld(this);

	// if the world is not locked before the operation
	// unlock it
	if(!lockState) unlock();
}

void PhysicsWorld::removeEntity( PhysicsWorldObject* entity )
{
	bool lockState = m_locked;

	lock();
	// remove the rigid body and its collider
	m_entityList.removeOne(entity);
	m_colliderList.removeAt(m_colliderList.indexOf(dynamic_cast<RigidBody*>(entity)->getCollider().data()));

	// if the world is not locked before the operation
	// unlock it
	if(!lockState) unlock();
}

void PhysicsWorld::addCollider( ICollider* collider )
{
	bool lockState = m_locked;

	lock();
	// add the rigid body and its collider
	m_colliderList << collider;
	
	// if the world is not locked before the operation
	// unlock it
	if(!lockState) unlock();
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
			ICollider* c1 = m_colliderList[i];
			ICollider* c2 = m_colliderList[j];
			
			// mark the bounding volume as green
			c1->setColor(Qt::green);
			c2->setColor(Qt::green);

			BroadPhaseCollisionFeedback result = c1->onBroadPhase(c2);

			if (result.isColliding())
			{
				// mark the bounding volume as red
				c1->setColor(Qt::red);
				c2->setColor(Qt::red);

// 				RigidBody* rb1 = c1->getRigidBody();
// 				RigidBody* rb2 = c2->getRigidBody();
// 
// 				float m1 = rb1->getMass();
// 				float m2 = rb2->getMass();
// 				vec3 v1 = rb1->getLinearVelocity();
// 				vec3 v2 = rb2->getLinearVelocity();
// 
// 				// Momentum Conservation Principle
// 				// in this case, the system does not lose kinematics energy
// 				vec3 v1Prime = v1*(m1-m2)/(m1+m2) + v2*(2*m2)/(m1+m2);
// 				vec3 v2Prime = v1*(2*m1)/(m1+m2) - v2*(m1-m2)/(m1+m2);
// 
// 				rb1->setLinearVelocity(v1Prime);
// 				rb2->setLinearVelocity(v2Prime);

				// go to narrow phase
				NarrowPhaseCollisionFeedback n;
				GJKSolver solver;
				ModelPtr model = c1->getRigidBody()->gameObject()->getComponent("Model").dynamicCast<IModel>();
				ConvexHullColliderPtr ch1 = model->getConvexHullCollider();

				model = c2->getRigidBody()->gameObject()->getComponent("Model").dynamicCast<IModel>();
				ConvexHullColliderPtr ch2 = model->getConvexHullCollider();

				if (solver.checkCollision(ch1.data(), ch2.data(), n, true))
				{
					//qDebug() << "normal vector =" << n.impulseDirectionOnALocal;
// 					qDebug() << "contact point" << ch1->gameObject()->objectName() << n.contactPntALocal;
// 					qDebug() << "contact point" << ch2->gameObject()->objectName() << n.contactPntBLocal;

					qDebug() << "contact point" << ch1->gameObject()->objectName() << n.contactPntAWorld;
					qDebug() << "contact point" << ch2->gameObject()->objectName() << n.contactPntBWorld;
					qDebug() << "contact normal" << (n.contactPntAWorld - n.contactPntBWorld).normalized();
					RigidBody* rb1 = c1->getRigidBody();
					RigidBody* rb2 = c2->getRigidBody();
					if (rb1->getMotionType() != RigidBody::MOTION_FIXED)
					{
						rb1->applyPointImpulse(-(n.contactPntAWorld - n.contactPntBWorld).normalized(), n.contactPntAWorld);
					}
					if (rb2->getMotionType() != RigidBody::MOTION_FIXED)
					{
						rb2->applyPointImpulse(-(n.contactPntBWorld - n.contactPntAWorld).normalized(), n.contactPntBWorld);
					}
				}
			}
// 			boarderCheck(c1->getRigidBody());
// 			boarderCheck(c2->getRigidBody());

		}
	}
}

void PhysicsWorld::boarderCheck( RigidBody* rb )
{
	vec3 pos = rb->getPosition();
	if (pos.x() > 5)
	{
		rb->setLinearVelocity(Math::Vector3::reflect(rb->getLinearVelocity(), vec3(-1,0,0)));
		vec3 v = rb->getLinearVelocity();
		if (qAbs(v.x()) < 1.0f)
		{
			rb->setLinearVelocity(vec3( - 1.0f, v.y(), v.z()));
		}
	}
	if ( pos.x() < -5)
	{
		rb->setLinearVelocity(Math::Vector3::reflect(rb->getLinearVelocity(), vec3(1,0,0)));
		vec3 v = rb->getLinearVelocity();
		if (qAbs(v.x()) < 1.0f)
		{
			rb->setLinearVelocity(vec3(+ 1.0f, v.y(), v.z()));
		}
	}
	if (pos.y() > 10)
	{
		rb->setLinearVelocity(Math::Vector3::reflect(rb->getLinearVelocity(), vec3(0,-1,0)));
		vec3 v = rb->getLinearVelocity();
		if (qAbs(v.y()) < 1.0f)
		{
			rb->setLinearVelocity(vec3(v.x(), - 1.0f, v.z()));
		}
	}
	if ( pos.y() < 1)
	{
		rb->setLinearVelocity(Math::Vector3::reflect(rb->getLinearVelocity(), vec3(0,1,0)));
		vec3 v = rb->getLinearVelocity();
		if (qAbs(v.y()) < 1.0f)
		{
			rb->setLinearVelocity(vec3(v.x(), + 1.0f, v.z()));
		}
	}
	if (pos.z() > 5)
	{
		rb->setLinearVelocity(Math::Vector3::reflect(rb->getLinearVelocity(), vec3(0,0,-1)));
		vec3 v = rb->getLinearVelocity();
		if (qAbs(v.z()) < 1.0f)
		{
			rb->setLinearVelocity(vec3(v.x(), v.y(), - 1.0f));
		}
	}
	if ( pos.z() < -5)
	{
		rb->setLinearVelocity(Math::Vector3::reflect(rb->getLinearVelocity(), vec3(0,0,1)));
		vec3 v = rb->getLinearVelocity();
		if (qAbs(v.z()) < 1.0f)
		{
			rb->setLinearVelocity(vec3(v.x(), v.y(), 1.0f));
		}
	}
}

void PhysicsWorld::reset()
{
	lock();
	m_entityList.clear();
	m_colliderList.clear();
}
