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
#include <Utility/Math.h>

using namespace Math;

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
	addCollider(dynamic_cast<RigidBody*>(entity)->getCollider().data());
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
	
	// generate the collision pairs
	generateCollisionPairs();

	// if the world is not locked before the operation
	// unlock it
	if(!lockState) unlock();
}

void PhysicsWorld::generateCollisionPairs()
{
	m_collisionPairs.clear();

	for (int i = 0; i < m_colliderList.size() - 1; ++i)
		for (int j = i + 1; j < m_colliderList.size(); ++j)
			m_collisionPairs << CollisionPairPtr(new CollisionPair(m_colliderList[i], m_colliderList[j]));
}

int PhysicsWorld::entitiesCount()
{
	return m_entityList.count();
}

void PhysicsWorld::handleCollisions()
{
	foreach(CollisionPairPtr pair, m_collisionPairs)
	{
		ICollider* cA = pair->pColliderA;
		ICollider* cB = pair->pColliderB;

		// mark the bounding volume as green
		cA->setColor(Qt::green);
		cB->setColor(Qt::green);

		// collision detection on broad phase
		BroadPhaseCollisionFeedback broadPhaseResult = cA->onBroadPhase(cB);

		// if not colliding on broad phase
		if (!broadPhaseResult.isColliding())
		{
			// mark the pair as un-checked
			pair->bChecked = false;
			continue;
		}

		// if colliding on broad phase and the pair is not checked before
		if (broadPhaseResult.isColliding() && !pair->bChecked)
		{
			// mark the bounding volume as red
			cA->setColor(Qt::red);
			cB->setColor(Qt::red);

			// go to narrow phase
			NarrowPhaseCollisionFeedback collisionInfo;
			GJKSolver solver;
			ModelPtr model = cA->getRigidBody()->gameObject()->getComponent("Model").dynamicCast<IModel>();
			ConvexHullColliderPtr chA = model->getConvexHullCollider();

			model = cB->getRigidBody()->gameObject()->getComponent("Model").dynamicCast<IModel>();
			ConvexHullColliderPtr chB = model->getConvexHullCollider();

			if (solver.checkCollision(chA.data(), chB.data(), collisionInfo, true))
			{
				RigidBody* bodyA = cA->getRigidBody();
				RigidBody* bodyB = cB->getRigidBody();
				vec3 A2B = (bodyB->getPosition() - bodyA->getPosition()).normalized();
				vec3 n = collisionInfo.contactNormalWorld;

				// separate the two objects by the penetration depth
				if (collisionInfo.penetrationDepth > 0.0f)
				{
					float offset = 0.5 * collisionInfo.penetrationDepth;
					bodyA->moveBy(-offset * A2B);
					bodyB->moveBy(offset * A2B);
				}

				float impuseMagnitude = computeContactImpulseMagnitude(&collisionInfo);
				qDebug() << cA->getRigidBody()->gameObject()->objectName() <<
					     cB->getRigidBody()->gameObject()->objectName() << impuseMagnitude;

				// apply impulse based on the direction
				if (vec3::dotProduct(A2B, n) > 0)
				{
					bodyA->applyPointImpulse(-impuseMagnitude * n, collisionInfo.closestPntAWorld);
					bodyB->applyPointImpulse(impuseMagnitude * n, collisionInfo.closestPntBWorld);
				}
				else
				{
					bodyA->applyPointImpulse(impuseMagnitude * n, collisionInfo.closestPntAWorld);
					bodyB->applyPointImpulse(-impuseMagnitude * n, collisionInfo.closestPntBWorld);
				}

				// mark the pair as checked
				pair->bChecked = true;

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
			}
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

float PhysicsWorld::computeContactImpulseMagnitude( const NarrowPhaseCollisionFeedback* pCollisionInfo )
{
	RigidBody* A = pCollisionInfo->pObjA->getRigidBody();
	RigidBody* B = pCollisionInfo->pObjB->getRigidBody();

	vec3 n = pCollisionInfo->contactNormalWorld;

// 	vec3 linearA = A->getLinearVelocity();
// 	vec3 angularA = A->getAngularVelocity();
// 	vec3 comA = A->getCenterOfMassInWorld();
// 	vec3 offsetA = pCollisionInfo->closestPntAWorld - comA;
// 	vec3 crossA = vec3::crossProduct(angularA, offsetA);
// 
// 	vec3 linearB = B->getLinearVelocity();
// 	vec3 angularB = B->getAngularVelocity();
// 	vec3 comB = B->getCenterOfMassInWorld();
// 	vec3 offsetB = pCollisionInfo->closestPntBWorld - comB;
// 	vec3 crossB = vec3::crossProduct(angularB, offsetB);

	vec3 pointVelocityA = A->getPointVelocityWorld(pCollisionInfo->closestPntAWorld);
	vec3 pointVelocityB = B->getPointVelocityWorld(pCollisionInfo->closestPntBWorld);
	float Vrel = vec3::dotProduct(n, pointVelocityA - pointVelocityB);
	qDebug() << "point A =" << pCollisionInfo->closestPntAWorld;
	qDebug() << "point B =" << pCollisionInfo->closestPntBWorld;
	// check if Vrel == 0
	if (Vrel == 0.0f) return 0.0f;

	float k = -(1 + 0.5*(A->getRestitution() + B->getRestitution()));

	float massInvSum = A->getMassInv() + B->getMassInv();
	vec3 rA = pCollisionInfo->closestPntAWorld - A->getCenterOfMassInWorld();
	vec3 rB = pCollisionInfo->closestPntBWorld - B->getCenterOfMassInWorld();

	float componentA = vec3::dotProduct(n, vec3::crossProduct(Math::Vector3::setMul(vec3::crossProduct(rA, n), A->getInertiaInvWorld()), rA));
	float componentB = vec3::dotProduct(n, vec3::crossProduct(Math::Vector3::setMul(vec3::crossProduct(rB, n), B->getInertiaInvWorld()), rB));
		
	float denominator = massInvSum + componentA + componentB;
	
	
	return qAbs(k * Vrel / denominator);
}
