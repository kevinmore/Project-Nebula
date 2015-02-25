#pragma once
#include "PhysicsWorldConfig.h"
#include "PhysicsWorldObject.h"
#include "Physicis/Collision/Collider/ICollider.h"

struct CollisionPair
{
	ICollider* pColliderA;
	ICollider* pColliderB;
	vec3 contactPoint;
	uint contactCount;

	CollisionPair(ICollider* a, ICollider* b)
	{
		pColliderA = a;
		pColliderB = b;
		contactPoint = vec3(1000, 1000, 1000);
	}
};

struct ImpulsePair 
{
	float magnitudeA;
	float magnitudeB;

	ImpulsePair()
	{
		magnitudeA = 0.0f;
		magnitudeB = 0.0f;
	}
};

typedef QSharedPointer<CollisionPair> CollisionPairPtr;

class NarrowPhaseCollisionFeedback;
class PhysicsWorld
{
public:
	PhysicsWorld(const PhysicsWorldConfig& config);
	~PhysicsWorld();

	PhysicsWorldConfig getConfig() const;

	void simulate(const float deltaTime);
	void handleCollisions();


	bool isLocked();
	void lock();
	void unlock();

	void addEntity(PhysicsWorldObject* entity);
	void removeEntity(PhysicsWorldObject* entity);
	void addBroadPhaseCollider(ICollider* collider);
	void removeBroadPhaseCollider(ICollider* collider);

	void generateCollisionPairs();

	int entitiesCount();

	void reset();


	ImpulsePair computeContactImpulseMagnitude(const NarrowPhaseCollisionFeedback* pCollisionInfo);

	void backToTimeOfImpact(RigidBody* rb1, RigidBody* rb2);
	void elasticCollisionResponse(RigidBody* rb1, RigidBody* rb2);

private:
	PhysicsWorldConfig m_config;
	float m_timeStep;
	bool m_locked;
	QList<PhysicsWorldObject*> m_entityList;
	QVector<ICollider*> m_broadPhaseColliderList;
	QList<CollisionPairPtr> m_collisionPairs;

	void boarderCheck(RigidBody* rb);
};

