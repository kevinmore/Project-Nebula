#pragma once
#include "PhysicsWorldConfig.h"
#include "PhysicsWorldObject.h"
#include "Physicis/Collision/Collider/ICollider.h"

struct CollisionPair
{
	ICollider* pColliderA;
	ICollider* pColliderB;
	float contactTime;

	CollisionPair(ICollider* a, ICollider* b)
	{
		pColliderA = a;
		pColliderB = b;
		contactTime = 0.0f;
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

	// Execute the start conditions, only once
	void start();

	// Step the physics world
	void simulate(const float deltaTime);

	// Handle collisions
	void handleCollisions();


	bool isLocked();
	void lock();
	void unlock();

	void addEntity(PhysicsWorldObject* entity);
	void removeEntity(PhysicsWorldObject* entity);
	void addBroadPhaseCollider(ICollider* collider);
	void removeBroadPhaseCollider(ICollider* collider);

	void generateCollisionPairs();
	void handleNarrowPhase(CollisionPairPtr pair);

	int entitiesCount();

	void reset();

	ImpulsePair computeContactImpulseMagnitude(const NarrowPhaseCollisionFeedback& collisionInfo);

	NarrowPhaseCollisionFeedback backToTimeOfImpact(RigidBody* rb1, RigidBody* rb2);
	void resetRigidBodyBeforeBackTrack(RigidBody* body, const Transform& trans, const vec3& linearVelocity, const vec3& angularVelocity);

	void elasticCollisionResponse(RigidBody* rb1, RigidBody* rb2);
	void generalCollisionResponse(const NarrowPhaseCollisionFeedback& collisionInfo);

	inline void setCurrentTime(const float time) { m_currentTime = time; }

	QList<PhysicsWorldObject*> getEntityList() const { return m_entityList; }

private:
	PhysicsWorldConfig m_config;
	
	float m_timeStep;
	float m_currentTime;
	bool m_locked;
	QList<PhysicsWorldObject*> m_entityList;
	QVector<ICollider*> m_broadPhaseColliderList;
	QList<CollisionPairPtr> m_broadPhaseCollisionPairs;

	void boarderCheck(RigidBody* rb);
};

