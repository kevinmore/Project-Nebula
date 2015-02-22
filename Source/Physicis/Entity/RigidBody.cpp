#include "RigidBody.h"
#include <Physicis/World/PhysicsWorld.h>
#include <Physicis/World/PhysicsWorldObject.inl>
#include <Physicis/Collision/Collider/ICollider.h>
#include <Primitives/GameObject.h>

RigidBody::RigidBody(const vec3& position, const quat& rotation, QObject* parent)
	: PhysicsWorldObject(parent)
{
	m_linearVelocity = Math::Vector3::ZERO;
	m_angularVelocity = Math::Vector3::ZERO;

	m_centerOfMass = Math::Vector3::ZERO;
	m_mass = 1.0f;
	m_massInv = 1.0f;

	m_linearDamping = 0.0f;
	m_angularDamping = 0.05f;
	m_gravityFactor = 1.0f;
	m_friction = 0.5f;
	m_restitution = 0.4f;
	m_maxLinearVelocity = 200.0f;
	m_maxAngularVelocity = 200.0f;
	m_timeFactor = 1.0f;
	m_canSleep = false;
	m_isAwake = true;
}

RigidBody::~RigidBody()
{
	 // remove it from the world
	if (m_world)
	{
		m_world->removeEntity(this);
	}
}

void RigidBody::update( const float dt )
{
	if (!m_isAwake) return;

	// only update the linear properties as an abstract rigid body
	m_linearVelocity += m_gravityFactor * getWorld()->getConfig().m_gravity * dt;
	m_deltaPosition = m_linearVelocity * dt;
	m_transform.translate(m_deltaPosition);

	// update the angular properties
	quat rotation = quat::fromAxisAndAngle(m_angularVelocity, qRadiansToDegrees(m_angularVelocity.length() * dt));
	m_transform.rotate(rotation);

	// sync the center position for the collider
	m_BroadPhaseCollider->setCenter(m_transform.getPosition());
	m_NarrowPhaseCollider->setCenter(m_transform.getPosition());

	// Update the kinetic energy store, and possibly put the body to sleep.
	if (m_canSleep) 
	{
		float currentMotion = m_linearVelocity.lengthSquared() + m_angularVelocity.lengthSquared();

		float bias = pow(0.5f, dt);
		m_motionEnergy = bias * m_motionEnergy + (1 - bias) * currentMotion;

		if (m_motionEnergy < 0.3f) setAwake(false);
		else if (m_motionEnergy > 10 * 0.3f) m_motionEnergy = 10 * 0.3f;
	}
}

void RigidBody::setMassProperties( const MassProperties& mp )
{
	m_mass = mp.m_mass;
	m_centerOfMass = mp.m_centerOfMass;
}

void RigidBody::setPosition( const vec3& pos )
{
	m_transform.setPosition(pos);
}

void RigidBody::setRotation( const quat& rot )
{
	m_transform.setRotation(rot);
}


void RigidBody::setMass( float m )
{
	float massInv;
	if (m == 0.0f)
		massInv = 0.0f;
	else
		massInv = 1.0f / m;
	setMassInv(massInv);
}

void RigidBody::setMassInv( float mInv )
{
	m_mass = mInv;
}

// Explicit center of mass in local space.
void RigidBody::setCenterOfMassLocal(const vec3& centerOfMass)
{	
	m_centerOfMass = centerOfMass;
}

void RigidBody::setPositionAndRotation( const vec3& position, const quat& rotation )
{
	m_transform.setPosition(position);
	m_transform.setRotation(rotation);
}

void RigidBody::setRestitution( float newRestitution )
{
	m_restitution = newRestitution;
}

void RigidBody::setFriction( float newFriction )
{
	m_friction = newFriction;
}

void RigidBody::attachBroadPhaseCollider( ColliderPtr col )
{
	m_BroadPhaseCollider = col;
	col->setRigidBody(this);
}

void RigidBody::attachNarrowPhaseCollider( ColliderPtr col )
{
	m_NarrowPhaseCollider = col;
	col->setRigidBody(this);
}

void RigidBody::syncTransform( const Transform& transform )
{
	// only sync the transform when the physics world is locked
	// this means that the engine is in editor mode
	if (m_world->isLocked())
	{
		m_transform = transform;
	}
}

void RigidBody::setAwake( const bool awake /*= true*/ )
{
	if (awake) 
	{
		m_isAwake= true;

		// Add a bit of motion to avoid it falling asleep immediately.
		m_motionEnergy = 0.3f * 2.0f;
	} 
	else 
	{
		m_isAwake = false;
		m_linearVelocity = Vector3::ZERO;
		m_angularVelocity = Vector3::ZERO;
	}
}

void RigidBody::setCanSleep( const bool canSleep /*= true*/ )
{
	m_canSleep = canSleep;
	if (!m_canSleep && !m_isAwake) setAwake();
}
