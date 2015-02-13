#include "RigidBody.h"
#include <Physicis/World/PhysicsWorld.h>
#include <Physicis/World/PhysicsWorldObject.inl>
#include <Physicis/Collision/Collider/ICollider.h>
#include <Primitives/GameObject.h>

RigidBody::RigidBody(const vec3& position, const quat& rotation, QObject* parent)
	: PhysicsWorldObject(parent)
{
	m_position = position;
	m_rotation = rotation;
	m_transformMatrix.translate(position);
	m_transformMatrix.rotate(m_rotation);

	m_linearVelocity = Math::Vector3::ZERO;
	m_angularVelocity = Math::Vector3::ZERO;

	m_centerOfMass = Math::Vector3::ZERO;
	m_mass = 1.0f;
	m_massInv = 1.0f;
	m_forceAccum = Math::Vector3::ZERO;

	m_linearDamping = 0.0f;
	m_angularDamping = 0.05f;
	m_gravityFactor = 1.0f;
	m_friction = 0.5f;
	m_restitution = 0.4f;
	m_maxLinearVelocity = 200.0f;
	m_maxAngularVelocity = 200.0f;
	m_timeFactor = 1.0f;

	m_eulerAngles = Math::Vector3::ZERO;
	m_objectRadius = 1.0f;
	m_rotationMatrix.setToIdentity();
	m_inertiaTensor.setToIdentity();
	m_inertiaTensorInv.setToIdentity();
	Math::Matrix3::setInverse(m_inertiaTensorInv);
}

RigidBody::~RigidBody()
{
	 // remove it from the world
	 m_world->removeEntity(this);
}

void RigidBody::setMassProperties( const MassProperties& mp )
{
	m_mass = mp.m_mass;
	m_inertiaTensor = mp.m_inertiaTensor;
	m_centerOfMass = mp.m_centerOfMass;
}

void RigidBody::setPosition( const vec3& pos )
{
	m_position = pos;
	m_transformMatrix.setToIdentity();
	m_transformMatrix.translate(pos);
}

void RigidBody::setRotation( const quat& rot )
{
	m_rotation = rot;
	m_transformMatrix.setToIdentity();
	m_transformMatrix.rotate(rot);
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
	m_position = position;
	m_rotation = rotation;
}

void RigidBody::setRestitution( float newRestitution )
{
	m_restitution = newRestitution;
}

void RigidBody::setFriction( float newFriction )
{
	m_friction = newFriction;
}

void RigidBody::update( const float dt )
{
	// only update the linear properties as an abstract rigid body
	m_linearVelocity += m_gravityFactor * getWorld()->getConfig().m_gravity * dt;
	m_deltaPosition = m_linearVelocity * dt;
	m_position += m_deltaPosition;
	m_transformMatrix.translate(m_position);

	// sync the center position for the collider
	m_collider->setCenter(m_position);
}

void RigidBody::attachCollider( ColliderPtr col )
{
	m_collider = col;
	col->setRigidBody(this);

	// attach the collider to the game object
//	gameObject()->attachComponent(col);
}
