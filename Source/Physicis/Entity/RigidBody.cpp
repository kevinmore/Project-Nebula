#include "RigidBody.h"
#include <Utility/Math.h>

RigidBody::RigidBody(QObject* parent)
	: PhysicsWorldObject(parent)
{
	m_shape = NULL;

	m_position = Math::Vector3D::ZERO;
	m_rotation = Math::Quaternion::ZERO;
	m_linearVelocity = Math::Vector3D::ZERO;
	m_angularVelocity = Math::Vector3D::ZERO;

	m_centerOfMass = Math::Vector3D::ZERO;
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

	m_deltaAngle = Math::Vector3D::ZERO;
	m_objectRadius = 1.0f;
}

void RigidBody::setShape( const AbstractShape* shape )
{
	m_shape = shape;
}

const AbstractShape* RigidBody::getShape() const
{
	return m_shape;
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
}

void RigidBody::setRotation( const quart& rot )
{
	m_rotation = rot;
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

float RigidBody::getMass() const
{
	return 1.0f / m_massInv;
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

void RigidBody::setPositionAndRotation( const vec3& position, const quart& rotation )
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