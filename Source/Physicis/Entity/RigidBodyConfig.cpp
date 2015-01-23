#include "RigidBodyConfig.h"
#include <Utility/Math.h>

RigidBodyConfig::RigidBodyConfig()
{
	m_rotation = Math::Quaternion::ZERO;
	m_position = Math::Vector3D::ZERO;
	m_linearVelocity = Math::Vector3D::ZERO;
	m_angularVelocity = Math::Vector3D::ZERO;
	m_centerOfMass = Math::Vector3D::ZERO;
	m_mass = 1.0f;
	m_linearDamping = 0.0f;
	m_angularDamping = 0.05f;
	m_gravityFactor = 1.0f;
	m_friction = 0.5f;
	m_restitution = 0.4f;
	m_maxLinearVelocity = 200.0f;
	m_maxAngularVelocity = 200.0f;
	m_shape = NULL;
	m_timeFactor = 1.0f;
}

RigidBodyConfig::~RigidBodyConfig()
{
}

void RigidBodyConfig::setMassProperties( const MassProperties& mp )
{
	m_mass = mp.m_mass;
	m_inertiaTensor = mp.m_inertiaTensor;
	m_centerOfMass = mp.m_centerOfMass;
}

void RigidBodyConfig::setPosition( const vec3& pos )
{
	m_position = pos;
}

void RigidBodyConfig::setRotation( const quart& rot )
{
	m_rotation = rot;
}
