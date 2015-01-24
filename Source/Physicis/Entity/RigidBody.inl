#include "RigidBody.h"
#include <Utility/Math.h>

inline float RigidBody::getMass() const
{
	return m_mass;
}

inline float RigidBody::getMassInv() const
{
	return m_massInv;
}

inline const vec3& RigidBody::getCenterOfMassLocal() const
{
	return m_centerOfMass;
}

inline const vec3& RigidBody::getCenterOfMassInWorld() const
{
	return m_centerOfMass;
}

inline const vec3& RigidBody::getPosition() const
{
	return m_position;
}

inline const quart& RigidBody::getRotation() const
{
	return m_rotation;
}

inline const vec3& RigidBody::getLinearVelocity() const
{
	return m_linearVelocity;
}

inline const vec3& RigidBody::getAngularVelocity() const
{
	return m_angularVelocity;
}

void RigidBody::setAngularVelocity( const vec3& newVel )
{
	m_angularVelocity = newVel;
}

inline vec3& RigidBody::getPointVelocity( const vec3& p ) const
{
	m_linearVelocity + vec3::crossProduct(m_angularVelocity, p - m_centerOfMass);
}

inline void RigidBody::applyLinearImpulse( const vec3& imp )
{
	m_linearVelocity += m_massInv * imp;
}

inline float RigidBody::getLinearDamping()
{
	return m_linearDamping;
}

inline void RigidBody::setLinearDamping( float d )
{
	m_linearDamping = d;
}

inline float RigidBody::getAngularDamping()
{
	return m_angularDamping;
}

inline void RigidBody::setAngularDamping( float d )
{
	m_angularDamping = d;
}

inline float RigidBody::getTimeFactor()
{
	return m_timeFactor;
}

inline void RigidBody::setTimeFactor( float f )
{
	m_timeFactor = f;
}

inline float RigidBody::getFriction() const
{
	return m_friction;
}

inline float RigidBody::getGravityFactor()
{
	return m_gravityFactor;
}

inline void RigidBody::setGravityFactor( float gravityFactor )
{
	m_gravityFactor = gravityFactor;
}
