#include "BoxRigidBody.h"
#include <Utility/Math.h>


BoxRigidBody::BoxRigidBody( const vec3& position, const quart& rotation )
	: RigidBody(position, rotation)
{
	m_MotionType = RigidBody::MOTION_BOX_INERTIA;
	vec3 boxSize(50, 50, 50);
	m_shape = new BoxShape(m_centerOfMass, boxSize);
}

mat3 BoxRigidBody::getInertiaLocal() const
{
	return m_inertiaTensor;
}

void BoxRigidBody::setInertiaLocal( const mat3& inertia )
{
	m_inertiaTensor = inertia;

	m_inertiaTensorInv = inertia;
	Math::Matrix3::setInverse(m_inertiaTensorInv);
}

mat3 BoxRigidBody::getInertiaInvLocal() const
{
	return m_inertiaTensorInv;
}

void BoxRigidBody::setInertiaInvLocal( const mat3& inertiaInv )
{
	m_inertiaTensorInv = inertiaInv;
}


mat3 BoxRigidBody::getInertiaWorld() const
{
	mat3 temp = m_inertiaTensorInvWorld;
	Math::Matrix3::setInverse(temp);
	
	return temp;
}

mat3 BoxRigidBody::getInertiaInvWorld() const
{
	return m_inertiaTensorInvWorld;
}

void BoxRigidBody::applyPointImpulse( const vec3& imp, const vec3& p )
{
	// PSEUDOCODE IS m_linearVelocity += m_massInv * imp;
	// PSEUDOCODE IS m_angularVelocity += getWorldInertiaInv() * (p - centerOfMassWorld).cross(imp);
	m_linearVelocity += m_massInv * imp;
	m_angularVelocity += m_inertiaTensorInvWorld * vec3::crossProduct(p - m_centerOfMass, imp);
	//m_angularVelocity +=
	//getInertiaWorld() * vec3::crossProduct(p - m_centerOfMass, imp)
}

void BoxRigidBody::applyAngularImpulse( const vec3& imp )
{
	// PSEUDOCODE IS m_angularVelocity += m_worldInertiaInv * imp;
}

void BoxRigidBody::applyForce( const float deltaTime, const vec3& force )
{
	applyLinearImpulse(force * deltaTime);
}

void BoxRigidBody::applyForce( const float deltaTime, const vec3& force, const vec3& p )
{
	applyPointImpulse(force * deltaTime, p);
}

void BoxRigidBody::applyTorque( const float deltaTime, const vec3& torque )
{
	applyAngularImpulse(torque * deltaTime);
}

void BoxRigidBody::update( const float dt )
{
	// update the linear properties in the parent
	RigidBody::update(dt);

	// update the angular properties

	//qDebug() << "Position:" << m_position << "Linear Velocity:" << m_linearVelocity;
}






