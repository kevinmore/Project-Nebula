#include "SphereRigidBody.h"
#include <Physicis/World/PhysicsWorld.h>
#include <Physicis/World/PhysicsWorldObject.inl>
using namespace Math;

SphereRigidBody::SphereRigidBody(const vec3& position, const quat& rotation)
	: RigidBody(position, rotation),
	  m_radius(0.5f)
{
	m_MotionType = RigidBody::MOTION_SPHERE_INERTIA;

	// fill the tensor with the default size
	mat3 tensor;
	Matrix3::setSphereInertiaTensor(tensor, m_radius, m_mass);
	m_inertiaTensorInv = glm::inverse(Converter::toGLMMat3(tensor));
}


void SphereRigidBody::setMass( float m )
{
	float massInv;
	if (m == 0.0f)
		massInv = 0.0f;
	else
	{
		m_mass = m;
		massInv = 1.0f / m;
	}
	setMassInv(massInv);
}

void SphereRigidBody::setMassInv( float mInv )
{
	m_massInv = mInv;
	m_mass = 1.0f / mInv;

	mat3 tensor;
	Matrix3::setSphereInertiaTensor(tensor, m_radius, m_mass);
	m_inertiaTensorInv = glm::inverse(Converter::toGLMMat3(tensor));
}


void SphereRigidBody::setSphereRadius( float newRadius )
{
	// re compute the inertia tensor
	m_radius = newRadius;
	mat3 tensor;
	Matrix3::setSphereInertiaTensor(tensor, m_radius, m_mass);
	m_inertiaTensorInv = glm::inverse(Converter::toGLMMat3(tensor));
}

void SphereRigidBody::applyPointImpulse( const vec3& imp, const vec3& p )
{
	// PSEUDOCODE IS m_linearVelocity += m_massInv * imp;
	// PSEUDOCODE IS m_angularVelocity += getWorldInertiaInv() * (p - centerOfMassWorld).cross(imp);
	applyLinearImpulse(imp);
	applyAngularImpulse(vec3::crossProduct(p - getCenterOfMassInWorld(), imp));
}

void SphereRigidBody::applyAngularImpulse( const vec3& imp )
{
	// PSEUDOCODE IS m_angularVelocity += m_worldInertiaInv * imp;
	vec3 dw = Converter::toQtVec3(getInertiaInvWorld() * Converter::toGLMVec3(imp));
	m_angularVelocity += dw;
}

void SphereRigidBody::applyForce( const float deltaTime, const vec3& force )
{
	applyLinearImpulse(force * deltaTime);
}

void SphereRigidBody::applyForce( const float deltaTime, const vec3& force, const vec3& p )
{
	applyPointImpulse(force * deltaTime, p);
}

void SphereRigidBody::applyTorque( const float deltaTime, const vec3& torque )
{
	applyAngularImpulse(torque * deltaTime);
}