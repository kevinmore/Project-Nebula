#include "SphereRigidBody.h"
#include <Physicis/World/PhysicsWorld.h>
#include <Physicis/World/PhysicsWorldObject.inl>
using namespace Math;

SphereRigidBody::SphereRigidBody(const vec3& position, const quat& rotation)
	: RigidBody(position, rotation),
	  m_radius(0.5f)
{
	m_MotionType = RigidBody::MOTION_SPHERE_INERTIA;

	Matrix3::setSphereInertiaTensor(m_inertiaTensor, m_radius, m_mass);
	m_inertiaTensorInv = Matrix3::inversed(m_inertiaTensor);
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

	Matrix3::setSphereInertiaTensor(m_inertiaTensor, m_radius, m_mass);
	m_inertiaTensorInv = Matrix3::inversed(m_inertiaTensor);
}


void SphereRigidBody::setSphereRadius( float newRadius )
{
	// re compute the inertia tensor
	m_radius = newRadius;
	Matrix3::setSphereInertiaTensor(m_inertiaTensor, m_radius, m_mass);
	m_inertiaTensorInv = Matrix3::inversed(m_inertiaTensor);
}

mat3 SphereRigidBody::getInertiaLocal() const
{
	return m_inertiaTensor;
}

void SphereRigidBody::setInertiaLocal( const mat3& inertia )
{
	m_inertiaTensor = inertia;

	m_inertiaTensorInv = Matrix3::inversed(m_inertiaTensor);
}

mat3 SphereRigidBody::getInertiaInvLocal() const
{
	return m_inertiaTensorInv;
}

void SphereRigidBody::setInertiaInvLocal( const mat3& inertiaInv )
{
	m_inertiaTensorInv = inertiaInv;
	m_inertiaTensor = Matrix3::inversed(m_inertiaTensorInv);
}


mat3 SphereRigidBody::getInertiaWorld() const
{
	return Matrix3::inversed(m_inertiaTensorInvWorld);
}

mat3 SphereRigidBody::getInertiaInvWorld() const
{
	return m_inertiaTensorInvWorld;
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
	vec3 dw = Vector3::setMul(imp, m_inertiaTensorInvWorld);
	
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

void SphereRigidBody::update( const float dt )
{
	// update the linear properties in the parent first
	RigidBody::update(dt);

	// update the angular properties
	vec3 angularVelocityInDegrees(qRadiansToDegrees(m_angularVelocity.x()), qRadiansToDegrees(m_angularVelocity.y()), qRadiansToDegrees(m_angularVelocity.z()));
	m_transform.setRotation(m_transform.getEulerAngles() + angularVelocityInDegrees * dt);

	mat3 rotationMatrix = m_transform.getRotationMatrix();
	m_inertiaTensorInvWorld =  rotationMatrix * m_inertiaTensorInv * rotationMatrix.transposed();
}