#include "BoxRigidBody.h"
#include <Physicis/World/PhysicsWorld.h>
#include <Physicis/World/PhysicsWorldObject.inl>

BoxRigidBody::BoxRigidBody( const vec3& position, const quat& rotation )
	: RigidBody(position, rotation),
	  m_halfExtents(vec3(0.5f, 0.5f, 0.5f))
{
	m_MotionType = RigidBody::MOTION_BOX_INERTIA;

	// fill the tensor with the default size
	Math::Matrix3::setBoxInertiaTensor(m_inertiaTensor, m_halfExtents, m_mass);
	m_inertiaTensorInv = m_inertiaTensor;
	Math::Matrix3::setInverse(m_inertiaTensorInv);
}

void BoxRigidBody::setMass( float m )
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

void BoxRigidBody::setMassInv( float mInv )
{
	m_massInv = mInv;

	Math::Matrix3::setBoxInertiaTensor(m_inertiaTensor, m_halfExtents, m_mass);
	m_inertiaTensorInv = m_inertiaTensor;
	Math::Matrix3::setInverse(m_inertiaTensorInv);
}


void BoxRigidBody::setBoxHalfExtents( const vec3& halfExtents )
{
	// re compute the inertia tensor
	m_halfExtents = halfExtents;
	Math::Matrix3::setBoxInertiaTensor(m_inertiaTensor, halfExtents, m_mass);
	m_inertiaTensorInv = m_inertiaTensor;
	Math::Matrix3::setInverse(m_inertiaTensorInv);
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
	vec3 angularMoment = vec3::crossProduct(p - m_centerOfMass, imp);
	vec3 dv =  Vector3::setMul(angularMoment, m_inertiaTensorInvWorld);
	
	m_angularVelocity += dv;
}

void BoxRigidBody::applyAngularImpulse( const vec3& imp )
{
	// PSEUDOCODE IS m_angularVelocity += m_worldInertiaInv * imp;
	vec3 dv = Vector3::setMul(imp, m_inertiaTensorInvWorld);
	
	m_angularVelocity += dv;
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
	// update the linear properties in the parent first
	RigidBody::update(dt);

	// update the angular properties
	m_transform.setRotation(m_transform.getEulerAngles() + m_angularVelocity * dt);

	mat3 rotationMatrix = m_transform.getRotationMatrix();
	m_inertiaTensorInvWorld =  rotationMatrix * m_inertiaTensorInv * rotationMatrix.transposed();
}

