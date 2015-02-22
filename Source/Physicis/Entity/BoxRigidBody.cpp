#include "BoxRigidBody.h"
#include <Physicis/World/PhysicsWorld.h>
#include <Physicis/World/PhysicsWorldObject.inl>
using namespace Math;

BoxRigidBody::BoxRigidBody( const vec3& position, const quat& rotation )
	: RigidBody(position, rotation),
	  m_halfExtents(vec3(0.5f, 0.5f, 0.5f))
{
	m_MotionType = RigidBody::MOTION_BOX_INERTIA;

	// fill the tensor with the default size
	mat3 tensor;
	Matrix3::setBoxInertiaTensor(tensor, m_halfExtents, m_mass);
	m_inertiaTensorInv = glm::inverse(Converter::toGLMMat3(tensor));
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
	m_mass = 1.0f / mInv;

	mat3 tensor;
	Matrix3::setBoxInertiaTensor(tensor, m_halfExtents, m_mass);
	m_inertiaTensorInv = glm::inverse(Converter::toGLMMat3(tensor));
}

void BoxRigidBody::setBoxHalfExtents( const vec3& halfExtents )
{
	// re compute the inertia tensor
	m_halfExtents = halfExtents;

	mat3 tensor;
	Matrix3::setBoxInertiaTensor(tensor, m_halfExtents, m_mass);
	m_inertiaTensorInv = glm::inverse(Converter::toGLMMat3(tensor));
}

void BoxRigidBody::applyPointImpulse( const vec3& imp, const vec3& p )
{
	// PSEUDOCODE IS m_linearVelocity += m_massInv * imp;
	// PSEUDOCODE IS m_angularVelocity += getWorldInertiaInv() * (p - centerOfMassWorld).cross(imp);

	applyLinearImpulse(imp);
	applyAngularImpulse(vec3::crossProduct(p - getCenterOfMassInWorld(), imp));
}

void BoxRigidBody::applyAngularImpulse( const vec3& imp )
{
	// PSEUDOCODE IS m_angularVelocity += m_worldInertiaInv * imp;
	vec3 dw = Converter::toQtVec3(getInertiaInvWorld() * Converter::toGLMVec3(imp));
	m_angularVelocity += dw;
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

