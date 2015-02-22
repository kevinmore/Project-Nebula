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