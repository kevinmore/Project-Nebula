#include "SphereRigidBody.h"
#include <Utility/Math.h>
#include <Physicis/World/PhysicsWorld.h>
#include <Physicis/World/PhysicsWorldObject.inl>
#include <Physicis/Geometry/SphereShape.h>

SphereRigidBody::SphereRigidBody(const vec3& position, const quart& rotation)
	: RigidBody(position, rotation)
{
	m_MotionType = RigidBody::MOTION_SPHERE_INERTIA;
	float radius = 0.5f;
	m_shape = new SphereShape(m_centerOfMass, radius);

	Math::Matrix3::setSphereInertiaTensor(m_inertiaTensor, radius, m_mass);
	m_inertiaTensorInv = m_inertiaTensor;
	Math::Matrix3::setInverse(m_inertiaTensorInv);
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

	SphereShape* sphere = (SphereShape*)m_shape;

	Math::Matrix3::setSphereInertiaTensor(m_inertiaTensor, sphere->getRadius(), m_mass);
	m_inertiaTensorInv = m_inertiaTensor;
	Math::Matrix3::setInverse(m_inertiaTensorInv);
}


void SphereRigidBody::setSphereRadius( float newRadius )
{
	SphereShape* sphere = (SphereShape*)m_shape;
	sphere->setRadius(newRadius);

	Math::Matrix3::setSphereInertiaTensor(m_inertiaTensor, newRadius, m_mass);
	m_inertiaTensorInv = m_inertiaTensor;
	Math::Matrix3::setInverse(m_inertiaTensorInv);
}

mat3 SphereRigidBody::getInertiaLocal() const
{
	return m_inertiaTensor;
}

void SphereRigidBody::setInertiaLocal( const mat3& inertia )
{
	m_inertiaTensor = inertia;

	m_inertiaTensorInv = inertia;
	Math::Matrix3::setInverse(m_inertiaTensorInv);
}

mat3 SphereRigidBody::getInertiaInvLocal() const
{
	return m_inertiaTensorInv;
}

void SphereRigidBody::setInertiaInvLocal( const mat3& inertiaInv )
{
	m_inertiaTensorInv = inertiaInv;
}


mat3 SphereRigidBody::getInertiaWorld() const
{
	mat3 temp = m_inertiaTensorInvWorld;
	Math::Matrix3::setInverse(temp);

	return temp;
}

mat3 SphereRigidBody::getInertiaInvWorld() const
{
	return m_inertiaTensorInvWorld;
}


void SphereRigidBody::applyPointImpulse( const vec3& imp, const vec3& p )
{
	// PSEUDOCODE IS m_linearVelocity += m_massInv * imp;
	// PSEUDOCODE IS m_angularVelocity += getWorldInertiaInv() * (p - centerOfMassWorld).cross(imp);
	m_linearVelocity += m_massInv * imp;
	vec3 angularMoment = vec3::crossProduct(p - m_centerOfMass, imp);

	vec3 dv = Math::Vector3::setMul(angularMoment, m_inertiaTensorInvWorld);

	m_angularVelocity += dv;
}

void SphereRigidBody::applyAngularImpulse( const vec3& imp )
{
	// PSEUDOCODE IS m_angularVelocity += m_worldInertiaInv * imp;
	vec3 dv = Math::Vector3::setMul(imp, m_inertiaTensorInvWorld);

	m_angularVelocity += dv;
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
	m_transformMatrix.setToIdentity();

	// update the angular properties
	m_eularAngles += m_angularVelocity * dt;

	quart oldRotation = m_rotation;
	m_rotation = quart::fromAxisAndAngle(Math::Vector3::UNIT_X, m_eularAngles.x())
		* quart::fromAxisAndAngle(Math::Vector3::UNIT_Y, m_eularAngles.y())
		* quart::fromAxisAndAngle(Math::Vector3::UNIT_Z, m_eularAngles.z());

	m_transformMatrix.rotate(m_rotation);

	mat3 rotX, rotY, rotZ;
	rotX.m[1][1] =  qCos(m_eularAngles.x());
	rotX.m[1][2] = -qSin(m_eularAngles.x());
	rotX.m[2][1] =  qSin(m_eularAngles.x());
	rotX.m[2][2] =  qCos(m_eularAngles.x());

	rotY.m[0][0] =  qCos(m_eularAngles.y());
	rotY.m[0][2] =  qSin(m_eularAngles.y());
	rotY.m[2][0] = -qSin(m_eularAngles.y());
	rotY.m[2][2] =  qCos(m_eularAngles.y());

	rotZ.m[0][0] =  qCos(m_eularAngles.z());
	rotZ.m[0][1] = -qSin(m_eularAngles.z());
	rotZ.m[1][0] =  qSin(m_eularAngles.z());
	rotZ.m[1][1] =  qCos(m_eularAngles.z());

	m_rotationMatrix = rotX * rotY * rotZ;
	m_inertiaTensorInvWorld = m_inertiaTensorInv * m_rotationMatrix;

	// update the linear properties in the parent
	RigidBody::update(dt);
}