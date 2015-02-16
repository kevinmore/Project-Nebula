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
	Matrix3::setBoxInertiaTensor(m_inertiaTensor, m_halfExtents, m_mass);
	m_inertiaTensorInv = Matrix3::inversed(m_inertiaTensor);
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

	Matrix3::setBoxInertiaTensor(m_inertiaTensor, m_halfExtents, m_mass);
	m_inertiaTensorInv = Matrix3::inversed(m_inertiaTensor);
}


void BoxRigidBody::setBoxHalfExtents( const vec3& halfExtents )
{
	// re compute the inertia tensor
	m_halfExtents = halfExtents;
	Matrix3::setBoxInertiaTensor(m_inertiaTensor, m_halfExtents, m_mass);
	m_inertiaTensorInv = Matrix3::inversed(m_inertiaTensor);
}

mat3 BoxRigidBody::getInertiaLocal() const
{
	return m_inertiaTensor;
}

void BoxRigidBody::setInertiaLocal( const mat3& inertia )
{
	m_inertiaTensor = inertia;

	m_inertiaTensorInv = Matrix3::inversed(m_inertiaTensor);
}

mat3 BoxRigidBody::getInertiaInvLocal() const
{
	return m_inertiaTensorInv;
}

void BoxRigidBody::setInertiaInvLocal( const mat3& inertiaInv )
{
	m_inertiaTensorInv = inertiaInv;
	m_inertiaTensor = Matrix3::inversed(m_inertiaTensorInv);
}

mat3 BoxRigidBody::getInertiaWorld() const
{
	return Matrix3::inversed(m_inertiaTensorInvWorld);
}

mat3 BoxRigidBody::getInertiaInvWorld() const
{
	return m_inertiaTensorInvWorld;
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
	vec3 dw = Vector3::setMul(imp, m_inertiaTensorInvWorld);
	
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

void BoxRigidBody::update( const float dt )
{
	// if the body is sleeping, skip
	if(m_bSleep) return;

	// update the linear properties in the parent first
	RigidBody::update(dt);

	// update the angular properties
	vec3 angularVelocityInDegrees(qRadiansToDegrees(m_angularVelocity.x()), qRadiansToDegrees(m_angularVelocity.y()), qRadiansToDegrees(m_angularVelocity.z()));
	m_transform.setRotation(m_transform.getEulerAngles() + angularVelocityInDegrees * dt);

	mat3 rotationMatrix = m_transform.getRotationMatrix();
	m_inertiaTensorInvWorld =  rotationMatrix * m_inertiaTensorInv * rotationMatrix.transposed();

	// check the status to decide sleep
	if(m_linearVelocity.lengthSquared() < 3e-4 && m_angularVelocity.lengthSquared() < 3e-4)
		m_bSleep = true;
}

