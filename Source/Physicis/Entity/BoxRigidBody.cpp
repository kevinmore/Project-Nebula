#include "BoxRigidBody.h"
#include <Utility/Math.h>
#include <Physicis/World/PhysicsWorld.h>
#include <Physicis/World/PhysicsWorldObject.inl>
#include <Physicis/Geometry/BoxShape.h>

BoxRigidBody::BoxRigidBody( const vec3& position, const quart& rotation )
	: RigidBody(position, rotation)
{
	m_MotionType = RigidBody::MOTION_BOX_INERTIA;
	vec3 halfSize(0.5, 0.5, 0.5);
	m_shape = new BoxShape(m_centerOfMass, halfSize);

	Math::Matrix3::setBoxInertiaTensor(m_inertiaTensor, halfSize, m_mass);
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

	BoxShape* box = (BoxShape*)m_shape;

	Math::Matrix3::setBoxInertiaTensor(m_inertiaTensor, box->getHalfExtents(), m_mass);
	m_inertiaTensorInv = m_inertiaTensor;
	Math::Matrix3::setInverse(m_inertiaTensorInv);
}


void BoxRigidBody::setBoxHalfExtents( const vec3& halfExtents )
{
	// resize the shape
	BoxShape* box = (BoxShape*)m_shape;
	box->setHalfExtents(halfExtents);

	// re compute the inertia tensor
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

	vec3 dv = Math::Vector3::setMul(angularMoment, m_inertiaTensorInvWorld);

	m_angularVelocity += dv;
}

void BoxRigidBody::applyAngularImpulse( const vec3& imp )
{
	// PSEUDOCODE IS m_angularVelocity += m_worldInertiaInv * imp;
	vec3 dv = Math::Vector3::setMul(imp, m_inertiaTensorInvWorld);

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
	m_transformMatrix.setToIdentity();

	// update the linear properties in the parent first
	RigidBody::update(dt);

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
}

