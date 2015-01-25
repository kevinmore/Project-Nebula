#include "BoxRigidBody.h"
#include <Utility/Math.h>


BoxRigidBody::BoxRigidBody( const vec3& position, const quart& rotation )
	: RigidBody(position, rotation)
{
	m_MotionType = RigidBody::MOTION_BOX_INERTIA;
	vec3 halfSize(0.5, 0.5, 0.5);
	m_shape = new BoxShape(m_centerOfMass, halfSize);

	Math::Matrix3::setBlockInertiaTensor(m_inertiaTensor, halfSize, m_mass);
	m_inertiaTensorInv = m_inertiaTensor;
	Math::Matrix3::setInverse(m_inertiaTensorInv);
}


void BoxRigidBody::setMass( float m )
{
	float massInv;
	if (m == 0.0f)
		massInv = 0.0f;
	else
		massInv = 1.0f / m;
	setMassInv(massInv);
}

void BoxRigidBody::setMassInv( float mInv )
{
	m_mass = mInv;

	BoxShape* box = (BoxShape*)m_shape;
	Math::Matrix3::setBlockInertiaTensor(m_inertiaTensor, box->getHalfExtents(), m_mass);
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
	//m_angularVelocity += m_inertiaTensorInvWorld * 
	//m_angularVelocity +=
	//getInertiaWorld() * vec3::crossProduct(p - m_centerOfMass, imp)
	vec3 cross = vec3::crossProduct(p - m_centerOfMass, imp);
	vec3 dv(m_inertiaTensorInv.m[0][0] * cross.x(),
		    m_inertiaTensorInv.m[1][1] * cross.y(),
		    m_inertiaTensorInv.m[2][2] * cross.z());

	m_angularVelocity += dv;
}

void BoxRigidBody::applyAngularImpulse( const vec3& imp )
{
	// PSEUDOCODE IS m_angularVelocity += m_worldInertiaInv * imp;
	vec3 dv(m_inertiaTensorInv.m[0][0] * imp.x(),
			m_inertiaTensorInv.m[1][1] * imp.y(),
			m_inertiaTensorInv.m[2][2] * imp.z());
	if (m_angularVelocity.lengthSquared() > m_maxAngularVelocity)
	{
		return;
	}
	m_angularVelocity += dv;
	qDebug() << m_angularVelocity;
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
	
	m_deltaAngle = m_angularVelocity * dt;
	//qDebug() << m_angularVelocity;
	//qDebug() << "Position:" << m_position << "Linear Velocity:" << m_linearVelocity;
}






