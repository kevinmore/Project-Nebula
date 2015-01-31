#include "RigidBody.h"
#include <Physicis/World/PhysicsWorld.h>
#include <Physicis/World/PhysicsWorldObject.inl>


/**
 * Internal function to do an intertia tensor transform by a quaternion.
 * Note that the implementation of this function was created by an
 * automated code-generator and optimizer.
 */
static inline void _transformInertiaTensor(mat3 &iitWorld,
                                           const quart &q,
                                           const mat3 &iitBody,
                                           const mat4 &rotmat)
{
    float t4 = rotmat(0, 0)*iitBody.m[0][0]+
        rotmat(0, 1)*iitBody.m[1][0]+
        rotmat(0, 2)*iitBody.m[2][0];
    float t9 = rotmat(0, 0)*iitBody.m[0][1]+
        rotmat(0, 1)*iitBody.m[1][1]+
        rotmat(0, 2)*iitBody.m[2][1];
    float t14 = rotmat(0, 0)*iitBody.m[0][2]+
        rotmat(0, 1)*iitBody.m[1][2]+
        rotmat(0, 2)*iitBody.m[2][2];
    float t28 = rotmat(1, 0)*iitBody.m[0][0]+
        rotmat(1, 1)*iitBody.m[1][0]+
        rotmat(1, 2)*iitBody.m[2][0];
    float t33 = rotmat(1, 0)*iitBody.m[0][1]+
        rotmat(1, 1)*iitBody.m[1][1]+
        rotmat(1, 2)*iitBody.m[2][1];
    float t38 = rotmat(1, 0)*iitBody.m[0][2]+
        rotmat(1, 1)*iitBody.m[1][2]+
        rotmat(1, 2)*iitBody.m[2][2];
    float t52 = rotmat(2, 0)*iitBody.m[0][0]+
        rotmat(2, 1)*iitBody.m[1][0]+
        rotmat(2, 2)*iitBody.m[2][0];
    float t57 = rotmat(2, 0)*iitBody.m[0][1]+
        rotmat(2, 1)*iitBody.m[1][1]+
        rotmat(2, 2)*iitBody.m[2][1];
    float t62 = rotmat(2, 0)*iitBody.m[0][2]+
        rotmat(2, 1)*iitBody.m[1][2]+
        rotmat(2, 2)*iitBody.m[2][2];

    iitWorld.m[0][0] = t4*rotmat(0, 0)+
        t9*rotmat(0, 1)+
        t14*rotmat(0, 2);
    iitWorld.m[0][1] = t4*rotmat(1, 0)+
        t9*rotmat(1, 1)+
        t14*rotmat(1, 2);
    iitWorld.m[0][2] = t4*rotmat(2, 0)+
        t9*rotmat(2, 1)+
        t14*rotmat(2, 2);
    iitWorld.m[1][0] = t28*rotmat(0, 0)+
        t33*rotmat(0, 1)+
        t38*rotmat(0, 2);
    iitWorld.m[1][1] = t28*rotmat(1, 0)+
        t33*rotmat(1, 1)+
        t38*rotmat(1, 2);
    iitWorld.m[1][2] = t28*rotmat(2, 0)+
        t33*rotmat(2, 1)+
        t38*rotmat(2, 2);
    iitWorld.m[2][0] = t52*rotmat(0, 0)+
        t57*rotmat(0, 1)+
        t62*rotmat(0, 2);
    iitWorld.m[2][1] = t52*rotmat(1, 0)+
        t57*rotmat(1, 1)+
        t62*rotmat(1, 2);
    iitWorld.m[2][2] = t52*rotmat(2, 0)+
        t57*rotmat(2, 1)+
        t62*rotmat(2, 2);
}





RigidBody::RigidBody(const vec3& position, const quart& rotation, QObject* parent)
	: PhysicsWorldObject(parent)
{
	m_position = position;
	m_rotation = rotation;
	m_transformMatrix.translate(position);
	m_transformMatrix.rotate(m_rotation);

	m_linearVelocity = Math::Vector3::ZERO;
	m_angularVelocity = Math::Vector3::ZERO;

	m_centerOfMass = Math::Vector3::ZERO;
	m_mass = 1.0f;
	m_massInv = 1.0f;
	m_forceAccum = Math::Vector3::ZERO;

	m_linearDamping = 0.0f;
	m_angularDamping = 0.05f;
	m_gravityFactor = 1.0f;
	m_friction = 0.5f;
	m_restitution = 0.4f;
	m_maxLinearVelocity = 200.0f;
	m_maxAngularVelocity = 200.0f;
	m_timeFactor = 1.0f;

	m_eularAngles = Math::Vector3::ZERO;
	m_objectRadius = 1.0f;
	m_rotationMatrix.setToIdentity();
	m_inertiaTensor.setToIdentity();
	m_inertiaTensorInv.setToIdentity();
	Math::Matrix3::setInverse(m_inertiaTensorInv);
}

void RigidBody::setShape( const AbstractShape* shape )
{
	m_shape = shape;
}

const AbstractShape* RigidBody::getShape() const
{
	return m_shape;
}

void RigidBody::setMassProperties( const MassProperties& mp )
{
	m_mass = mp.m_mass;
	m_inertiaTensor = mp.m_inertiaTensor;
	m_centerOfMass = mp.m_centerOfMass;
}

void RigidBody::setPosition( const vec3& pos )
{
	m_position = pos;
	m_transformMatrix.setToIdentity();
	m_transformMatrix.translate(pos);
}

void RigidBody::setRotation( const quart& rot )
{
	m_rotation = rot;
	m_transformMatrix.setToIdentity();
	m_transformMatrix.rotate(rot);
}


void RigidBody::setMass( float m )
{
	float massInv;
	if (m == 0.0f)
		massInv = 0.0f;
	else
		massInv = 1.0f / m;
	setMassInv(massInv);
}

void RigidBody::setMassInv( float mInv )
{
	m_mass = mInv;
}

// Explicit center of mass in local space.
void RigidBody::setCenterOfMassLocal(const vec3& centerOfMass)
{	
	m_centerOfMass = centerOfMass;
}

void RigidBody::setPositionAndRotation( const vec3& position, const quart& rotation )
{
	m_position = position;
	m_rotation = rotation;
}

void RigidBody::setRestitution( float newRestitution )
{
	m_restitution = newRestitution;
}

void RigidBody::setFriction( float newFriction )
{
	m_friction = newFriction;
}

void RigidBody::update( const float dt )
{
	// only update the linear properties as an abstract rigid body
	m_linearVelocity += m_gravityFactor * getWorld()->getConfig().m_gravity * dt;
	m_deltaPosition = m_linearVelocity * dt;
	m_position += m_deltaPosition;
	m_transformMatrix.translate(m_position);
}
