#include "RigidBody.h"
#include <Physicis/World/PhysicsWorld.h>
#include <Physicis/World/PhysicsWorldObject.inl>
#include <Physicis/Collision/Collider/SphereCollider.h>
#include <Physicis/Collision/Collider/BoxCollider.h>
#include <Primitives/GameObject.h>

RigidBody::RigidBody(const vec3& position, const quat& rotation, QObject* parent)
	: PhysicsWorldObject(parent)
{

	m_motionType = RigidBody::MOTION_BOX_INERTIA;

	m_linearVelocity = Math::Vector3::ZERO;
	m_angularVelocity = Math::Vector3::ZERO;

	m_centerOfMass = Math::Vector3::ZERO;
	m_mass = 1.0f;
	m_massInv = 1.0f;

	m_linearDamping = 0.0f;
	m_angularDamping = 0.05f;
	m_gravityFactor = 1.0f;
	m_friction = 0.5f;
	m_restitution = 0.4f;
	m_maxLinearVelocity = 200.0f;
	m_maxAngularVelocity = 200.0f;
	m_timeFactor = 1.0f;
	m_canSleep = false;
	m_isAwake = true;
}

RigidBody::~RigidBody()
{
	 // remove it from the world
	if (m_world)
	{
		m_world->removeEntity(this);
	}
}

void RigidBody::computeInertiaTensor()
{
	// prepare variables
	SphereColliderPtr sphere = m_BroadPhaseCollider.dynamicCast<SphereCollider>();
	BoxColliderPtr box = m_BroadPhaseCollider.dynamicCast<BoxCollider>();

	switch(m_motionType)
	{
	case MOTION_BOX_INERTIA:
		if (box)
		{
			Matrix3::setBoxInertiaTensor(m_inertiaTensorInv, box->getHalfExtents(), m_mass);
			m_inertiaTensorInv = glm::inverse(m_inertiaTensorInv);
		}
		break;

	case MOTION_SPHERE_INERTIA:
		if (sphere)
		{
			Matrix3::setSphereInertiaTensor(m_inertiaTensorInv, sphere->getRadius(), m_mass);
			m_inertiaTensorInv = glm::inverse(m_inertiaTensorInv);
		}
		break;

	case MOTION_FIXED:
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				m_inertiaTensorInv[i][j] = 0;
		break;
	}
}

void RigidBody::update( const float dt )
{
	if (!m_isAwake) return;

	m_timeStep = dt;

	// update the linear properties
	m_lastLinerVelocity = m_linearVelocity;
	m_linearVelocity += m_gravityFactor * getWorld()->getConfig().m_gravity * dt;
	m_deltaPosition = m_linearVelocity * dt;
	m_transform.translate(m_deltaPosition);

	// update the angular properties
	m_lastAngularVelocity = m_angularVelocity;
	m_deltaRotation = quat::fromAxisAndAngle(m_angularVelocity, qRadiansToDegrees(m_angularVelocity.length() * dt));
	m_transform.rotate(m_deltaRotation);

	// sync the center position for the collider
	m_BroadPhaseCollider->setCenter(m_transform.getPosition());
	m_NarrowPhaseCollider->setCenter(m_transform.getPosition());

	// Update the kinetic energy store, and possibly put the body to sleep.
	if (m_canSleep) 
	{
		float currentMotion = m_linearVelocity.lengthSquared() + m_angularVelocity.lengthSquared();

		float bias = pow(0.5f, dt);
		m_motionEnergy = bias * m_motionEnergy + (1 - bias) * currentMotion;

		if (m_motionEnergy < 0.3f) setAwake(false);
		else if (m_motionEnergy > 10 * 0.3f) m_motionEnergy = 10 * 0.3f;
	}
}

void RigidBody::applyPointImpulse( const vec3& imp, const vec3& p )
{
	// PSEUDOCODE IS m_linearVelocity += m_massInv * imp;
	// PSEUDOCODE IS m_angularVelocity += getWorldInertiaInv() * (p - centerOfMassWorld).cross(imp);
	applyLinearImpulse(imp);
	applyAngularImpulse(vec3::crossProduct(p - getCenterOfMassInWorld(), imp));
}

void RigidBody::applyAngularImpulse( const vec3& imp )
{
	// PSEUDOCODE IS m_angularVelocity += m_worldInertiaInv * imp;
	vec3 dw = Converter::toQtVec3(getInertiaInvWorld() * Converter::toGLMVec3(imp));
	m_angularVelocity += dw;
}

void RigidBody::applyForce( const float deltaTime, const vec3& force )
{
	applyLinearImpulse(force * deltaTime);
}

void RigidBody::applyForce( const float deltaTime, const vec3& force, const vec3& p )
{
	applyPointImpulse(force * deltaTime, p);
}

void RigidBody::applyTorque( const float deltaTime, const vec3& torque )
{
	applyAngularImpulse(torque * deltaTime);
}

void RigidBody::setMassProperties( const MassProperties& mp )
{
	m_mass = mp.m_mass;
	m_centerOfMass = mp.m_centerOfMass;
}

void RigidBody::setMass( float m )
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

void RigidBody::setMotionType( MotionType type )
{
	m_motionType = type;
	computeInertiaTensor();
}

void RigidBody::setMassInv( float mInv )
{
	m_massInv = mInv;
	m_mass = 1.0f / mInv;

	computeInertiaTensor();
}


void RigidBody::setPosition( const vec3& pos )
{
	m_transform.setPosition(pos);
}

void RigidBody::setRotation( const quat& rot )
{
	m_transform.setRotation(rot);
}

// Explicit center of mass in local space.
void RigidBody::setCenterOfMassLocal(const vec3& centerOfMass)
{	
	m_centerOfMass = centerOfMass;
}

void RigidBody::setPositionAndRotation( const vec3& position, const quat& rotation )
{
	m_transform.setPosition(position);
	m_transform.setRotation(rotation);
}

void RigidBody::setRestitution( float newRestitution )
{
	m_restitution = newRestitution;
}

void RigidBody::setFriction( float newFriction )
{
	m_friction = newFriction;
}

void RigidBody::attachBroadPhaseCollider( ColliderPtr col )
{
	m_BroadPhaseCollider = col;
	col->setRigidBody(this);
	computeInertiaTensor();
}

void RigidBody::attachNarrowPhaseCollider( ColliderPtr col )
{
	m_NarrowPhaseCollider = col;
	col->setRigidBody(this);
}

void RigidBody::syncTransform( const Transform& transform )
{
	// only sync the transform when the physics world is locked
	// this means that the engine is in editor mode
	if (m_world->isLocked())
	{
		m_transform = transform;
	}
}

void RigidBody::setAwake( const bool awake /*= true*/ )
{
	if (awake) 
	{
		m_isAwake= true;

		// Add a bit of motion to avoid it falling asleep immediately.
		m_motionEnergy = 0.3f * 2.0f;
	} 
	else 
	{
		m_isAwake = false;
		m_linearVelocity = Vector3::ZERO;
		m_angularVelocity = Vector3::ZERO;
	}
}

void RigidBody::setCanSleep( const bool canSleep /*= true*/ )
{
	m_canSleep = canSleep;
	if (!m_canSleep && !m_isAwake) setAwake();
}

void RigidBody::backTrack( const float duration )
{
	backTrackPosition(duration);
	backTrackRotation(duration);
	backTrackLinearVelocity(duration);
	backTrackAngularVelocity(duration);
}

void RigidBody::backTrackPosition( const float duration )
{
	vec3 amount = -m_deltaPosition * duration / m_timeStep;
	m_transform.translate(amount);

	// sync the center position for the collider
	m_BroadPhaseCollider->setCenter(m_transform.getPosition());
	m_NarrowPhaseCollider->setCenter(m_transform.getPosition());
}

void RigidBody::backTrackRotation( const float duration )
{
	quat curRotation = m_transform.getRotation();
	quat lastRotation = m_deltaRotation.conjugate() * curRotation;
	quat amount = quat::slerp(curRotation, lastRotation, duration / m_timeStep);
	m_transform.rotate(amount);
}

void RigidBody::backTrackLinearVelocity( const float duration )
{
	m_linearVelocity = Spline::lerp(m_linearVelocity, m_lastLinerVelocity, duration / m_timeStep);
}

void RigidBody::backTrackAngularVelocity( const float duration )
{
	m_angularVelocity = Spline::lerp(m_angularVelocity, m_lastAngularVelocity, duration / m_timeStep);
}


/************************************************************************/
/*          SLOTS                                                       */
/************************************************************************/
void RigidBody::setMotionType_SLOT( const QString& type )
{
	if (type == "Box")
		setMotionType(MOTION_BOX_INERTIA);
	else if (type == "Sphere")
		setMotionType(MOTION_SPHERE_INERTIA);
	else if (type == "Fixed")
		setMotionType(MOTION_FIXED);
}

void RigidBody::setMass_SLOT( double val )
{
	setMass(val);
}

void RigidBody::setGravityFactor_SLOT( double val )
{
	m_gravityFactor = val;
}

void RigidBody::setRestitution_SLOT( double val )
{
	m_restitution = val;
}

void RigidBody::setLinearVelocityX_SLOT( double val )
{
	m_linearVelocity.setX(val);
}

void RigidBody::setLinearVelocityY_SLOT( double val )
{
	m_linearVelocity.setY(val);
}

void RigidBody::setLinearVelocityZ_SLOT( double val )
{
	m_linearVelocity.setZ(val);
}

void RigidBody::setAngularVelocityX_SLOT( double val )
{
	m_angularVelocity.setX(val);
}

void RigidBody::setAngularVelocityY_SLOT( double val )
{
	m_angularVelocity.setY(val);
}

void RigidBody::setAngularVelocityZ_SLOT( double val )
{
	m_angularVelocity.setZ(val);
}

void RigidBody::setRadius_SLOT( double val )
{
	SphereColliderPtr sphere = m_BroadPhaseCollider.dynamicCast<SphereCollider>();
	if (sphere)
	{
		sphere->setRadius(val);
		computeInertiaTensor();
	}
}

void RigidBody::setExtentsX_SLOT( double val )
{
	BoxColliderPtr box = m_BroadPhaseCollider.dynamicCast<BoxCollider>();
	if (box)
	{
		vec3 halfExtents = box->getHalfExtents();
		halfExtents.setX(0.5 * val);
		box->setHalfExtents(halfExtents);
		computeInertiaTensor();
	}
}

void RigidBody::setExtentsY_SLOT( double val )
{
	BoxColliderPtr box = m_BroadPhaseCollider.dynamicCast<BoxCollider>();
	if (box)
	{
		vec3 halfExtents = box->getHalfExtents();
		halfExtents.setY(0.5 * val);
		box->setHalfExtents(halfExtents);
		computeInertiaTensor();
	}
}

void RigidBody::setExtentsZ_SLOT( double val )
{
	BoxColliderPtr box = m_BroadPhaseCollider.dynamicCast<BoxCollider>();
	if (box)
	{
		vec3 halfExtents = box->getHalfExtents();
		halfExtents.setZ(0.5 * val);
		box->setHalfExtents(halfExtents);
		computeInertiaTensor();
	}
}
