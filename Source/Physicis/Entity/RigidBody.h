#pragma once
#include <Physicis/World/PhysicsWorldObject.h>
#include <Physicis/Geometry/IShape.h>
#include <Utility/Math.h>
#include <Primitives/Transform.h>

struct MassProperties
{
	/// Initialize (to zero data).
	MassProperties() 
		: m_volume(0),
		m_mass(0),
		m_centerOfMass(vec3(0, 0, 0)) 
	{  m_inertiaTensor.setToIdentity(); }

	void scaleToDensity(float density)
	{
		scaleToMass(m_volume * density);
	}
	void scaleToMass(float newMass)
	{
		if(newMass < 0) return;
		float k = newMass / m_mass;

		m_mass = newMass;
		m_inertiaTensor *= k;
	}

	/// The volume of an object.
	float m_volume;

	/// The mass of an object.
	float m_mass;

	/// The center of mass.
	vec3 m_centerOfMass;

	/// The inertia tensor.
	mat3 m_inertiaTensor;
};

class ICollider;
typedef QSharedPointer<ICollider> ColliderPtr;

class RigidBody : public PhysicsWorldObject
{
	Q_OBJECT

public:

	enum MotionType
	{
		MOTION_SPHERE_INERTIA,
		MOTION_BOX_INERTIA,
		MOTION_FIXED,
		MOTION_MAX_ID
	};

	/// Construct a default rigid body
	RigidBody(const vec3& position = Math::Vector3::ZERO, 
		      const quat& rotation = Math::Quaternion::IDENTITY, 
			  QObject* parent = 0);

	/// Inherited from Component
	virtual QString className() { return "RigidBody"; }

	virtual ~RigidBody();
	void update(const float dt);

public slots:
	void setMotionType_SLOT(const QString& type);
	void setMass_SLOT(double val);
	void setGravityFactor_SLOT(double val);
	void setRestitution_SLOT(double val) ;
	void setLinearVelocityX_SLOT(double val);
	void setLinearVelocityY_SLOT(double val);
	void setLinearVelocityZ_SLOT(double val);
	void setAngularVelocityX_SLOT(double val);
	void setAngularVelocityY_SLOT(double val);
	void setAngularVelocityZ_SLOT(double val);
	void setRadius_SLOT(double val);
	void setExtentsX_SLOT(double val);
	void setExtentsY_SLOT(double val);
	void setExtentsZ_SLOT(double val);
	void setPointImpulseX_SLOT(double val);
	void setPointImpulseY_SLOT(double val);
	void setPointImpulseZ_SLOT(double val);
	void setPointImpulsePositionX_SLOT(double val);
	void setPointImpulsePositionY_SLOT(double val);
	void setPointImpulsePositionZ_SLOT(double val);
	void setAngularImpulseX_SLOT(double val);
	void setAngularImpulseY_SLOT(double val);
	void setAngularImpulseZ_SLOT(double val);

public:
	//
	// Motion Type
	//
	inline MotionType getMotionType() const { return m_motionType; }
	void setMotionType(MotionType type);

	//
	// Collider
	//
	void attachBroadPhaseCollider(ColliderPtr col);
	inline ColliderPtr getBroadPhaseCollider() const { return m_BroadPhaseCollider; };

	void attachNarrowPhaseCollider(ColliderPtr col);
	inline ColliderPtr getNarrowPhaseCollider() const { return m_NarrowPhaseCollider; };

	//
	// MASS, INERTIA AND DENSITY PROPERTIES.
	//

	/// Sets the mass properties
	void setMassProperties(const MassProperties& mp);

	/// Gets the mass of the rigid body.
	inline float getMass() const { return m_mass; }

	/// Gets the 1.0/mass of the rigid body.
	inline float getMassInv() const { return m_massInv; }

	/// Sets the mass of the rigid body.
	virtual void setMass(float m);

	/// Sets the inverse mass of the rigid body.
	virtual void setMassInv(float mInv);

	/// Gets the inverse inertia tensor in local space.
	inline glm::mat3 getInertiaInvLocal() { return m_inertiaTensorInv; }
	
	/// Gets the inverse inertia tensor in world space.
	inline glm::mat3 getInertiaInvWorld()
	{
		glm::mat3 rotationMatrix = m_transform.getRotationMatrix();
		m_inertiaTensorInvWorld = rotationMatrix * m_inertiaTensorInv * glm::transpose(rotationMatrix);
		return m_inertiaTensorInvWorld;
	}

	//
	// CENTER OF MASS.
	//

	/// Explicitly sets the center of mass of the rigid body in local space.
	void setCenterOfMassLocal(const vec3& centerOfMass);

	/// Gets the center of mass of the rigid body in the rigid body's local space.
	inline const vec3& getCenterOfMassLocal() const { return m_centerOfMass; }

	/// Gets the center of mass of the rigid body in world space.
	inline const vec3 getCenterOfMassInWorld() const {	return m_transform.getPosition() + m_centerOfMass; }

	//
	// POSITION ACCESS.
	//

	/// Returns the transform for this rigid body, in world space.
	inline const Transform& getTransform() const { return m_transform; }

	/// Sets the transform for this rigid body, in world space.
	inline void setTransform(const Transform& trans) { m_transform = trans; }

	/// Returns the position (the local space origin) for this rigid body, in world space.
	inline const vec3& getPosition() const { return m_transform.getPosition(); }

	// Returns the position changed (the local space origin) for this rigid body, in world space.
	inline const vec3& getDeltaPosition() const { return m_deltaPosition; }

	/// Sets the position (the local space origin) of this rigid body, in world space.
	void setPosition(const vec3& position);

	/// Returns the rotation from local to world space for this rigid body.
	inline const quat& getRotation() const { return m_transform.getRotation(); }

	/// Returns the rotation changed from local to world space for this rigid body.
	inline const quat& getDeltaRotation() const { return m_deltaRotation;	}


	inline vec3 getEulerAngles() const { return m_transform.getEulerAngles();}
	inline glm::mat3 getRotationMatrix() const { return m_transform.getRotationMatrix(); }
	inline mat4 getTransformMatrix() const { return m_transform.getTransformMatrix(); }

	/// Sets the rotation from local to world Space for this rigid body.
	/// This activates the body and its simulation island if it is inactive.
	void setRotation(const quat& rotation);

	/// Sets the position and rotation of the rigid body, in world space.
	void setPositionAndRotation(const vec3& position, const quat& rotation);

	/// Move the rigid body by a certain offset
	inline void moveBy(const vec3& offset) { setPosition(getPosition() + offset); }

	//
	// VELOCITY ACCESS.
	//

	/// Returns the linear velocity of the center of mass of the rigid body, in world space.
	inline const vec3& getLinearVelocity() const { return m_linearVelocity;	}


	/// Sets the linear velocity at the center of mass, in world space.
	/// This activates the body and its simulation island if it is inactive.
	inline void	setLinearVelocity(const vec3& newVel) { m_linearVelocity = newVel; }

	/// Returns the angular velocity around the center of mass, in world space.
	inline const vec3& getAngularVelocity() const {	return m_angularVelocity; }

	/// Sets the angular velocity around the center of mass, in world space.
	/// This activates the body and its simulation island if it is inactive.
	inline void	setAngularVelocity(const vec3& newVel) { m_angularVelocity = newVel; }

	/// Gets the velocity of point p on the rigid body in local space.
	inline vec3 getPointVelocityLocal(const vec3& p) const { return m_linearVelocity + vec3::crossProduct(m_angularVelocity, p - m_centerOfMass); }

	/// Gets the velocity of point p on the rigid body in world space.
	inline vec3 getPointVelocityWorld(const vec3& p) const { return m_linearVelocity + vec3::crossProduct(m_angularVelocity, p - getCenterOfMassInWorld()); }

	//
	// IMPULSE APPLICATION.
	//

	/// Run the user input at the first step of the simulation (impulse, force and torque)
	void executeUserInput();

	/// Applies an impulse (in world space) to the center of mass.
	/// This activates the body and its simulation island if it is inactive.
	inline void applyLinearImpulse(const vec3& imp) { m_linearVelocity += m_massInv * imp; }

	/// Applies an impulse (in world space) at the point p in world space.
	/// This activates the body and its simulation island if it is inactive.
	void applyPointImpulse(const vec3& imp, const vec3& p);

	/// Applies an instantaneous change in angular velocity (in world space) around
	/// the center of mass.
	/// This activates the body and its simulation island if it is inactive.
	void applyAngularImpulse(const vec3& imp);

	//
	// FORCE AND TORQUE APPLICATION
	//

	/// Applies a force (in world space) to the rigid body. The force is applied to the
	/// center of mass.
	void applyForce(const float deltaTime, const vec3& force);

	/// Applies a force (in world space) to the rigid body at the point p in world space.
	void applyForce(const float deltaTime, const vec3& force, const vec3& p);

	/// Applies the specified torque (in world space) to the rigid body. (Note: the inline
	/// is for internal use only).
	void applyTorque(const float deltaTime, const vec3& torque);

	//
	// DAMPING
	//

	/// Naive momentum damping.
	inline float getLinearDamping() { return m_linearDamping; }

	/// Naive momentum damping.
	inline void setLinearDamping( float d ) { m_linearDamping = d; }

	/// Naive momentum damping.
	inline float getAngularDamping() { return m_angularDamping;	}

	/// Naive momentum damping.
	inline void setAngularDamping( float d ) { m_angularDamping = d; }

	/// Time factor.
	inline float getTimeFactor() { return m_timeFactor;	}

	/// Time factor.
	inline void setTimeFactor( float f ) { m_timeFactor = f; }


	//
	// Friction and Restitution
	//

	/// Returns the friction coefficient (dynamic and static) from the material.
	inline float getFriction() const { return m_friction; }

	/// Sets the friction coefficient of the material. Note: Setting this will not update existing contact information.
	void setFriction( float newFriction );

	/// Returns the default restitution from the material.
	//  restitution = bounciness (1 should give object all its energy back, 0 means it just sits there, etc.).
	inline float getRestitution() const { return m_restitution; }

	/// Sets the restitution coefficient of the material. Note: Setting this will not update existing contact information.
	void setRestitution( float newRestitution );

	//
	// GRAVITY FACTOR
	//

	/// Get the current gravity factor.
	inline float getGravityFactor() { return m_gravityFactor; }

	/// Set the gravity factor.
	inline void setGravityFactor(float gravityFactor) { m_gravityFactor = gravityFactor; }

    /// Returns true if the body is awake and responding to integration.
    inline bool getAwake() const { return m_isAwake; }


	//
	// SLEEP STATUS
	//

	/// Sets the awake state of the body. If the body is set to be
	/// not awake, then its velocities are also canceled, since
	/// a moving body that is not awake can cause problems in the
	/// simulation.
	void setAwake(const bool awake = true);

    /// Returns true if the body is allowed to go to sleep at any time.
    inline bool getCanSleep() const { return m_canSleep; }

    /// Sets whether the body is ever allowed to go to sleep. Bodies
    /// under the player's control, or for which the set of
    /// transient forces applied each frame are not predictable,
    /// should be kept awake.
    void setCanSleep(const bool canSleep = true);

	//
	// BACKTRACKING
	//

	/// Back tracks all the properties for a given time duration
	void backTrack(const float duration);

	/// Back tracks the world position for a given time duration
	void backTrackPosition(const float duration);

	/// Back tracks the world rotation for a given time duration
	void backTrackRotation(const float duration);

	/// Back tracks the linear velocity for a given time duration
	void backTrackLinearVelocity(const float duration);

	/// Back tracks the angular velocity for a given time duration
	void backTrackAngularVelocity(const float duration);

protected:

	/// overload function from component
	virtual void syncTransform(const Transform& transform);

	void computeInertiaTensor();

	//
	// Members
	//

	/// The motion type of the rigid body.
	MotionType	m_motionType;

	/// The mass of the body.
	/// This defaults to 1.
	float m_mass;

	/// The inverse mass of the body.
	/// This defaults to 1.
	float m_massInv;

	/// Transform of the body
	/// The initial position of the body defaults to (0, 0, 0)
	/// The initial rotation of the body defaults to the Identity quaternion
	Transform m_transform;

	/// The position changed of the body after each update.
	/// This defaults to 0,0,0.
	vec3 m_deltaPosition;

	/// The rotation changed of the body after each update.
	/// This defaults to the Identity quaternion.
	quat m_deltaRotation;

	/// The time duration from the last frame
	float m_timeStep;

	/// The initial linear velocity of the body.
	/// This defaults to 0,0,0.
	vec3 m_linearVelocity;

	/// The linear velocity of the body in the last frame.
	vec3 m_lastLinerVelocity;

	/// The initial angular velocity of the body in world space.
	/// This defaults to 0,0,0.
	vec3 m_angularVelocity;

	/// The angular velocity of the body in the last frame.
	vec3 m_lastAngularVelocity;

	/// The inverse of inertia tensor of the rigid body.
	glm::mat3 m_inertiaTensorInv;

	/// The inverse of inertia tensor of the rigid body in world space.
	glm::mat3 m_inertiaTensorInvWorld;

	/// The center of mass in the local space of the rigid body.
	/// This defaults to 0,0,0.
	vec3 m_centerOfMass;

	/// The initial linear damping of the body.
	/// This defaults to 0.
	float m_linearDamping;

	/// The initial angular damping of the body.
	/// This defaults to 0.05.
	float m_angularDamping;

	/// Gravity factor used to control gravity on a per body basis. Defaults to 1.0
	float m_gravityFactor;

	/// The initial friction of the body.
	/// This defaults to 0.5.
	float m_friction;

	/// The initial restitution of the body.
	/// This defaults to 0.4.
	float m_restitution;

	/// The maximum linear velocity of the body (in m/s).
	/// This defaults to 200.
	float m_maxLinearVelocity;

	/// The maximum angular velocity of the body (in rad/s).
	/// This defaults to 200.
	float m_maxAngularVelocity;

	/// The initial time factor of the body.
	/// This defaults to 1.
	float m_timeFactor;

	/// The pointer to the broad phase collider that is attached to the rigid body
	ColliderPtr m_BroadPhaseCollider;

	/// The pointer to the narrow phase collider that is attached to the rigid body
	ColliderPtr m_NarrowPhaseCollider;

	/// A body can be put to sleep to avoid it being updated
	/// by the integration functions or affected by collisions
	/// with the world.
    bool m_isAwake;

    /// Some bodies may never be allowed to fall asleep.
    /// User controlled bodies, for example, should be
    /// always awake.
    bool m_canSleep;

    /// Holds the amount of motion of the body. This is a recency
    /// weighted mean that can be used to put a body to sleep.
    float m_motionEnergy;

	//
	// User defined variables
	//
	bool m_userInputeExecuted;
	vec3 m_userPointImpulse;
	vec3 m_userImpulsePosition;
	vec3 m_userAngularImpulse;
};

typedef QSharedPointer<RigidBody> RigidBodyPtr;

