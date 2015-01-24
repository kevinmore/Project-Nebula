#pragma once
#include <Physicis/World/PhysicsWorldObject.h>
#include <Physicis/Geometry/BoxShape.h>
#include <Physicis/Geometry/SphereShape.h>

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

class RigidBody : public PhysicsWorldObject
{
public:

	enum MotionType
	{
		/// 
		MOTION_INVALID,

		/// A fully-simulated, movable rigid body. At construction time the engine checks
		/// the input inertia and selects MOTION_SPHERE_INERTIA or MOTION_BOX_INERTIA as
		/// appropriate.
		MOTION_DYNAMIC,

		/// Simulation is performed using a sphere inertia tensor. (A multiple of the
		/// Identity matrix). The highest value of the diagonal of the rigid body's
		/// inertia tensor is used as the spherical inertia.
		MOTION_SPHERE_INERTIA,

		/// Simulation is performed using a box inertia tensor. The non-diagonal elements
		/// of the inertia tensor are set to zero. This is slower than the
		/// MOTION_SPHERE_INERTIA motions, however it can produce more accurate results,
		/// especially for long thin objects.
		MOTION_BOX_INERTIA,

		/// Simulation is not performed as a normal rigid body. During a simulation step,
		/// the velocity of the rigid body is used to calculate the new position of the
		/// rigid body, however the velocity is NOT updated. The user can keyframe a rigid
		/// body by setting the velocity of the rigid body to produce the desired keyframe
		/// positions. The hkpKeyFrameUtility class can be used to simply apply keyframes
		/// in this way. The velocity of a keyframed rigid body is NOT changed by the
		/// application of impulses or forces. The keyframed rigid body has an infinite
		/// mass when viewed by the rest of the system.
		MOTION_KEYFRAMED,

		/// This motion type is used for the static elements of a game scene, e.g., the
		/// landscape. Fixed rigid bodies are treated in a special way by the system. They
		/// have the same effect as a rigid body with a motion of type MOTION_KEYFRAMED
		/// and velocity 0, however they are much faster to use, incurring no simulation
		/// overhead, except in collision with moving bodies.
		MOTION_FIXED,

		/// A box inertia motion which is optimized for thin boxes and has less stability problems
		MOTION_THIN_BOX_INERTIA,

		/// A specialized motion used for character controllers
		MOTION_CHARACTER,

		/// 
		MOTION_MAX_ID
	};

	// Construct a default rigid body
	RigidBody(QObject* parent = 0);
	virtual ~RigidBody() {}

	//
	// Shape
	//

	void setShape(const AbstractShape* shape);
	const AbstractShape* getShape() const;

	//
	// MASS, INERTIA AND DENSITY PROPERTIES.
	//

	/// Sets the mass properties
	void setMassProperties(const MassProperties& mp);

	/// Gets the mass of the rigid body.
	inline float getMass() const;

	/// Gets the 1.0/mass of the rigid body.
	inline float getMassInv() const;

	/// Sets the mass of the rigid body. N.B. This does NOT automatically update other dependent mass properties i.e., the inertia tensor.
	void setMass(float m);

	/// Sets the inverse mass of the rigid body.
	void setMassInv(float mInv);

	/// Gets the inertia tensor (around the center of mass) in local space.
	inline mat3 getInertiaLocal() const;

	/// Gets the inertia tensor (around the center of mass) in world space.
	inline mat3 getInertiaWorld() const;

	/// Gets the inverse inertia tensor in local space.
	inline mat3 getInertiaInvLocal() const;

	/// Gets the inverse inertia tensor in world space.
	inline mat3 getInertiaInvWorld() const;

	//
	// CENTER OF MASS.
	//

	/// Explicitly sets the center of mass of the rigid body in local space.
	void setCenterOfMassLocal(const vec3& centerOfMass);

	/// Gets the center of mass of the rigid body in the rigid body's local space.
	inline const vec3& getCenterOfMassLocal() const;

	/// Gets the center of mass of the rigid body in world space.
	inline const vec3& getCenterOfMassInWorld() const;

	//
	// POSITION ACCESS.
	//

	/// Returns the position (the local space origin) for this rigid body, in world space.
	inline const vec3& getPosition() const;

	/// Sets the position (the local space origin) of this rigid body, in world space.
	void setPosition(const vec3& position);

	/// Returns the rotation from local to world space for this rigid body.
	inline const quart& getRotation() const;

	/// Sets the rotation from local to world Space for this rigid body.
	/// This activates the body and its simulation island if it is inactive.
	void setRotation(const quart& rotation);

	/// Sets the position and rotation of the rigid body, in world space.
	void setPositionAndRotation(const vec3& position, const quart& rotation);

	//
	// VELOCITY ACCESS.
	//

	/// Returns the linear velocity of the center of mass of the rigid body, in world space.
	inline const vec3& getLinearVelocity() const;

	/// Sets the linear velocity at the center of mass, in world space.
	/// This activates the body and its simulation island if it is inactive.
	inline void	setLinearVelocity(const vec3& newVel);

	/// Returns the angular velocity around the center of mass, in world space.
	inline const vec3& getAngularVelocity() const;

	/// Sets the angular velocity around the center of mass, in world space.
	/// This activates the body and its simulation island if it is inactive.
	inline void	setAngularVelocity(const vec3& newVel);

	/// Gets the velocity of point p on the rigid body in world space.
	inline vec3& getPointVelocity(const vec3& p) const;

	//
	// IMPULSE APPLICATION.
	//

	/// Applies an impulse (in world space) to the center of mass.
	/// This activates the body and its simulation island if it is inactive.
	inline void applyLinearImpulse(const vec3& imp);

	/// Applies an impulse (in world space) at the point p in world space.
	/// This activates the body and its simulation island if it is inactive.
	virtual void applyPointImpulse(const vec3& imp, const vec3& p) = 0;

	/// Applies an instantaneous change in angular velocity (in world space) around
	/// the center of mass.
	/// This activates the body and its simulation island if it is inactive.
	virtual void applyAngularImpulse(const vec3& imp) = 0;

	//
	// FORCE AND TORQUE APPLICATION
	//

	/// Applies a force (in world space) to the rigid body. The force is applied to the
	/// center of mass.
	virtual void applyForce(const float deltaTime, const vec3& force) = 0;

	/// Applies a force (in world space) to the rigid body at the point p in world space.
	virtual void applyForce(const float deltaTime, const vec3& force, const vec3& p) = 0;

	/// Applies the specified torque (in world space) to the rigid body. (Note: the inline
	/// is for internal use only).
	virtual void applyTorque(const float deltaTime, const vec3& torque) = 0;

	//
	// DAMPING
	//

	/// Naive momentum damping.
	inline float getLinearDamping();

	/// Naive momentum damping.
	inline void setLinearDamping( float d );

	/// Naive momentum damping.
	inline float getAngularDamping();

	/// Naive momentum damping.
	inline void setAngularDamping( float d );

	/// Time factor.
	inline float getTimeFactor();

	/// Time factor.
	inline void setTimeFactor( float f );

	//
	// Friction and Restitution
	//

	/// Returns the friction coefficient (dynamic and static) from the material.
	inline float getFriction() const;

	/// Sets the friction coefficient of the material. Note: Setting this will not update existing contact information.
	void setFriction( float newFriction );

	/// Returns the default restitution from the material.
	//  restitution = bounciness (1 should give object all its energy back, 0 means it just sits there, etc.).
	inline float getRestitution() const;

	/// Sets the restitution coefficient of the material. Note: Setting this will not update existing contact information.
	void setRestitution( float newRestitution );

	//
	// GRAVITY FACTOR
	//

	/// Get the current gravity factor.
	inline float getGravityFactor();

	/// Set the gravity factor.
	inline void setGravityFactor( float gravityFactor );

public:
	MotionType	m_MontionType;

	//
	// Members
	//

	/// The collision detection representation for this entity.
	const AbstractShape* m_shape;

	/// The initial position of the body.
	/// This defaults to 0,0,0.
	vec3 m_position;

	/// The initial rotation of the body.
	/// This defaults to the Identity quaternion.
	quart m_rotation;

	/// The initial linear velocity of the body.
	/// This defaults to 0,0,0.
	vec3 m_linearVelocity;

	/// The initial angular velocity of the body in world space.
	/// This defaults to 0,0,0.
	vec3 m_angularVelocity;

	/// The inertia tensor of the rigid body. Use the hkpInertiaTensorComputer class to
	/// set the inertia to suitable values.
	/// This defaults to the identity matrix.
	mat3 m_inertiaTensor;

	/// The center of mass in the local space of the rigid body.
	/// This defaults to 0,0,0.
	vec3 m_centerOfMass;

	/// The mass of the body.
	/// This defaults to 1.
	float m_mass;

	/// The inverse mass of the body.
	/// This defaults to 1.
	float m_massInv;

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

	/// The angular velocity * dt which was used in the last integration step.
	vec3 m_deltaAngle; 

	/// A sphere around the center of mass which completely encapsulates the object
	float m_objectRadius;
};

