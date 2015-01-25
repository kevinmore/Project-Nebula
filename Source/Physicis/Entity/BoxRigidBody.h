#pragma once
#include "RigidBody.h"

class BoxRigidBody : public RigidBody
{
public:

	/// Default constructor - sets the initial velocity to zero
	BoxRigidBody(const vec3& position = Math::Vector3D::ZERO,
		         const quart& rotation = Math::Quaternion::ZERO);

	// Updates the properties of the rigid body
	virtual void update(const float dt);

	/// Gets the inertia tensor of the rigid body in local space.
	virtual mat3 getInertiaLocal() const;

	/// Sets the inertia tensor of the rigid body in local space. Advanced use only.
	virtual void setInertiaLocal(const mat3& inertia);

	/// Gets the inverse inertia tensor in local space.
	virtual mat3 getInertiaInvLocal() const;

	/// Sets the inertia tensor of the rigid body by supplying its inverse. Advanced use only.
	virtual void setInertiaInvLocal(const mat3& inertiaInv);

	/// Get the inertia tensor of the rigid body in world space
	virtual mat3 getInertiaWorld() const;

	/// Get the inverse inertia tensor in local space
	virtual mat3 getInertiaInvWorld() const;

	/// Applies an impulse (in world space) at the point p in world space.
	virtual void applyPointImpulse(const vec3& imp, const vec3& p);

	/// Applies an instantaneous change in angular velocity (in world space) around the center of mass.
	virtual void applyAngularImpulse(const vec3& imp);

	/// Applies a force (in world space) to the rigid body. The force is applied to the center of mass.
	virtual void applyForce(const float deltaTime, const vec3& force);

	/// Applies a force (in world space) to the rigid body at the point p in world space.
	virtual void applyForce(const float deltaTime, const vec3& force, const vec3& p);

	/// Applies the specified torque (in world space) to the rigid body. (note: the inline is for internal use only)
	virtual void applyTorque(const float deltaTime, const vec3& torque);
};

typedef QSharedPointer<BoxRigidBody> BoxRigidBodyPtr;