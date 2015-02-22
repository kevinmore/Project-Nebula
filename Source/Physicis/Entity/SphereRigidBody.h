#pragma once
#include "RigidBody.h"

class SphereRigidBody : public RigidBody
{
public:
	/// Default constructor - sets the initial velocity to zero
	SphereRigidBody(const vec3& position = Math::Vector3::ZERO,
				    const quat& rotation = Math::Quaternion::IDENTITY);

	/// Sets the mass of the rigid body.
	void setMass(float m);

	/// Sets the inverse mass of the rigid body.
	void setMassInv(float mInv);

	/// Sets the radius of the sphere
	void setSphereRadius(float newRadius);

private:

	/// Radius of the sphere, used for compute the inertia tensor, default to 0.5
	float m_radius;
};

typedef QSharedPointer<SphereRigidBody> SphereRigidBodyPtr;