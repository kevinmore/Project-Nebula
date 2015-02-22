#pragma once
#include "RigidBody.h"

class BoxRigidBody : public RigidBody
{
public:

	/// Default constructor - sets the initial velocity to zero
	BoxRigidBody(const vec3& position = Math::Vector3::ZERO,
		         const quat& rotation = Math::Quaternion::IDENTITY);

	/// Sets the mass of the rigid body.
	void setMass(float m);

	/// Sets the inverse mass of the rigid body.
	void setMassInv(float mInv);

	/// Sets the half extents, this will update the inertia tensor
	void setBoxHalfExtents(const vec3& halfExtents);

private:

	/// Half extents of the box, used for compute the inertia tensor
	/// This defaults to (0.5, 0.5, 0.5)
	vec3 m_halfExtents;
};

typedef QSharedPointer<BoxRigidBody> BoxRigidBodyPtr;