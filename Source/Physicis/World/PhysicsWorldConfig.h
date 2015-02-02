#pragma once
#include <Utility/EngineCommon.h>

class PhysicsWorldConfig
{
public:

	enum BroadPhaseBorderBehaviour
	{
		/// Cause an assert and set the motion type to be fixed (default).
		BROADPHASE_BORDER_ASSERT,

		/// Set the motion type to be fixed and raise a warning.
		BROADPHASE_BORDER_FIX_ENTITY,

		/// Remove the entity from the hkpWorld and raise a warning.
		BROADPHASE_BORDER_REMOVE_ENTITY,

		/// Do not do anything, just continue to work.
		/// If many (>20) objects leave the broadphase,
		/// serious memory and CPU can be wasted.
		BROADPHASE_BORDER_DO_NOTHING,
	};

	/// Sets the broadphase size to be a cube centered on the origin of side
	/// sideLength. See also m_broadPhaseWorldSize.
	void setBroadPhaseWorldSize(const vec3& size);

	/// Default constructor
	PhysicsWorldConfig();
	~PhysicsWorldConfig();

	//
	// Members
	//
	/// The gravity for the world. The default is (0, -9.8, 0).
	vec3 m_gravity; //+default(0,-9.8f,0)
};

