#pragma once
#include <Utility/EngineCommon.h>

class PhysicsWorldConfig
{
public:

	/// Default constructor
	PhysicsWorldConfig();
	~PhysicsWorldConfig();

	//
	// Members
	//
	/// The gravity for the world. The default is (0, -9.8, 0).
	vec3 m_gravity; //+default(0,-9.8f,0)
};

