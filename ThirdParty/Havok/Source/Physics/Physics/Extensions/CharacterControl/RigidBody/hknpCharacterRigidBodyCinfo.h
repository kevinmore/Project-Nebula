/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CHARACTER_RIGID_BODY_CINFO__H
#define HKNP_CHARACTER_RIGID_BODY_CINFO__H

#include <Common/Base/Object/hkReferencedObject.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>

extern const class hkClass hknpCharacterRigidBodyCinfoClass;

class hknpShape;
class hknpWorld;

/// Information used to construct a hknpCharacterRigidBody.
struct hknpCharacterRigidBodyCinfo : public hkReferencedObject
{
	//+version(1)
	HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
	HK_DECLARE_REFLECTION();

	/// Constructor. Sets some defaults.
	hknpCharacterRigidBodyCinfo();

	/// Serialization constructor.
	hknpCharacterRigidBodyCinfo( hkFinishLoadedObjectFlag flag ) : hkReferencedObject(flag) {}


	/// The collision filter info. See hknpBodyCinfo for details.
	hkUint32 m_collisionFilterInfo; //+hk.Ui(visible=false)

	/// The shape. See hknpBodyCinfo for details.
	const hknpShape* m_shape; //+hk.Ui(visible=false)

	hknpWorld* m_world; //+hk.Ui(visible=false) +nosave

	/// Initial position. See hknpBodyCinfo for details.
	hkVector4 m_position; //+hk.Ui(visible=false)

	/// Initial rotation. See hknpBodyCinfo for details.
	hkQuaternion m_orientation; //+hk.Ui(visible=false)

	///	The mass of character.
	hkReal m_mass;	//+default(100.0f)
					//+hk.RangeReal(absmin=0.0,absmax=1000.0)
					//+hk.Description("The character rigid body mass.")

	/// Set dynamic friction of character.
	hkReal m_dynamicFriction; //+default(0.0f)
							  //+hk.RangeReal(absmin=0.0,absmax=4.0)
							  //+hk.Description("The dynamic friction of the character rigid body.")

	/// Set static friction of character.
	hkReal m_staticFriction; //+default(0.0f)
							  //+hk.RangeReal(absmin=0.0,absmax=4.0)
							  //+hk.Description("The static friction of the character rigid body.")

	/// Set the character material's welding tolerance.
	hkReal m_weldingTolerance; //+default(0.1f)

	/// Optional: use a specific preallocated ID for this body.
	hknpBodyId m_reservedBodyId;	//+hk.Ui(visible=false)
									//+overridetype(hkUint32)

	/// Flags specifing when/how to add the body to the world.
	hknpWorld::AdditionFlags m_additionFlags;	//+default(0)
												//+overridetype(hkUint8)

	//
	// Character controller specific values
	//

	/// Set up direction.
	hkVector4 m_up;	//+hk.Ui(visible=false)

	/// Set maximal slope.
	hkReal m_maxSlope;	//+default(1.04719755f)
						//+hk.RangeReal(absmin=0.0,absmax=1.57079633)
						//+hk.Description("The maximum slope that the character can walk up. This angle is measured in radians from the horizontal. The default value is pi / 3.")

	/// Set maximal force of character.
	hkReal m_maxForce;	//+default(1000.0f)
						//+hk.RangeReal(absmin=0.0,absmax=100000.0)
						//+hk.Description(" The maximum force of character.")

	//
	// Parameters used by checkSupport
	//

	/// Set maximal speed for simplex solver.
	hkReal m_maxSpeedForSimplexSolver;	//+default(10.0f)
										//+hk.RangeReal(absmin=0.0,absmax=100.0)
										//+hk.Description("The maximum speed for the simplex solver.")

	/// A character is considered supported if it is less than this distance above its supporting planes.
	hkReal m_supportDistance;	//+default(0.1f)
								//+hk.RangeReal(absmin=0.0,absmax=1.0)
								//+hk.Description("A character is considered supported if it is less than this distance above its supporting planes.")

	/// A character should keep falling until it is this distance or less from its supporting planes.
	hkReal m_hardSupportDistance;	//+default(0.0f)
									//+hk.RangeReal(absmin=0.0,absmax=1.0)
									//+hk.Description("A character should keep falling until it is this distance or less from its supporting planes.")
};

#endif // HKNP_CHARACTER_RIGID_BODY_CINFO__H

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
