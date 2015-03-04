/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MOTION_CINFO_H
#define HKNP_MOTION_CINFO_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Dynamics/Body/hknpBody.h>


/// Construction info for a motion.
/// \sa hknpMotion
struct hknpMotionCinfo
{
	//+version(1)

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMotionCinfo );
		HK_DECLARE_REFLECTION();

		/// Constructor. Initializes the members with default values for motion of a dynamic body.
		hknpMotionCinfo();

		/// Serialization constructor.
		hknpMotionCinfo( hkFinishLoadedObjectFlag flag ) {}

		/// Initialize from an array of hknpBodyCinfo's using the density from the shapes.
		void initialize( const hknpBodyCinfo* bodyCinfos, int numCinfos );

		/// Initialize from an array of hknpBodyCinfo's and a total mass.
		void initializeWithMass( const hknpBodyCinfo* bodyCinfos, int numCinfos, hkReal mass );

		/// Initialize from an array of hknpBodyCinfo's and a density.
		void initializeWithDensity( const hknpBodyCinfo* bodyCinfos, int numCinfos, hkReal density );

		/// Initialize with infinite mass and inertia, and m_motionPropertiesId = hknpMotionPropertiesId::KEYFRAMED.
		void initializeAsKeyFramed( const hknpBodyCinfo* bodyCinfos, int numCinfos );

		/// Stops any angular movement by setting the invInertia to 0.
		void fixRotation();

		/// Stops any linear movement by setting the invMass to 0.
		void fixPosition();

		/// Helper function to set the inertia for a box like shape.
		/// More helper functions can be found in hkInertiaTensorComputer.
		void setBoxInertia( hkVector4Parameter halfExtents, hkReal mass );

		/// Sets the orientation and the inertia defined in local space.
		void setOrientationAndInertiaLocal( hkQuaternionParameter orientation, const hkMatrix3& inertia );

		/// Initialize from an existing body. Advanced use only.
		///		- If massOrNegativeDensity > 0, the supplied value is used as the mass.
		///		- If massOrNegativeDensity < 0, the mass will be taken from the shape volume * -massOrNegativeDensity
		///		- If massOrNegativeDensity == 0, the hknpMotionCinfo will be set to keyframed.
		HK_FORCE_INLINE void initializeFromBody( const hknpBody& body, hkReal massOrNegativeDensity );

		/// Initialize from an existing body. Advanced use only. Use mass specified by the shape.
		HK_FORCE_INLINE void initializeFromBody( const hknpBody& body );

	public:

		/// Identifier to the shared integration settings for the motion.
		hknpMotionPropertiesId m_motionPropertiesId; //+overridetype(hkUint16)

		/// Can be this motion be deactivated.
		hkBool m_enableDeactivation; //+default(true)

		/// Inverse mass.
		hkReal m_inverseMass; //+default(1.0f)

		/// The mass of the object relative to the mass of its shape(s).
		/// This is used to recalculate the mass (for example in hknpWorld::detachBody()).
		hkReal m_massFactor; //+default(1.0f)

		/// The maximum linear acceleration of a body that can be applied to it in a single physics step.
		/// See hknpBody::getCollisionLookAheadDistance().
		/// E.g. if you are running physics at 30Hz and you set this value to 0.5f,
		/// the body can accelerate by 15m/sec every physics step.
		/// Note that bodies can have higher velocity, but will 'lose' time in the frame of clipped acceleration.
		/// Note: if the hknpBody has m_collisionLookAheadDistance set, this value will be
		///       the min( m_collisionLookAheadDistance, m_maxLinearAccelerationDistancePerStep ).
		hkReal m_maxLinearAccelerationDistancePerStep;	//+default(HK_REAL_HIGH)

		/// This clips the angular velocity to prevent especially thin highly tessellated objects from tunneling.
		/// Set this to the minimum angle between two neighboring faces to
		/// dramatically reduce the likelihood of thin objects tunneling through other thin objects.
		/// Typically you do not have to set this value by hand, simply set
		/// hknpBodyQuality::m_bodyFlags |= hknpBodyQuality::CLIP_ANGULAR_VELOCITY.
		hkReal m_maxRotationToPreventTunneling; //+default(HK_REAL_HIGH)

		/// Inverse inertia tensor in motion space.
		/// Motion space is the space where the inertia tensor becomes a diagonal.
		hkVector4 m_inverseInertiaLocal;

		/// Mass center in world space.
		hkVector4 m_centerOfMassWorld; //+default(0.0f,0.0f,0.0f,0.0f)

		/// Orientation.
		hkQuaternion m_orientation; //+default(0.0f,0.0f,0.0f,1.0f)

		/// Linear velocity.
		hkVector4 m_linearVelocity; //+default(0.0f,0.0f,0.0f,0.0f)

		/// Angular velocity.
		hkVector4 m_angularVelocity; //+default(0.0f,0.0f,0.0f,0.0f)
};

#include <Physics/Physics/Dynamics/Motion/hknpMotionCinfo.inl>


#endif // HKNP_MOTION_CINFO_H

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
