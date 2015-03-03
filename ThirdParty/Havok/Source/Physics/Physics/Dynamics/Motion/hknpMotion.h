/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MOTION_H
#define HKNP_MOTION_H

#include <Physics/Physics/Dynamics/Motion/hknpMotionProperties.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionCinfo.h>

#if defined(HK_PLATFORM_WIN32)
#	pragma warning( 3 : 4820 )	
#endif

struct hkMassProperties;

/// Information on how one or more rigid bodies move.
///
/// Motions are allocated and owned by the world. The world creates a persistent Id for each motion.
/// Nothing can be assumed about the value of the Id, but it is guaranteed not to change during the lifetime of the motion.
/// Motion space is always the space where the inertia tensor becomes a diagonal matrix.
/// To be able to still support full body inertias, there is a relative transform between motions and bodies,
/// which is stored in a body.
///
/// Bodies which are supposed to move together can share the motion. All bodies sharing the same motion
/// are linked together.
///
/// Note that the angular velocity and the inertia is stored in local space.
///
/// This structure requires 128 Bytes of memory. In addition, 16 bytes of memory are needed for the deactivation info
/// and the motion properties. The latter are shared between motions.
///
/// \sa hknpBody hknpWorld hknpMotionManager hknpDeactivationMotionState hknpMotionProperties
class hknpMotion
{
	//+version(2)

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMotion );
		HK_DECLARE_REFLECTION();
		HK_DECLARE_POD_TYPE();

		/// Construct an uninitialized motion.
		HK_FORCE_INLINE hknpMotion();

		/// Copy operator.
		HK_FORCE_INLINE void operator=( const hknpMotion& other );

		/// Reset the motion to a static one.
		HK_FORCE_INLINE void reset();

		/// Returns true if the motion is a static one.
		HK_FORCE_INLINE bool isStatic() const;

		/// Returns non-zero if the motion is not a static one.
		HK_FORCE_INLINE hkBool32 isDynamic() const;

		/// Returns true if the motion is dynamic and active.
		HK_FORCE_INLINE bool isActive() const;

		/// Returns true if the motion is in use and not marked for deletion.
		HK_FORCE_INLINE bool isValid() const;

		/// Returns true if the inverse mass and inertia are zero (ie. the motion is static or keyframed).
		HK_FORCE_INLINE bool hasInfiniteMass() const;

		//
		// Mass and inertia functions
		//

		/// Set the inverse mass, and rescale the inertia.
		/// The motion needs to already have a finite mass, so that the inertia can be scaled.
		HK_FORCE_INLINE void setInverseMass( hkSimdRealParameter invMass );

		/// Sets the inverse inertia moments values from the first three vector coordinates, leaving the mass unchanged.
		HK_FORCE_INLINE void setInverseInertia( hkVector4Parameter invInertia );

		/// Sets the inverse inertia moment values from the first three vector coordinates, and the inverse mass
		/// from the fourth.
		HK_FORCE_INLINE void setInverseInertiaAndMass( hkVector4Parameter invInertiaAndMass );

		/// Set an infinite mass and inertia, making this a keyframed motion.
		HK_FORCE_INLINE void setInfiniteInertiaAndMass();

		/// Sets the mass, inertia and center of mass from the given mass properties.
		/// A mass of 0 is considered infinite.
		void setFromMassProperties( const hkMassProperties& mp, const hkTransform& massPropertiesTransform );

		/// Set the center of mass in world space.
		HK_FORCE_INLINE void setCenterOfMassInWorld( hkVector4Parameter com );

		/// Get the center of mass in world space.
		HK_FORCE_INLINE const hkVector4& getCenterOfMassInWorld() const;

		/// Get the world transform of this motion.
		HK_FORCE_INLINE void getWorldTransform( hkQTransform& transform ) const;

		/// Get the inverse mass.
		HK_FORCE_INLINE hkSimdReal getInverseMass() const;

		/// Get the mass.
		/// Returns 0 for infinite mass.
		HK_FORCE_INLINE hkSimdReal getMass() const;

		/// Get the inverse inertia in local space. The W component is the inverse mass.
		HK_FORCE_INLINE void getInverseInertiaLocal( hkVector4& invInertiaOut ) const;

		/// Get the inertia in local space. The W component is the mass.
		HK_FORCE_INLINE void getInertiaLocal( hkVector4& inertiaOut ) const;

		/// Get the inverse inertia in world space. This is relatively slow.
		void getInverseInertiaWorld( hkMatrix3& invInertiaOut ) const;

		/// Get the inertia in world space. This is relatively slow.
		void getInertiaWorld( hkMatrix3& inertiaOut ) const;

		/// Set the mass factor (used only to help in some API methods, may move elsewhere)
		HK_FORCE_INLINE void setMassFactor( hkSimdRealParameter massFactor );

		/// Get the mass factor (used only to help in some API methods, may move elsewhere)
		HK_FORCE_INLINE hkSimdReal getMassFactor() const;

		//
		// Velocity functions
		//

		/// Get the linear velocity of the center of mass.
		HK_FORCE_INLINE const hkVector4& getLinearVelocity() const;

		/// Get the angular velocity in local space.
		HK_FORCE_INLINE const hkVector4& getAngularVelocityLocal() const;

		/// Get the angular velocity in world space.
		void getAngularVelocity( hkVector4& angularVelWorldOut ) const;
		HK_FORCE_INLINE void _getAngularVelocity( hkVector4& angularVelWorldOut ) const; ///< inline version

		/// Get the point velocity, projected onto the given \a normal, using this motion.
		HK_FORCE_INLINE hkSimdReal getProjectedPointVelocity( hkVector4Parameter position, hkVector4Parameter normal ) const;

		/// Get the point velocity for the given body parameters using this motion.
		void getPointVelocity( hkVector4Parameter position, hkVector4& velOut ) const;
		HK_FORCE_INLINE void _getPointVelocity( hkVector4Parameter position, hkVector4& velOut ) const;///< inline version

		/// Gets the point velocity for this motion using the input velocities in world space.
		/// Use this function if you need to get multiple point velocities for one motion.
		HK_FORCE_INLINE void _getPointVelocityUsingVelocity(
			hkVector4Parameter linVel, hkVector4Parameter angVelWorld, hkVector4Parameter position, hkVector4& velOut ) const;

		/// Calculate the projected inverse mass.
		HK_FORCE_INLINE hkSimdReal _calcProjectedInverseMass( hkVector4Parameter direction, hkVector4Parameter position ) const;

		/// Apply an \a impulse (world space) to a motion at a specified \a position (world space).
		/// You can not do this while the solver is running.
		/// If you call this function while the collision system is running and you have restitution !=0,
		/// you get non-deterministic behavior.
		void applyPointImpulse( hkVector4Parameter impulse, hkVector4Parameter position );
		HK_FORCE_INLINE void _applyPointImpulse( hkVector4Parameter impulse, hkVector4Parameter position ); ///< inline version

		/// Apply an angular impulse.
		void applyAngularImpulse( hkVector4Parameter angImpulseWorld );
		HK_FORCE_INLINE void _applyAngularImpulse( hkVector4Parameter angImpulseWorld ); ///< inline version

		/// Apply a linear impulse.
		void applyLinearImpulse( hkVector4Parameter impulse );
		HK_FORCE_INLINE void _applyLinearImpulse( hkVector4Parameter impulse ); ///< inline version

		/// Set the linear velocity.
		HK_FORCE_INLINE void setLinearVelocity( hkVector4Parameter velocity );

		/// Set the velocity at a given point.
		void setPointVelocity( hkVector4Parameter velocity, hkVector4Parameter position );

		/// Set the angular velocity (world space).
		void setAngularVelocity( hkVector4Parameter angVelocity );
		HK_FORCE_INLINE void _setAngularVelocity( hkVector4Parameter angVelocityWorld );

		/// Sets the linear and angular velocities. Typically used to inherit the velocities of another motion.
		void setVelocities(
			hkVector4Parameter linearVelocity, hkVector4Parameter angularVelocity,
			hkVector4Parameter centerOfLinearVelocityInWorld );

		/// Internal function to override the integrated velocities.
		void setPreviousStepVelocities(
			hkVector4Parameter linearVelocity, hkVector4Parameter angularVelocity,
			hkVector4Parameter centerOfLinearVelocityInWorld );

		//
		// Utility functions
		//

		/// Back-step the motion backward in time. t is [0.0f ... 1.0f] and always relative
		/// to interval [lastFrame, currentFrame].
		/// if you reintegrate(0.5f) and then reintegrate(0.5f), you get the same results as a single call to reintegrate(0.25f).
		/// It will modify m_previousStepLinearVelocity and m_previousStepAngularVelocity to match the reduced movement.
		/// m_integrationFactor can be used to track how much it has been back stepped.
		void reintegrate( hkSimdRealParameter t, hkSimdRealParameter deltaTime );

		/// Build the mass matrix for a given point in space. If the mass is infinite (but inertia is finite),
		/// the mass matrix will be computed as if the mass was equal to 1.0;
		void buildEffMassMatrixAt( hkVector4Parameter position, hkMatrix3& effMassMatrixOut) const;

		/// Set the space splitter weight. It can be used to tell the space splitter how this motion should be weighted when updating the cell divisions.
		/// The weight must be greater than 0 and the default value for a motion is 1.
		/// For expensive motions (like compounds with many instances) setting this to a larger value can improve the multithreading performance.
		HK_FORCE_INLINE void setSpaceSplitterWeight( hkUint8 weight );

		/// In debug this checks if all the values in the motion are not NAN
		HK_ON_CPU( void checkConsistency() const );
		HK_ON_SPU( HK_FORCE_INLINE void checkConsistency() const {} );

	protected:

		/// XYZ: Position of the mass center in world space.
		///   W: Mass factor.
		hkVector4 m_centerOfMassAndMassFactor;

		
		friend struct hknpInternalMotionUtilFunctions2;
		friend class hknpInternalMotionUtil;

	public:

		

		/// Orientation.
		hkQuaternion m_orientation;

		

		/// A packed representation of the inertia tensor and mass.
		/// To retrieve and manipulate these values, use the methods provided in this class.
		/// The stored values are (in sequence):
		/// - inverse inertia tensor first axis extent
		/// - inverse inertia tensor second axis extent
		/// - inverse inertia tensor third axis extent
		/// - inverse mass
		hkHalf m_inverseInertia[4];

		/// A body identifier to a group of bodies using this motion.
		hknpBodyId m_firstAttachedBodyId;	//+overridetype(hkUint32)

		/// An index into the solver velocities (owned by the motion manager).
		///   - is invalid for inactive bodies or bodies not added to the world yet.
		///   - is 0 for static bodies.
		///   - else references the entry in the solverVelocity array of the local grid cell.
		hknpSolverId m_solverId;	//+overridetype(hkUint32)	//+nosave

		

		///
		HK_ALIGN16(hkHalf) m_linearVelocityCage[3];

		/// An identifier for the motion properties.
		hknpMotionPropertiesId m_motionPropertiesId; //+overridetype(hkUint16)

		/// The maximum linear acceleration of a body that can be applied to a body in a single solver step.
		/// E.g. if you are running physics at 30Hz and you set this value to 0.5f,
		/// the body can accelerate by 15m/sec every physics step.
		/// Note that bodies can have higher velocity, but will 'lose' time in the frame of clipped acceleration.
		hkHalf m_maxLinearAccelerationDistancePerStep;

		/// Clip the angular rotation per frame. This is needed for continuous physics.
		/// This could be set by hand, but a better way is to do this automatically by setting
		/// the CLIP_ANGULAR_VELOCITY flag in a body's hknpBodyQuality.
		hkHalf m_maxRotationToPreventTunneling;

		/// Helper value to indicate by how much this motion was integrated.
		/// This is set to 1 when the motion is integrated, and is reduced by reintegrate().
		hkHalf m_integrationFactor;

		/// The cell index of this motion in the world grid structures.
		hknpCellIndex m_cellIndex;

		/// The space splitter weight is used by the space splitter when dividing the world into cells.
		/// Set to zero if the motion is not in use or is marked for deletion.
		hkUint8 m_spaceSplitterWeight;

		

		/// The current linear velocity (world space).
		hkVector4 m_linearVelocity;

		/// The current angular velocity (local space).
		hkVector4 m_angularVelocity;

		/// The linear velocity (world space) which was used to integrate this motion during the previous step.
		hkVector4 m_previousStepLinearVelocity;

		/// The angular velocity (local space) which was used to integrate this motion during the previous step.
		hkVector4 m_previousStepAngularVelocity;

		
};

#if defined(HK_PLATFORM_WIN32)
#	pragma warning( disable : 4820 )
#endif

#include <Physics/Physics/Dynamics/Motion/hknpMotion.inl>


#endif // HKNP_MOTION_H

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
