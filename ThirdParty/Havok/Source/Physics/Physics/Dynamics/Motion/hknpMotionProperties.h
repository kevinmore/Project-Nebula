/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MOTION_PROPERTIES_H
#define HKNP_MOTION_PROPERTIES_H

#include <Physics/Physics/hknpTypes.h>

#if defined(HK_PLATFORM_WIN32)
	//#pragma warning( 3 : 4820 )	
#endif


/// Integration settings for a motion.
/// This set of values defines how a motion is integrated and is referenced by ID from hknpMotion.
///
/// A typical simulation will have a small set of these motion properties to define how certain object types move.
/// The maximum number of motion properties available in the world can be small for reasons of optimization,
/// so try to build a small library of properties yourself.
///
/// This class stores the parameters for the motion solver deactivation. It reduces the velocity of slow objects.
/// As a result, static stacks of bodies get more stable. Attention: the motion solver deactivation should not be
/// confused with the body deactivation facility of the world.
///
/// Setting sensible values for the motion solver deactivation can be challenging. This class provides a set of safe
/// deactivation settings for the setMotionSolverDeactivation() method. We recommend using them in favor of using
/// custom values.
HK_CLASSALIGN16(class) hknpMotionProperties
{
	//+version(1)

	public:

		/// Motion Property Flags.
		enum FlagsEnum
		{
			ENABLE_GRAVITY_MODIFICATION		= 1 << 0,	// call modifyMotionGravity() on any modifiers registered with this flag
			NEVER_REBUILD_MASS_PROPERTIES	= 1 << 1	
		};

		typedef hkFlags<FlagsEnum, hkUint8> Flags;

		/// Predefined solver stabilization settings.
		enum SolverStabilizationType
		{
			SOLVER_STABILIZATION_OFF,			///< solver deactivation off
			SOLVER_STABILIZATION_LOW,			///< solver deactivation only for very slow objects (about 10cm/sec)
			SOLVER_STABILIZATION_MEDIUM,		///< solver deactivation for all slow objects (about 17cm/sec)
			SOLVER_STABILIZATION_HIGH,			///< solver deactivation for most of the slower objects (about 20cm/sec)
			SOLVER_STABILIZATION_AGGRESSIVE,	///< solver deactivation already for moderately fast objects (about 25cm/sec)
		};

		/// Deactivation type determines what to do when an object is slow.
		/// An object is recognized as a slow object if
		///     - all components of normalizedVelocity are less than 1
		///		- normalizedVelocity = m_linearSolverDeactFactor * linearVelocity + m_angularSolverDeactFactor * angularVelocity
		///   the solver deactivation then does bodyVel *= slowObjectVelocityMultiplier
		enum DeactivationStrategy
		{
			/// A more aggressive deactivation strategy than the ones below.
			/// Does not wait as long, at the cost of deactivating objects at higher velocities.
			DEACTIVATION_STRATEGY_AGGRESSIVE = 3,

			/// The deactivation period is the time required to find minimal jitter within
			/// hknpMotionProperties_DEFAULT_REFERENCE_DISTANCE at hknpMotionProperties_DEFAULT_MINIMUM_PATHING_VELOCITY.
			DEACTIVATION_STRATEGY_BALANCED,

			 /// Increased the period of waiting for deactivation to detect pathological jitter cases that
			 /// have minimal jitter and maximal movement within the deactivation boundaries.
			DEACTIVATION_STRATEGY_ACCURATE,
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMotionProperties );
		HK_DECLARE_REFLECTION();

		/// Construct a motion properties. Defaults to predefined settings for a dynamic motion.
		HK_FORCE_INLINE hknpMotionProperties( hknpMotionPropertiesId::Preset preset = hknpMotionPropertiesId::DYNAMIC );

		/// Comparison operator.
		HK_FORCE_INLINE bool operator==( const hknpMotionProperties& other ) const;

		/// Reset the parameters to one of the predefined settings. Does not change the m_isExclusive flag.
		void setPreset( hknpMotionPropertiesId::Preset preset, hkReal gravityHint = 9.81f, hkReal objectScaleHint = 1.0f );

		/// Set solver stabilization parameters, based on a type and a gravity magnitude.
		void setSolverStabilization( SolverStabilizationType type, hkReal gravity );

		/// Set basic deactivation parameters, defaults correspond to objects of one meter of roughly equal dimensions
		/// (one should at least linearly scale referenceDistance upwards based on the increased radius)
		void setDeactivationParameters(
			hkReal referenceDistance = 0.05f //Detect and deactivate jitter up to a maximum magnitude. An increasingly high setting has the side effect of disabling increasingly faster moving objects
			, hkReal referenceRotation = 0.05f * HK_REAL_PI //in radians, the maximum allowed rotation for jitter. Clamped to hknpMotionProperties_MAX_ROTATION_IN_PI_RAD
			, DeactivationStrategy deactivationStrategy = DEACTIVATION_STRATEGY_BALANCED //The duration of the deactivation checks, making it more aggressive will also disable increasingly faster moving objects
			 );

		///
		void setAdvancedDeactivationParameters(
			hkReal deactivationVelocity = 0.02f	//The velocity below which objects not moving in a path are immediately/aggressively deactivated.
												//Automatically saturates to minimumPathingVelocity, which is the deactivation velocity at the end of
												//the full deactivation routine (represented by m_minimumPathingVelocityScaleSquare).
			, hkReal pathingUpperThreshold = 0.8125f //the pathing threshold above which objects with significant movement are never deactivated(*), in the range [-1..1]
			, hkReal pathingLowerThreshold = -0.8f //the pathing threshold below which objects are immediately/aggressively deactivated, in the range [-1..1]
			, hkUint8 spikingVelocityThreshold = 3 //the velocity increase (in terms of their exponent) that will be considered a spike (ie. 3 would represent a spike 8 times bigger than the previous velocities)
			, hkReal minimumSpikingVelocity = 0.016f ); //the minimum velocity above which spiking analysis is performed

		// (*)pathingUpperThreshold's value is based on a maximum number of 2.0 attainable points during a frequency pass, of which multiple are performed during the full deactivation period.
		// The first period lasts 16 frames, where a velocity flip will reduce the maximum score of 1.0 by 2/16.
		// Allowing 1.5 velocity flips per pass therefore puts the threshold at 1-3/16=0.8125.
		// For DEACTIVATION_STRATEGY_BALANCED the longest pass lasts 128 frames, so allowing 1.5 velocity flips will deactivate objects jittering at frequencies
		// starting from 60/128*(1.5/2) = 0.35hz in a 60hz simulation.

		/// In debug builds, check that all member values are sane. Fire asserts if any are not.
		HK_FORCE_INLINE void checkConsistency() const;

		// Internal: Operations for hkFreeListArray.
		struct FreeListArrayOperations
		{
			HK_FORCE_INLINE static void setEmpty( hknpMotionProperties& mp, hkUint32 next );
			HK_FORCE_INLINE static hkUint32 getNext( const hknpMotionProperties& mp );
			HK_FORCE_INLINE static hkBool32 isEmpty( const hknpMotionProperties& mp );
		};

	public:

		/// If set to TRUE, this motion properties is not allowed to be merged with other identical motion properties.
		HK_ALIGN_REAL(hkBool32) m_isExclusive;

		//
		// General
		//

		/// Flags. See FlagsEnum.
		Flags m_flags;

		/// The factor of the world's gravity vector to apply.
		hkReal m_gravityFactor;

		/// The maximum linear speed that is allowed.
		/// Note that if we do not limit body speeds, their expanded AABBs could become so large that the system
		/// will run out of memory because of all the extra broad phase overlaps.
		hkReal m_maxLinearSpeed;

		/// The maximum angular speed that is allowed.
		hkReal m_maxAngularSpeed;

		/// The linear damping factor.
		hkReal m_linearDamping;

		/// The angular damping factor.
		hkReal m_angularDamping;

		//
		// Solver stabilization
		//

		/// A linear speed threshold below which the solver is allowed to reduce the speed of a body,
		/// by m_solverStabilizationSpeedReduction per second.
		hkReal m_solverStabilizationSpeedThreshold; //+default(1.0)

		/// See m_solverStabilizationSpeedThreshold.
		hkReal m_solverStabilizationSpeedReduction;	//+default(0.0)

		//
		// Deactivation
		//

		/// Square of the maximum distance a deactivation candidate is allowed to move
		hkReal m_maxDistSqrd;

		/// Same as m_maxDistSqrd, but instead of using the distance traveled by the mass center
		/// this is the angle the object is allowed to rotate.
		hkReal m_maxRotSqrd;

		/// The discretization block size for position compression in hknpDeactivationState.
		hkReal m_invBlockSize;

		/// Upper and lower threshold for the pathing algorithm of hknpDeactivationState,
		/// compressed from the arguments of setAdvancedDeactivationParameters
		hkInt16 m_pathingUpperThreshold;
		hkInt16 m_pathingLowerThreshold;

		/// The number of times where instead of deactivating the body, the deactivation window can be increased
		/// (under certain conditions) and the deactivation checks are repeated in the new deactivation window.
		hkUint8 m_numDeactivationFrequencyPasses;

		// The minimum velocity for which objects which are not moving in a path are immediately deactivated at the end
		// of each frequency pass. Objects that are moving faster are kept active for further analysis - provided that there
		// is a subsequent frequency pass.
		hkUint8 m_deactivationVelocityScaleSquare; //used for the intermediate passes (value should be <= m_minimumPathingVelocityScaleSquare)
		hkUint8 m_minimumPathingVelocityScaleSquare; //used for the final pass

		// Velocity spikes are those velocities which have a significantly higher value than all previous velocities.
		// When a velocity spike is detected, the deactivation analysis is restarted.
		// To be considered a spike, the velocity scale (exponent) should be larger than the highest exponent of the
		// squared previous velocities + m_spikingVelocityScaleThresholdSquared
		hkUint8 m_spikingVelocityScaleThresholdSquared;

		// When checking for velocity spikes, the current velocity scale (exponent) should be at least m_minimumSpikingVelocityScaleSquared.
		hkUint8 m_minimumSpikingVelocityScaleSquared;
};

#if defined(HK_PLATFORM_WIN32)
	//#pragma warning( disable : 4820 )
#endif

#include <Physics/Physics/Dynamics/Motion/hknpMotionProperties.inl>


#endif // HKNP_MOTION_PROPERTIES_H

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
