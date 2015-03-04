/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_DEACTIVATION_MOTION_STATE_H
#define HKNP_DEACTIVATION_MOTION_STATE_H

class hknpMotion;
class hknpMotionProperties;

//#define DEACTIVATION_FREEFALL_DETECTION
//#define DEACTIVATION_DEBUG_ACTIVATION_REASON


/// Class which holds information to deactivate a motion
struct hknpDeactivationState
{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, hknpDeactivationState);

		/// Init
		void initDeactivationState(int random, bool enableDeactivation=true);

		/// Reset the counters.
		HK_FORCE_INLINE void resetCounters(int random);

		/// Return -1 if it is OK to deactivate, 0 else.
		int checkMotionForDeactivation(hknpMotion& motion, const hknpMotionProperties* allMotionProperties, hkVector4Parameter prevLinearVelocity);

		/// Returns the number of frames that the motion has been inactive. In multiples of 4.
		HK_FORCE_INLINE int getNumInactiveFrames() const;

		/// return -1 if it is OK to deactivate, 0 else.
		HK_FORCE_INLINE hkUint32 isOkToDeactivateAsMask(const hknpMotionProperties& motionProperties) const;

		/// return whether or not deactivation is enabled for this motion.
		HK_FORCE_INLINE hkBool32 isDeactivationEnabled() const;


		/// the packed reference position for the deactivator.
		/// to deactivate deactivation, simply set m_numDeactivationChecks to 0xff
		HK_ALIGN16( hkUint32 ) m_referencePosition;

		/// the packed m_deactivationRefOrientation (stored in the motionState motion (so that sizeof(hkpEntity) < 512))
		hkUint32 m_referenceOrientation;

		/// m_accumVelocityDiff is used for the pathing accumulation
		hkInt16 m_accumVelocityDiff;

		/// Instead of a timer t, we use two 8-bit counters, m_frameCounter and m_numDeactivationChecks.
		/// m_frameCounter increases every frame and gets reset to 0 at every frequency pass,
		/// so it represents the number of frames that have passed since the start of the last frequency pass.
		hkUint8	m_frameCounter;

		/// m_numDeactivationChecks is the global counter for the whole deactivation period and beyond. Setting it to 0 resets deactivation.
		/// To support long deactivation periods, it only gets incremented every time a position/orientation/pathing check is performed.
		/// There are four such checks for each frequency pass so at the end of the deactivation period the following relationship holds:
		/// m_numDeactivationChecks = 4 * hknpMotionProperties::m_numDeactivationFrequencyPasses
		hkUint8	m_numDeactivationChecks;

		/// m_velocityScaleSquared is used for recording the maximum velocity. One velocity can be recording maxima while the
		/// other one is being checked (eg for spiking velocities) at the same time. Together they yield the maximum velocity
		/// over a frequency pass.
		hkUint8 m_velocityScaleSquared[2];

#ifdef DEACTIVATION_FREEFALL_DETECTION
		enum ActivationFlags
		{
			ACTIVATION_FLAG_FREEFALL = 1
		};

		hkFlags<ActivationFlags, hkUint8> m_activationFlags;
#endif


#ifdef DEACTIVATION_DEBUG_ACTIVATION_REASON
		enum ActivationReason
		{
			ACTIVATIONREASON_NOT_FULLY_TESTED,
			ACTIVATIONREASON_POSITION_CHANGED,
			ACTIVATIONREASON_ORIENTATION_CHANGED,
			ACTIVATIONREASON_MEDIUM_TO_HIGH_VELOCITY_WITH_UNCLASSIFIED_JITTER,
			ACTIVATIONREASON_SMOOTH_PATH,
			ACTIVATIONREASON_HIGH_VELOCITY_CONTINUOUSLY_SMOOTH_PATH,
			ACTIVATIONREASON_VELOCITY_SPIKE,
			ACTIVATIONREASON_NO_CONTACTS,
			ACTIVATIONREASON_ISLAND_ACTIVE
		};
		hkEnum<ActivationReason,hkUint8> m_activationReason;
		hkVector4 m_lastReferencePosition;
		hkVector4 m_lastCompressedPosition;
		hkQuaternion m_lastReferenceOrientation;
		hkQuaternion m_lastCompressedOrientation;
		hkVector4 m_previousLinearVelocity;
		hkVector4 m_accelerationDirection;
		hkUint8 m_compressedAccelerationDirection;
		hkInt8 m_accumAccelerationDiff;
#endif
};

#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationState.inl>


#endif // HKNP_DEACTIVATION_MOTION_STATE_H

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
