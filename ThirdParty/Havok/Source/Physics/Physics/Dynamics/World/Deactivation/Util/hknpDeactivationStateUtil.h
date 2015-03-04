/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_DEACTIVATION_STATE_UTIL_H
#define HKNP_DEACTIVATION_STATE_UTIL_H

#include <Common/Base/Types/hkBaseTypes.h>

/// Util which helps deactivation to unpack/pack different attributes
class hknpDeactivationStateUtil
{
public:
	// Methods for packing/unpacking the position and orientation.

	static HK_FORCE_INLINE void storeReferencePosition(hkVector4Parameter refPosition, hkSimdRealParameter invBlockSize, hkUint32& compressedPosition);

	static HK_FORCE_INLINE void storeReferenceOrientation(const hkQuaternion& refOrientation, hkUint32& compressedOrientation);

	static HK_FORCE_INLINE void extractReferencePosition(hkVector4& refPositionOut, hkSimdRealParameter invBlockSize, hkUint32& compressedPosition);

	static HK_FORCE_INLINE void extractReferenceOrientation(hkQuaternion& refOrientationOut, hkUint32& compressedOrientation);

	static HK_FORCE_INLINE void positionAfterCompression(hkVector4Parameter positionIn, hkVector4& positionOut, hkSimdRealParameter invBlockSize);

	static HK_FORCE_INLINE void orientationAfterCompression(const hkQuaternion& orientationIn, hkQuaternion& orientationOut);

	// Methods to accumulate a velocity

	static HK_FORCE_INLINE void accumVelocityChange(hkSimdRealParameter velocityDiff, hkInt16& accumVelocityDiff);

	//methods to accumulate acceleration

	static HK_FORCE_INLINE void storeAccelerationDirection(hkVector4Parameter accelerationDir, hkUint8& accelerationDirection);

	static HK_FORCE_INLINE void extractAccelerationDirection(hkVector4& accelerationDirOut, hkUint8& accelerationDirection);

	static HK_FORCE_INLINE void accumAccelerationChange(hkSimdRealParameter accelerationDiff, hkInt8& accumAccelerationDiff);
};



#include <Physics/Physics/Dynamics/World/Deactivation/Util/hknpDeactivationStateUtil.inl>


#endif // HKNP_DEACTIVATION_STATE_UTIL_H

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
