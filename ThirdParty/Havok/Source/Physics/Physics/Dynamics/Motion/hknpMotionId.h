/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MOTION_ID_H
#define HKNP_MOTION_ID_H

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/hkHandle.h>


/// Motion identifier.
HK_DECLARE_HANDLE( hknpMotionIdBase, hkUint32, HK_INT32_MAX );	
struct hknpMotionId : public hknpMotionIdBase
{
	/// Some preset IDs.
	enum Preset
	{
		INVALID = InvalidValue,
		STATIC = 0,		///< A special motion used by all static bodies
		NUM_PRESETS
	};

	/// Empty constructor
	HK_FORCE_INLINE hknpMotionId() {}

	/// Construct from int
	explicit HK_FORCE_INLINE hknpMotionId(int i) { m_value = (Type)i; }

	/// Construct from enum
	HK_FORCE_INLINE hknpMotionId(Preset p) { m_value = (Type)p; }

	/// Construct from base
	HK_FORCE_INLINE hknpMotionId(const hknpMotionIdBase& b) { m_value = b.valueUnchecked(); }

	/// Comparison operator
	HK_FORCE_INLINE bool operator ==(hknpMotionId other) const { return m_value == other.m_value; }

	/// Comparison operator
	HK_FORCE_INLINE bool operator !=(hknpMotionId other) const { return m_value != other.m_value; }
};


#endif	// HKNP_MOTION_ID_H

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
