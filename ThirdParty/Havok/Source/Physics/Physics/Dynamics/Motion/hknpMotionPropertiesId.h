/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MOTION_PROPERTIES_ID_H
#define HKNP_MOTION_PROPERTIES_ID_H

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/hkHandle.h>


/// Motion Properties identifier.
HK_DECLARE_HANDLE( hknpMotionPropertiesIdBase, hkUint16, 0xffff );
struct hknpMotionPropertiesId : public hknpMotionPropertiesIdBase
{
	/// Some preset IDs.
	enum Preset
	{
		INVALID = InvalidValue,

		STATIC = 0,		///< No velocity allowed
		DYNAMIC,		///< For regular dynamic bodies, undamped and gravity factor = 1
		KEYFRAMED,		///< like DYNAMIC, but gravity factor = 0
		FROZEN,			///< like KEYFRAMED, but lots of damping
		DEBRIS,			///< like DYNAMIC, but aggressive deactivation

		NUM_PRESETS
	};

	/// Empty constructor
	HK_FORCE_INLINE hknpMotionPropertiesId() {}

	/// Construct from int
	explicit HK_FORCE_INLINE hknpMotionPropertiesId(int i) { m_value = (Type)i; }

	/// Construct from enum
	HK_FORCE_INLINE hknpMotionPropertiesId(Preset p) { m_value = (Type)p; }

	/// Construct from base
	HK_FORCE_INLINE hknpMotionPropertiesId(const hknpMotionPropertiesIdBase& b) { m_value = b.valueUnchecked(); }
};


#endif	// HKNP_MOTION_PROPERTIES_ID_H

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
