/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MATERIAL_ID_H
#define HKNP_MATERIAL_ID_H

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/hkHandle.h>


/// Material identifier.
HK_DECLARE_HANDLE( hknpMaterialIdBase, hkUint16, 0xffff );
struct hknpMaterialId : public hknpMaterialIdBase
{
	HK_DECLARE_POD_TYPE();

	/// Some preset IDs.
	enum Preset
	{
		INVALID = InvalidValue,

		/// Default material settings
		///		- dynamic friction = 0.5
		///		- static friction = 0.5
		///		- restitution = 0.0
		///		- no special behaviors
		DEFAULT = 0,

		NUM_PRESETS
	};

	/// Empty constructor
	HK_FORCE_INLINE hknpMaterialId() {}

	/// Construct from int
	explicit HK_FORCE_INLINE hknpMaterialId(int i) { m_value = (Type)i; }

	/// Construct from enum
	HK_FORCE_INLINE hknpMaterialId(Preset p) { m_value = (Type)p; }

	/// Construct from base
	HK_FORCE_INLINE hknpMaterialId(const hknpMaterialIdBase& b) { m_value = b.valueUnchecked(); }

	/// Comparison operator
	HK_FORCE_INLINE bool operator ==(hknpMaterialId other) const { return m_value == other.m_value; }

	/// Comparison operator
	HK_FORCE_INLINE bool operator !=(hknpMaterialId other) const { return m_value != other.m_value; }
};


#endif	// HKNP_MATERIAL_ID_H

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
