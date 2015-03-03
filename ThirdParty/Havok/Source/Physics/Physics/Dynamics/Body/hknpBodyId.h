/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_BODY_ID_H
#define HKNP_BODY_ID_H

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/hkHandle.h>


/// Body identifier.
HK_DECLARE_HANDLE( hknpBodyIdBase, hkUint32, HK_INT32_MAX );	
struct hknpBodyId : public hknpBodyIdBase
{
	/// Some preset IDs.
	enum Preset
	{
		INVALID = InvalidValue,
		WORLD = 0,	///< Preset ID for the static world "body"
		NUM_PRESETS
	};

	/// Empty constructor
	HK_FORCE_INLINE hknpBodyId() {}

	explicit HK_FORCE_INLINE hknpBodyId(hkFinishLoadedObjectFlag flag) {}

	/// Construct from int
	explicit HK_FORCE_INLINE hknpBodyId(int i) { m_value = (Type)i; }

	/// Construct from enum
	HK_FORCE_INLINE hknpBodyId(Preset p) { m_value = (Type)p; }

	/// Construct from base
	HK_FORCE_INLINE hknpBodyId(const hknpBodyIdBase& b) { m_value = b.valueUnchecked(); }
};


/// A serializable, reference counted body ID (i.e. for content tools / Destruction)


class hknpBodyReference : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS);
		HK_DECLARE_REFLECTION();

	public:

		HK_FORCE_INLINE hknpBodyReference(hknpBodyId bodyId)				: hkReferencedObject(),		m_id(bodyId){}
		HK_FORCE_INLINE hknpBodyReference(hkFinishLoadedObjectFlag flag)	: hkReferencedObject(flag), m_id(flag)	{}

	public:

		/// The body ID to serialize
		hknpBodyId m_id;	//+overridetype(hkUint32) // default( hknpBodyId::invalid() )
};


/// Holds a pair of body IDs.
/// This structure is optimized for comparison.
struct hknpBodyIdPair
{
	HK_DECLARE_POD_TYPE();

	/// Compare two body pairs. Body B has higher priority.
	HK_FORCE_INLINE bool operator < ( const hknpBodyIdPair& other ) const
	{
		HK_COMPILE_TIME_ASSERT( sizeof(hkUint64) == 2*sizeof(hknpBodyId));
		return (*(const hkUint64*)this) < (*(const hkUint64*)&other);
	}

	HK_FORCE_INLINE bool operator == ( const hknpBodyIdPair& other ) const
	{
		HK_COMPILE_TIME_ASSERT( sizeof(hkUint64) == 2*sizeof(hknpBodyId));
		return (*(const hkUint64*)this) == (*(const hkUint64*)&other);
	}

#if HK_ENDIAN_BIG 	// PPC, ARM
	hknpBodyId m_bodyB;
	hknpBodyId m_bodyA;
#else				// INTEL
	hknpBodyId m_bodyA;
	hknpBodyId m_bodyB;
#endif
};


#endif	// HKNP_BODY_ID_H

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
