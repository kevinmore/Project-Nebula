/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SPARSE_COMPACT_MAP_H
#define HKNP_SPARSE_COMPACT_MAP_H

#include <Common/Base/Container/String/hkStringBuf.h>


namespace hknpSparseCompactMapUtil
{
	struct Entry
	{
		hkUint32 m_key;
		hkUint32 m_value;
		bool operator<(const Entry& rhs) const {return m_key<rhs.m_key;}
	};

	void sort( Entry* entries, int numEntries );
}

/// Used for compactly storing a constant sparse map (requires a rebuild if entries change)
/// Since it uses hkArray for storage it supports serialization.
/// Currently only used by composite shape for edge welding.
template <typename StoreT>
class hknpSparseCompactMap
{
public:
	typedef hknpSparseCompactMap<StoreT> ThisType;
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, ThisType );
	HK_DECLARE_REFLECTION();

	/// Serialization constructor.
	hknpSparseCompactMap( class hkFinishLoadedObjectFlag flag );

	hknpSparseCompactMap();

	/// Build the map.
	void buildMap(int keyBits, int primaryKeyBits, int valueBits, hknpSparseCompactMapUtil::Entry* entries, int numEntries);

	/// Lookup a value. Returns 0xffffffff if not found.
	hkUint32 HK_FORCE_INLINE lookup(hkUint32 key) const;

	hkUint32 m_secondaryKeyMask;
	hkUint32 m_sencondaryKeyBits;

	hkArray<hkUint16> m_primaryKeyToIndex;
	hkArray<StoreT> m_valueAndSecondaryKeys;
};

#include <Physics/Physics/Collide/Shape/Composite/hknpSparseCompactMap.inl>


#endif // HKNP_SPARSE_COMPACT_MAP_H

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
