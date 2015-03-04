/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpConfig.h>

template <typename SHAPE_CLASS, int SHAPE_TYPE>
HK_FORCE_INLINE void hknpShapeVirtualTableUtil::registerShape()
{
	hkUint8 buffer[HKNP_MAX_SHAPE_SIZE_ON_SPU];
	hknpShapeVirtualTableUtilDummy dummy;
	new (buffer) SHAPE_CLASS(dummy);
	s_vTables[SHAPE_TYPE] = *reinterpret_cast<void**>(buffer);
}

HK_FORCE_INLINE void** hknpShapeVirtualTableUtil::getVTables()
{
	return s_vTables;
}

#if defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE void hknpShapeVirtualTableUtil::setPpuVTables(void*const* vTablesPpu)
{
	s_vTablesPpu = vTablesPpu;
}

template <int SHAPE_TYPE>
HK_FORCE_INLINE void hknpShapeVirtualTableUtil::patchVirtualTableWithType(hknpShape* shape)
{
	HK_ASSERT2(0x6d27e54e, SHAPE_TYPE >= 0 && SHAPE_TYPE < hknpShapeType::NUM_SHAPE_TYPES, "Invalid shape type");
	HK_ASSERT2(0x48d1447a, s_vTables[SHAPE_TYPE], "NULL v-table pointer. You must call hknpShapeVirtualTableUtil::registerShape() for this shape type on SPU.");

	// Set the patched v-table pointer using the shape type
	void** vTablePtr = reinterpret_cast<void**>(shape);
	*vTablePtr = s_vTables[SHAPE_TYPE];
}

#if defined(HK_PLATFORM_SIM)

HK_FORCE_INLINE void hknpShapeVirtualTableUtil::clearVTables()
{
	for (int k = hknpShapeType::NUM_SHAPE_TYPES - 1; k >= 0; k--)
	{
		s_vTables[k] = HK_NULL;
	}
}

#endif

#endif // defined(HK_PLATFORM_SPU)

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
