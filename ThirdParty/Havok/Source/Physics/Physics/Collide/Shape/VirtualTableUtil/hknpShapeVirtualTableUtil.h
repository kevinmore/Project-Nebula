/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_VIRTUAL_TABLE_UTIL_H
#define HKNP_SHAPE_VIRTUAL_TABLE_UTIL_H

#define HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR(SHAPE_CLASS_NAME, BASE_CLASS_NAME)
#define HKNP_DECLARE_SHAPE_KEY_MASK_VTABLE_UTIL_CONSTRUCTOR(SHAPE_CLASS_NAME, BASE_CLASS_NAME)

#if defined(HK_PLATFORM_HAS_SPU)

#include <Physics/Physics/Collide/Shape/hknpShapeType.h>

/// Dummy type used to prevent accidental calls to the shape constructors created with HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR
struct hknpShapeVirtualTableUtilDummy {};

#if defined(HK_PLATFORM_PPU)
#	undef HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR
#	define HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR(SHAPE_CLASS_NAME, BASE_CLASS_NAME) \
		friend class hknpShapeVirtualTableUtil; \
		/* Constructor to be used only by hknpShapeVirtualTableUtil to obtain the v-table pointer */ \
		HK_FORCE_INLINE SHAPE_CLASS_NAME(hknpShapeVirtualTableUtilDummy dummy) : BASE_CLASS_NAME(dummy) {}
#endif

/// Dummy type used to prevent accidental calls to the shape constructors created with HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR
struct hknpShapeKeyMaskVirtualTableDummy {};

#	undef HKNP_DECLARE_SHAPE_KEY_MASK_VTABLE_UTIL_CONSTRUCTOR
#	define HKNP_DECLARE_SHAPE_KEY_MASK_VTABLE_UTIL_CONSTRUCTOR(SHAPE_CLASS_NAME, BASE_CLASS_NAME) \
		/* Constructor to be used only on SPU to obtain the v-table pointer */ \
		HK_FORCE_INLINE SHAPE_CLASS_NAME(hknpShapeKeyMaskVirtualTableDummy dummy) : BASE_CLASS_NAME(dummy) {}

class hknpShape;


/// Singleton class used in PlayStation(R)3 to patch the v-table pointers of shapes transfered to SPU.
/// DMAed shapes can be patched on SPU calling patchVirtualTable() as long as its shape type has been previously
/// registered on PPU and SPU via registerShape().
class hknpShapeVirtualTableUtil
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeVirtualTableUtil );

	public:

		/// Creates a shape of SHAPE_CLASS using the dummy constructor to obtain its v-table pointer and registers it
		/// for the SHAPE_TYPE. In SPU it should be called with partial shapes to prevent linking in unnecessary methods
		/// (see hknpNarrowPhaseJobPartialShapes.h and hknpSolverJobPartialShapes.h).
		template <typename SHAPE_CLASS, int SHAPE_TYPE>
		HK_FORCE_INLINE static void registerShape();

		/// Returns the array of local v-table pointers. One entry per shape type, up to hknpShapeType::NUM_SHAPE_TYPES.
		HK_FORCE_INLINE static void** getVTables();

#if defined(HK_PLATFORM_PPU)

		/// Calls registerShape() for all non-custom shapes.
		static void HK_CALL registerAllShapes();

#endif

#if defined(HK_PLATFORM_SPU)

		/// Sets the pointer to the PPU shape v-table pointers array.
		HK_FORCE_INLINE static void setPpuVTables(void*const* vTablesPpu);

		/// Patches the v-table pointer of the given shape with the v-table corresponding to its current PPU
		/// v-table pointer. Returns the dynamic type of the shape or hknpShapeType::INVALID in case of error.
		static int patchVirtualTable(hknpShape* shape);

		/// Patches the v-table pointer of the given shape with the v-table of the specified shape type.
		template <int SHAPE_TYPE>
		HK_FORCE_INLINE static void patchVirtualTableWithType(hknpShape* shape);

	#if defined(HK_PLATFORM_SIM)

		/// Clears the static array of v-tables on the SPU simulator.
		HK_FORCE_INLINE static void clearVTables();

	#endif

#endif // defined(HK_PLATFORM_SPU)

	protected:

		/// Local shape v-table pointers. One entry per shape type. Unregistered types default to NULL.
		static void* s_vTables[hknpShapeType::NUM_SHAPE_TYPES];

		/// PPU shape v-table pointers.
		HK_ON_SPU(static void*const* s_vTablesPpu);
};

#include <Physics/Physics/Collide/Shape/VirtualTableUtil/hknpShapeVirtualTableUtil.inl>

#endif // HK_PLATFORM_HAS_SPU

#endif // HKNP_SHAPE_VIRTUAL_TABLE_UTIL_H

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
