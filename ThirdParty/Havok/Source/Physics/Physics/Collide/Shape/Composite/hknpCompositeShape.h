/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COMPOSITE_SHAPE_H
#define HKNP_COMPOSITE_SHAPE_H

#include <Common/Base/Container/Array/hkFixedCapacityArray.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Physics/Physics/Collide/Shape/hknpShape.h>
#include <Physics/Physics/Collide/Shape/Composite/hknpSparseCompactMap.h>

extern const hkClass hknpCompositeShapeClass;

struct hknpAabbQuery;


/// The base class for all shapes that are built from a set of sub shapes.
class hknpCompositeShape : public hknpShape
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor.
		hknpCompositeShape();

#if !defined(HK_PLATFORM_SPU)

		/// Serialization constructor.
		HK_FORCE_INLINE hknpCompositeShape( hkFinishLoadedObjectFlag flag );

		/// hkReferencedObject implementation
		virtual const hkClass* getClassType() const HK_OVERRIDE;

#endif

		/// Destructor.
		HK_FORCE_INLINE virtual ~hknpCompositeShape() {}

		//
		// hknpShape implementation
		//

		/// Get all enabled shape keys.
		/// This implementation uses the shape key iterator to gather the keys.
#if !defined(HK_PLATFORM_SPU)
		virtual void getAllShapeKeys(
			const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,
			hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const HK_OVERRIDE;
#else
		virtual void getAllShapeKeys(
			const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,
			hkUint8* shapeBuffer, int shapeBufferSize,
			hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const HK_OVERRIDE;
#endif

		//
		// Edge welding markup
		//

		struct EdgeWeld
		{
			hknpShapeKey m_shapeKey;
			hkUint8 m_edges; // The bits determine which edges require welding. LSB is edge 0. Only 4 bits used.
		};

		/// Build the edge welding map.
		void buildEdgeWeldingMap( const EdgeWeld* entries, int numEntries );

		/// Return a bit pattern that determines which edges in the triangle should be considered internal for welding (use triangle normal)
		HK_FORCE_INLINE hkUint8 getEdgeWeldingInfo( hknpShapeKey key ) const;

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpCompositeShape, hknpShape );

	protected:

		hknpSparseCompactMap<hkUint16> m_edgeWeldingMap;

	public:

		/// Can be used by shape tag codecs to store information to help with shape tag decoding.
		/// Defaults to HKNP_INVALID_SHAPE_TAG_CODEC_INFO.
		hknpShapeTagCodecInfo m_shapeTagCodecInfo;
};

#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.inl>

#endif // HKNP_COMPOSITE_SHAPE_H

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
