/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MIN_MAX_QUAD_TREE_H
#define HKNP_MIN_MAX_QUAD_TREE_H

#if defined( HK_PLATFORM_SPU )
#	include <Common/Base/Spu/Config/hkSpuConfig.h>
#	include <Common/Base/Memory/PlatformUtils/Spu/SpuDmaCache/hkSpu4WayCache.h>
extern class hkSpu4WayCache* g_SpuCollideUntypedCache;
#endif


/// A min-max tree that is used by the height field shape as a bounding volume hierarchy.
struct hknpMinMaxQuadTree
{
	public:

		/// Structure used to hold one level of the min-max tree.
		struct MinMaxLevel
		{
			public:

				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpMinMaxQuadTree::MinMaxLevel );
				HK_DECLARE_REFLECTION();

				/// Constructor.
				HK_FORCE_INLINE MinMaxLevel();

				/// Constructor.
				HK_FORCE_INLINE MinMaxLevel( const MinMaxLevel& l );

				/// Serialization constructor.
				HK_FORCE_INLINE MinMaxLevel( hkFinishLoadedObjectFlag f );

			public:

				/// Storage for the level. Min-max values are interleaved and stored for a quad (4 points) for each vector.
				/// See getMinMax for details.
				hkArray<hkUint32> m_minMaxData;

				/// Number of quads in the x direction.
				hkUint16 m_xRes;

				/// Number of quads in the z direction.
				hkUint16 m_zRes;
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMinMaxQuadTree );
		HK_DECLARE_REFLECTION();

		/// Empty constructor
		hknpMinMaxQuadTree() {}

		/// Serialization constructor
		hknpMinMaxQuadTree( hkFinishLoadedObjectFlag f ) : m_coarseTreeData(f) {}

		/// Set the size of the lowest level in the tree and min and max height values.
		void init( int sizeX, int sizeZ, hkSimdRealParameter minHeight, hkSimdRealParameter maxHeight );

		/// Set the min-max at the lowest level. Use updateRegion to update the higher levels of the tree or
		/// use buildTree to update the entire tree.
		HK_FORCE_INLINE void setMinMax( int x, int z, hkSimdRealParameter min, hkSimdRealParameter max );

		/// Used by the height field raycast. Returns 4 minimum and 4 maximum height values for 4 cells
		/// packed into the xyzw components of the vectors. The four cells have coordinates (2*x,2*z), (2*x+1,2*z),
		/// (2*x+1,2*z+1) and (2*x,2*z+1) in the grid with length 1<<level (two to the level'th power).
		HK_FORCE_INLINE void getMinMax( int level, int x, int z, hkVector4* HK_RESTRICT minOut, hkVector4* HK_RESTRICT maxOut ) const;

		/// Update part of the quadtree (after setting the lowest level with setMinMax).
		void updateRegion( int x0, int z0, int x1, int z1, hkReal* newMinOut, hkReal* newMaxOut );

		/// Build the entire tree. The lowest level should be set using setMinMax.
		HK_FORCE_INLINE void buildTree( hkReal* newMinOut, hkReal* newMaxOut );

	public:

		/// Storage for the levels of the min-max tree.
		hkArray<MinMaxLevel> m_coarseTreeData;

		/// Used when converting to and from internal uint16 representation and floats. Equal to min height of the tree.
		hkVector4 m_offset;

		/// Used when converting to and from internal uint16 representation and floats. Equal to max height - min height times 1/0xffff.
		hkReal m_multiplier;

		/// Reciprocal value of m_multiplier.
		hkReal m_invMultiplier;
};

#include <Physics/Physics/Collide/Shape/Composite/HeightField/hknpMinMaxQuadTree.inl>


#endif // HKNP_MIN_MAX_QUAD_TREE_H

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
