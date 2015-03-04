/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_HEIGHT_FIELD_SHAPE_UTILS_H
#define HKNP_HEIGHT_FIELD_SHAPE_UTILS_H

#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>

struct hkcdRay;


// Utilities used by hknpHeightFieldShape.
namespace hknpHeightFieldShapeUtils
{
	//
	struct QuadTreeWalkerStackElement
	{
		hkAabb m_aabb;
		int m_level;
		int m_x;
		int m_z;
	};


	//
	template<int N>
	struct HeightCacheT
	{
		public:

			enum
			{
				Size      = (1<<N),
				IndexMask = (1<<N)-1,
			};

			struct CacheElement
			{
				int m_x;
				int m_z;
			};

		public:

			HK_FORCE_INLINE void clear();

			HK_FORCE_INLINE bool get(
				int x, int z,
				hkVector4 *heightOut, hknpShapeTag *shapeTagOut, hkBool32 *triangleFlipOut ) const;

			HK_FORCE_INLINE void put(
				int x, int z,
				hkVector4Parameter height, hknpShapeTag shapeTag, hkBool32 triangleFlip );

		public:

			CacheElement m_elements[Size][Size];
			hkVector4 m_heights[Size][Size];
			hknpShapeTag m_shapeTags[Size][Size];
			hkBool m_triangleFlips[Size][Size];
	};


	//
	struct NoCacheT
	{
		HK_FORCE_INLINE void clear();

		HK_FORCE_INLINE bool get(
			int x, int z,
			hkVector4 *heightOut, hknpShapeTag *shapeTagOut, hkBool32 *triangleFlipOut ) const;

		HK_FORCE_INLINE void put(
			int x, int z,
			hkVector4Parameter height, hknpShapeTag shapeTag, hkBool32 triangleFlip );
	};


	//
	template <typename T, typename CacheT = NoCacheT>
	struct QuadCollector
	{
		public:

			struct Quad
			{
				hkVector4		m_cornerHeights;
				hknpShapeTag	m_shapeTag;
				hkBool32		m_triangleFlip;
			};

		public:

			HK_FORCE_INLINE QuadCollector( const T* obj );

			HK_FORCE_INLINE void collect4Heights(
				int x, int z,
				hkVector4* heightOut, hknpShapeTag* shapeTagOut, hkBool32 *triangleFlipOut );

			HK_FORCE_INLINE void collect16Heights(
				int x, int z,
				hkVector4* heightsOut, hknpShapeTag* shapeTagOut, hkBool32 *triangleFlipOut );

		public:

			const T* m_obj;
			CacheT m_cache;

	};


	// Class used to evaluate heights and normals for a height field.
	template<typename T, typename CacheT = NoCacheT>
	struct TriangleEvaluator : public QuadCollector<T, CacheT>
	{
		public:

			HK_FORCE_INLINE TriangleEvaluator( const T* obj );

			HK_FORCE_INLINE void gatherHeightsAndInfo( int x, int z );

			HK_FORCE_INLINE hkVector4 evalHeightAndDiffHeight( hkVector4Parameter fracPos );

		public:

			hkVector4		m_heights;
			hknpShapeTag	m_shapeTag;
			hkBool32		m_triangleFlip;
	};


	HK_FORCE_INLINE static hkVector4 calcFractionXZ(
		int baseX, int baseZ,
		hkVector4Parameter hitPos );


	HK_FORCE_INLINE static hkVector4 calcFractionXZWithScale(
		int baseX, int baseZ,
		hkVector4Parameter hitPos, hkVector4Parameter baseScale, hkVector4Parameter invBaseScale );


	/// Used by height fields to descend the quadtree with a ray. Will only report nodes that
	/// hit the ray and in ray order.
	HK_FORCE_INLINE static void descendQuadTreeWithRay(
		QuadTreeWalkerStackElement* stack, int* stackSize,
		const hkAabb& aabb, hkVector4Parameter minY, hkVector4Parameter maxY,
		int x, int z, int nextlevel,
		const hkcdRay& ray, int rotOffset, hkSimdRealParameter earlyOutHitFraction,
		hkVector4ComparisonParameter mask0, hkVector4ComparisonParameter mask1,
		hkVector4ComparisonParameter mask2, hkVector4ComparisonParameter mask3 );


	/// Used by height fields to descend the quadtree with an aabb. Will only report nodes that
	/// hit the aabb.
	template<bool USE_NMP>
	HK_FORCE_INLINE static void descendQuadTreeWithAabb(
		QuadTreeWalkerStackElement* stack, int* stackSize,
		const hkAabb& aabb, hkVector4Parameter minY, hkVector4Parameter maxY,
		int x, int z, int nextlevel,
		const hkAabb& intAabb,
		hknpQueryAabbNmp* HK_RESTRICT nmpInOut);


	HK_FORCE_INLINE static void rayTriangleQuadCheckHelper(
		const hkcdRay& ray, hkAabb& aabb, hkVector4Parameter heights, hkVector4ComparisonParameter flipSelect,
		hkVector4Parameter floatToIntScale,hkcdRayQueryFlags::Enum flags, hkSimdRealParameter earlyOutFraction, hkSimdReal *earlyOutHitFractionOut0, hkSimdReal *earlyOutHitFractionOut1,
		hkVector4* normalOut0, hkInt32* hit0, hkVector4* normalOut1, hkInt32* hit1 );

	HK_FORCE_INLINE static hknpShapeKey shapeKeyFromCoordinates( int numBitsX, int x, int z, int triIndex );


	HK_FORCE_INLINE static void coordinatesFromShapeKey( hknpShapeKey key, int bitsPerX, int* xOut, int* zOut,
														 int* triIndexOut );

} //namespace hknpHeightFieldShapeUtils


#include <Physics/Physics/Collide/Shape/Composite/HeightField/hknpHeightFieldShapeUtils.inl>


#endif // HKNP_HEIGHT_FIELD_SHAPE_UTILS_H

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
