/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONVEX_POLYTOPE_SHAPE_H
#define HKNP_CONVEX_POLYTOPE_SHAPE_H

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>

extern const hkClass hknpConvexPolytopeShapeClass;

/// Base size of a convex polytope shape, NOT including vertices, faces, etc.
#define HKNP_CONVEX_POLYTOPE_BASE_SIZE	HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeof(hknpConvexPolytopeShape))


/// A convex shape which stores face and plane information as well as vertices.
class hknpConvexPolytopeShape : public hknpConvexShape
{
	//+version(1)

	public:

		/// A face
		struct Face
		{
			HK_DECLARE_POD_TYPE();
			HK_DECLARE_REFLECTION();

			hkUint8		m_minHalfAngle;	///< Minimum half angle between this and neighboring faces. (255 = 90deg).
			hkUint8		m_numIndices;	///< Number of indices.
			hkUint16	m_firstIndex;	///< First vertex index (hknpConvexPolytopeShape::m_indices + m_firstIndex)
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Serialization constructor.
		hknpConvexPolytopeShape( hkFinishLoadedObjectFlag flag );

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::CONVEX_POLYTOPE; }
		virtual int calcSize() const HK_OVERRIDE;

		virtual			int		getNumberOfFaces() const HK_OVERRIDE { return m_faces.getSize(); }
		virtual			int		getFaceVertices( const int faceIndex, hkVector4& planeOut, hkcdVertex* vertexBufferOut ) const HK_OVERRIDE;
		HK_FORCE_INLINE void	getFaceInfo( const int index, hkVector4& planeOut, int& minAngleOut ) const;
		HK_FORCE_INLINE int		getSupportingFace( hkVector4Parameter direction, const hkcdGsk::Cache* gskCache, bool useB, hkVector4& planeOut, int& minAngleOut, hkUint32 &prevFaceId ) const;
		HK_FORCE_INLINE int		getSupportingFaceFromNormal( hkVector4Parameter normal, hkVector4& planeOut, int& minAngleOut, hkUint32 &prevFaceId ) const;
		virtual			hkReal	calcMinAngleBetweenFaces() const HK_OVERRIDE;

		virtual void getSignedDistances( const hknpShape::SdfQuery& query, SdfContactPoint* contactsOut ) const HK_OVERRIDE;

	#if !defined(HK_PLATFORM_SPU)
		virtual void checkConsistency() const HK_OVERRIDE;
	#endif

	public:

		// Plane accessors
		HK_FORCE_INLINE int					getNumPlanes() const		{ return m_planes.getSize(); }
		HK_FORCE_INLINE hkVector4*			getPlanes()					{ return m_planes.begin(); }
		HK_FORCE_INLINE const hkVector4*	getPlanes() const			{ return m_planes.begin(); }
		HK_FORCE_INLINE void				getPlanes( hkVector4* HK_RESTRICT planesOut ) const;
		HK_FORCE_INLINE const hkVector4&	getPlane( int index ) const	{ return getPlanes()[index]; }

		// Face accessors
		HK_FORCE_INLINE int					getNumFaces() const			{ return m_faces.getSize(); }
		HK_FORCE_INLINE Face*				getFaces()					{ return m_faces.begin(); }
		HK_FORCE_INLINE const Face*			getFaces() const			{ return m_faces.begin(); }
		HK_FORCE_INLINE const Face&			getFace( int index ) const	{ return m_faces[index]; }

		// Indices accessors
		HK_FORCE_INLINE int					getNumIndices() const		{ return m_indices.getSize(); }
		HK_FORCE_INLINE VertexIndex*		getIndices()				{ return m_indices.begin(); }
		HK_FORCE_INLINE const VertexIndex*	getIndices() const			{ return m_indices.begin(); }

		/// Returns the size of a convex polytope shape with the given characteristics.
		static HK_FORCE_INLINE int calcConvexPolytopeShapeSize(
			int numVertices, int numFaces, int numIndices, int sizeofBaseClass = HKNP_CONVEX_POLYTOPE_BASE_SIZE );

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpConvexPolytopeShape, hknpConvexShape );

		/// Protected constructor. Use the hknpConvexShape::createFromXxx() methods instead.
		HK_FORCE_INLINE hknpConvexPolytopeShape(
			int numVertices, int numFaces, int numIndices, hkReal radius, int sizeOfBaseClass = HKNP_CONVEX_POLYTOPE_BASE_SIZE );

		/// Initializes the convex polytope part of the shape. Required to construct shapes in place in SPU.
		HK_FORCE_INLINE void init(
			int numVertices, int numFaces, int numIndices, int sizeOfBaseClass  = HKNP_CONVEX_POLYTOPE_BASE_SIZE);

		/// Allocates space for a convex polytope shape with the given characteristics.
		static HK_FORCE_INLINE void* allocateConvexPolytopeShape(
			int numVertices, int numFaces, int numIndices, int sizeOfBaseClass, int& shapeSizeOut );

	private:

		// Copying is not allowed
		HK_FORCE_INLINE hknpConvexPolytopeShape( const hknpConvexPolytopeShape& );
		HK_FORCE_INLINE void operator=( const hknpConvexPolytopeShape& );

#if !defined(HK_PLATFORM_SPU)

		/// Construct a convex shape from a point cloud.
		static hknpConvexShape* HK_CALL	createFromVerticesInternal(
			const struct hkStridedVertices& vertices, hkReal radius = HKNP_SHAPE_DEFAULT_CONVEX_RADIUS,
			const BuildConfig& config = BuildConfig() );

#endif

	protected:

		hkRelArray<hkVector4>	m_planes;	///< Offset to planes stream.
		hkRelArray<Face>		m_faces;	///< Offset to faces stream.
		hkRelArray<VertexIndex>	m_indices;	///< Offset to indices stream.

		friend class hknpConvexShape;
		friend class hknpConvexShapeUtil;
};

#include <Physics/Physics/Collide/Shape/Convex/Polytope/hknpConvexPolytopeShape.inl>


#endif // HKNP_CONVEX_POLYTOPE_SHAPE_H

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
