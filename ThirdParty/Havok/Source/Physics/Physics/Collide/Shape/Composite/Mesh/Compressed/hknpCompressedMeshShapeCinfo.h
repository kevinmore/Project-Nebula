/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COMPRESSED_MESH_SHAPE_CINFO_H
#define HKNP_COMPRESSED_MESH_SHAPE_CINFO_H

#include <Physics/Physics/Collide/Shape/Convex/Scaled/hknpScaledConvexShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpCompoundShape.h>

struct hkGeometry;
class hknpConvexShape;


/// Interface used to construct a hknpCompressedMeshShape.
class hknpCompressedMeshShapeCinfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_SHAPE, hknpCompressedMeshShapeCinfo );

		hknpCompressedMeshShapeCinfo();

		virtual ~hknpCompressedMeshShapeCinfo() {}

		//
		// Triangles interface
		//

		/// Return the total number of vertices in the geometry.
		virtual	int getNumVertices() const = 0;

		/// Return the total number of triangles in the geometry.
		virtual	int getNumTriangles() const = 0;

		/// Retrieve a vertex given its index.
		virtual void getVertex( int vertexIndex, hkVector4& vertexOut ) const = 0;

		/// Retrieve the three vertex indices of a triangle given its index.
		virtual void getIndices( int triangleIndex, int* indices ) const = 0;

		/// Return the shape tag for a given triangle index.
		virtual hknpShapeTag getTriangleShapeTag( int triangleIndex ) const { return static_cast<hknpShapeTag>(0); }

		//
		// Convex shapes interface
		//

		/// Return the total number of convex shapes.
		virtual int getNumConvexShapes() const = 0;

		/// Return a convex shape instance given its index.
		/// Note that if the shape convex radius if different to m_convexRadius, an extra storage cost of 2 bytes will occur.
		virtual const hknpShapeInstance& getConvexShape( int convexIndex ) const = 0;

		/// Return the shape tag for a given convex shape index.
		virtual hknpShapeTag getConvexShapeTag( int convexIndex ) const { return static_cast<hknpShapeTag>(0); }

	public:

		//
		// Inputs
		//

		/// Convex radius applied to each primitive, unless the primitive provides its own.
		/// Defaults to zero.
		hkReal m_convexRadius;

		/// If set, pairs of triangles are merged if:
		///		- they are in the same quad
		///		- they are coplanar (see m_mergeCoplanarTrianglesTolerance)
		///		- they share the same hknpShapeTag
		/// Defaults to true.
		hkBool m_mergeCoplanarTriangles;

		/// Tolerance used to detect flat quads.
		/// This is the distance from the last vertex to the plane formed by the other 3 vertices.
		/// Defaults to 0.01.
		hkReal m_mergeCoplanarTrianglesTolerance;

		/// Preserve vertex ordering in triangles.
		/// Enabling this may increase the size of the resulting mesh in memory.
		/// Defaults to false.
		hkBool m_preserveVertexOrder;

		/// If enabled, duplicate edges (i.e. their corresponding triangles) are internally flagged,
		/// this enables the hknpBodyQuality::ALLOW_CONCAVE_TRIANGLE_COLLISIONS optimization.
		/// Defaults to true.
		hkBool m_flagConcaveTriangles;

		/// Maximum allowed vertex position error for convex shape compression.
		/// Defaults to HK_REAL_MAX.
		hkReal m_maxConvexShapeError;

		/// If enabled, favor queries speed over memory consumption.
		/// Expected speedup: up to 3x faster.
		/// Expected memory increase: about 4x larger.
		/// Defaults to false.
		/// Note that on PlayStation(R)3 this optimization is only available through the PPU code path.
		hkBool m_optimizeForSpeed;

		//
		// Outputs
		//

		/// If not NULL, this will be filled with a mapping between the input triangle indices
		/// and the leaf shape keys that were generated.
		/// Defaults to NULL.
		hkArray<hknpShapeKey>* m_triangleIndexToShapeKeyMap;

		/// If not NULL, this will be filled with the vertex ordering for a given triangle index.
		/// This establish the following mapping between a compressed triangle vertices and the original one with:
		/// When ordering is 0, compressed triangle indices [0,1,2] maps to [0,1,2] in the input.
		/// When ordering is 1, compressed triangle indices [0,1,2] maps to [1,2,0] in the input.
		/// When ordering is 2, compressed triangle indices [0,1,2] maps to [2,0,1] in the input.
		/// More generally, a compressed vertex index 'I' with ordering 'O' maps to the '(O+I)%3' index in the input triangle.
		/// Note that the ordering value can only be 0, 1 or 2, thus can be stored in 2 bits.
		hkArray<hkUint8>* m_triangleIndexToVertexOrderMap;
};


/// Default implementation of hknpCompressedMeshShapeCinfo. Takes an hkGeometry as the input geometry and uses the
/// material in the each triangle as the shape tag. Allows as well to specify convex shapes instances that will be
/// stored compressed inside the shape.
class hknpDefaultCompressedMeshShapeCinfo : public hknpCompressedMeshShapeCinfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SHAPE, hknpDefaultCompressedMeshShapeCinfo);

		hknpDefaultCompressedMeshShapeCinfo(
			const hkGeometry* geometry, const hknpShapeInstance* convexShapes = HK_NULL, int numConvexShapes = 0 )
			: m_geometry(geometry), m_convexShapes(convexShapes), m_numConvexShapes(numConvexShapes)
		{}

		//
		// hknpCompressedMeshShapeCinfo interface implementation
		//

		virtual int getNumVertices() const;
		virtual int getNumTriangles() const;
		virtual void getVertex( int vi, hkVector4& vertexOut ) const;
		virtual void getIndices( int ti, int* indices ) const;
		virtual hknpShapeTag getTriangleShapeTag( int triangleIndex ) const;
		virtual int getNumConvexShapes() const;
		virtual const hknpShapeInstance& getConvexShape( int convexIndex ) const;
		virtual hknpShapeTag getConvexShapeTag( int convexIndex ) const;

	public:

		/// Input geometry
		const hkGeometry* m_geometry;

		/// Convex shape instances
		const hknpShapeInstance* m_convexShapes;
		int m_numConvexShapes;
};


#endif // HKNP_COMPRESSED_MESH_SHAPE_CINFO_H

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
